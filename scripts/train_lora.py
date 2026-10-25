# -*- coding: utf-8 -*-
"""
Qwen3-0.6B LoRA 微调（面向 GTX 1060 / Pascal 架构优化）

关键点
------
1. Pascal(sm_61) 的 fp16 算力只有 fp32 的 1/64，因此**关闭 autocast**，全程 fp32。
2. 只训练 LoRA 参数，基座权重冻结，显存占用 ≈ 权重 2.4G + 激活 1~1.5G。
3. 逐段 tokenize 构造 labels，只对 assistant 段算 loss（含 <think></think> 头，保证跨推理端一致）。
4. 梯度检查点 + 梯度累积，让 6G 显存也能跑 1024 长度。
"""
import json
import math
import os
import time

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments, default_data_collator)

BASE = os.environ.get("BASE_MODEL", "models/Qwen3-0.6B")
OUT = os.environ.get("OUT_DIR", "train/lora_out")
MAX_LEN = int(os.environ.get("MAX_LEN", "768"))
BS = int(os.environ.get("BS", "2"))
GRAD_ACC = int(os.environ.get("GRAD_ACC", "8"))
LR = float(os.environ.get("LR", "1.2e-4"))
EPOCHS = float(os.environ.get("EPOCHS", "2"))
LORA_R = int(os.environ.get("LORA_R", "32"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "300"))
LOG_STEPS = int(os.environ.get("LOG_STEPS", "10"))
RESUME = os.environ.get("RESUME_FROM_CHECKPOINT", "0") == "1"

# 精度：Pascal(1060) 必须 float32；A100/H100/4090 用 bfloat16 又快又省显存
_DTYPES = {"float32": torch.float32, "fp32": torch.float32,
           "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
           "float16": torch.float16, "fp16": torch.float16}
DTYPE = _DTYPES[os.environ.get("DTYPE", "float32").lower()]

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 设备：cuda（默认）或 cpu。3B 在 6G 显存放不下，CPU + 32G 内存是唯一可行路径
USE_CPU = os.environ.get("DEVICE", "cuda").lower() == "cpu"
if USE_CPU:
    _th = int(os.environ.get("CPU_THREADS", "0")) or (os.cpu_count() or 8)
    torch.set_num_threads(_th)
    print(f"[cpu] 使用 {_th} 个线程训练")

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, padding_side="right")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Qwen3 需要空的 think 块；Qwen2.5 没有 think 机制，THINK_HEAD=0 关闭
THINK_HEAD = os.environ.get("THINK_HEAD", "1") == "1"
HEAD = "<|im_start|>assistant\n<think>\n\n</think>\n\n" if THINK_HEAD else "<|im_start|>assistant\n"
TAIL = "<|im_end|>\n"


class SFTDataset(Dataset):
    def __init__(self, path):
        self.rows = []
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                msgs = json.loads(line)["messages"]
                item = self._encode(msgs)
                if item is None:
                    skipped += 1
                    continue
                self.rows.append(item)
        self.lens = [len(r["input_ids"]) for r in self.rows]
        print(f"  {path}: 可用 {len(self.rows)} 条，跳过 {skipped} 条，"
              f"平均长度 {sum(self.lens)/max(1,len(self.lens)):.0f}")

    def _encode(self, messages):
        ids, labels = [], []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "assistant":
                h = tok(HEAD, add_special_tokens=False)["input_ids"]
                b = tok(content, add_special_tokens=False)["input_ids"]
                t = tok(TAIL, add_special_tokens=False)["input_ids"]
                ids += h + b + t
                labels += h + b + t          # 头部也计入 loss，保证任何推理端都能接上
            else:
                seg = tok(f"<|im_start|>{role}\n{content}<|im_end|>\n",
                          add_special_tokens=False)["input_ids"]
                ids += seg
                labels += [-100] * len(seg)
        if len(ids) < 12:
            return None
        # 超长样本：保留开头 1/4 + 结尾 3/4，保证问题与答案都还在
        if len(ids) > MAX_LEN:
            keep_head = MAX_LEN // 4
            ids = ids[:keep_head] + ids[-(MAX_LEN - keep_head):]
            labels = labels[:keep_head] + labels[-(MAX_LEN - keep_head):]
        return {"input_ids": ids, "labels": labels}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    """动态 padding 到 8 的倍数，减少碎片"""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        maxlen = min(MAX_LEN, ((maxlen + 7) // 8) * 8)
        input_ids, labels, attn = [], [], []
        for b in batch:
            ids = b["input_ids"][:maxlen]
            lb = b["labels"][:maxlen]
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lb + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class BucketSampler(torch.utils.data.Sampler):
    """按长度分桶采样：同一 batch 内长度接近，padding 浪费最小
    做法是"先随机打乱，再在每 100 条的窗口内按长度排序"，兼顾随机性与局部性。
    """

    def __init__(self, lengths, batch_size, window=100, seed=42):
        self.lengths = lengths
        self.bs = batch_size
        self.window = max(window, batch_size * 20)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.lengths), generator=g).tolist()
        out = []
        for i in range(0, len(order), self.window):
            part = order[i:i + self.window]
            part.sort(key=lambda j: self.lengths[j])
            out.extend(part)
        return iter(out)


class BucketTrainer(Trainer):
    def __init__(self, *a, sampler=None, **kw):
        super().__init__(*a, **kw)
        self._sampler = sampler

    def _get_train_sampler(self, *a, **kw):
        return self._sampler


def main():
    smoke = os.environ.get("SMOKE") == "1"
    print("== 加载分词器与数据 ==")
    train_ds = SFTDataset(os.environ.get("TRAIN_FILE", "data/train.jsonl"))
    val_ds = SFTDataset(os.environ.get("VAL_FILE", "data/val.jsonl"))
    if smoke:
        train_ds.rows = train_ds.rows[:200]
        train_ds.lens = train_ds.lens[:200]
        val_ds.rows = val_ds.rows[:16]
        val_ds.lens = val_ds.lens[:16]
        print("  [冒烟模式] 仅用 200 条训练 / 16 条验证")

    print(f"== 加载基座 {BASE}（{DTYPE}）==")
    LOAD_4BIT = os.environ.get("LOAD_4BIT", "0") == "1"
    if LOAD_4BIT:
        # QLoRA：基座 4bit 量化后放显存，只有 LoRA 参数参与训练。
        # 3B 在 6G 显存上的唯一可行路径（fp32 需要 12G、fp16 需要 6G 且 Pascal fp16 极慢）。
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,   # Pascal 的 fp16 算力是 fp32 的 1/64，用 fp32 算
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE, quantization_config=bnb_cfg, attn_implementation="sdpa",
            trust_remote_code=True, device_map={"": 0},
        )
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        print("[4bit] 基座已按 NF4 量化加载")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE, torch_dtype=DTYPE, attn_implementation="sdpa", trust_remote_code=True
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    from peft import LoraConfig, PeftModel, get_peft_model
    resume = os.environ.get("RESUME_ADAPTER", "")
    if resume and os.path.isdir(resume):
        print(f"== 从已有 adapter 继续训练（增量）: {resume} ==")
        model = PeftModel.from_pretrained(model, resume, is_trainable=True)
    else:
        # 默认挂全部 7 个投影层；显存吃紧时可只挂注意力层（q/k/v/o），
        # 省掉 MLP 的反向计算与激活，显存约降 1G、速度约快 30%
        _MAP = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj",
                "gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
        _t = os.environ.get("LORA_TARGETS", "q,k,v,o,gate,up,down").split(",")
        targets = [_MAP[t.strip()] for t in _t if t.strip() in _MAP]
        print(f"[lora] target_modules = {targets}")
        lcfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_R * 2, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUT,
        per_device_train_batch_size=BS,
        gradient_accumulation_steps=GRAD_ACC,
        per_device_eval_batch_size=1,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=LOG_STEPS,
        save_total_limit=1,
        eval_strategy="steps",
        fp16=False,
        bf16=False,
        use_cpu=USE_CPU,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        dataloader_num_workers=int(os.environ.get("DL_WORKERS", "2")),
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=12 if smoke else -1,
        save_steps=99999 if smoke else SAVE_STEPS,
        eval_steps=99999 if smoke else SAVE_STEPS,
    )

    sampler = BucketSampler(train_ds.lens, BS)
    trainer = BucketTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=Collator(tok.pad_token_id),
        sampler=sampler,
    )

    print(f"== 开始训练 | 样本 {len(train_ds)} | 等效 batch {BS * GRAD_ACC} "
          f"| 预计优化步数 {math.ceil(len(train_ds) / (BS * GRAD_ACC) * EPOCHS)} ==")
    t0 = time.time()
    # RESUME_FROM_CHECKPOINT=1 时从 OUT 目录里最新的 checkpoint 续训
    trainer.train(resume_from_checkpoint=True if RESUME else None)
    print(f"== 训练完成，耗时 {(time.time() - t0) / 60:.1f} 分钟 ==")
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print("LoRA 已保存到", OUT)


if __name__ == "__main__":
    main()
