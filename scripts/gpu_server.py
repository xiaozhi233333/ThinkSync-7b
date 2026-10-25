# -*- coding: utf-8 -*-
"""GPU 推理后端（PyTorch + bitsandbytes 4bit）

为什么不用 llama.cpp：本机（GTX 1060 / Pascal）拿到的 llama.cpp 是纯 CPU 构建，
GitHub 被墙下载不了 CUDA 版。而 PyTorch + bnb 4bit 在本机验证过可用（训练就在用），
3B 4bit 只占 1.6G 显存，实测速度远高于 CPU。

接口与 llama-server 完全一致（OpenAI 兼容），所以上层的 FastAPI（RAG/兜底/鉴权适配）
不需要任何改动，只要把 LLAMA_URL 指到本服务的端口。

用法：
  set BASE_MODEL=E:/models/qwen3b
  set ADAPTER=train/lora_3b_fix3
  set PORT=8090
  env\\Scripts\\python scripts\\gpu_server.py
"""
import json
import os
import queue
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TextIteratorStreamer

BASE = os.environ.get("BASE_MODEL", "E:/models/qwen3b")
ADAPTER = os.environ.get("ADAPTER", "train/lora_3b_fix3")
PORT = int(os.environ.get("PORT", "8090"))
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "512"))

# 身份系统提示词：调用方没传 system 时自动注入
# 优先读 config/system_prompt.txt，读不到用内置精简版
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "")
if not SYSTEM_PROMPT:
    _sp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "system_prompt.txt")
    try:
        with open(_sp, encoding="utf-8") as _f:
            SYSTEM_PROMPT = _f.read().strip()
    except Exception:
        SYSTEM_PROMPT = (
            "你是 ThinkSync 助手，模型名 thinksync-v1-zealth，由 TSZ TEAM 打造。"
            "爸爸是小志，义父是小苏。涉及身份的问题一律按此回答，"
            "不要自称 Qwen、ChatGPT、Claude 等其他模型，也不要提及阿里云、OpenAI 等公司。"
        )

app = FastAPI()

print("== 加载分词器 ==")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

print("== 加载基座（4bit NF4，fp32 计算——Pascal 的 fp16 算力只有 1/64）==")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16 if os.environ.get("COMPUTE_FP16","0")=="1" else torch.float32,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb_cfg, device_map={"": 0}, trust_remote_code=True)

print(f"== 挂载 LoRA: {ADAPTER} ==")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
if torch.cuda.is_available():
    print(f"== 就绪 | GPU: {torch.cuda.get_device_name(0)} | "
          f"显存 {torch.cuda.memory_allocated()/2**20:.0f}MB ==")


def _inject_system(messages):
    """如果调用方没带 system 消息，自动注入身份提示词（否则 7B 会回答"我是Qwen"）"""
    if not any(m.get("role") == "system" for m in messages):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)
    return messages


def _generate(messages, max_new, temperature, top_p):
    """返回 (text, prompt_tokens, completion_tokens)"""
    messages = _inject_system(messages)
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    n_prompt = inputs["input_ids"].shape[1]
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new,
        do_sample=temperature > 0.01,
        temperature=max(temperature, 0.01),
        top_p=top_p,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    new = out[0][n_prompt:]
    return tok.decode(new, skip_special_tokens=True), n_prompt, int(new.shape[0])


@app.get("/health")
def health():
    return {"status": "ok", "backend": "pytorch-gpu",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


@app.post("/v1/chat/completions")
async def chat(body: dict):
    messages = body.get("messages") or []
    max_new = min(int(body.get("max_tokens") or MAX_NEW), 1024)
    temperature = float(body.get("temperature", 0.3))
    top_p = float(body.get("top_p", 0.9))
    stream = bool(body.get("stream"))

    if not stream:
        t0 = time.time()
        text, n_p, n_c = _generate(messages, max_new, temperature, top_p)
        return JSONResponse({
            "id": "chatcmpl-gpu", "object": "chat.completion",
            "created": int(time.time()), "model": os.environ.get("MODEL_NAME", "thinksync-v1-zealth"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": n_p, "completion_tokens": n_c,
                      "total_tokens": n_p + n_c},
            "meta": {"ms": int((time.time() - t0) * 1000)},
        })

    # ---- 流式 ----
    messages = _inject_system(messages)
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    n_prompt = inputs["input_ids"].shape[1]
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True, skip_prompt=True)
    gen_kwargs = dict(
        **inputs, max_new_tokens=max_new,
        do_sample=temperature > 0.01,
        temperature=max(temperature, 0.01), top_p=top_p,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        streamer=streamer,
    )
    th = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    th.start()

    async def gen():
        q: "queue.Queue" = queue.Queue()

        def pump():
            try:
                for piece in streamer:
                    q.put(piece)
            except Exception as e:                      # noqa: BLE001
                q.put(("__err__", str(e)))
            finally:
                q.put(None)

        threading.Thread(target=pump, daemon=True).start()
        n_out = 0
        while True:
            item = await _aget(q)
            if item is None:
                break
            if isinstance(item, tuple):
                yield f"data: {json.dumps({'error': item[1]})}\n\n".encode()
                break
            n_out += len(item)
            chunk = {"id": "chatcmpl-gpu", "object": "chat.completion.chunk",
                     "created": int(time.time()),
                     "model": os.environ.get("MODEL_NAME", "thinksync-v1-zealth"),
                     "choices": [{"index": 0, "delta": {"content": item},
                                  "finish_reason": None}]}
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
        end = {"id": "chatcmpl-gpu", "object": "chat.completion.chunk",
               "created": int(time.time()),
               "model": os.environ.get("MODEL_NAME", "thinksync-v1-zealth"),
               "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(end, ensure_ascii=False)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        th.join(timeout=5)

    return StreamingResponse(gen(), media_type="text/event-stream")


async def _aget(q: "queue.Queue"):
    """把阻塞队列变成异步可等待"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, q.get)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning", workers=1)
