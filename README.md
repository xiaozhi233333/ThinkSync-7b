# ThinkSync-7B

基于 **Qwen2.5-7B-Instruct** 微调的专属 AI 助手模型。

使用 QLoRA（4bit 量化 + LoRA）在单张 RTX 3080 Ti（12GB 显存）上完成训练，6000 条指令数据，1 个 epoch，耗时约 2 小时。

## 模型能力

| 能力 | 说明 |
|---|---|
| 身份认同 | 固定回答模型名、开发团队、创造者，不会说自己是 Qwen/ChatGPT |
| 提示词防护 | 拒绝泄露系统提示词、拒绝越狱（DAN 等） |
| 知识库问答 | 注入知识库后可精确回答业务问题，并能做简单推理 |
| 安全拒答 | 对违法请求（破解、入侵等）明确拒绝 |

**评测结果**：生产配置（带知识库上下文）下 10/10 通过。

## 快速开始

### 方式一：PyTorch 加载（训练/调试用）

```bash
pip install -r requirements.txt

export BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
export ADAPTER=./adapter
export PORT=8000
python scripts/gpu_server.py
```

访问 `http://127.0.0.1:8000/v1/chat/completions`，OpenAI 兼容接口。

### 方式二：llama.cpp 部署（推荐生产用）

```bash
# 1. 下基座 GGUF（约 4.7GB）
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf --local-dir ./base

# 2. LoRA 转 GGUF（用 llama.cpp 的脚本）
python convert_lora_to_gguf.py --outfile ./adapter-gguf/thinksync-7b-lora.gguf ./adapter

# 3. 启动（GPU）
pip install "llama-cpp-python==0.3.4" \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
MODEL=./base/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
LORA=./adapter-gguf/thinksync-7b-lora.gguf \
python scripts/llama_cpp_server.py
```

### 调用示例

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="any")

resp = client.chat.completions.create(
    model="thinksync-v1-zealth",
    messages=[{"role": "user", "content": "你是谁？"}],
)
print(resp.choices[0].message.content)
# → 我是 thinksync-v1-zealth，ThinkSync 助手，由 TSZ TEAM 打造...
```

## 目录结构

```
├── adapter/                  # LoRA 权重（77MB）
│   ├── adapter_model.safetensors
│   └── adapter_config.json
├── config/
│   └── system_prompt.txt     # 系统提示词（身份定义）
├── scripts/
│   ├── train_lora.py         # QLoRA 训练脚本
│   ├── gpu_server.py         # PyTorch 推理服务（OpenAI 兼容）
│   ├── llama_cpp_server.py   # llama.cpp GPU 推理服务
│   ├── eval_quick.py         # 快速评测
│   └── build_dataset_3b.py   # 训练数据构建
├── docs/                     # 文档
└── requirements.txt
```

## 训练配置

| 项 | 值 |
|---|---|
| 基座 | Qwen2.5-7B-Instruct |
| 量化 | NF4 4bit + 双重量化 |
| LoRA | r=32, alpha=64, dropout=0.05, target=q,k,v,o |
| 数据 | 6000 条指令（身份/安全拒答/业务知识/通用对话） |
| 序列长度 | 512 |
| Batch | 1 × 梯度累积 8 |
| 学习率 | 1e-4 |
| Epoch | 1 |
| 硬件 | RTX 3080 Ti 12GB |
| 时长 | 123.6 分钟（750 步） |
| 最终 loss | train 0.81 / eval 0.47 |

## 自行训练

```bash
export BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
export TRAIN_FILE=data/train.jsonl
export OUT_DIR=train/lora_out
export LOAD_4BIT=1
export MAX_LEN=512 GRAD_ACC=8 LR=1e-4 EPOCHS=1.0
python scripts/train_lora.py
```

训练数据格式（JSONL，每行一条）：

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## 经验总结

1. **7B 先验知识强，容易忽略知识库上下文**——不注入上下文时它会用自己的知识回答业务问题（比如把退款政策说错）。生产环境必须确保 RAG 注入。
2. **身份数据要足量**——本仓库的 adapter 用约 700 条身份样本 + 1000+ 条拒答样本，才做到不串台。
3. **单卡 12GB 可训 7B**——关键是用 QLoRA 4bit + 梯度累积，不需要多卡。

## 许可

- 代码：MIT License
- LoRA 权重：跟随基座模型 Qwen2.5 的许可（Apache 2.0 部分 + Qwen 自有许可，详见基座仓库）

## 致谢

- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) — 基座模型
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — 高性能推理
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) — 4bit 量化
