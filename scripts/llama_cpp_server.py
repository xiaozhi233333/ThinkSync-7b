# -*- coding: utf-8 -*-
"""GPU 推理后端（llama.cpp CUDA 版，通过 llama-cpp-python）

比 PyTorch 方案快得多：3B Q4_K_M + LoRA 全放显存，GTX 1060 上约 25~40 tok/s
（PyTorch bnb 4bit 只有 5 tok/s，CPU 版 llama.cpp 只有 1.5 tok/s）。

依赖（只需装一次）：
  pip install "llama-cpp-python==0.3.4" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
  pip install diskcache
CUDA 运行库：借用 torch 自带的（脚本会自动加入 DLL 搜索路径）；
  如果没装 torch，请另装 CUDA 12.4 runtime（cudart64_12.dll / cublas64_12.dll / cublasLt64_12.dll）。

用法：
  set MODEL=models\\qwen25-3b-gguf\\qwen2.5-3b-instruct-q4_k_m.gguf
  set LORA=models\\qwen25-3b-gguf\\thinksync-3b-lora.gguf
  set PORT=8090
  env\\Scripts\\python scripts\\llama_cpp_server.py

上层 FastAPI（app/server.py，含 RAG/兜底/鉴权适配）不用改，把 LLAMA_URL 指到本端口即可。
"""
import json
import os
import sys
import time

# ---- Windows：解决 DLL 加载问题 ----
# 1) 把 llama_cpp/lib（ggml*.dll）和 torch/lib（CUDA 运行库）加进 DLL 搜索路径
# 2) 先用 ctypes 预加载 llama.dll，否则 llama_cpp 内部用 RTLD_GLOBAL 加载会失败
if os.name == "nt":
    import ctypes
    import pathlib
    _here = pathlib.Path(__file__).resolve().parent.parent
    _lib = _here / "env" / "Lib" / "site-packages" / "llama_cpp" / "lib"
    _torch_lib = _here / "env" / "Lib" / "site-packages" / "torch" / "lib"
    for _d in (_lib, _torch_lib):
        if _d.is_dir():
            os.add_dll_directory(str(_d))
    if (_lib / "llama.dll").exists():
        ctypes.CDLL(str(_lib / "llama.dll"))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from llama_cpp import Llama

MODEL = os.environ.get(
    "MODEL", "models/qwen25-3b-gguf/qwen2.5-3b-instruct-q4_k_m.gguf")
LORA = os.environ.get("LORA", "models/qwen25-3b-gguf/thinksync-3b-lora.gguf")
PORT = int(os.environ.get("PORT", "8090"))
CTX = int(os.environ.get("CTX", "4096"))
THREADS = int(os.environ.get("THREADS", "8"))
GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "99"))
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "512"))
NAME = os.environ.get("MODEL_NAME", "thinksync-v1-zealth")

app = FastAPI()

print(f"== 加载模型（GPU 层数 {GPU_LAYERS}）==")
_kw = dict(model_path=MODEL, n_gpu_layers=GPU_LAYERS, n_ctx=CTX,
           n_threads=THREADS, n_threads_batch=THREADS,
           n_batch=int(os.environ.get("N_BATCH", "512")), n_ubatch=int(os.environ.get("N_UBATCH", "128")), verbose=False)
if LORA and os.path.exists(LORA):
    _kw["lora_path"] = LORA
    print(f"   LoRA: {LORA}")
llm = Llama(**_kw)
print("== 就绪 ==")


@app.get("/health")
def health():
    return {"status": "ok", "backend": "llama.cpp-cuda", "gpu_layers": GPU_LAYERS}


@app.post("/v1/chat/completions")
async def chat(body: dict):
    messages = body.get("messages") or []
    max_new = min(int(body.get("max_tokens") or MAX_NEW), 2048)
    temperature = float(body.get("temperature", 0.3))
    top_p = float(body.get("top_p", 0.9))
    stream = bool(body.get("stream"))
    kw = dict(messages=messages, max_tokens=max_new, temperature=temperature, top_p=top_p)

    if not stream:
        t0 = time.time()
        out = llm.create_chat_completion(**kw)
        u = out.get("usage", {})
        return JSONResponse({
            "id": out.get("id", "chatcmpl-gpu"), "object": "chat.completion",
            "created": int(time.time()), "model": NAME,
            "choices": out["choices"],
            "usage": {"prompt_tokens": u.get("prompt_tokens", 0),
                      "completion_tokens": u.get("completion_tokens", 0),
                      "total_tokens": u.get("total_tokens", 0)},
            "meta": {"ms": int((time.time() - t0) * 1000)},
        })

    def gen():
        for chunk in llm.create_chat_completion(**kw, stream=True):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning", workers=1)
