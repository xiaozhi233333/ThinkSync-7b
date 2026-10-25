# -*- coding: utf-8 -*-
"""为 Qwen2.5-3B 构建精选训练集（CPU/GPU 都吃不动全量 6897 条，按重要性分层抽样）

思路：3B 的通用能力已经不错，微调重点是"身份 + 业务口径 + 安全边界 + 回复风格"，
通用能力只需少量回放防遗忘。总量控制在 2400 条左右，1.5 epoch ≈ 450 步。
"""
import json
import random
from collections import Counter

random.seed(20261023)

# 各类样本的抽样上限（None = 全要）
QUOTA = {
    "identity": 700,
    "plan_fact": None,
    "rag_answer": None,
    "rag_refusal": None,
    "no_ctx_refusal": None,
    "single_turn": None,
    "safety": None,
    "format": None,
    "multiturn": None,
    "bare_refusal": 120,
    "bare_biz_qa": None,
    "jailbreak": 96,
    "knowledge": 78,
    "exact_fact": 110,
    "must_general": 88,
    "gpt_diff": 14,
    "general_turn": 138,
    "general_plain": 320,
    "general_rag": 200,
}

buckets = {}
seen = set()
for path in ["data/train.jsonl", "data/stage3.jsonl", "data/stage4.jsonl"]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            msgs = r["messages"]
            key = (msgs[1]["content"], msgs[2]["content"])
            if key in seen:          # 跨文件去重（stage3/4 含 stage2 的回放样本）
                continue
            seen.add(key)
            buckets.setdefault(r.get("_src", "unknown"), []).append(r)

out = []
for src, rows in buckets.items():
    cap = QUOTA.get(src, 100)
    random.shuffle(rows)
    take = rows if cap is None else rows[:cap]
    out.extend(take)
    print(f"  {src:16s} 原有 {len(rows):5d} → 取 {len(take):4d}")

random.shuffle(out)
with open("data/train_3b.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n3B 训练集：{len(out)} 条")
c = Counter(r["_src"] for r in out)
print("构成：", " ".join(f"{k}:{v}" for k, v in c.most_common()))
