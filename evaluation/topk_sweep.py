"""Top-K 参数扫描：Retrieve 20/30/50/100 → Rerank 5，对比 Recall@5 / MRR / Top1 / 耗时。

运行：PYTHONPATH=. python evaluation/topk_sweep.py
需要：宁德时代2025年报.pdf + 本地 BGE-M3 + bge-reranker-v2-m3

目标：找到 召回候选数 与 精排质量 的平衡点。
"""

from __future__ import annotations

import time
from pathlib import Path

from app.core.config import settings
from app.rag import retriever
from tests.eval_helpers import run_company_eval

_ROOT = Path(__file__).resolve().parent.parent
PDF = _ROOT / "宁德时代2025年报.pdf"
EVAL = _ROOT / "evaluation" / "catl_2025.json"
COMPANY = "宁德时代"
VERSION = "rerank-2026-08-05"

settings.rag_embedding_device = "cuda"
from app.rag.reranker.cross_encoder import DEFAULT_RERANKER_PATH

settings.rag_reranker_model = DEFAULT_RERANKER_PATH

# 首次运行建存档（GPU ~2min），之后各参数复用（快）
run_company_eval(str(PDF), str(EVAL), COMPANY, VERSION)

print(f"\n{'Retrieve':>10} | {'Recall@5':>8} | {'MRR':>5} | {'Top1':>5} | {'耗时':>6}")
print("-" * 55)
for fetch_k in (20, 30, 50, 100):
    retriever._FETCH_K = fetch_k
    settings.rag_retrieve_top_k = fetch_k
    t0 = time.time()
    top1, mrr, recall5, n = run_company_eval(str(PDF), str(EVAL), COMPANY, VERSION)
    elapsed = time.time() - t0
    print(f"{fetch_k:>10} | {recall5:>7.0%} | {mrr:>5.3f} | {top1:>4.0%} | {elapsed:>5.0f}s")
