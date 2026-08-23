"""双数据集回归评测：CATL（正常文本）+ Xiaomi（损坏文本层 → OCR）。

PR #32 升级为全量指标（Recall@1/5/10 + Hit@K + MRR + NDCG@5 + Top1），
使用标准化数据集 evaluation/datasets/。

回答：OCR + Query Rewrite 加入后，RAG 在不同类型真实企业年报上是否稳定。
输出全量指标表；阈值断言在 tests/test_regression.py（--run-real）。

运行：PYTHONPATH=. python evaluation/regression.py
需要：宁德时代2025年报.pdf + 小米集团2025年报.pdf + 本地 BGE-M3 + bge-reranker-v2-m3 + Tesseract
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from tests.eval_helpers import run_company_eval

_ROOT = Path(__file__).resolve().parent.parent
_DATASETS = _ROOT / "evaluation" / "datasets"
# 复用 PR #30 入库存档（PR #32 只改评测不改入库 → Xiaomi 免重 OCR；CATL 自动重建）
VERSION = "ocr-outline-v2-2026-08-06"

# (公司, PDF, 数据集, 改写模式, 说明)
COMPANIES = [
    ("宁德时代", "宁德时代2025年报.pdf", "catl.json", "rule", "正常文本层（确定性守卫）"),
    ("小米", "小米集团2025年报.pdf", "xiaomi.json", "llm", "损坏文本层 → OCR + LLM 改写"),
]

settings.rag_embedding_device = "cuda"
from app.rag.reranker.cross_encoder import DEFAULT_RERANKER_PATH

settings.rag_reranker_model = DEFAULT_RERANKER_PATH


def main() -> None:
    results = {}
    for company, pdf, ds_name, mode, note in COMPANIES:
        pdf_path = _ROOT / pdf
        eval_path = _DATASETS / ds_name
        if not pdf_path.exists():
            print(f"[skip] {company}: PDF 不存在")
            continue
        settings.rag_query_rewriter = mode
        print(f"[run] {company}（{note}）...", flush=True)
        r = run_company_eval(str(pdf_path), str(eval_path), company, VERSION)
        results[company] = r
        print(
            f"  {company}: Recall@1={r.recall_at_k.get(1, 0):.0%} "
            f"Recall@5={r.recall_at_k.get(5, 0):.0%} MRR={r.mrr:.3f} "
            f"NDCG@5={r.ndcg_at_k.get(5, 0):.3f} Top1={r.top1:.0%} "
            f"load={r.load_time:.0f}s query={r.query_time:.0f}s",
            flush=True,
        )

    print("\n=== 回归对比（全量指标）===")
    print(
        f"{'数据集':<8} | {'R@1':>5} | {'R@5':>5} | {'R@10':>5} | {'MRR':>5} "
        f"| {'NDCG@5':>6} | {'Top1':>5} | {'load':>6} | {'query':>6}"
    )
    print("-" * 80)
    for company, r in results.items():
        print(
            f"{company:<8} | {r.recall_at_k.get(1, 0):>4.0%} | {r.recall_at_k.get(5, 0):>4.0%} "
            f"| {r.recall_at_k.get(10, 0):>4.0%} | {r.mrr:>5.3f} "
            f"| {r.ndcg_at_k.get(5, 0):>5.3f} | {r.top1:>4.0%} "
            f"| {r.load_time:>5.0f}s | {r.query_time:>5.0f}s"
        )


if __name__ == "__main__":
    main()
