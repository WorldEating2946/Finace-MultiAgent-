"""真实宁德时代年报评测（real 标记，默认跳过）。

流程：load → split → BGE embed → Hybrid + CrossEncoder → retrieve，
记录 Recall@5 / MRR / Top1（expected_sections 命中任意）。
存档复用：二次运行跳过重 embed（仅查询）。

运行：pytest --run-real tests/test_real_catl_eval.py -s
"""

from pathlib import Path

import pytest

from tests.eval_helpers import run_company_eval

_ROOT = Path(__file__).resolve().parent.parent
_PDF = _ROOT / "宁德时代2025年报.pdf"
_EVAL = _ROOT / "evaluation" / "catl_2025.json"
_PIPELINE_VERSION = "ocr-outline-2026-08-05"  # PR #30（PDF outline 优先 TOC）→ 强制重入库


@pytest.mark.real
def test_real_catl_recall_and_mrr(xiaomi_pdf_path):
    """真实 CATL 年报：Recall@5 / MRR / Top1（expected_sections 多期望章节）。

    显式 rule 模式（防小米测试的 llm 模式泄漏）；CATL 查询精确，不做 LLM 改写
    （确定性回归守卫）。实测 100% / 0.950 / 90%。
    """
    from app.core.config import settings

    settings.rag_query_rewriter = "rule"
    r = run_company_eval(
        pdf_path=str(_PDF),
        eval_path=str(_EVAL),
        company="宁德时代",
        pipeline_version=_PIPELINE_VERSION,
    )
    print(f"\n[EVAL] Recall@5={r.recall5:.0%}  MRR={r.mrr:.3f}  Top1={r.top1:.0%}  (n={r.n})  load={r.load_time:.0f}s  query={r.query_time:.0f}s")

    assert r.recall5 >= 0.5, f"Recall@5 过低: {r.recall5:.0%}"
    assert r.top1 >= 0.3, f"Top1 过低: {r.top1:.0%}"
