"""真实小米年报评测（real 标记，默认跳过）。

小米 PDF 文本层损坏（ToUnicode 乱码，质量分 43% < 80% 门禁）时自动跳过；
评测集与指标逻辑已就绪，待获取干净版 PDF（或 OCR）后启用。

运行：pytest --run-real tests/test_real_xiaomi_eval.py -s
"""

from pathlib import Path

import pytest

from tests.eval_helpers import run_company_eval

_ROOT = Path(__file__).resolve().parent.parent
_PDF = _ROOT / "小米集团2025年报.pdf"
_EVAL = _ROOT / "evaluation" / "xiaomi_2025.json"
# 版本不变 → 复用缓存库（Query Rewrite 不改入库，无需重 embed）
_PIPELINE_VERSION = "ocr-outline-v2-2026-08-06"


@pytest.mark.real
def test_real_xiaomi_recall_and_mrr(xiaomi_pdf_path):
    """真实小米年报：LLM Query Rewrite 下 Recall@5 / MRR / Top1。

    实测（deepseek-v4-flash，2026-08-06）：Recall@5=80% MRR=0.310 Top1=10%
    （vs 基线 70% / 0.265 / 10%）。Top1 天花板来自 OCR 损坏的 MD&A 章节 + 评测数据
    期望的"未来展望"章节不存在，见 evaluation/RESULTS.md。
    """
    from app.core.config import settings

    settings.rag_query_rewriter = "llm"  # PR #31 目标路径（LLM 改写）
    r = run_company_eval(
        pdf_path=str(_PDF),
        eval_path=str(_EVAL),
        company="小米",
        pipeline_version=_PIPELINE_VERSION,
    )
    print(f"\n[EVAL] Recall@5={r.recall5:.0%}  MRR={r.mrr:.3f}  Top1={r.top1:.0%}  (n={r.n})  load={r.load_time:.0f}s  query={r.query_time:.0f}s")

    assert r.recall5 >= 0.5
    assert r.mrr >= 0.2  # rank 质量（0.310 实测）
    assert r.top1 >= 0.1  # 回归地板（Top1 结构化天花板 10%）
