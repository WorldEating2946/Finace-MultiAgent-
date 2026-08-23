"""评测体系测试（PR #32）。

两部分：
1. 单元测试（默认跑）：metrics 正确性 / 数据集加载 / RagEvaluator（dummy store）；
2. 回归评测（@pytest.mark.real，--run-real）：双数据集阈值断言——任何 commit
   不得让 CATL Recall@5 < 100% 或 Xiaomi Recall@5 < 80%。

阈值来源：evaluation/RESULTS.md 最新实测（deepseek-v4-flash，2026-08-06）。
"""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_DATASETS = _ROOT / "evaluation" / "datasets"

# 回归阈值（从 RESULTS.md 实测固化；低于此即判回归）。
# PR #33：生产用 metadata 模式（per-company 权重：小米 metadata 增强、CATL 纯 CE 直通）。
# 查询改写用 DETERMINISTIC 的 rule 模式（LLM 改写随机，不做硬门禁）。
# 实测（metadata 模式，2026-08-06）：Xiaomi R@5=80% MRR=0.423 NDCG@5=0.474；
# CATL 纯 CE 直通 R@5=100% MRR=0.950。
_XIAOMI_RECALL5_FLOOR = 0.75  # metadata 实测 80%（留缓冲）
_XIAOMI_MRR_FLOOR = 0.30      # metadata 实测 0.423
_XIAOMI_NDCG5_FLOOR = 0.35    # metadata 实测 0.474
_CATL_RECALL5_FLOOR = 1.0     # 实测 100%
_CATL_MRR_FLOOR = 0.9         # 实测 0.950


# ── 单元测试：metrics ───────────────────────────────────────────

def test_metrics_recall_mrr():
    """Recall@K / MRR：首命中位置决定，后续命中不算 Recall。"""
    from app.rag.evaluation.metrics import compute_metrics

    # 3 条 query：query1 命中 rank3+4（rank3 是首命中）；query2 命中 rank1；query3 未命中
    m = compute_metrics([[3, 4], [1], []], n_queries=3)

    assert m.recall_at_k[1] == 1 / 3   # 只有 query2 在 top-1 命中
    assert m.recall_at_k[5] == 2 / 3   # query1(rank3) + query2(rank1)
    assert m.hit_at_k[5] == m.recall_at_k[5]
    assert abs(m.mrr - (1 / 3 + 1.0 + 0.0) / 3) < 1e-9
    assert m.top1 == 1 / 3


def test_metrics_ndcg_captures_multiple_hits():
    """NDCG 与 MRR 的差异：MRR 只看第一个命中，NDCG 把后续命中也给分。

    query1 命中 rank3+4：
        MRR 贡献 1/3；NDCG@5 = (1/log2(4) + 1/log2(5)) / (1/log2(2) + 1/log2(3))
        = 0.9307 / 1.6310 = 0.5707（高于 MRR 的 0.333，因为奖励了第二个命中）。
    """
    from app.rag.evaluation.metrics import compute_metrics

    m = compute_metrics([[3, 4], [1], []], n_queries=3)

    # query1 NDCG = 0.5707, query2 = 1.0, query3 = 0.0 → 均值
    expected = (0.5707 + 1.0) / 3
    assert abs(m.ndcg_at_k[5] - expected) < 0.01


def test_metrics_empty_dataset():
    """空数据集不崩溃，指标为 0。"""
    from app.rag.evaluation.metrics import compute_metrics

    m = compute_metrics([], n_queries=0)
    assert m.n == 0
    assert m.recall_at_k == {}
    assert m.mrr == 0.0


# ── 单元测试：数据集加载 ─────────────────────────────────────────

def test_load_dataset_new_format():
    """新格式数据集（{name, company, items}）正确加载。"""
    from app.rag.evaluation.dataset import load_dataset

    ds = load_dataset(str(_DATASETS / "xiaomi.json"))
    assert ds.name == "Xiaomi 2025 Annual Report"
    assert ds.company == "小米"
    assert len(ds.items) == 10
    item = ds.items[0]
    assert item.id == "xiaomi_001"
    assert item.query
    assert item.expected_sections
    assert item.expected_keywords  # PR #32 新增字段


def test_load_dataset_catl():
    from app.rag.evaluation.dataset import load_dataset

    ds = load_dataset(str(_DATASETS / "catl.json"))
    assert ds.company == "宁德时代"
    assert len(ds.items) == 10
    assert all(it.id for it in ds.items)


# ── 单元测试：RagEvaluator（dummy store，无需 BGE）────────────────

def _dummy_store():
    from app.rag.document import DocumentChunk
    from app.rag.embedding import DummyEmbeddingModel
    from app.rag.vector_store import FAISSVectorStore

    model = DummyEmbeddingModel(dim=8)
    store = FAISSVectorStore(dim=8)
    chunks = [
        DocumentChunk(
            chunk_id="a", company="测试公司", doc_type="pdf",
            source="/t.txt", source_name="t", page=1,
            text="公司智能电动汽车业务营收增长迅速",
            metadata={"chapter": "管理层讨论及分析"},
        ),
        DocumentChunk(
            chunk_id="b", company="测试公司", doc_type="pdf",
            source="/t.txt", source_name="t", page=2,
            text="公司研发费用达到101亿元",
            metadata={"chapter": "董事会报告"},
        ),
        DocumentChunk(
            chunk_id="c", company="测试公司", doc_type="pdf",
            source="/t.txt", source_name="t", page=3,
            text="公司现金及现金等价物充裕",
            metadata={"chapter": "财务报表"},
        ),
    ]
    store.add(chunks, model.embed([c.text for c in chunks]))
    return model, store


def test_evaluator_with_dummy_store():
    """RagEvaluator 端到端（dummy store）：命中检测 + 指标计算正确。"""
    from app.rag.evaluation import RagEvaluator
    from app.rag.evaluation.dataset import DatasetItem, EvaluationDataset

    model, store = _dummy_store()
    ds = EvaluationDataset(
        name="dummy", company="测试公司",
        items=[
            DatasetItem(id="q1", company="测试公司", query="智能电动汽车业务",
                        expected_sections=["管理层讨论及分析"]),
            DatasetItem(id="q2", company="测试公司", query="研发费用",
                        expected_sections=["董事会报告"]),
        ],
    )

    evaluator = RagEvaluator("测试公司", _model=model, _store=store)
    metrics, details = evaluator.evaluate(ds)

    assert metrics.n == 2
    # q1 期望"管理层讨论及分析"→ 命中含该 chapter 的 chunk（BM25 关键词匹配）
    assert details[0]["hit_ranks"]  # 至少命中一个
    # q2 期望"董事会报告"→ 命中研发费用 chunk
    assert details[1]["hit_ranks"]
    assert metrics.recall_at_k[5] == 1.0
    assert metrics.mrr > 0.0


# ── 回归评测（--run-real）───────────────────────────────────────

def _run_regression(pdf_name: str, dataset_name: str, company: str, rewrite_mode: str):
    """跑单公司回归（metadata 精排 + per-company 权重 + 指定改写模式）。"""
    from app.core.config import settings
    from tests.eval_helpers import run_company_eval

    settings.rag_query_rewriter = rewrite_mode
    # PR #33 生产精排：metadata 模式（小米走融合增强、CATL 未配置直通纯 CE）
    settings.rag_reranker_model = "metadata"
    settings.rag_metadata_company_weights = {"小米": "0.90,0.08,0.02"}
    pdf = _ROOT / pdf_name
    if not pdf.exists():
        pytest.skip(f"缺少真实 PDF: {pdf}")
    return run_company_eval(
        pdf_path=str(pdf),
        eval_path=str(_DATASETS / dataset_name),
        company=company,
        pipeline_version="ocr-outline-v2-2026-08-06",  # 复用 PR #30 入库存档
    )


@pytest.mark.real
def test_regression_xiaomi(xiaomi_pdf_path):
    """小米回归：metadata 精排 + rule 改写下 Recall@5 / MRR / NDCG 不得低于基线。"""
    r = _run_regression("小米集团2025年报.pdf", "xiaomi.json", "小米", "rule")
    print(
        f"\n[REGRESSION Xiaomi] Recall@5={r.recall_at_k.get(5, 0):.0%} "
        f"MRR={r.mrr:.3f} NDCG@5={r.ndcg_at_k.get(5, 0):.3f} Top1={r.top1:.0%}"
    )
    assert r.recall_at_k.get(5, 0) >= _XIAOMI_RECALL5_FLOOR
    assert r.mrr >= _XIAOMI_MRR_FLOOR
    assert r.ndcg_at_k.get(5, 0) >= _XIAOMI_NDCG5_FLOOR


@pytest.mark.real
def test_regression_catl(xiaomi_pdf_path):
    """CATL 回归：metadata 模式下未配置公司直通纯 CE，R@5 / MRR 不得低于基线。"""
    r = _run_regression("宁德时代2025年报.pdf", "catl.json", "宁德时代", "rule")
    print(
        f"\n[REGRESSION CATL] Recall@5={r.recall_at_k.get(5, 0):.0%} "
        f"MRR={r.mrr:.3f} NDCG@5={r.ndcg_at_k.get(5, 0):.3f} Top1={r.top1:.0%}"
    )
    assert r.recall_at_k.get(5, 0) >= _CATL_RECALL5_FLOOR
    assert r.mrr >= _CATL_MRR_FLOOR
