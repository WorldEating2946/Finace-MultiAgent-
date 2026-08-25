"""企业问题评测集（xiaomi_eval.json）测试。

评测集用于后续 splitter / embedding / reranker 改进的统一评价：
    结构校验 → 默认跑（快）；
    真实 BGE 检索命中率 → 标记 @pytest.mark.real，`pytest -m real` 才跑（慢，需本地模型）。
"""

import json
from pathlib import Path

import pytest

_EVAL_PATH = Path(__file__).parent / "data" / "xiaomi_eval.json"

# 合成年报的节标题（与 conftest._CHAPTERS 对齐，防评测集与数据漂移）
_SYNTHETIC_SECTIONS = {
    "公司简介", "主营业务范围", "公司治理",
    "经营成果", "财务状况", "现金流",
    "发展战略", "智能电动车业务升级目标",
    "智能手机业务", "IoT业务", "互联网服务",
    "主要会计数据", "财务指标", "风险提示",
}


def _load_eval() -> list[dict]:
    return json.loads(_EVAL_PATH.read_text(encoding="utf-8"))


def test_eval_dataset_valid():
    """评测集结构：question 与 expected_section 均非空。"""
    data = _load_eval()
    assert len(data) > 0
    for item in data:
        assert item["question"].strip()
        assert item["expected_section"].strip()


def test_eval_sections_match_synthetic_pdf():
    """expected_section 都应能在合成年报章节结构中找到。"""
    data = _load_eval()
    for item in data:
        assert any(item["expected_section"] in s for s in _SYNTHETIC_SECTIONS), item


@pytest.mark.real
def test_eval_real_retrieval_hit_rate(xiaomi_pdf_path):
    """真实 BGE 检索命中率：问题应命中期望章节（慢，需 -m real 且本地模型）。

    流程：ingest 合成年报 → 逐题 retrieve → 校验 top chunk 的 section 是否含期望章节。
    """
    import shutil
    import tempfile

    from app.core.config import settings
    from app.rag import ingest, retrieve

    base = Path(tempfile.gettempdir()) / "rag_eval_smoke"
    shutil.rmtree(base, ignore_errors=True)
    settings.rag_vector_store_path = str(base)

    ingest(str(xiaomi_pdf_path), company="小米")

    data = _load_eval()
    hits = 0
    for item in data:
        result = retrieve(item["question"], company="小米", top_k=3)
        assert len(result.chunks) >= 1
        # top-3 命中：期望章节出现在任一返回 chunk 的 section 中（retrieve 交付 3 条给 LLM）
        sections = [c.metadata["section"] for c in result.chunks]
        if any(item["expected_section"] in s for s in sections):
            hits += 1

    rate = hits / len(data)
    print(f"\n[EVAL] 命中 {hits}/{len(data)} = {rate:.0%}")
    assert rate >= 0.6, f"命中率过低: {rate:.0%}（< 60%）"
