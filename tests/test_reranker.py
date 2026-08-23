"""Reranker 精排器单元测试。

测试 Reranker ABC + DummyReranker 直通 + get_reranker 工厂；
真实 bge-reranker-v2-m3 排序用 --run-real 跑（加载 2.2GB 模型）。
"""

import pytest

from app.rag.document import DocumentChunk
from app.rag.reranker import (
    DummyReranker,
    Reranker,
    get_reranker,
)


def _make_chunk(text: str, chunk_id: str = "test-0") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        company="测试公司",
        doc_type="text",
        source="/tmp/test.txt",
        source_name="测试文档",
        page=0,
        text=text,
        metadata={},
    )


# ── A. 抽象接口 ───────────────────────────────────────────────────

def test_reranker_is_abstract():
    """Reranker 为抽象类，不可直接实例化。"""
    with pytest.raises(TypeError):
        Reranker()  # type: ignore[abstract]


# ── B. DummyReranker（直通）──────────────────────────────────────

def test_dummy_reranker_passthrough():
    reranker = DummyReranker()
    chunks = [
        _make_chunk("研发费用达到100亿", chunk_id="a"),
        _make_chunk("公司历史介绍", chunk_id="b"),
        _make_chunk("研发人员数量", chunk_id="c"),
    ]

    result = reranker.rerank("研发投入", chunks)

    assert [c.chunk_id for c in result] == ["a", "b", "c"]  # 直通保持原序


def test_dummy_reranker_empty_list():
    assert DummyReranker().rerank("查询", []) == []


def test_dummy_reranker_single_chunk():
    chunk = _make_chunk("唯一文档")
    assert DummyReranker().rerank("查询", [chunk]) == [chunk]


def test_dummy_reranker_does_not_modify_chunks():
    reranker = DummyReranker()
    chunks = [_make_chunk(f"原文内容 {i}", chunk_id=f"id-{i}") for i in range(5)]

    result = reranker.rerank("查询", chunks)

    for original, reranked in zip(chunks, result):
        assert reranked.text == original.text
        assert reranked.chunk_id == original.chunk_id


# ── C. 工厂（测试默认 DummyReranker）─────────────────────────────

def test_get_reranker_returns_singleton():
    """get_reranker() 返回同一实例（测试环境强制 dummy）。"""
    r1 = get_reranker()
    r2 = get_reranker()
    assert r1 is r2
    assert isinstance(r1, DummyReranker)


def test_get_reranker_uses_config(monkeypatch):
    """配置指向模型路径时返回 CrossEncoderReranker（不加载模型，构造惰性）。"""
    import app.rag.reranker as rr
    from app.rag.reranker.cross_encoder import CrossEncoderReranker

    monkeypatch.setattr(rr.settings, "rag_reranker_model", "app/models/reranker/bge-reranker-v2-m3")
    monkeypatch.setattr(rr, "_default_reranker", None)

    assert isinstance(rr.get_reranker(), CrossEncoderReranker)


# ── D. 真实 CrossEncoder 排序（--run-real）──────────────────────

@pytest.mark.real
def test_cross_encoder_reranks_relevant_first():
    """真实 bge-reranker-v2-m3：查询"研发投入"应把研发相关排在公司历史之前。"""
    from app.rag.reranker.cross_encoder import CrossEncoderReranker

    chunks = [
        _make_chunk("报告期内公司研发费用达到101亿元，同比增长20%。", chunk_id="A"),
        _make_chunk("公司研发人员数量为2.3万人，占比持续提升。", chunk_id="B"),
        _make_chunk("公司自成立以来的发展历史与重大事件回顾。", chunk_id="C"),
    ]
    result = CrossEncoderReranker().rerank("研发投入", chunks)

    order = [c.chunk_id for c in result]
    # 研发相关（A 研发费用 / B 研发人员）应排在 C（公司历史）之前
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("C")
