"""检索管线集成测试：query → Hybrid → Reranker → 结果。"""

from app.rag.document import DocumentChunk
from app.rag.embedding import DummyEmbeddingModel
from app.rag.reranker.dummy import DummyReranker
from app.rag.vector_store import FAISSVectorStore


def _chunk(cid: str, text: str, company: str = "测试公司") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid, company=company, doc_type="text",
        source="/tmp/t.txt", source_name="t", page=0, text=text,
    )


class _RecordingReranker(DummyReranker):
    """直通 + 记录被调用。"""

    def __init__(self):
        self.calls = 0

    def rerank(self, query, chunks):
        self.calls += 1
        return list(chunks)


def _make_pipeline():
    model = DummyEmbeddingModel(dim=8)
    store = FAISSVectorStore(dim=8)
    texts = [
        "报告期内公司研发费用达到101亿元。",
        "公司主营业务涵盖动力电池与储能系统。",
        "公司治理结构完善，独立董事占比合理。",
    ]
    store.add(
        [_chunk(f"id-{i}", t) for i, t in enumerate(texts)],
        model.embed(texts),
    )
    return model, store


def test_retriever_hybrid_rerank_loop():
    """retriever 层：Dense + BM25 → RRF → Reranker（spy 验证被调用）。"""
    from app.rag.retriever import retrieve

    model, store = _make_pipeline()
    reranker = _RecordingReranker()

    results = retrieve(
        "研发费用", k=2, company="测试公司",
        _model=model, _store=store, _reranker=reranker,
    )

    assert reranker.calls == 1  # reranker 在链路中被调用
    assert 1 <= len(results) <= 2
    for chunk, score in results:
        assert 0.0 <= score <= 1.0
    # 研发相关 chunk 应被召回（Dense 或 BM25）
    assert any("研发" in c.text for c, _ in results)


def test_pipeline_returns_retrieval_result():
    """pipeline 层：query → Hybrid → Reranker → RetrievalResult。"""
    from app.rag.pipeline import retrieve

    model, store = _make_pipeline()

    result = retrieve(
        "研发费用", company="测试公司", top_k=2, _model=model, _store=store
    )

    assert result.query == "研发费用"
    assert len(result.chunks) == len(result.scores)
    assert len(result.chunks) <= 2
    assert result.confidence >= 0.0


def test_pipeline_result_chunks_traceable():
    """结果 chunk 应带 dense_vector 与来源（接口设计：chunk 自包含）。"""
    from app.rag.pipeline import retrieve

    model, store = _make_pipeline()

    result = retrieve(
        "研发费用", company="测试公司", top_k=2, _model=model, _store=store
    )

    assert len(result.chunks) >= 1
    chunk = result.chunks[0]
    assert chunk.dense_vector  # 回填稠密向量
    assert chunk.source  # 来源（类型化字段）
    assert chunk.company == "测试公司"
