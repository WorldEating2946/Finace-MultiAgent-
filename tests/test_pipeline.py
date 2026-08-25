"""RAG Pipeline 入口单元测试。

测试 pipeline.retrieve() 对外接口 —— 从 query 到 RetrievalResult 的完整链路。
"""

from app.rag.document import DocumentChunk, RetrievalResult
from app.rag.embedding import DummyEmbeddingModel
from app.rag.pipeline import retrieve
from app.rag.vector_store import FAISSVectorStore

DIM = 128


# ── 测试辅助函数 ──────────────────────────────────────────────────

def _make_chunk(text: str, *, company: str = "", chunk_id: str = "test-0") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        company=company,
        doc_type="text",
        source="/tmp/test.txt",
        source_name="测试文档",
        page=0,
        text=text,
        metadata={},
    )


def _make_model(dim: int = DIM) -> DummyEmbeddingModel:
    return DummyEmbeddingModel(dim=dim)


def _make_store(dim: int = DIM) -> FAISSVectorStore:
    return FAISSVectorStore(dim=dim)


def _populate(store: FAISSVectorStore, texts: list[str], *, company: str = "测试公司") -> list[DocumentChunk]:
    model = _make_model()
    chunks = [_make_chunk(t, company=company, chunk_id=f"id-{i}") for i, t in enumerate(texts)]
    store.add(chunks, model.embed(texts))
    return chunks


# ── 核心链路 ──────────────────────────────────────────────────────

def test_retrieve_returns_retrieval_result():
    """retrieve() 应返回 RetrievalResult 实例。"""
    model = _make_model()
    store = _make_store()
    _populate(store, ["基金赎回通常需要T+1到T+3个工作日到账"])

    result = retrieve("基金赎回多久到账", company="测试公司", _model=model, _store=store)

    assert isinstance(result, RetrievalResult)


def test_retrieve_returns_correct_query():
    """返回结果的 query 字段应与输入一致。"""
    model = _make_model()
    store = _make_store()
    _populate(store, ["基金赎回规则说明"])

    result = retrieve("基金赎回多久到账", company="测试公司", _model=model, _store=store)

    assert result.query == "基金赎回多久到账"


def test_retrieve_top_k_limits_results():
    """top_k 应限制返回数量。"""
    model = _make_model()
    store = _make_store()
    _populate(store, [f"文档片段 {i}" for i in range(10)])

    result = retrieve("文档", company="测试公司", top_k=3, _model=model, _store=store)

    assert len(result.chunks) == 3
    assert len(result.scores) == 3


def test_retrieve_empty_store_returns_empty():
    """空库检索应返回空结果，confidence=0。"""
    model = _make_model()
    store = _make_store()

    result = retrieve("任意查询", company="测试公司", _model=model, _store=store)

    assert result.chunks == []
    assert result.scores == []
    assert result.confidence == 0.0


def test_retrieve_scores_match_chunks():
    """scores 与 chunks 序号一一对应。"""
    model = _make_model()
    store = _make_store()
    _populate(store, ["基金申购规则", "基金赎回规则", "基金转换规则"])

    result = retrieve("赎回", company="测试公司", top_k=3, _model=model, _store=store)

    assert len(result.scores) == len(result.chunks)
    for score in result.scores:
        assert 0.0 <= score <= 1.0


def test_retrieve_confidence_is_top1_score():
    """confidence 应等于 top-1 的 score。"""
    model = _make_model()
    store = _make_store()
    _populate(store, ["基金赎回T+1到账", "基金申购确认", "基金分红规则"])

    result = retrieve("赎回到账时间", company="测试公司", top_k=3, _model=model, _store=store)

    if result.scores:
        assert result.confidence == result.scores[0]
        # 分数应降序
        assert result.scores == sorted(result.scores, reverse=True)


def test_retrieve_chunks_have_source_info():
    """返回的 chunk 应包含来源溯源信息。"""
    model = _make_model()
    store = _make_store()
    _populate(store, ["基金产品说明书摘要"])

    result = retrieve("基金", company="测试公司", _model=model, _store=store)

    if result.chunks:
        chunk = result.chunks[0]
        assert chunk.source
        assert chunk.source_name
        assert chunk.chunk_id


# ── 真实验证：from app.rag import retrieve ────────────────────────

def test_public_api_import_works():
    """验证 from app.rag import retrieve 可用且返回 RetrievalResult。"""
    from app.rag import retrieve as public_retrieve

    model = _make_model()
    store = _make_store()
    _populate(store, ["基金赎回一般在T+2个工作日内完成"])

    result = public_retrieve("基金赎回多久到账", company="测试", _model=model, _store=store)

    assert isinstance(result, RetrievalResult)
    assert result.query == "基金赎回多久到账"
