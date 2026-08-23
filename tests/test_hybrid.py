"""Hybrid 检索（Dense + BM25 + RRF）单元测试。"""

from app.rag.document import DocumentChunk
from app.rag.embedding import DummyEmbeddingModel
from app.rag.fusion import rrf_fuse
from app.rag.retriever import retrieve
from app.rag.sparse_retriever import SparseRetriever
from app.rag.vector_store import FAISSVectorStore


def _chunk(cid: str, text: str, company: str = "宁德时代") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid, company=company, doc_type="text",
        source="/tmp/t.txt", source_name="t", page=0, text=text,
    )


# ── RRF 融合 ───────────────────────────────────────────────────

def test_rrf_fuse_orders_by_rank_sum():
    """用户示例：Dense(A,B,C) + BM25(B,D,A) → 融合 B > A > D > C。"""
    dense = [
        (_chunk("A", "甲"), 0.9),
        (_chunk("B", "乙"), 0.8),
        (_chunk("C", "丙"), 0.5),
    ]
    sparse = [
        (_chunk("B", "乙"), 8.0),
        (_chunk("D", "丁"), 6.0),
        (_chunk("A", "甲"), 4.0),
    ]

    fused = rrf_fuse(dense, sparse)

    order = [c.chunk_id for c, _ in fused]
    assert order == ["B", "A", "D", "C"]
    # 分数归一化到 [0,1] 且降序
    scores = [s for _, s in fused]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_rrf_fuse_single_route():
    dense = [(_chunk("A", "甲"), 0.9), (_chunk("B", "乙"), 0.8)]
    fused = rrf_fuse(dense, [])
    assert [c.chunk_id for c, _ in fused] == ["A", "B"]


def test_rrf_fuse_empty():
    assert rrf_fuse([], []) == []


# ── BM25 稀疏检索 ───────────────────────────────────────────────

def test_sparse_bm25_ranks_keyword_matches():
    chunks = [
        _chunk("a", "宁德时代动力电池系统销量全球第一"),
        _chunk("b", "公司研发投入持续增长"),
        _chunk("c", "智能电动车业务快速推进"),
    ]
    retriever = SparseRetriever().build(chunks)

    results = retriever.search("动力电池销量", top_k=2, company="宁德时代")

    assert results[0][0].chunk_id == "a"  # 关键词命中最高


def test_sparse_filters_by_company():
    chunks = [
        _chunk("a", "动力电池出货量领先", company="宁德时代"),
        _chunk("b", "动力电池出货量领先", company="比亚迪"),
    ]
    retriever = SparseRetriever().build(chunks)

    results = retriever.search("动力电池", top_k=5, company="宁德时代")

    assert all(c.company == "宁德时代" for c, _ in results)
    assert len(results) == 1


def test_sparse_empty_store_returns_empty():
    assert SparseRetriever().build([]).search("查询", top_k=5) == []


# ── Hybrid 检索结果回填 ────────────────────────────────────────

def test_retrieve_returns_chunk_with_dense_and_sparse():
    """retrieve 结果 chunk 应回填 dense_vector + sparse_tokens（接口设计）。"""
    model = DummyEmbeddingModel(dim=8)
    store = FAISSVectorStore(dim=8)
    texts = ["宁德时代动力电池系统销量全球第一", "公司研发投入持续增长"]
    store.add(
        [_chunk(f"id-{i}", t, company="宁德时代") for i, t in enumerate(texts)],
        model.embed(texts),
    )

    results = retrieve("动力电池销量", k=2, company="宁德时代", _model=model, _store=store)

    assert len(results) >= 1
    chunk, score = results[0]
    assert chunk.dense_vector  # 回填稠密向量
    assert chunk.sparse_tokens  # 回填 BM25 词
    assert 0.0 <= score <= 1.0
