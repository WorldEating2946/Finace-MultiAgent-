"""检索器编排：Dense + Sparse → RRF 融合 → reranker → top-k。

    query
      ├── DenseRetriever（BGE-M3 + FAISS）
      ├── SparseRetriever（BM25 + jieba）
      ├── rrf_fuse（Reciprocal Rank Fusion）
      └── Reranker 精排

Hybrid 扩大召回候选，解决"语义近但关键词远 / 关键词近但语义远"互补问题。
内部模块，不对外暴露（外部通过 pipeline.retrieve() 调用）。
"""

from __future__ import annotations

import logging
import time
from weakref import WeakKeyDictionary

from app.core.config import settings
from app.rag.dense_retriever import DenseRetriever
from app.rag.document import DocumentChunk
from app.rag.embedding import EmbeddingModel, get_embedding_model
from app.rag.fusion import rrf_fuse
from app.rag.query import QueryRewriter, get_query_rewriter
from app.rag.reranker import Reranker, get_reranker
from app.rag.sparse_retriever import SparseRetriever
from app.rag.vectorstore import VectorStore, get_store

_FETCH_K = settings.rag_retrieve_top_k  # 单路召回候选数（供融合 + CrossEncoder 精排）
_EXTRA_K = 10  # 扩展查询补充候选上限（base 之外的新 chunk 数）

logger = logging.getLogger(__name__)

# 稀疏检索器缓存（弱引用 key：store 被 GC 即自动移除，杜绝 id 复用误命中旧语料）
_sparse_cache: WeakKeyDictionary = WeakKeyDictionary()


def _get_sparse_retriever(store: VectorStore) -> SparseRetriever:
    """按 store 实例缓存 BM25 检索器（语料不变则不重建）。

    WeakKeyDictionary 而非 ``dict[int, ...]``（key=id(store)）——store 被 GC 后
    id 可能被新 store 复用，dict 会误返回旧语料（BM25 语料错公司 → 召回漂移）。
    """
    if store not in _sparse_cache:
        records = store.all_chunks()
        if isinstance(store, VectorStore):
            # 新接口：all_chunks() 返回 VectorRecord → 桥接为 DocumentChunk
            # （SparseRetriever.build() 仍消费 DocumentChunk，独立 PR 再迁移）
            chunks = [
                r.to_document_chunk(r.metadata.get("company_id", ""))
                for r in records
            ]
        else:
            # 旧接口（测试 seam）：all_chunks() 已返回 DocumentChunk
            chunks = list(records)
        _sparse_cache[store] = SparseRetriever().build(chunks)
    return _sparse_cache[store]


def retrieve(
    query: str,
    k: int = settings.rag_rerank_top_k,
    company: str = "",
    *,
    _model: EmbeddingModel | None = None,
    _store: VectorStore | None = None,
    _reranker: Reranker | None = None,
    _rewriter: QueryRewriter | None = None,
) -> list[tuple[DocumentChunk, float]]:
    """Hybrid 检索：query → (Rewrite → 补充召回) → Dense+Sparse → RRF → reranker → top-k。

    Query Rewrite 将单条查询扩展为多条变体（弥合词汇鸿沟）。
    原始 query 的结果保持权威序（base-first）；扩展 query 只补充 base 之外的
    新候选（recall booster），交给 reranker 用原始 query 裁决。
    无匹配时 pass-through（单查询路径，行为与原实现一致）。

    Args:
        query:   查询文本。
        k:       精排后返回的 chunk 数量，默认 5。
        company: 一级过滤字段（空串表示不过滤），由 pipeline 层传入。

    Keyword Args:
        _model:    测试 seam —— 注入 EmbeddingModel。
        _store:    测试 seam —— 注入 VectorStore。
        _reranker: 测试 seam —— 注入 Reranker。
        _rewriter: 测试 seam —— 注入 QueryRewriter（默认 RuleBasedQueryRewriter）。

    Returns:
        list[tuple[DocumentChunk, float]]:
            (chunk, 融合分数) 列表，精排后按相关度降序；分数 ∈ [0, 1]。
    """
    model = _model or get_embedding_model()
    store = _store or get_store(company_id=company)
    reranker = _reranker or get_reranker()
    rewriter = _rewriter or get_query_rewriter()

    # 0. Query Rewrite：扩展查询弥合词汇鸿沟（无匹配时返回 [query] 直通）
    t0 = time.time()
    sub_queries = rewriter.rewrite(query)
    original, expansions = sub_queries[0], sub_queries[1:]
    logger.info("[retrieve] rewrite %d 子查询 +%.0fms", len(sub_queries), (time.time() - t0) * 1000)
    t_rewrite = time.time() - t0

    def _recall(sq: str) -> tuple[list[tuple[DocumentChunk, float]], bool]:
        """单查询召回：Dense + Sparse → RRF → top-k。返回 (fused, sparse_ok)。"""
        t = time.time()
        dense_candidates = DenseRetriever(model, store).search(
            sq, top_k=_FETCH_K, company=company
        )
        t_dense = time.time() - t
        # Sparse：BM25 召回（jieba 中文分词；依赖缺失时降级纯 dense）
        sparse_candidates: list[tuple[DocumentChunk, float]] = []
        sparse_ok = False
        if settings.rag_hybrid and dense_candidates:
            try:
                t = time.time()
                sparse_candidates = _get_sparse_retriever(store).search(
                    sq, top_k=_FETCH_K, company=company
                )
                logger.info("[retrieve]   sparse +%.0fms (ok=%s)", (time.time() - t) * 1000, bool(sparse_candidates))
                sparse_ok = bool(sparse_candidates)
            except ImportError:
                sparse_candidates = []
        logger.info("[retrieve]   %s dense +%.0fms, fused", sq[:20], t_dense * 1000)
        return rrf_fuse(dense_candidates, sparse_candidates)[:_FETCH_K], sparse_ok

    # 1. 原始 query 主路径：结果保持权威序（不被扩展查询稀释）
    base_fused, sparse_available = _recall(original)

    # 2. 扩展 query 只补充 base 之外的新候选（recall booster）
    base_ids = {c.chunk_id for c, _ in base_fused}
    extra: list[tuple[DocumentChunk, float]] = []
    for sq in expansions:
        q_fused, _ = _recall(sq)
        for chunk, score in q_fused:
            if chunk.chunk_id not in base_ids:
                extra.append((chunk, score))
                base_ids.add(chunk.chunk_id)
        if len(extra) >= _EXTRA_K:
            break

    if not base_fused and not extra:
        return []

    # 3. 候选集：base 在前（保持原始召回序），extra 补充在后
    fused = base_fused + extra

    # 4. reranker 精排（始终用原始 query：扩展只负责召回，精排以用户意图为准）
    candidate_chunks = [c for c, _ in fused]
    t = time.time()
    reranked = reranker.rerank(query, candidate_chunks)
    logger.info("[retrieve] rerank %d 候选 +%.0fms", len(candidate_chunks), (time.time() - t) * 1000)

    # 5. 保留融合分数 + 回填 dense_vector / sparse_tokens（接口设计：chunk 自包含）
    #    DummyReranker 直通时分数不变；CrossEncoder 需扩展接口。
    score_map = {c.chunk_id: s for c, s in base_fused}
    score_map.update({c.chunk_id: s for c, s in extra})
    results: list[tuple[DocumentChunk, float]] = []
    t = time.time()
    for chunk in reranked[:k]:
        enriched = chunk.model_copy(update={"dense_vector": store.vector_of(chunk.chunk_id) or []})
        if sparse_available:  # sparse 可用时回填 BM25 稀疏词
            try:
                enriched = enriched.model_copy(
                    update={"sparse_tokens": SparseRetriever.tokenize(chunk.text)}
                )
            except ImportError:
                pass
        results.append((enriched, score_map.get(chunk.chunk_id, 0.0)))
    logger.info("[retrieve] vector_of/tokenize ×%d +%.0fms", len(reranked[:k]), (time.time() - t) * 1000)

    return results
