"""RAG 对外统一入口（唯一公共接口）。

Research Agent 等外部调用方只依赖本模块的 ``retrieve()``，
不感知底层 loader / splitter / embedding / vector_store / retriever / reranker
的实现细节（Phase 1 FAISS / Phase 2 Milvus 切换对调用方零影响）。

对外访问方式：
    from app.rag import retrieve
"""

from __future__ import annotations

from app.core.config import settings
from app.rag.document import RetrievalResult
from app.rag.embedding import EmbeddingModel
from app.rag.query.rewriter import QueryRewriter
from app.rag.retriever import retrieve as _retrieve
from app.rag.vectorstore import VectorStore


def retrieve(
    query: str,
    company: str,
    top_k: int = settings.rag_default_top_k,
    doc_type: str | None = None,
    *,
    _model: EmbeddingModel | None = None,
    _store: VectorStore | None = None,
    _rewriter: QueryRewriter | None = None,
) -> RetrievalResult:
    """RAG 检索唯一入口（ADR-001 接口契约）。

    Args:
        query:    分析问题（如"商业模式与竞争壁垒"）。
        company:  目标公司，一级过滤维度（必填），避免跨公司噪声。
        top_k:    精排后返回条数，默认取自 settings.rag_default_top_k。
        doc_type: 可选，限定文档类型（招股书/财报/行业报告/政策）。

    Keyword Args:
        _model:    测试 seam —— 注入 EmbeddingModel。
        _store:    测试 seam —— 注入 VectorStore。
        _rewriter: 测试 seam —— 注入 QueryRewriter（默认 RuleBasedQueryRewriter）。

    Returns:
        RetrievalResult：精排后片段 + 各片段置信度 + Top-1 置信度 + 来源引用。

    Note:
        Phase 1：company / doc_type 过滤留待后续阶段实现；
        当前直接委托 retriever → vector_store 完成 Dense 检索。
    """
    # 1. 委托 retriever 完成 query → embedding → vector search
    raw = _retrieve(
        query,
        k=top_k,
        company=company,
        _model=_model,
        _store=_store,
        _rewriter=_rewriter,
    )

    # 2. 组装 RetrievalResult
    chunks = [c for c, _ in raw]
    scores = [s for _, s in raw]
    confidence = scores[0] if scores else 0.0

    return RetrievalResult(
        query=query,
        chunks=chunks,
        scores=scores,
        confidence=confidence,
    )
