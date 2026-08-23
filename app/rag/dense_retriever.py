"""Dense 稠密检索（BGE-M3 + FAISS）。

封装 query → embedding → FAISS 检索 的稠密通路，
与 SparseRetriever（BM25）并列，供 retriever 做双路召回。
"""

from __future__ import annotations

from app.rag.document import DocumentChunk
from app.rag.embedding import EmbeddingModel
from app.rag.vectorstore import VectorStore


class DenseRetriever:
    """基于 Embedding + VectorStore 的稠密检索器。"""

    def __init__(self, model: EmbeddingModel, store):
        # 类型注解放宽：同时接受新接口 store（生产 get_store()）与旧接口 store
        # （测试 seam 注入的旧 FAISSVectorStore），search() 内按类型分派。
        self._model = model
        self._store = store

    def search(
        self,
        query: str,
        top_k: int,
        company: str = "",
    ) -> list[tuple[DocumentChunk, float]]:
        """稠密检索：query → embedding → FAISS 召回 → 按 company 过滤 → top_k。

        Args:
            query:   查询文本。
            top_k:   返回条数。
            company: 一级过滤字段（空串不过滤）。

        Returns:
            (chunk, 相似度分数) 列表，按分数降序；分数 ∈ [0, 1]。
        """
        query_vec = self._model.embed([query])[0]
        if isinstance(self._store, VectorStore):
            # 新接口：结构化 filters + SearchResult → 桥接回旧返回类型
            # （RFfuse / reranker / pipeline 仍消费 (DocumentChunk, score) 元组）
            filters = {"company_id": company} if company else None
            results = self._store.search(query_vec, top_k=top_k, filters=filters)
            return [(r.to_document_chunk(company), r.score) for r in results]
        # 旧接口（测试 seam 注入的旧 FAISSVectorStore）：company 位置参数
        return self._store.search(query_vec, company=company, top_k=top_k)
