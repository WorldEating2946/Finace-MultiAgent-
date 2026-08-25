"""直通精排器（开发/测试占位，不重排）。

用于未配置真实模型（rag_reranker_model="dummy"）时的降级，
以及单元测试的显式注入。
"""

from __future__ import annotations

from app.rag.document import DocumentChunk
from app.rag.reranker.base import Reranker


class DummyReranker(Reranker):
    """直通：保持候选原序，不做精排。"""

    def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        return list(chunks)
