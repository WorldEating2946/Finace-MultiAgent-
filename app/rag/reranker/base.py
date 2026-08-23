"""精排器抽象接口。

主流程只依赖本抽象（retriever / pipeline），不感知底层模型。
切换 Dummy → CrossEncoder 时只改工厂，主流程零改动。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.rag.document import DocumentChunk


class Reranker(ABC):
    """精排器抽象接口。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """对候选 chunk 按相关度重排序。

        Args:
            query:  查询文本。
            chunks: 召回候选（Hybrid 融合后，通常 30~50 条）。

        Returns:
            重排序后的 chunk 列表（相关度降序）。
        """
        raise NotImplementedError
