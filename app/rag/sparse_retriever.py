"""BM25 稀疏检索（rank_bm25 + jieba 中文分词）。

企业年报中文场景，BM25 与 Dense 语义检索互补：
    - Dense 召回"语义近但关键词远"；
    - BM25 召回"关键词命中但语义远"（如精确指标名、公司名）。

接口：
    retriever = SparseRetriever().build(chunks)
    results = retriever.search(query, top_k, company="")
"""

from __future__ import annotations

from app.rag.document import DocumentChunk


class SparseRetriever:
    """基于 BM25 的稀疏检索器。"""

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._index = None

    def build(self, chunks: list[DocumentChunk]) -> SparseRetriever:
        """用 chunk 列表构建 BM25 语料（jieba 中文分词）。

        Args:
            chunks: 语料 chunk（通常来自向量库的全部 chunk）。

        Returns:
            self（支持链式调用）。

        Raises:
            ImportError: 未安装 jieba / rank_bm25。
        """
        self._chunks = list(chunks)
        if not self._chunks:
            self._index = None  # 空语料不建索引
            return self

        import jieba
        from rank_bm25 import BM25Okapi

        corpus = [jieba.lcut(c.text) for c in self._chunks]
        self._index = BM25Okapi(corpus)
        return self

    def search(
        self,
        query: str,
        top_k: int,
        company: str = "",
    ) -> list[tuple[DocumentChunk, float]]:
        """BM25 检索：query 分词 → 打分 → 按 company 过滤 → top_k。

        Args:
            query:   查询文本。
            top_k:   返回条数。
            company: 一级过滤字段（空串不过滤）。

        Returns:
            (chunk, BM25 分数) 列表，按分数降序。
        """
        if not self._chunks or self._index is None:
            return []
        import jieba

        query_tokens = jieba.lcut(query)
        scores = self._index.get_scores(query_tokens)

        ranked: list[tuple[DocumentChunk, float]] = []
        for score, chunk in sorted(zip(scores, self._chunks), key=lambda x: -x[0]):
            if chunk.company != company:
                continue
            ranked.append((chunk, float(score)))
            if len(ranked) >= top_k:
                break
        return ranked

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """jieba 中文分词（供结果回填 sparse_tokens）。"""
        import jieba

        return jieba.lcut(text)
