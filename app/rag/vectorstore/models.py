"""PR44.1 统一数据模型（chunk_id + metadata + embedding 一致性）。

设计目标：把写入/读取的三要素（chunk_id / metadata / embedding）封装为单一
``VectorRecord``，从源头消除旧接口 ``add(chunks, vectors)`` 两参数分离传递
导致的位置对齐风险（见 PR44 spec §7 最大风险）。

与旧 ``DocumentChunk`` 的关系：
    - 写入：``VectorRecord.to_document_chunk()`` → 旧接口内部使用；
    - 读取：``VectorRecord.from_document_chunk()`` → wrapper 桥接；
    - 上层新代码只接触 VectorRecord / SearchResult，不感知 DocumentChunk。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 仅类型检查，避免循环导入
    from app.rag.document import DocumentChunk


@dataclass
class VectorRecord:
    """向量库写入/读取的统一记录。

    metadata 约定字段（enterprise 过滤能力的基础）：
        company_id    — 企业标识（一级过滤，如 "xiaomi"）
        document_id   — 文档标识（如 "2025_report"）
        chunk_index   — 切片序号
        page          — 页码（PDF）或 None（Markdown）
        section       — 章节路径（如 "财务分析"）
        doc_type      — 文档格式（pdf / markdown / text）
        source_type   — 文档语义类型（annual_report / research_report / ...）
        year          — 年份（数值过滤用）
    """

    chunk_id: str
    text: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)

    def to_document_chunk(self, company: str = "") -> DocumentChunk:
        """转换为旧接口的 DocumentChunk（wrapper 内部使用）。"""
        from app.rag.document import DocumentChunk

        meta = dict(self.metadata)
        return DocumentChunk(
            chunk_id=self.chunk_id,
            company=company or meta.get("company_id", ""),
            doc_type=meta.get("doc_type", ""),
            source=meta.get("source", ""),
            source_name=meta.get("source_name", ""),
            page=meta.get("page"),
            text=self.text,
            metadata=meta,
        )

    @classmethod
    def from_document_chunk(
        cls,
        chunk: DocumentChunk,
        embedding: list[float],
    ) -> VectorRecord:
        """从旧 DocumentChunk + embedding 构造（wrapper 内部使用）。"""
        meta = dict(chunk.metadata)
        # 统一企业标识：DocumentChunk.company 是顶层字段，metadata 缺省时注入
        # company_id，保证 enterprise 过滤契约在 add→search 全链路一致。
        if chunk.company and not meta.get("company_id"):
            meta["company_id"] = chunk.company
        return cls(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            embedding=embedding,
            metadata=meta,
        )


@dataclass
class SearchResult:
    """search() 返回的单条结果（取代裸 ``tuple[DocumentChunk, float]``）。"""

    chunk_id: str
    text: str
    score: float  # [0, 1]
    metadata: dict = field(default_factory=dict)

    def to_document_chunk(self, company: str = "") -> DocumentChunk:
        """转换为旧接口的 DocumentChunk（bridge：新接口 → retriever/RRF 管线）。

        dense_vector 留空——调用方后续用 store.vector_of() 回填（与旧行为一致）。
        """
        from app.rag.document import DocumentChunk

        meta = dict(self.metadata)
        return DocumentChunk(
            chunk_id=self.chunk_id,
            company=company or meta.get("company_id", ""),
            doc_type=meta.get("doc_type", ""),
            source=meta.get("source", ""),
            source_name=meta.get("source_name", ""),
            page=meta.get("page"),
            text=self.text,
            metadata=meta,
        )
