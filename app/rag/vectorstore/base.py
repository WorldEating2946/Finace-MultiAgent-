"""PR44.1 向量库抽象接口（分层：基础 + 本地持久化 + Hybrid）。

分层原则（PR44 spec §4）：
    - 基础接口（VectorStore ABC）：所有 backend 必须实现，保持最小集；
    - LocalVectorStoreMixin：本地文件持久化（FAISS 需要；Milvus 由服务端
      管理持久化，不需要 save/load，故不进 ABC）；
    - HybridSupportMixin：dense + sparse 混合检索 + 全量 chunk 访问
      （未来轻量 dense-only 后端不需要，故不进 ABC）。

对比旧接口（app/rag/vector_store.py）：
    - search():  company 位置参数 → 结构化 filters（未来 Milvus 可自然表达
      ``company_id=="xiaomi" AND year>=2024 AND document_type=="annual_report"``）；
    - add():      ``(chunks, vectors)`` 分离 → ``records: list[VectorRecord]``；
    - 返回值：    裸 tuple → SearchResult 结构化对象；
    - 新增：      delete() / update() / count()（旧接口无删除/更新能力）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.rag.vectorstore.models import SearchResult, VectorRecord

# ── 基础接口（所有 backend 必须实现）──────────────────────────


class VectorStore(ABC):
    """向量库抽象接口。

    所有后端（FAISS / Milvus / pgvector）必须实现这 5 个方法。
    save()/load()/hybrid_search()/all_chunks() 不在 ABC 中——
    它们属于扩展能力，由 mixin 提供。
    """

    @abstractmethod
    def add(self, records: list[VectorRecord]) -> None:
        """批量写入向量与元数据。

        Args:
            records: VectorRecord 列表（chunk_id + text + embedding + metadata 三合一）。

        Raises:
            ValueError: records 为空或长度不齐。
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Dense 向量检索 + 结构化元数据过滤。

        Args:
            query_embedding: 查询向量（dense，调用方负责 embed）。
            top_k:           返回条数，默认 10。
            filters:         结构化过滤条件，支持（未来 Milvus 原生表达）：
                - company_id:    str  企业标识（如 "xiaomi"）
                - year:          int  年份（如 2025，>= 语义）
                - document_type: str  文档类型（如 "annual_report"）
                - section:       str  章节（如 "财务分析"）
                None 表示不筛选。

        Returns:
            list[SearchResult]: 按分数降序排列，分数 ∈ [0, 1]。
        """
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> int:
        """按 chunk_id 删除记录。

        FAISS：逻辑删除（metadata["deleted"] = True），不重建索引。
        Milvus：物理删除。

        Returns:
            实际标记/删除的记录数。
        """
        ...

    @abstractmethod
    def update(self, record: VectorRecord) -> bool:
        """更新单条记录（= delete(chunk_id) + add(record)）。

        Returns:
            True 表示更新成功（原记录存在），False 表示原记录不存在。
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """返回活跃（未删除）记录数。"""
        ...


# ── 本地持久化 Mixin ──────────────────────────────────────────


class LocalVectorStoreMixin:
    """本地文件持久化能力（FAISS 需要；Milvus 由服务端管理，不需要）。

    不放入 VectorStore ABC——Milvus 的持久化由服务端负责，save()/load()
    对其无意义，放进 ABC 会强制无关实现承担空方法。
    """

    def save(self, dir_path: str | Path | None = None) -> None:
        """持久化到 dir_path（默认构造时配置的目录）。"""
        raise NotImplementedError

    def load(self, dir_path: str | Path | None = None) -> None:
        """从 dir_path 加载，覆盖当前内存状态。"""
        raise NotImplementedError

    def validate_integrity(self) -> dict:
        """校验索引完整性（索引行数与元数据条数一致）。

        Returns:
            dict: {"ntotal": int, "metadata_count": int, "active": int,
                   "deleted": int, "consistent": bool}
        """
        raise NotImplementedError


# ── Hybrid 支持 Mixin ─────────────────────────────────────────


class HybridSupportMixin:
    """Dense + Sparse 混合检索 + 全量 chunk 访问。

    不放入 VectorStore ABC——未来可能有只做 dense 的轻量后端。
    """

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: list[float] | None,
        top_k: int = 10,
        filters: dict | None = None,
        dense_weight: float = 0.7,
    ) -> list[SearchResult]:
        """Hybrid 召回（dense + sparse 双路融合）。

        Phase 1：sparse_vector 为 None 时退化为纯 dense 检索。
        """
        raise NotImplementedError

    def all_chunks(self) -> list[VectorRecord]:
        """返回库内全部活跃 chunk（供 BM25 稀疏检索构建语料等）。"""
        raise NotImplementedError

    def vector_of(self, chunk_id: str) -> list[float] | None:
        """返回指定 chunk 的稠密向量（供结果回填 dense_vector）。"""
        raise NotImplementedError
