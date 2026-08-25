"""PR44.1 FAISS 实现（包装旧 FAISSVectorStore，接口适配到新 VectorStore ABC）。

PR44.1 策略（AD1）：不重写 FAISS 逻辑，通过包装委托保证行为 100% 一致。
持久化格式不变（index.faiss + metadata.json）。

逻辑删除（AD4）：FAISS 索引不支持单行删除，故 delete() 在 chunk 的
metadata["deleted"] = True 上做标记（随 metadata.json 持久化），
search()/all_chunks()/count() 均排除已删除项。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.rag.vector_store import FAISSVectorStore  # 旧实现（facade 保留）
from app.rag.vectorstore.base import (
    HybridSupportMixin,
    LocalVectorStoreMixin,
    VectorStore,
)
from app.rag.vectorstore.models import SearchResult, VectorRecord

# 过滤键的别名解析：统一外部 filter 键 → 内部查找顺序
# document_type 在旧模型中分两层：doc_type（格式）与 metadata["source_type"]（语义类型）。
_FILTER_ALIASES: dict[str, tuple] = {
    "document_type": ("document_type", "source_type", "doc_type"),
    "company_id": ("company",),  # company_id 实际由 search 层翻译，此处仅为兜底
}


class FAISSStore(VectorStore, LocalVectorStoreMixin, HybridSupportMixin):
    """FAISS 向量库（包装旧 FAISSVectorStore，新接口适配）。

    Args:
        dim:      向量维度（占位，首次 add() 自动适配实际维度）。
        dir_path: 持久化目录（<root>/<company>/）。
        company:  所属企业（一级过滤上下文；空串表示根目录无隔离）。
    """

    def __init__(
        self,
        dim: int = 128,
        dir_path: str | Path | None = None,
        company: str = "",
    ):
        self._company = company
        self._dir_path = Path(dir_path) if dir_path else None
        self._legacy = FAISSVectorStore(dim=dim, dir_path=dir_path)

    # ── VectorStore ABC ────────────────────────────────────────

    def add(self, records: list[VectorRecord]) -> None:
        """批量写入（幂等：已存在的活跃 chunk_id 跳过）。"""
        if not records:
            return
        existing_active = {
            c.chunk_id
            for c in self._legacy._chunks
            if not c.metadata.get("deleted")
        }
        new_records = [r for r in records if r.chunk_id not in existing_active]
        if not new_records:
            return
        chunks = [r.to_document_chunk(self._company) for r in new_records]
        vectors = [r.embedding for r in new_records]
        self._legacy.add(chunks, vectors)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Dense 检索 + filters 过滤。

        company_id 由 search 层翻译为 company 后置过滤（FAISS 后端）；
        其余 filter（year / document_type / section 等）在候选集上做元数据匹配。
        """
        filters = filters or {}
        # company 翻译：filters.company_id 缺省用构造时公司；指定了别的公司 → 空
        company = filters.get("company_id", self._company)
        if self._company and company != self._company:
            return []

        # 多召回 + 后置过滤：向旧实现多取 top_k*5，为元数据过滤留余量
        if self._legacy._index is None or self._legacy._index.ntotal == 0:
            return []
        fetch_k = min(top_k * 5, self._legacy._index.ntotal)
        raw = self._legacy.search(
            query_embedding, company=company, top_k=fetch_k
        )

        results: list[SearchResult] = []
        for chunk, score in raw:
            if chunk.metadata.get("deleted"):
                continue
            if not self._match_chunk(chunk, filters):
                continue
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=score,
                    metadata=dict(chunk.metadata),
                )
            )
            if len(results) >= top_k:
                break
        return results

    def delete(self, ids: list[str]) -> int:
        """逻辑删除：标记 metadata["deleted"] = True（随存档持久化）。"""
        count = 0
        for cid in ids:
            if not cid:
                continue
            for chunk in self._legacy._chunks:
                if chunk.chunk_id == cid and not chunk.metadata.get("deleted"):
                    chunk.metadata["deleted"] = True
                    count += 1
                    break
        return count

    def update(self, record: VectorRecord) -> bool:
        """更新 = 删除旧 chunk_id（逻辑删除）+ 插入新记录。"""
        existed = any(c.chunk_id == record.chunk_id for c in self._legacy._chunks)
        self.delete([record.chunk_id])
        self.add([record])
        return existed

    def count(self) -> int:
        """返回活跃（未删除）记录数。"""
        return sum(
            1
            for c in self._legacy._chunks
            if not c.metadata.get("deleted")
        )

    def compact(self) -> int:
        """物理删除已标记 chunk，重建 FAISS 索引与 metadata.json。

        流程：
            1. 收集活跃（未删除）chunk 的索引位置；
            2. reconstruct 活跃向量 → 重建 IndexFlatIP；
            3. 重建 _chunks 列表（只保留活跃 chunk，逻辑删除标记随之消失）；
            4. 自动 save()（配置了目录时落盘）。

        Returns:
            被物理删除的 chunk 数（= compact 前的 deleted 标记数）。

        Note:
            重建成本正比于 ntotal（小米 ~5000 chunks 为秒级）。不自动触发——
            由人工 / 后台任务显式调用，避免在线查询被打断。
        """
        legacy = self._legacy
        if legacy._index is None or legacy._index.ntotal == 0:
            return 0

        active_indices = [
            i
            for i, c in enumerate(legacy._chunks)
            if not c.metadata.get("deleted")
        ]
        removed = len(legacy._chunks) - len(active_indices)
        if removed == 0:
            return 0

        # reconstruct 活跃向量（IndexFlatIP 行序 = 插入序，位置索引稳定）
        new_vectors = np.array(
            [legacy._index.reconstruct(i) for i in active_indices],
            dtype=np.float32,
        )
        legacy._chunks = [legacy._chunks[i] for i in active_indices]
        legacy._index = legacy._faiss_index_flat(legacy._dim)
        legacy._faiss_normalize_l2(new_vectors)
        legacy._index.add(new_vectors)

        self.save()
        return removed

    # ── LocalVectorStoreMixin ──────────────────────────────────

    def save(self, dir_path: str | Path | None = None) -> None:
        """持久化（metadata["deleted"] 标记随 metadata.json 一并落盘）。"""
        self._legacy.save(dir_path)

    def load(self, dir_path: str | Path | None = None) -> None:
        """从磁盘加载（deleted 标记在 chunk.metadata 中，随载入自动恢复）。"""
        self._legacy.load(dir_path)

    def validate_integrity(self) -> dict:
        """校验索引完整性（faiss ntotal == metadata 条数）。"""
        index_count = self._legacy._index.ntotal if self._legacy._index else 0
        meta_count = len(self._legacy._chunks)
        deleted = sum(
            1 for c in self._legacy._chunks if c.metadata.get("deleted")
        )
        return {
            "ntotal": index_count,
            "metadata_count": meta_count,
            "active": self.count(),
            "deleted": deleted,
            "consistent": index_count == meta_count,
        }

    # ── HybridSupportMixin ─────────────────────────────────────

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: list[float] | None,
        top_k: int = 10,
        filters: dict | None = None,
        dense_weight: float = 0.7,
    ) -> list[SearchResult]:
        """Phase 1：sparse 通路未实现，退化为纯 dense 检索。"""
        if sparse_vector is not None:
            raise NotImplementedError(
                "Hybrid 检索（sparse 通路）待 Phase 2 实现"
            )
        return self.search(dense_vector, top_k=top_k, filters=filters)

    def all_chunks(self) -> list[VectorRecord]:
        """返回全部活跃 chunk（排除已删除）。"""
        result: list[VectorRecord] = []
        for c in self._legacy._chunks:
            if c.metadata.get("deleted"):
                continue
            result.append(
                VectorRecord.from_document_chunk(
                    c, self._legacy.vector_of(c.chunk_id) or []
                )
            )
        return result

    def vector_of(self, chunk_id: str) -> list[float] | None:
        """返回指定 chunk 的稠密向量（供结果回填 dense_vector）。"""
        return self._legacy.vector_of(chunk_id)

    # ── 内部辅助 ───────────────────────────────────────────────

    @classmethod
    def _match_chunk(cls, chunk, filters: dict) -> bool:
        """检查 chunk 是否匹配所有 filter 条件（除 company_id 已由 search 处理）。"""
        for key, value in filters.items():
            if key == "company_id":
                continue  # company 过滤已在 search 层完成
            actual = cls._resolve_field(chunk, key)
            if actual is None:
                return False
            # 数值 filter 采用 >= 语义（如 year >= 2024）
            if isinstance(value, (int, float)) and isinstance(
                actual, (int, float)
            ):
                if actual < value:
                    return False
            elif str(actual) != str(value):
                return False
        return True

    @staticmethod
    def _resolve_field(chunk, key: str):
        """解析 filter 键对应的实际值（metadata dict 优先，回退 chunk 顶层字段）。"""
        if key in chunk.metadata and chunk.metadata[key] is not None:
            return chunk.metadata[key]
        aliases = _FILTER_ALIASES.get(key, (key,))
        for alias in aliases:
            if alias in chunk.metadata and chunk.metadata[alias] is not None:
                return chunk.metadata[alias]
            attr = getattr(chunk, alias, None)
            if attr is not None:
                return attr
        return None
