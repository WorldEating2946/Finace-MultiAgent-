"""向量库抽象层（写/查 + 持久化，预留 hybrid_search 接口）。

Phase 1：FAISS 本地文件（零服务依赖，IndexFlatIP + 后置 company 过滤）。
Phase 2：可切换 Milvus / pgvector（接口不变，上层零感知）。

设计约束（来自 docs/RAG_ARCHITECTURE.md §4.2 与 §5.2）：
    - 入库幂等由调用方（pipeline）保证，本层不负责去重；
    - company 作为一级过滤字段，先向量召回再元数据过滤；
    - hybrid_search 接口已预留（dense + sparse 双路入参），Phase 2 启用；
    - 持久化：save()/load() 读写 <settings.rag_vector_store_path>/<company>/
      （index.faiss + metadata.json），get_vector_store() 启动自动加载，
      避免每次启动重复 ingest；
    - 多公司隔离：每个 company 独立子目录，get_vector_store(company) 按公司缓存。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.rag.document import DocumentChunk


class VectorStore(ABC):
    """向量库抽象接口。

    上层（retriever / pipeline）只依赖本抽象，不感知底层实现。
    Phase 1 FAISS → Phase 2 Milvus 切换时上层零改动。
    """

    @abstractmethod
    def add(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        """批量写入向量与元数据。

        Args:
            chunks:  DocumentChunk 列表。
            vectors: 对应的 dense 向量列表，与 chunks 一一对应。

        Raises:
            ValueError: chunks 与 vectors 长度不一致。
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        company: str,
        top_k: int = 10,
    ) -> list[tuple[DocumentChunk, float]]:
        """Dense 向量检索 + company 元数据过滤。

        Args:
            query_vector: 查询向量（dense）。
            company:      一级过滤字段（必填），仅返回匹配的 chunk。
            top_k:        召回数量，默认 10（后续由 Reranker 精排至 3）。

        Returns:
            list[tuple[DocumentChunk, float]]:
                (chunk, 相似度分数) 列表，按分数降序；分数 ∈ [0, 1]。
        """
        raise NotImplementedError

    @abstractmethod
    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: list[float] | None,
        company: str,
        top_k: int = 10,
        dense_weight: float = 0.7,
    ) -> list[tuple[DocumentChunk, float]]:
        """Hybrid 召回（dense + sparse 双路融合）—— Phase 2 启用。

        Phase 1：sparse_vector 为 None 时退化为纯 dense 检索。

        Args:
            dense_vector:  Dense 查询向量（语义）。
            sparse_vector: Sparse 查询向量（关键词），Phase 1 传 None。
            company:       一级过滤字段。
            top_k:         召回数量。
            dense_weight:  Dense 权重 ∈ [0, 1]，sparse 权重 = 1 - dense_weight。

        Returns:
            list[tuple[DocumentChunk, float]]: 加权融合后的结果。
        """
        raise NotImplementedError

    @abstractmethod
    def all_chunks(self) -> list[DocumentChunk]:
        """返回库内全部 chunk（供 BM25 稀疏检索构建语料）。"""
        raise NotImplementedError

    @abstractmethod
    def vector_of(self, chunk_id: str) -> list[float] | None:
        """返回指定 chunk 的稠密向量（供 Hybrid 结果回填 dense_vector）。

        Returns:
            chunk 的 dense 向量；找不到时返回 None。
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, dir_path: str | None = None) -> None:
        """持久化索引与元数据到 dir_path（默认用构造时配置的目录）。

        Phase 1 FAISS：写入 ``index.faiss`` + ``metadata.json``。
        未配置目录时静默跳过（便于测试注入的临时 store 复用同一接口）。
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, dir_path: str | None = None) -> None:
        """从 dir_path 加载索引与元数据，覆盖当前内存状态。

        Phase 1 FAISS：读取 ``index.faiss`` + ``metadata.json``。
        未配置目录时静默跳过；目录存在但文件缺失时抛 FileNotFoundError。
        """
        raise NotImplementedError


class FAISSVectorStore(VectorStore):
    """FAISS 向量库实现（Phase 1）。

    内部维护：
        - ``faiss.IndexFlatIP``：内积索引（向量归一化后等价余弦相似度）。
        - ``_chunks``：并行元数据列表，与 FAISS 索引行一一对应。

    FAISS 不支持原生元数据过滤，因此采用"多召回 + 后置过滤"
    策略：先召回 top_k * 5 个候选，再按 company 过滤取 top_k。
    """

    def __init__(self, dim: int = 128, dir_path: str | Path | None = None):
        self._dim = dim
        self._dir_path = Path(dir_path) if dir_path is not None else None
        self._chunks: list[DocumentChunk] = []
        self._index: faiss.IndexFlatIP | None = None  # noqa: F821

    # ── 延迟导入 faiss ──────────────────────────────────────────
    # faiss-cpu 为 Phase 1 核心依赖，但导入失败时给出明确提示
    # 而非在模块顶层直接崩溃，便于其他模块（如 loader）独立使用。

    @staticmethod
    def _faiss_index_flat(dim: int) -> faiss.IndexFlatIP:  # noqa: F821
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu 未安装，请执行：pip install faiss-cpu"
            )
        return faiss.IndexFlatIP(dim)

    @staticmethod
    def _faiss_normalize_l2(x: np.ndarray) -> None:
        import faiss

        faiss.normalize_L2(x)

    # ── 公开接口 ─────────────────────────────────────────────────

    def add(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks 与 vectors 长度不一致：{len(chunks)} vs {len(vectors)}"
            )

        if self._index is None:
            # 维度由首批向量推导（Dummy 128 / BGE-M3 1024 自动适配，上层零感知）
            self._dim = len(vectors[0])
            self._index = self._faiss_index_flat(self._dim)

        np_vectors = np.array(vectors, dtype=np.float32)
        self._faiss_normalize_l2(np_vectors)

        self._index.add(np_vectors)
        self._chunks.extend(chunks)

    def search(
        self,
        query_vector: list[float],
        company: str,
        top_k: int = 10,
    ) -> list[tuple[DocumentChunk, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []

        q = np.array([query_vector], dtype=np.float32)
        self._faiss_normalize_l2(q)

        # FAISS 不支持原生元数据过滤 → 多召回 + 后置过滤
        fetch_k = min(top_k * 5, self._index.ntotal)
        distances, indices = self._index.search(q, fetch_k)

        results: list[tuple[DocumentChunk, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            if chunk.company != company:
                continue
            # 归一化后内积 ∈ [-1, 1] → 映射到 [0, 1]
            score = (float(dist) + 1.0) / 2.0
            results.append((chunk, score))
            if len(results) >= top_k:
                break

        return results

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: list[float] | None,
        company: str,
        top_k: int = 10,
        dense_weight: float = 0.7,
    ) -> list[tuple[DocumentChunk, float]]:
        """Phase 1：sparse 通路未实现，退化为纯 dense 检索。"""
        if sparse_vector is not None:
            raise NotImplementedError(
                "Hybrid 检索（sparse 通路）待 Phase 2 实现"
            )
        return self.search(dense_vector, company=company, top_k=top_k)

    # ── 数据访问 ────────────────────────────────────────────────

    def all_chunks(self) -> list[DocumentChunk]:
        """返回库内全部 chunk（供 BM25 稀疏检索构建语料）。"""
        return list(self._chunks)

    def vector_of(self, chunk_id: str) -> list[float] | None:
        """返回指定 chunk 的稠密向量（FAISS reconstruct）。"""
        if self._index is None:
            return None
        for i, chunk in enumerate(self._chunks):
            if chunk.chunk_id == chunk_id:
                return self._index.reconstruct(i).tolist()
        return None

    # ── 持久化 ──────────────────────────────────────────────────

    def save(self, dir_path: str | Path | None = None) -> None:
        """将当前索引与元数据写入磁盘（index.faiss + metadata.json）。

        Args:
            dir_path: 目标目录；缺省用构造时配置的目录。均未配置时静默跳过。

        Note:
            空库（未 add 过）时静默跳过，避免写入空存档。
        """
        target = Path(dir_path) if dir_path is not None else self._dir_path
        if target is None or self._index is None:
            return

        import faiss  # 惰性导入，与 _faiss_index_flat 保持一致

        target.mkdir(parents=True, exist_ok=True)
        # 用 serialize + Python 写盘：避免把含非 ASCII（如中文公司名）的路径
        # 传给 faiss C++ 层（Windows 下 fopen 窄字符转码会写错路径）
        blob = faiss.serialize_index(self._index).tobytes()
        (target / "index.faiss").write_bytes(blob)
        # dense_vector / sparse_tokens 为运行时字段（向量已在 index.faiss），不持久化
        metadata = [
            c.model_dump(exclude={"dense_vector", "sparse_tokens"})
            for c in self._chunks
        ]
        (target / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, dir_path: str | Path | None = None) -> None:
        """从磁盘加载索引与元数据，覆盖当前内存状态。

        Args:
            dir_path: 源目录；缺省用构造时配置的目录。均未配置时静默跳过。

        Raises:
            FileNotFoundError: 目录存在但 index.faiss / metadata.json 缺失。
        """
        source = Path(dir_path) if dir_path is not None else self._dir_path
        if source is None:
            return

        index_path = source / "index.faiss"
        meta_path = source / "metadata.json"
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"向量库存档缺失：{source}")

        import faiss  # 惰性导入

        raw = index_path.read_bytes()
        self._index = faiss.deserialize_index(np.frombuffer(raw, dtype=np.uint8))
        self._dim = self._index.d
        self._chunks = [
            DocumentChunk.model_validate(m)
            for m in json.loads(meta_path.read_text(encoding="utf-8"))
        ]


# ── 多公司单例入口 ─────────────────────────────────────────────

_default_stores: dict[str, VectorStore] = {}


def _guard_company_path(company: str) -> None:
    """company 会被用作知识库子目录名，禁止路径分隔符与越级访问。"""
    if company and ("/" in company or "\\" in company or company in (".", "..")):
        raise ValueError(
            f"company 含非法路径字符，不能作为知识库目录: {company!r}"
        )


def get_vector_store(company: str = "", dim: int = 128) -> VectorStore:
    """获取指定 company 的向量库实例（按 company 缓存单例）。

    每个 company 独立持久化于 ``<settings.rag_vector_store_path>/<company>/``：
    ``retrieve(query, company="company_a")`` 会自动加载 company_a 的知识库。
    首次访问自动加载已存索引，无需重复 ingest。

    Phase 2 切换 Milvus / pgvector 时，只修改本函数
    （如读取配置决定具体实现），上层代码无需改动。

    Args:
        company: 企业名（一级过滤字段），决定加载哪个知识库子目录；空串用根目录。
        dim:     向量维度（默认 128）。首次 add() 未建索引时会按实际向量
                 维度重建索引，故 Dummy(128) / BGE-M3(1024) 均自动适配。

    Raises:
        ValueError: company 含路径分隔符或 "." / ".."。
    """
    _guard_company_path(company)
    if company not in _default_stores:
        dir_path = Path(settings.rag_vector_store_path) / company
        store = FAISSVectorStore(dim=dim, dir_path=dir_path)
        try:
            store.load()
        except FileNotFoundError:
            pass  # 首次访问该公司，尚无持久化索引
        _default_stores[company] = store
    return _default_stores[company]


# ── PR44 facade ────────────────────────────────────────────────
# 本文件保留旧接口（VectorStore / FAISSVectorStore / get_vector_store）
# 以保持向后兼容，不修改任何现有代码。新代码请使用：
#
#   from app.rag.vectorstore import (
#       VectorStore,      # 新抽象接口（filters 结构化过滤 + delete/update/count）
#       FAISSStore,       # 包装本文件 FAISSVectorStore 的新实现
#       VectorRecord,     # 统一数据模型（chunk_id + text + embedding + metadata）
#       get_store,        # 新工厂（company_id + backend，PR44.2 起唯一入口）
#   )
#
# PR44.2 已完成调用方迁移（retriever / ingestion / evaluation / benchmark
# 全部走 app.rag.vectorstore）。本文件已降级为 deprecated 兼容层——
# 保留是为了不破坏既有测试与外部调用，新功能不再添加。
