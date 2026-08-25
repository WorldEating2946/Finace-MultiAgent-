"""PR44.1 向量库工厂（公司隔离 + 后端选择）。

与旧 ``app.rag.vector_store.get_vector_store`` 的关系：
    - 本工厂是新接口（VectorStore ABC + filters）的入口；
    - 旧工厂继续服务旧调用方（company 位置参数），两者独立并行；
    - 同一 company 的 FAISS 存档文件（index.faiss + metadata.json）共享，
      新旧实例读取同一磁盘目录。
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.rag.vectorstore.base import VectorStore
from app.rag.vectorstore.faiss_store import FAISSStore
from app.rag.vectorstore.milvus_store import MilvusStore

# 新单例缓存（key = f"{backend}:{company_id}"，与旧 _default_stores 独立）
_stores: dict[str, VectorStore] = {}


def _guard_company_path(company: str) -> None:
    """company 用作知识库子目录名，禁止路径分隔符与越级访问。"""
    if company and ("/" in company or "\\" in company or company in (".", "..")):
        raise ValueError(
            f"company 含非法路径字符，不能作为知识库目录: {company!r}"
        )


def _resolve_backend(backend: str | None) -> str:
    """解析后端：显式传参优先，否则读生产配置 rag_vector_backend（PR44.4）。"""
    return backend or settings.rag_vector_backend


def get_store(
    company_id: str = "",
    dim: int = 128,
    backend: str | None = None,
) -> VectorStore:
    """获取指定 company_id + backend 的向量库单例（新接口唯一入口）。

    Args:
        company_id: 企业标识（一级过滤维度），空串用根目录。
        dim:        向量维度（占位，首次 add() 自动适配实际维度）。
        backend:    None → 读 settings.rag_vector_backend（生产默认）；
                    "faiss"（默认值，离线评测/本地开发钉死使用）/
                    "milvus"（PR44.3.1，生产切换，懒加载 pymilvus）。

    Returns:
        VectorStore 实例（新接口：filters 结构化过滤 + delete/update/count）。

    Raises:
        ValueError: company_id 含非法路径字符，或 backend 未知。
    """
    backend = _resolve_backend(backend)
    _guard_company_path(company_id)
    key = f"{backend}:{company_id}"
    if key not in _stores:
        if backend == "faiss":
            dir_path = Path(settings.rag_vector_store_path) / company_id
            store = FAISSStore(dim=dim, dir_path=dir_path, company=company_id)
            try:
                store.load()
            except FileNotFoundError:
                pass  # 首次访问该公司，尚无持久化索引
            _stores[key] = store
        elif backend == "milvus":
            # dim 用 settings.milvus_dim（BGE-M3=1024）：Milvus 维度在 collection
            # 创建时固定，不能像 FAISS 那样运行时自适应（见设计文档 AD-2）。
            store = MilvusStore(
                dim=settings.milvus_dim,
                uri=settings.milvus_uri,
                collection_name=settings.milvus_collection_name,
                company_id=company_id,
                db_name=settings.milvus_db_name,
            )
            _stores[key] = store
        else:
            raise ValueError(f"未知后端: {backend}")
    return _stores[key]


def clear_store_cache() -> None:
    """清除所有单例缓存（测试 / 存档重建用）。"""
    _stores.clear()
