"""PR44.1 VectorStore 包——企业级向量库抽象层。

用法（新代码）：
    from app.rag.vectorstore import (
        VectorStore,       # 抽象接口（add/search/delete/update/count）
        FAISSStore,        # FAISS 实现（包装旧 FAISSVectorStore）
        MilvusStore,       # Milvus 实现（PR44.3.1，pymilvus 可选懒加载）
        VectorRecord,      # 统一数据模型
        SearchResult,      # search() 返回值
        get_store,         # 工厂（company_id + backend）
    )

旧代码继续使用 app.rag.vector_store（facade 保留，见该模块注释）。
"""

from app.rag.vectorstore.base import (
    HybridSupportMixin,
    LocalVectorStoreMixin,
    VectorStore,
)
from app.rag.vectorstore.factory import clear_store_cache, get_store
from app.rag.vectorstore.faiss_store import FAISSStore
from app.rag.vectorstore.health import check_backend_ready
from app.rag.vectorstore.milvus_store import MilvusStore
from app.rag.vectorstore.models import SearchResult, VectorRecord

__all__ = [
    "FAISSStore",
    "HybridSupportMixin",
    "LocalVectorStoreMixin",
    "MilvusStore",
    "SearchResult",
    "VectorRecord",
    "VectorStore",
    "check_backend_ready",
    "clear_store_cache",
    "get_store",
]
