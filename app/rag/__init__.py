"""RAG 模块。

对外仅暴露公共接口，内部实现（embedding / vector_store /
retriever / reranker）不对外暴露。

    from app.rag import retrieve          # 检索
    from app.rag import load_documents    # 文档加载
    from app.rag import split_documents   # 文档切片
"""

from app.rag.ingestion import ingest
from app.rag.loaders import load_documents
from app.rag.pipeline import retrieve
from app.rag.splitter import split_documents

__all__ = ["ingest", "load_documents", "retrieve", "split_documents"]
