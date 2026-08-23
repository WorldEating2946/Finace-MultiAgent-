"""Embedding 具体实现集合。

Phase 1：本地 BGE-M3（dense 1024 维）。
主流程（ingestion / retriever / pipeline）只依赖 app.rag.embedding
的抽象接口，不直接感知本包内部实现。
"""
