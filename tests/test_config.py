"""配置模块单元测试。"""

from app.core.config import Settings, settings


def test_settings_defaults():
    """未额外配置时使用合理默认值（与 .env 提供的值一致）。"""
    assert settings.rag_vector_store_path == "data/vector_store"
    assert settings.rag_default_top_k == 5
    assert settings.rag_retrieve_top_k == 20
    assert settings.rag_rerank_top_k == 5
    assert settings.rag_reranker_model == "dummy"
    assert settings.embedding_model == "bge-m3"


def test_settings_reads_env_var(monkeypatch):
    """环境变量可覆盖配置（优先级高于 .env）。"""
    monkeypatch.setenv("RAG_DEFAULT_TOP_K", "7")
    assert Settings().rag_default_top_k == 7


def test_settings_reads_embedding_model_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "dummy")
    assert Settings().embedding_model == "dummy"
