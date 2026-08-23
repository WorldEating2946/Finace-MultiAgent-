"""Embedding 抽象层单元测试。"""

import pytest

from app.rag.embedding import (
    DummyEmbeddingModel,
    EmbeddingModel,
    get_embedding_model,
)
from app.rag.embeddings.bge_m3 import BGE_M3EmbeddingModel


def test_embedding_model_is_abstract():
    # 抽象接口不可直接实例化
    with pytest.raises(TypeError):
        EmbeddingModel()


def test_dummy_embedding_returns_vector_per_text():
    model = DummyEmbeddingModel(dim=128)
    texts = ["宁德时代简介", "动力电池行业"]

    vectors = model.embed(texts)

    assert len(vectors) == len(texts)
    assert all(len(v) == 128 for v in vectors)


def test_dummy_embedding_is_deterministic():
    model = DummyEmbeddingModel()
    texts = ["宁德时代是全球动力电池龙头"]

    assert model.embed(texts) == model.embed(texts)


def test_dummy_embedding_empty_list():
    model = DummyEmbeddingModel()
    assert model.embed([]) == []


def test_get_embedding_model_returns_singleton():
    model = get_embedding_model()
    assert model is get_embedding_model()
    assert isinstance(model, EmbeddingModel)
    # 真实模型已默认接入（构造不加载模型，惰性加载）
    assert isinstance(model, BGE_M3EmbeddingModel)


def test_get_embedding_model_uses_config(monkeypatch):
    """settings.embedding_model='dummy' 时应返回 DummyEmbeddingModel（无 torch 依赖）。"""
    from app.rag import embedding as emb

    monkeypatch.setattr(emb.settings, "embedding_model", "dummy")
    monkeypatch.setattr(emb, "_default_model", None)

    assert isinstance(emb.get_embedding_model(), emb.DummyEmbeddingModel)
