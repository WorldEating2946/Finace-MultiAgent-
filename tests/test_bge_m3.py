"""BGE-M3 Embedding 实现单元测试（不加载真实模型，保持快速）。

说明：真实模型约 2.2GB，测试不触发模型加载（惰性加载设计保证
构造与 embed([]) 均不加载模型），真实链路由冒烟脚本验证。
"""

from pathlib import Path

from app.rag.embeddings.bge_m3 import (
    DEFAULT_MODEL_PATH,
    BGE_M3EmbeddingModel,
)


def test_default_model_path_points_to_local_bge_m3():
    """默认模型路径应指向仓库内已提供的本地模型目录。"""
    model_dir = Path(DEFAULT_MODEL_PATH)
    assert model_dir.is_dir(), f"模型目录不存在: {model_dir}"
    assert (model_dir / "config_sentence_transformers.json").exists()
    assert (model_dir / "modules.json").exists()


def test_bge_m3_construct_without_loading_model():
    """构造不加载 2.2GB 模型（惰性加载，避免拖慢单测）。"""
    model = BGE_M3EmbeddingModel()
    assert model._model is None


def test_bge_m3_dense_dim_is_1024():
    """BGE-M3 dense 向量维度应为 1024。"""
    assert BGE_M3EmbeddingModel().dim == 1024


def test_bge_m3_embed_empty_list_returns_empty():
    """空列表直接返回空（不触发模型加载）。"""
    model = BGE_M3EmbeddingModel()
    assert model.embed([]) == []
