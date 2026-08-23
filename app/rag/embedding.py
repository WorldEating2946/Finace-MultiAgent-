"""Embedding 抽象层（接口 + 工厂，不绑定具体模型）。

职责：文本 → 向量。屏蔽具体模型实现，使 RAG 主流程
（vector_store / retriever / pipeline）不依赖特定厂商。

具体实现：
    DummyEmbeddingModel → 开发占位（128 维，单元测试注入用）
    BGE_M3EmbeddingModel → 真实模型默认接入（app/rag/embeddings/bge_m3.py，
                           1024 维 dense），Phase 2 扩展 sparse 输出
切换模型时只改 get_embedding_model()，主流程不变。
"""

from abc import ABC, abstractmethod
from hashlib import md5

from app.core.config import settings


class EmbeddingModel(ABC):
    """Embedding 模型抽象接口。

    具体实现需提供 embed()：将文本批量编码为 dense 向量。
    主流程只依赖本抽象，不感知底层模型。
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量编码文本，返回 dense 向量列表（与 texts 一一对应）。

        Args:
            texts: 待编码的文本列表。

        Returns:
            list[list[float]]：每个文本对应的向量（float 列表）。
        """
        raise NotImplementedError


class DummyEmbeddingModel(EmbeddingModel):
    """开发占位实现（非生产）：基于 MD5 的确定性伪向量。

    用途：Phase 1 在真实模型接入前打通 pipeline 与单元测试；
    相同文本始终得到相同向量，便于调试。
    生产环境请替换为真实模型实现。
    """

    def __init__(self, dim: int = 128):
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = md5(text.encode("utf-8")).digest()
        return [float(digest[i % len(digest)]) / 255.0 for i in range(self._dim)]


_default_model: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    """获取当前 Embedding 模型（进程内单例）。

    模型由 ``settings.embedding_model`` 决定：
        - "bge-m3"（默认）→ 本地 BGE-M3（1024 维 dense）；
        - "dummy"        → DummyEmbeddingModel（开发占位，无 torch 依赖）。

    DummyEmbeddingModel 保留，供单元测试显式注入与离线环境使用。

    惰性 import BGE_M3EmbeddingModel：避免与 bge_m3.py 的循环依赖，
    且未安装 torch / sentence-transformers 时本模块仍可正常导入。
    """
    global _default_model
    if _default_model is None:
        if settings.embedding_model == "dummy":
            _default_model = DummyEmbeddingModel()
        else:
            from app.rag.embeddings.bge_m3 import BGE_M3EmbeddingModel

            _default_model = BGE_M3EmbeddingModel()
    return _default_model
