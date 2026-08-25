"""精排器集合（抽象 + 实现 + 工厂）。

由 settings.rag_reranker_model 决定：
    - "dummy" → DummyReranker（开发/测试，直通，无模型依赖）；
    - "metadata" → MetadataReranker（CrossEncoder + 元数据信号融合，PR #33）；
    - 其他（模型路径）→ CrossEncoderReranker（bge-reranker-v2-m3）。

对外：
    from app.rag.reranker import get_reranker
"""

from __future__ import annotations

from app.core.config import settings
from app.rag.reranker.base import Reranker
from app.rag.reranker.dummy import DummyReranker

__all__ = ["DummyReranker", "Reranker", "get_reranker"]

_default_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    """获取当前精排器（进程内单例）。

    真实模型默认接入：配置 rag_reranker_model 指向模型路径时用 CrossEncoder；
    "metadata" 时用 MetadataReranker（语义 + 章节/关键词信号融合，PR #33）；
    "dummy" 时用 DummyReranker（直通，无 2.2GB 模型加载）。
    """
    global _default_reranker
    if _default_reranker is None:
        if settings.rag_reranker_model == "dummy":
            _default_reranker = DummyReranker()
        elif settings.rag_reranker_model == "metadata":
            from app.rag.reranker.metadata_reranker import MetadataReranker

            _default_reranker = MetadataReranker()
        else:
            from app.rag.reranker.cross_encoder import CrossEncoderReranker

            _default_reranker = CrossEncoderReranker(model_path=settings.rag_reranker_model)
    return _default_reranker
