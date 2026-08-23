"""CrossEncoder 精排器（BAAI/bge-reranker-v2-m3）。

对企业年报场景：query 与候选 chunk 逐对打分，重排序。
与 BGE-M3 embedding 同生态、中文优秀、支持本地部署。

惰性加载：构造不加载模型（约 2.2GB），首次 rerank() 才加载。
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.rag.document import DocumentChunk
from app.rag.reranker.base import Reranker

# 本地 BGE-reranker-v2-m3 模型目录（app/models/reranker/bge-reranker-v2-m3）
DEFAULT_RERANKER_PATH = str(
    Path(__file__).resolve().parents[2] / "models" / "reranker" / "bge-reranker-v2-m3"
)

# 精排输入最大 token 数：截断长表格 chunk（几千字符）避免注意力耗时爆炸。
# 对重排序（非最终答案），前段上下文足以判相关性。
_MAX_RERANK_TOKENS = 1024


class CrossEncoderReranker(Reranker):
    """基于 CrossEncoder 的精排器（query, chunk 逐对打分）。"""

    def __init__(self, model_path: str = DEFAULT_RERANKER_PATH, device: str | None = None):
        self._model_path = model_path
        self._device = device or settings.rag_embedding_device
        self._model = None  # 惰性加载

    def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """对候选 chunk 逐对打分并按相关度降序。

        Args:
            query:  查询文本。
            chunks: 召回候选（30~50 条）。

        Returns:
            重排序后的 chunk 列表（相关度降序）。
        """
        if len(chunks) <= 1:
            return list(chunks)
        model = self._load_model()
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = model.predict(pairs, max_length=_MAX_RERANK_TOKENS)
        ordered = [
            chunk
            for _, chunk in sorted(zip(scores, chunks), key=lambda x: -float(x[0]))
        ]
        return ordered

    def _load_model(self):
        """惰性加载 CrossEncoder 模型（首次 rerank() 时触发）。"""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "CrossEncoder 精排需要 sentence-transformers，请执行："
                    "uv pip install -r requirements.txt"
                )
            device = self._resolve_device()
            self._model = CrossEncoder(self._model_path, device=device)
            if device == "cuda":
                # GPU 用 fp16 减半显存：与 BGE-M3(2.2G) 同载 8GB 显卡时避免 OOM 抖动
                self._model.model.half()
        return self._model

    def _resolve_device(self) -> str:
        """解析推理设备：auto/cuda 优先 GPU（BGE-M3 已转 fp16 减半显存，容得下），无 GPU 回退 cpu。"""
        if self._device == "cpu":
            return "cpu"
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
