"""BGE-M3 本地 Embedding 实现（Phase 1：仅启用 dense 通路）。

基于 sentence-transformers 加载本地 BGE-M3 模型，输出 1024 维 dense 向量。
BGE-M3 原生支持 dense + sparse + colbert 三路，Phase 1 只使用 dense，
sparse 检索（hybrid_search）留待 Phase 2（届时需扩展 vector_store / retriever）。

设计约束：
    - 惰性加载：构造时不加载模型（约 2.2GB），首次 embed() 才加载，
      避免单元测试与不依赖 torch 的模块被拖慢；
    - sentence-transformers / torch 为可选依赖：模块导入不硬依赖它们，
      在 embed() 内延迟 import，缺失时抛出明确提示。
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.rag.embedding import EmbeddingModel

# 本地 BGE-M3 模型目录（相对本项目 app/models/embedding/bge-m3）
DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parents[2] / "models" / "embedding" / "bge-m3"
)

_BGE_M3_DENSE_DIM = 1024  # BGE-M3 dense 向量维度

# 编码批大小（按最长文本自适应）：
#   短文本（≤800 字符）→ 64，提升吞吐；
#   长文本（表格可达数千字符）→ 16 / 8，避免小显存 GPU OOM 退化。
_BATCH_SHORT, _BATCH_MEDIUM, _BATCH_LONG = 64, 16, 8


def _resolve_batch_size(texts: list[str]) -> int:
    max_len = max((len(t) for t in texts), default=0)
    if max_len > 2000:
        return _BATCH_LONG
    if max_len > 800:
        return _BATCH_MEDIUM
    return _BATCH_SHORT


class BGE_M3EmbeddingModel(EmbeddingModel):
    """基于本地 BGE-M3 的 Dense Embedding 实现（1024 维）。

    由 DummyEmbeddingModel（128 维占位）切换为本实现时，主流程
    （ingestion / retriever / pipeline）无需改动 —— 抽象接口
    ``embed(texts)`` 保持不变。

    设备：默认 ``settings.rag_embedding_device``（cpu / cuda）；
    cuda 不可用（未装 CUDA torch / 无 GPU）时自动回退 cpu。
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str | None = None):
        self._model_path = model_path
        self._device = device or settings.rag_embedding_device
        self._model = None  # 惰性加载：构造时不加载模型

    @property
    def dim(self) -> int:
        """Dense 向量维度（BGE-M3 = 1024）。"""
        return _BGE_M3_DENSE_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量编码文本为 1024 维 dense 向量（已 L2 归一化）。

        归一化后与 FAISS ``IndexFlatIP``（内积）配合即等价余弦相似度，
        与 vector_store 的 ``normalize_L2`` 策略一致。
        """
        if not texts:
            return []
        model = self._load_model()
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=_resolve_batch_size(texts),
        )
        return vectors.tolist()

    def _load_model(self):
        """惰性加载 sentence-transformers 模型（首次 embed() 时触发）。"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "BGE-M3 需要 sentence-transformers + torch，请先执行："
                    "pip install -r requirements.txt"
                )
            device = self._resolve_device()
            self._model = SentenceTransformer(self._model_path, device=device)
            # GPU 用 fp16 减半显存：BGE-M3(fp32 3.4GB) + CrossEncoder 同载 8GB 卡会 OOM，
            # 转 fp16(~1.7GB) 才容得下 reranker 也上 GPU → 检索 rerank 秒级。
            if device == "cuda":
                try:
                    self._model.half()
                except Exception:  # noqa: BLE001 —— fp16 失败保持 fp32（质量优先）
                    pass
        return self._model

    def _resolve_device(self) -> str:
        """解析推理设备。

        cpu → 强制 cpu；cuda / auto（默认） → 优先 GPU，未装 CUDA torch 或无 GPU 则回退 cpu。
        """
        if self._device == "cpu":
            return "cpu"
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
