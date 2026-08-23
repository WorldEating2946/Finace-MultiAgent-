"""Metadata-aware Reranker（PR #33）—— Document Intelligence Rerank。

从"Text Rerank"升级：CrossEncoder 不再只看到 (query, chunk.text)，
而是 (query, [Company][Chapter][Section][Page][Content]) 结构化上下文，
并叠加元数据信号做 Hybrid Score Fusion：

    final_score = α * ce_score_norm + β * section_signal + γ * keyword_signal

**按公司配置权重**（PR #33 关键设计）：
    - CrossEncoder 语义分数对干净文本语料（如 CATL）已最优，任何元数据扰动都是噪声；
    - metadata 增强对分析师查询 + OCR 受损语料（如小米）有显著增益（MRR 0.26→0.42）。
    - 故权重按公司配置：配置的公司走 metadata 融合，未配置的公司走纯 CrossEncoder（零回归）。

不训练模型：信号可解释、权重可调、回归可测（PR #32 门禁保护）。
"""

from __future__ import annotations

from app.core.config import settings
from app.rag.document import DocumentChunk
from app.rag.reranker.context_builder import RerankContextBuilder
from app.rag.reranker.cross_encoder import (
    _MAX_RERANK_TOKENS,
    DEFAULT_RERANKER_PATH,
    CrossEncoderReranker,
)
from app.rag.reranker.section_priority import get_section_priority


class MetadataReranker(CrossEncoderReranker):
    """基于 CrossEncoder + 元数据信号融合的精排器。

    继承 CrossEncoderReranker（复用 _load_model / _resolve_device 惰性加载）。
    融合权重按公司配置：未配置公司 → (1.0, 0.0, 0.0) 纯 CE，保证零回归。
    """

    def __init__(
        self,
        model_path: str = "",
        device: str | None = None,
        company_weights: dict[str, tuple[float, float, float]] | None = None,
    ) -> None:
        super().__init__(model_path or DEFAULT_RERANKER_PATH, device)
        self._context_builder = RerankContextBuilder()
        # company_weights: {"公司": (α, β, γ)}；None 时从 settings.rag_metadata_company_weights 读
        self._company_weights = company_weights or self._load_company_weights()

    @staticmethod
    def _load_company_weights() -> dict[str, tuple[float, float, float]]:
        """从配置读 per-company 权重 {"小米": "0.90,0.08,0.02"}。"""
        out: dict[str, tuple[float, float, float]] = {}
        for company, w in settings.rag_metadata_company_weights.items():
            parts = [float(x) for x in str(w).split(",")]
            if len(parts) == 3:
                out[company] = (parts[0], parts[1], parts[2])
        return out

    def _weights_for(self, company: str) -> tuple[float, float, float]:
        """按公司取融合权重；未配置公司 → 纯 CrossEncoder（α=1, β=0, γ=0）。"""
        return self._company_weights.get(company, (1.0, 0.0, 0.0))

    def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """上下文构造 → CrossEncoder 打分 → 元数据信号 → 按公司融合排序。

        未配置公司 → 完全直通 CrossEncoderReranker（原始 chunk.text，无 context 标签、
        无归一化）——保证干净文本语料（如 CATL）零回归。

        Args:
            query:  查询文本。
            chunks: 召回候选（同一公司）。

        Returns:
            按融合分数降序的 chunk 列表。
        """
        if len(chunks) <= 1:
            return list(chunks)

        # 关键：未配置公司必须直通纯 CE。即使 α=1/β=0/γ=0，格式化 context 的
        # [Company][Chapter] 标签也会改变 CE 打分（标签非原始文本，CE 训练未见过），
        # 对已最优的排序是噪声。故未配置时走父类原始路径（chunk.text + 原始分数）。
        company = chunks[0].company if chunks else ""
        if company not in self._company_weights:
            return CrossEncoderReranker.rerank(self, query, chunks)

        model = self._load_model()

        # 1. 结构化上下文（含公司/章节/小节/页码）
        contexts = [self._context_builder.build(c) for c in chunks]
        pairs = [(query, ctx) for ctx in contexts]
        ce_scores = [float(s) for s in model.predict(pairs, max_length=_MAX_RERANK_TOKENS)]

        # 2. 元数据信号
        section_signal = [self._section_signal(c) for c in chunks]
        keyword_signal = [self._keyword_signal(query, c) for c in chunks]

        # 3. 按公司取权重 + 融合（CE 分数按 max 归一化到 [0,1]，与信号同尺度）
        alpha, beta, gamma = self._weights_for(company)
        ce_max = max(ce_scores) if ce_scores else 1.0
        fused = []
        for i, chunk in enumerate(chunks):
            ce_norm = ce_scores[i] / ce_max if ce_max > 0 else 0.0
            final = (
                alpha * ce_norm
                + beta * section_signal[i]
                + gamma * keyword_signal[i]
            )
            fused.append((chunk, final))

        fused.sort(key=lambda x: -x[1])
        return [c for c, _ in fused]

    # ── 元数据信号 ───────────────────────────────────────────────
    def _section_signal(self, chunk: DocumentChunk) -> float:
        """章节优先级信号：chunk 的 section/chapter 在优先级表中的值，无匹配=0。

        section（小节路径）优先于 chapter（章名）——小节更具体。
        """
        priority = get_section_priority()
        meta = chunk.metadata or {}
        for field in ("section", "chapter"):
            name = meta.get(field) or ""
            if name and name in priority:
                return priority[name]
        return 0.0

    def _keyword_signal(self, query: str, chunk: DocumentChunk) -> float:
        """关键词命中信号：query 分词在 chunk.text 的重叠率 ∈ [0, 1]。

        hits / max(3, len(query_tokens))，封顶 1.0。
        jieba 不可用时（纯数字/英文）返回 0（无词级命中信息）。
        """
        try:
            import jieba

            tokens = [t for t in jieba.lcut(query) if t.strip() and len(t) > 1]
        except ImportError:
            return 0.0
        if not tokens:
            return 0.0
        text = chunk.text or ""
        hits = sum(1 for t in tokens if t in text)
        return min(1.0, hits / max(3, len(tokens)))
