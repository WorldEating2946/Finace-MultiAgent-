"""研究执行工具集（PR #37）。

对已有能力的薄抽象层 —— Executor 只依赖这些工具签名，不直接调 RAG/Profile/Fusion。
可 mock（测试）、可替换（未来 LangGraph Tool 化零改动）。

工具：
    profile_lookup    企业知识画像（build_profile）
    knowledge_search  知识检索（retrieve）
    evidence_search   证据检索（retrieve → EvidenceRef，带 source/page/quote）
    conflict_analysis 跨源冲突检测（SourceFusion）
"""

from __future__ import annotations

from app.rag.profile.schema import CompanyProfile, EvidenceRef
from app.rag.research.state import Finding


class ResearchTools:
    """执行工具集 —— 每个工具都是已有能力的封装。"""

    # ── 企业知识画像 ───────────────────────────────────────────
    def profile_lookup(self, company: str) -> CompanyProfile:
        """企业知识画像（多源融合的独立画像）。"""
        from app.rag.profile import build_profile

        return build_profile(company)

    # ── 知识检索 ───────────────────────────────────────────────
    def knowledge_search(
        self,
        query: str,
        company: str,
        source_types: list[str] | None = None,
        top_k: int = 5,
    ):
        """知识检索：RAG retrieve()（source_types 留作后续原生过滤增强）。"""
        from app.rag import retrieve

        return retrieve(query, company=company, top_k=top_k)

    # ── 证据检索 ───────────────────────────────────────────────
    def evidence_search(
        self,
        query: str,
        company: str,
        source_types: list[str] | None = None,
        top_k: int = 5,
    ) -> list[EvidenceRef]:
        """证据检索：RAG → DocumentChunk → EvidenceRef（带 source/page/quote）。

        source_type 过滤在工具层做（post-retrieval），不侵入检索管线。
        """
        result = self.knowledge_search(query, company, source_types, top_k * 2)
        refs: list[EvidenceRef] = []
        for c in result.chunks:
            meta = c.metadata or {}
            refs.append(
                EvidenceRef(
                    source=c.source or meta.get("source", ""),
                    source_type=meta.get("source_type", "") or "",
                    chapter=meta.get("chapter", "") or "",
                    section=meta.get("section", "") or "",
                    page=c.page,
                    quote=(c.text or "").strip()[:200],  # 前 200 字作为原文引用
                    chunk_id=c.chunk_id,
                )
            )
        if source_types:
            refs = [r for r in refs if r.source_type in source_types]
        return refs[:top_k]

    # ── 跨源冲突检测 ───────────────────────────────────────────
    def conflict_analysis(
        self,
        company: str,
        sources: list[str] | None = None,
    ) -> list:
        """跨源冲突检测：SourceFusion 融合画像并返回冲突列表。"""
        from app.rag.source import SourceFusion

        return SourceFusion(company, sources=sources).build().conflicts


# ── 便捷：由 ResearchStep 构建 Finding（executor 复用）─────────
def step_finding(step_order: int, step_name: str,
                 evidence: list[EvidenceRef],
                 source_types: list[str]) -> Finding:
    """构建步骤 Finding（含 source_types 元信息）。"""
    return Finding(
        step_order=step_order,
        step_name=step_name,
        evidence=evidence,
        source_types=source_types,
    )
