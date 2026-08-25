"""研究报告质量评测（PR37.5）。

在 RAG 检索评测（PR #32，app/rag/evaluation/）之上，增加"研究产出质量"评测：
报告层的证据覆盖率 / 引用准确性 / 计划完整度 / 步骤产出率。

指标（规则驱动，零 LLM 成本）：
    - Evidence Coverage = 有证据 claim 数 / 总 claim 数
    - Citation Accuracy = 引用证据四字段完整度（chunk_id / source / quote / page）
    - Claim Alignment   = claim ↔ quote 的 token 对齐度（jieba Jaccard，复用 conflict.py）
    - Completeness      = 实际完成步骤 / 计划步骤（PR38 动态补步才有 <1.0 的价值）
    - Step Yield        = 有证据产出的检索步骤占比（low_yield_steps = PR38 补步候选）

纯函数，不依赖 LLM / 向量库，可独立测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.rag.profile.schema import EvidenceRef
from app.rag.research.report import ResearchReport
from app.rag.research.state import ResearchState
from app.rag.source.conflict import _jaccard, _tokens  # noqa: WPS450 复用 jieba tokenizer

# claim↔quote 对齐度阈值：claim 是 LLM 摘要、quote 是原文，允许适度改写，
# 比冲突检测（0.3）更宽松——只拦截明显的"引错 chunk"（完全无关 token → Jaccard≈0）。
_ALIGN_THRESHOLD = 0.2


@dataclass
class ClaimEval:
    """单条 claim 的评测明细（可审计）。"""

    claim: str
    n_evidence: int
    supported: bool               # 至少 1 条证据支撑
    citation_ok: bool             # 有证据且全部四字段完整
    alignment: float              # claim↔quote max jieba Jaccard
    alignment_ok: bool            # alignment >= _ALIGN_THRESHOLD
    source_types: list[str] = field(default_factory=list)


@dataclass
class ResearchMetrics:
    """研究报告质量指标汇总（纯 dataclass，JSON 可序列化）。"""

    # ── Evidence Coverage ──
    total_claims: int = 0
    supported_claims: int = 0
    evidence_coverage: float = 0.0
    # ── Citation Accuracy ──
    citation_total: int = 0       # 所有 claim 引用的证据总数
    citation_ok: int = 0          # 四字段完整的证据数
    citation_accuracy: float = 0.0
    aligned_claims: int = 0
    claim_alignment: float = 0.0  # aligned / total（无证据 claim 视为未对齐）
    # ── Completeness ──
    plan_steps: int = 0
    completed_steps: int = 0
    completeness: float = 0.0     # 无 state 时 = 0.0
    # ── Step Yield（PR38 信号）──
    search_steps: int = 0
    yield_steps: int = 0
    step_yield: float = 0.0
    low_yield_steps: list[int] = field(default_factory=list)  # 0 证据步骤 order
    # ── 明细（审计）──
    claim_evals: list[ClaimEval] = field(default_factory=list)


def evaluate_report(
    report: ResearchReport,
    *,
    state: ResearchState | None = None,
) -> ResearchMetrics:
    """评测研究报告质量（纯函数，零副作用）。

    Args:
        report: ResearchReport（LLM 合成后的结构化报告）。
        state:  ResearchState（可选——不传则 completeness / yield 退化为 0）。

    Returns:
        ResearchMetrics（全部指标 + claim 级明细）。
    """
    claim_evals = _evaluate_claims(report)
    total = len(claim_evals)
    supported = sum(1 for c in claim_evals if c.supported)
    aligned = sum(1 for c in claim_evals if c.alignment_ok)

    # 引用准确度（per-evidence）：四字段完整证据数 / 总证据数
    citation_total = 0
    citation_ok = 0
    for claim in list(report.advantages) + list(report.risks):
        for e in claim.evidence:
            citation_total += 1
            if _evidence_field_ok(e):
                citation_ok += 1

    comp = _evaluate_completeness(state)
    return ResearchMetrics(
        total_claims=total,
        supported_claims=supported,
        evidence_coverage=supported / total if total else 0.0,
        citation_total=citation_total,
        citation_ok=citation_ok,
        citation_accuracy=citation_ok / citation_total if citation_total else 0.0,
        aligned_claims=aligned,
        claim_alignment=aligned / total if total else 0.0,
        plan_steps=comp["plan_steps"],
        completed_steps=comp["completed_steps"],
        completeness=comp["completeness"],
        search_steps=comp["search_steps"],
        yield_steps=comp["yield_steps"],
        step_yield=comp["step_yield"],
        low_yield_steps=comp["low_yield_steps"],
        claim_evals=claim_evals,
    )


# ── claim 级评估 ────────────────────────────────────────────────

def _evaluate_claims(report: ResearchReport) -> list[ClaimEval]:
    """逐 claim 评估：证据支撑 + 引用完整度 + claim↔quote 对齐。"""
    return [_eval_claim(claim) for claim in list(report.advantages) + list(report.risks)]


def _eval_claim(claim) -> ClaimEval:
    """单条 claim 的评测（advantages / risks 共用 ReportClaim）。"""
    evs = claim.evidence
    n = len(evs)
    # claim↔quote 对齐度：取该 claim 所有证据 quote 中的最大 token Jaccard
    alignment = max(
        (_jaccard(_tokens(claim.claim), _tokens(e.quote)) for e in evs if e.quote),
        default=0.0,
    )
    return ClaimEval(
        claim=claim.claim,
        n_evidence=n,
        supported=n > 0,
        citation_ok=n > 0 and all(_evidence_field_ok(e) for e in evs),
        alignment=alignment,
        alignment_ok=n > 0 and alignment >= _ALIGN_THRESHOLD,
        source_types=list(claim.source_types),
    )


def _evidence_field_ok(e: EvidenceRef) -> bool:
    """引用四字段完整：chunk_id（解析到真实 chunk）+ source + quote + page。"""
    return bool(e.chunk_id) and bool(e.source) and bool(e.quote) and e.page is not None


# ── state 级评估（completeness + step yield）────────────────────

def _evaluate_completeness(state: ResearchState | None) -> dict:
    """从 state 提取 completeness + step yield 指标。

    无 state → 全部退化为 0。
    画像步骤（profile 入 state、无 finding）不计入 search_steps。
    """
    if state is None:
        return {
            "plan_steps": 0, "completed_steps": 0, "completeness": 0.0,
            "search_steps": 0, "yield_steps": 0, "step_yield": 0.0,
            "low_yield_steps": [],
        }
    plan_steps = len(state.plan.steps)
    completed = len(state.completed_steps)
    findings_by_order = {f.step_order: f for f in state.findings}
    search_steps = len(findings_by_order)
    yield_steps = sum(1 for f in findings_by_order.values() if f.evidence)
    low_yield = sorted(o for o, f in findings_by_order.items() if not f.evidence)
    return {
        "plan_steps": plan_steps,
        "completed_steps": completed,
        "completeness": completed / plan_steps if plan_steps else 0.0,
        "search_steps": search_steps,
        "yield_steps": yield_steps,
        "step_yield": yield_steps / search_steps if search_steps else 0.0,
        "low_yield_steps": low_yield,
    }
