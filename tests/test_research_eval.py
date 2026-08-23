"""研究报告质量评测测试（PR37.5）。

覆盖：evidence coverage / citation 字段完整度 / claim↔quote 对齐 /
completeness / step yield / 无 state / 空报告 / 端到端。
纯函数，全 mock，零 LLM / 向量库依赖。
"""

from datetime import datetime, timezone

from app.rag.profile.schema import EvidenceRef
from app.rag.research import (
    ClaimEval,
    ResearchMetrics,
    ResearchReport,
    ReportClaim,
    build_research_plan,
    evaluate_report,
)
from app.rag.research.state import Finding, ResearchState


# ── 构造 helper ────────────────────────────────────────────────

def _ref(quote: str = "小米汽车采用CTB一体化车身架构", *, page: int | None = 1,
         chunk_id: str = "c1", source: str = "x.pdf") -> EvidenceRef:
    return EvidenceRef(source=source, source_type="annual_report",
                       chapter="管理层讨论及分析", page=page, quote=quote, chunk_id=chunk_id)


def _report(advantages: list[list[EvidenceRef]],
            risks: list[list[EvidenceRef]] | None = None) -> ResearchReport:
    """构造报告：advantages = 每条 claim 的证据列表（claim 文本用同 quote 保证对齐）。"""
    return ResearchReport(
        title="测试报告", summary="综合研判",
        advantages=[ReportClaim(claim=evs[0].quote if evs else f"论点{i}",
                                evidence=evs) for i, evs in enumerate(advantages)],
        risks=[ReportClaim(claim=evs[0].quote if evs else f"风险{i}",
                           evidence=evs) for i, evs in enumerate(risks or [])],
        uncertainties=[], evidence=[],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def _state(plan_steps: int, findings: list[Finding]) -> ResearchState:
    plan = build_research_plan("分析小米汽车竞争力")
    # 裁出 plan_steps 步（测试只需 step order 对齐）
    plan.steps = plan.steps[:plan_steps]
    return ResearchState(
        request="分析小米汽车竞争力",
        intent=plan.intent.value,
        target=plan.target,
        plan=plan,
        findings=findings,
        completed_steps=list(range(1, plan_steps + 1)),
    )


# ── Evidence Coverage ──────────────────────────────────────────

def test_coverage_all_supported():
    """全部 claim 有证据 → coverage = 1.0。"""
    r = _report(
        advantages=[[_ref("CTB车身")], [_ref("大压铸")], [_ref("超高强度钢")]],
        risks=[[_ref("网络安全")]],
    )
    m = evaluate_report(r)
    assert m.total_claims == 4
    assert m.supported_claims == 4
    assert m.evidence_coverage == 1.0


def test_coverage_partial():
    """3 claim 中 2 个有证据 → coverage = 2/3。"""
    r = _report(
        advantages=[[_ref("CTB车身")], [], [_ref("大压铸")]],
        risks=[],
    )
    m = evaluate_report(r)
    assert m.total_claims == 3
    assert m.supported_claims == 2
    assert abs(m.evidence_coverage - 2 / 3) < 1e-9


# ── Citation Accuracy ──────────────────────────────────────────

def test_citation_field_completeness():
    """四字段完整度：chunk_id/source/quote/page 缺任一 → 不算完整。"""
    r = _report(advantages=[[
        _ref("完整引用"),                                        # 全部具备
        _ref("缺页码", page=None),                                # page 缺失
        _ref("", chunk_id="c3"),                                  # quote 缺失
    ]])
    m = evaluate_report(r)
    assert m.citation_total == 3
    assert m.citation_ok == 1
    assert abs(m.citation_accuracy - 1 / 3) < 1e-9
    # claim 级：全证据完整才为 citation_ok（此处 3 条证据只 1 条完整）
    assert m.claim_evals[0].citation_ok is False


# ── Claim ↔ quote 对齐（chunk 匹配）────────────────────────────

def test_claim_alignment_high():
    """claim 与 quote 文本重叠 → alignment > 0.5 且 alignment_ok。"""
    text = "小米汽车采用CTB一体化车身架构实现轻量化与安全协同"
    r = _report(advantages=[[_ref(text)]])
    m = evaluate_report(r)
    c = m.claim_evals[0]
    assert c.alignment > 0.5
    assert c.alignment_ok is True
    assert m.aligned_claims == 1


def test_claim_alignment_low():
    """claim 与 quote 完全无关 → alignment = 0.0，不通过。"""
    r = _report(advantages=[[_ref("小米汽车采用CTB一体化车身架构")]])
    # 覆写 claim 文本为完全无关的内容
    r.advantages[0].claim = "这个论点与证据完全无关"
    m = evaluate_report(r)
    c = m.claim_evals[0]
    assert c.alignment == 0.0
    assert c.alignment_ok is False
    assert m.aligned_claims == 0
    assert m.claim_alignment == 0.0


# ── Completeness ───────────────────────────────────────────────

def test_completeness_full():
    """所有步骤完成 → completeness = 1.0。"""
    state = _state(plan_steps=5, findings=[])
    r = _report(advantages=[[_ref()]])
    m = evaluate_report(r, state=state)
    assert m.plan_steps == 5
    assert m.completed_steps == 5
    assert m.completeness == 1.0


# ── Step Yield（PR38 信号）─────────────────────────────────────

def test_step_yield_detects_low():
    """空证据步骤 → low_yield_steps 给出 order，yield 正确。"""
    f1 = Finding(step_order=2, step_name="行业竞争", evidence=[_ref("CTB车身")],
                 source_types=["annual_report"])
    f2 = Finding(step_order=3, step_name="政策分析", evidence=[], source_types=["policy"])
    state = _state(plan_steps=4, findings=[f1, f2])
    r = _report(advantages=[[_ref()]])
    m = evaluate_report(r, state=state)
    assert m.search_steps == 2
    assert m.yield_steps == 1
    assert abs(m.step_yield - 0.5) < 1e-9
    assert m.low_yield_steps == [3]


# ── 无 state / 空报告（退化路径）───────────────────────────────

def test_eval_without_state():
    """仅传 report → completeness / yield 退化为 0，不崩溃。"""
    r = _report(advantages=[[_ref()]])
    m = evaluate_report(r)
    assert m.plan_steps == 0
    assert m.completeness == 0.0
    assert m.search_steps == 0
    assert m.step_yield == 0.0
    assert m.low_yield_steps == []
    # coverage 不受 state 缺失影响
    assert m.evidence_coverage == 1.0


def test_empty_report():
    """空报告（0 claims）→ 全部为 0，不崩溃。"""
    r = _report(advantages=[], risks=[])
    m = evaluate_report(r)
    assert m.total_claims == 0
    assert m.evidence_coverage == 0.0
    assert m.citation_accuracy == 0.0
    assert m.claim_alignment == 0.0
    assert m.claim_evals == []


# ── 端到端（mock state + report）───────────────────────────────

def test_evaluate_report_integration():
    """构造完整 state + report → 全部指标正确填充。"""
    text = "小米自研2200MPa级超高强度钢并联合高校突破材料技术"
    f1 = Finding(step_order=2, step_name="技术分析",
                 evidence=[_ref(text, chunk_id="tech-1")], source_types=["annual_report"])
    f2 = Finding(step_order=3, step_name="政策分析", evidence=[], source_types=["policy"])
    state = _state(plan_steps=4, findings=[f1, f2])
    state.profile = None  # 画像步骤不入 findings

    r = ResearchReport(
        title="小米竞争力", summary="综合研判",
        advantages=[ReportClaim(claim=text, evidence=[_ref(text, chunk_id="tech-1")])],
        risks=[ReportClaim(claim="无证据的风险论点", evidence=[])],
        uncertainties=["盈利周期未知"], evidence=[],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    m = evaluate_report(r, state=state)
    assert isinstance(m, ResearchMetrics)
    assert isinstance(m.claim_evals[0], ClaimEval)
    # coverage：2 claim 中 1 有证据
    assert abs(m.evidence_coverage - 0.5) < 1e-9
    # citation：唯一证据四字段完整
    assert m.citation_accuracy == 1.0
    # alignment：claim 与 quote 同文本 → 1.0
    assert abs(m.claim_evals[0].alignment - 1.0) < 1e-9
    # completeness：4/4；yield：1/2，low=[3]
    assert m.completeness == 1.0
    assert m.step_yield == 0.5
    assert m.low_yield_steps == [3]
