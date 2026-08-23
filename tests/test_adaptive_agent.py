"""自适应研究 Agent 测试（PR38）。

覆盖：graph 结构 / 3 个验收案例（正常 1 轮 / 缺失触发补步 2 轮 / 最大轮次强制结束）/
AgentState schema / router 决策 / add_replan_step / executor.resume 增量执行 / 入口。
全 mock，无真实 LLM / 向量库依赖（LangGraph 编译本地完成）。
"""

from app.rag.profile.schema import CompanyProfile, EvidenceRef
from app.rag.research import ResearchReport, ReportClaim, build_research_plan
from app.rag.research.evaluate import ResearchMetrics
from app.rag.research.executor import ResearchExecutor
from app.rag.research.planner import add_replan_step


# ── Mock 依赖 ─────────────────────────────────────────────────

class _MockTools:
    """有状态 mock 工具：可按关键词控制证据产出（首失败 + 后续成功）。"""

    def __init__(self, *, fail_keyword: str | None = None, always_empty: bool = False):
        self.fail_keyword = fail_keyword
        self.always_empty = always_empty
        self._failed = False
        self.calls: list[tuple] = []

    def profile_lookup(self, company: str) -> CompanyProfile:
        self.calls.append(("profile", company))
        return CompanyProfile(company_name=company, industry="智能硬件")

    def evidence_search(self, query, company, source_types=None, top_k=5):
        self.calls.append(("search", query))
        if self.always_empty:
            return []
        if self.fail_keyword and self.fail_keyword in query and not self._failed:
            self._failed = True
            return []
        return [
            EvidenceRef(source="x.pdf", source_type="annual_report", chapter="c",
                        page=1, quote=query, chunk_id=f"c{len(self.calls)}")
        ]


class _MockReportBuilder:
    """根据 evidence_pool 合成报告：有证据 → 全部有证据 claim；无证据 → 空。"""

    def build(self, state):
        evs = state.evidence_pool or []
        advantages = [ReportClaim(claim=e.quote, evidence=[e]) for e in evs]
        return ResearchReport(
            title="t", summary="s", advantages=advantages,
            risks=[], uncertainties=[], evidence=evs,
        )


def _run(request: str, tools: _MockTools, report_builder=None) -> "AgentState":
    """跑自适应 Agent（run_adaptive_research 入口），返回最终 AgentState。"""
    from app.rag.agent import run_adaptive_research

    return run_adaptive_research(request, _tools=tools, _report_builder=report_builder or _MockReportBuilder())


def _good_metrics() -> ResearchMetrics:
    return ResearchMetrics(
        total_claims=2, supported_claims=2, evidence_coverage=1.0,
        citation_total=2, citation_ok=2, citation_accuracy=1.0,
        plan_steps=8, completed_steps=8, completeness=1.0,
    )


def _bad_metrics() -> ResearchMetrics:
    return ResearchMetrics(total_claims=2, supported_claims=0, evidence_coverage=0.0)


# ── Graph 结构 ─────────────────────────────────────────────────

def test_graph_structure():
    """build_graph 编译成功，6 节点齐全。"""
    from app.rag.agent import build_graph

    g = build_graph(tools=_MockTools(), report_builder=_MockReportBuilder())
    nodes = g.get_graph().nodes
    for name in ("intent", "planning", "execute", "report", "evaluate", "replan"):
        assert name in nodes, f"节点 {name} 缺失"


# ── AgentState schema ──────────────────────────────────────────

def test_agent_state_schema():
    """AgentState 继承 ResearchState + 自适应字段默认值。"""
    from app.rag.agent import AgentState

    s = AgentState.from_request("分析小米汽车未来竞争力")
    assert s.request == "分析小米汽车未来竞争力"
    assert s.iteration == 0
    assert s.next_action == "continue"
    assert s.missing_dimensions == []
    assert s.replanned_dimensions == []
    assert s.evaluation is None
    # 继承 ResearchState 字段
    assert s.completed_steps == []
    assert s.evidence_pool == []
    assert s.findings == []


# ── Case 1：正常研究（1 轮结束）───────────────────────────────

def test_case1_normal_one_round():
    """所有步骤产出证据 → iteration=1, next_action=end, quality_ok。"""
    state = _run("分析小米汽车未来竞争力", _MockTools())
    assert state.iteration == 1
    assert state.next_action == "end"
    assert state.missing_dimensions == []
    assert state.evaluation is not None
    from app.rag.agent.router import quality_ok
    assert quality_ok(state.evaluation)
    # 报告已生成
    assert state.current_report is not None
    assert state.current_report.advantages


# ── Case 2：缺失触发补步（2 轮）───────────────────────────────

def test_case2_missing_triggers_replan():
    """风险步骤首轮无证据 → 补步 → 第 2 轮覆盖风险后结束。"""
    state = _run("分析小米汽车未来竞争力", _MockTools(fail_keyword="风险"))
    assert state.iteration == 2
    assert state.next_action == "end"
    # 补充步骤已添加（competitive 模板 8 步 + 补步 = 9）
    assert len(state.plan.steps) == 9
    assert "risk" in state.replanned_dimensions
    # 风险已覆盖：evidence_pool 含风险相关引用
    assert any("风险" in (e.quote or "") for e in state.evidence_pool)


# ── Case 3：最大轮次强制结束（3 轮）────────────────────────────

def test_case3_max_iteration_force_finish():
    """证据持续缺失 → iteration=3, next_action=end, quality 不达标（强制结束）。"""
    state = _run("分析小米汽车未来竞争力", _MockTools(always_empty=True))
    assert state.iteration == 3
    assert state.next_action == "end"
    from app.rag.agent.router import quality_ok
    assert not quality_ok(state.evaluation)
    # 报告为空（无证据可引用）
    assert state.current_report is not None
    assert state.current_report.advantages == []


# ── Router 决策（单元）─────────────────────────────────────────

def test_router_decisions():
    """decide_next_action：质量达标→end；有缺失→replan（优先于质量）；迭代满→end。"""
    from app.rag.agent.router import decide_next_action

    good, bad = _good_metrics(), _bad_metrics()
    # 质量达标 + 无缺失 → end
    assert decide_next_action(good, 1, []) == "end"
    # 有缺失（即使质量达标）→ replan
    assert decide_next_action(good, 1, ["risk"]) == "replan"
    # 质量不足 + 无缺失 → end（不盲目循环）
    assert decide_next_action(bad, 1, []) == "end"
    # 迭代满 → 强制 end
    assert decide_next_action(bad, 3, ["risk"]) == "end"
    assert decide_next_action(good, 3, ["risk"]) == "end"


def test_router_quality_ok_none():
    """quality_ok(None) = False（未评测不通过）。"""
    from app.rag.agent.router import quality_ok

    assert quality_ok(None) is False


# ── add_replan_step（planner 扩展）─────────────────────────────

def test_add_replan_step():
    """缺失维度 → 补充步骤；无模板维度 → None。"""
    plan = build_research_plan("分析小米汽车未来竞争力")
    step = add_replan_step(plan, "risk")
    assert step is not None
    assert step.order == len(plan.steps) + 1
    assert step.dimensions == ["risk"]
    assert "风险" in step.retrieval_query
    # 无模板维度 → None
    assert add_replan_step(plan, "business") is None


# ── executor.resume 增量执行 ───────────────────────────────────

def test_executor_resume_incremental_noop():
    """已完成 plan 再 resume → 零新增执行。"""
    tools = _MockTools()
    ex = ResearchExecutor(tools=tools)  # type: ignore[arg-type]
    plan = build_research_plan("分析小米汽车竞争力")
    state = ex.execute(plan)
    n_findings, n_evidence = len(state.findings), len(state.evidence_pool)
    n_searches = sum(1 for c in tools.calls if c[0] == "search")

    state2 = ex.resume(state)
    assert len(state2.findings) == n_findings
    assert len(state2.evidence_pool) == n_evidence
    assert sum(1 for c in tools.calls if c[0] == "search") == n_searches


def test_executor_resume_runs_only_new_step():
    """追加补步步骤 → resume 只执行新步骤。"""
    tools = _MockTools()
    ex = ResearchExecutor(tools=tools)  # type: ignore[arg-type]
    plan = build_research_plan("分析小米汽车竞争力")
    state = ex.execute(plan)
    n_findings = len(state.findings)

    n_evidence_before = len(state.evidence_pool)
    state.plan.steps = [*state.plan.steps, add_replan_step(state.plan, "risk")]
    state2 = ex.resume(state)
    assert len(state2.findings) == n_findings + 1
    assert len(state2.evidence_pool) == n_evidence_before + 1


# ── 入口端到端（mock）──────────────────────────────────────────

def test_run_adaptive_research_entry():
    """run_adaptive_research 返回完整 AgentState。"""
    from app.rag.agent import AgentState

    state = _run("分析小米汽车未来竞争力", _MockTools())
    assert isinstance(state, AgentState)
    assert state.current_report is not None
    assert state.evaluation is not None
    assert state.target.company == "小米"
    assert state.intent == "competitive_analysis"
