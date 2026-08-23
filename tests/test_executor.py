"""研究执行引擎测试（PR #37）。

覆盖：state schema / executor（步骤分派、证据聚合、去重）/ tools（chunk→EvidenceRef）/
report（LLM JSON 解析、越界过滤）/ run_research 端到端（mock tools + mock report）。
全 mock，无真实向量库 / LLM 依赖。
"""

from datetime import datetime, timezone

from app.rag.profile.schema import CompanyProfile, EvidenceRef
from app.rag.research import (
    ReportBuilder,
    ResearchExecutor,
    ResearchPlan,
    ResearchReport,
    ResearchTarget,
    ResearchTools,
    build_research_plan,
    run_research,
)
from app.rag.research.schema import ResearchIntent, ResearchStep
from app.rag.research.state import Finding, ResearchState


# ── State schema ───────────────────────────────────────────────

def test_research_state_schema():
    """ResearchState 结构化正确。"""
    plan = build_research_plan("分析小米汽车竞争力")
    state = ResearchState(
        request="分析小米汽车竞争力",
        intent=plan.intent.value,
        target=plan.target,
        plan=plan,
    )
    assert state.intent == "competitive_analysis"
    assert state.target.company == "小米"
    assert state.completed_steps == []
    assert state.evidence_pool == []


def test_finding_schema():
    f = Finding(step_order=1, step_name="画像", evidence=[], source_types=["annual_report"])
    assert f.step_order == 1
    assert f.source_types == ["annual_report"]


# ── Mock tools ─────────────────────────────────────────────────

class _MockTools:
    """mock 工具集：可控返回，记录调用。"""

    def __init__(self, chunks: list[EvidenceRef] | None = None):
        self._chunks = chunks or [
            EvidenceRef(source="x.pdf", source_type="annual_report",
                        chapter="管理层讨论及分析", page=1, quote="小米智能驾驶投入增长", chunk_id="c1"),
            EvidenceRef(source="y.pdf", source_type="research_report",
                        chapter="行业", page=2, quote="汽车业务短期亏损", chunk_id="c2"),
        ]
        self.calls: list[str] = []
        self.profile_calls: list[str] = []

    def profile_lookup(self, company: str) -> CompanyProfile:
        self.profile_calls.append(company)
        return CompanyProfile(company_name=company, industry="智能硬件")

    def evidence_search(self, query, company, source_types=None, top_k=5):
        self.calls.append(query)
        # 简单 source_type 过滤模拟
        refs = self._chunks
        if source_types:
            refs = [r for r in refs if r.source_type in source_types]
        return refs[:top_k]


# ── Executor ───────────────────────────────────────────────────

def test_executor_runs_all_steps_in_order():
    """执行所有步骤，completed_steps 顺序完整。"""
    tools = _MockTools()
    ex = ResearchExecutor(tools=tools)  # type: ignore[arg-type]
    plan = build_research_plan("分析小米汽车竞争力")

    state = ex.execute(plan)

    assert state.completed_steps == list(range(1, len(plan.steps) + 1))
    # 画像步骤：profile 填充，无 finding
    assert state.profile is not None
    assert state.profile.company_name == "小米"
    assert tools.profile_calls == ["小米"]
    # 其他步骤各收集 finding
    assert len(state.findings) == len(plan.steps) - 1
    # 全部证据入池
    assert len(state.evidence_pool) > 0


def test_executor_dedupes_evidence():
    """同一 chunk_id 的 evidence 只保留一条。"""
    chunks = [
        EvidenceRef(source="a.pdf", source_type="annual_report", page=1, quote="q1", chunk_id="dup"),
        EvidenceRef(source="b.pdf", source_type="annual_report", page=2, quote="q2", chunk_id="dup"),
        EvidenceRef(source="c.pdf", source_type="annual_report", page=3, quote="q3", chunk_id="uniq"),
    ]
    ex = ResearchExecutor(tools=_MockTools(chunks=chunks))  # type: ignore[arg-type]
    plan = build_research_plan("小米竞争力")
    state = ex.execute(plan)
    ids = [e.chunk_id for e in state.evidence_pool]
    assert ids.count("dup") == 1
    assert "uniq" in ids


def test_executor_respects_source_types():
    """步骤的 source_types 传给 evidence_search（policy 步骤过滤）。"""
    tools = _MockTools()
    ex = ResearchExecutor(tools=tools)  # type: ignore[arg-type]
    plan = build_research_plan("小米汽车竞争力")
    state = ex.execute(plan)
    # 政策步骤 source_types=['policy'] → mock 过滤后无结果，但该步骤有 finding
    policy_findings = [f for f in state.findings if "policy" in f.source_types]
    assert policy_findings


# ── Tools：chunk → EvidenceRef ─────────────────────────────────

def test_evidence_search_converts_chunks():
    """evidence_search 把 DocumentChunk 转成 EvidenceRef（source/page/quote）。"""
    from app.rag.document import DocumentChunk

    class _FakeRetrieveResult:
        chunks = [
            DocumentChunk(
                chunk_id="x1", company="小米", doc_type="pdf",
                source="/abs/x.pdf", source_name="x",
                page=19, text="小米智能电动汽车业务快速发展",
                metadata={"source_type": "annual_report", "chapter": "管理层讨论及分析"},
            )
        ]

    class _RetrievalTools(ResearchTools):
        def knowledge_search(self, query, company, source_types=None, top_k=5):
            return _FakeRetrieveResult()

    refs = _RetrievalTools().evidence_search("业务", "小米")
    assert len(refs) == 1
    r = refs[0]
    assert r.source == "/abs/x.pdf"
    assert r.source_type == "annual_report"
    assert r.chapter == "管理层讨论及分析"
    assert r.page == 19
    assert "智能电动汽车" in r.quote
    assert r.chunk_id == "x1"


def test_evidence_search_filters_source_type():
    """source_types 过滤在工具层生效。"""
    from app.rag.document import DocumentChunk

    class _FakeRetrieveResult:
        chunks = [
            DocumentChunk(chunk_id="a", company="小米", doc_type="pdf",
                          source="a", source_name="a", text="年报文",
                          metadata={"source_type": "annual_report"}),
            DocumentChunk(chunk_id="b", company="小米", doc_type="pdf",
                          source="b", source_name="b", text="政策文",
                          metadata={"source_type": "policy"}),
        ]

    class _RetrievalTools(ResearchTools):
        def knowledge_search(self, query, company, source_types=None, top_k=5):
            return _FakeRetrieveResult()

    refs = _RetrievalTools().evidence_search("x", "小米", source_types=["policy"])
    assert len(refs) == 1
    assert refs[0].chunk_id == "b"


# ── Report ─────────────────────────────────────────────────────

def _state_with_evidence() -> ResearchState:
    plan = build_research_plan("分析小米汽车")
    state = ResearchState(request="分析小米汽车", intent=plan.intent.value,
                          target=plan.target, plan=plan)
    state.evidence_pool = [
        EvidenceRef(source="a.pdf", source_type="annual_report", chapter="c", page=1,
                    quote="小米智能驾驶投入增长", chunk_id="c1"),
        EvidenceRef(source="b.pdf", source_type="annual_report", chapter="d", page=2,
                    quote="小米汽车海外扩张", chunk_id="c2"),
    ]
    return state


def test_report_parse_resolves_evidence():
    """LLM JSON → ReportClaim，evidence_refs 解析为真实 EvidenceRef。"""
    builder = ReportBuilder(_call=lambda prompt: (
        '{"title":"小米汽车竞争力","summary":"整体较强","advantages":'
        '[{"claim":"智能驾驶投入领先","evidence_refs":[0]}],"risks":[],"uncertainties":["盈利周期未知"]}'
    ))
    report = builder.build(_state_with_evidence())
    assert report.title == "小米汽车竞争力"
    assert len(report.advantages) == 1
    assert report.advantages[0].claim == "智能驾驶投入领先"
    assert report.advantages[0].evidence[0].chunk_id == "c1"
    assert report.uncertainties == ["盈利周期未知"]


def test_report_discards_out_of_range_refs():
    """越界 evidence_refs 被丢弃，不崩溃。"""
    builder = ReportBuilder(_call=lambda prompt: (
        '{"title":"t","summary":"s","advantages":'
        '[{"claim":"无证据论点","evidence_refs":[9]}],"risks":[],"uncertainties":[]}'
    ))
    report = builder.build(_state_with_evidence())
    # 越界 ref → claim 保留但 evidence 为空
    assert report.advantages[0].evidence == []


def test_report_extract_json_codeblock():
    """```json``` 包裹 → 仍能解析。"""
    builder = ReportBuilder(_call=lambda prompt: (
        '```json\n{"title":"t","summary":"s","advantages":[],"risks":[],"uncertainties":[]}\n```'
    ))
    report = builder.build(_state_with_evidence())
    assert report.title == "t"


def test_report_empty_llm_output():
    """LLM 空输出 → 空报告（不崩溃）。"""
    builder = ReportBuilder(_call=lambda prompt: "")
    report = builder.build(_state_with_evidence())
    assert report.title == ""
    assert report.advantages == []


# ── run_research 端到端（mock）─────────────────────────────────

def test_run_research_end_to_end_mocked():
    """run_research 全链路：plan → executor → report（mock tools + mock report）。"""
    class _MockReportBuilder:
        def build(self, state: ResearchState) -> ResearchReport:
            return ResearchReport(
                title="小米汽车竞争力分析",
                summary="综合研判",
                advantages=[], risks=[], uncertainties=[],
                evidence=state.evidence_pool,
                generated_at=datetime.now(timezone.utc).isoformat(),
            )

    report = run_research(
        "分析小米汽车未来竞争力",
        _tools=_MockTools(),  # type: ignore[arg-type]
        _report_builder=_MockReportBuilder(),  # type: ignore[arg-type]
    )
    assert isinstance(report, ResearchReport)
    assert report.title == "小米汽车竞争力分析"
    assert len(report.evidence) > 0  # 证据链贯穿到报告
