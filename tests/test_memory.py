"""Research Memory + Checkpoint 测试（PR39）。

覆盖：store 工厂 / Memory Boundary 派生 / 3 个验收案例
（服务重启恢复 / 人工暂停审核继续 / 多任务线程隔离）/ 未知线程 / resume action 构造。
SqliteSaver 用 pytest tmp_path（测试后自动清理）；全 mock，零真实 LLM / 向量库。
"""

import pytest

from app.rag.agent.state import AgentState
from app.rag.memory import (
    RecordStatus,
    ResearchCheckpointer,
    ResumeAction,
    get_checkpointer,
    to_record,
)
from app.rag.profile.schema import CompanyProfile, EvidenceRef
from app.rag.research import ResearchReport, ReportClaim


# ── Mock 依赖 ─────────────────────────────────────────────────

class _MockTools:
    """company 感知 mock：证据 quote 带公司前缀，供线程隔离断言。"""

    def profile_lookup(self, company: str) -> CompanyProfile:
        return CompanyProfile(company_name=company, industry="智能硬件")

    def evidence_search(self, query, company, source_types=None, top_k=5):
        return [
            EvidenceRef(source="x.pdf", source_type="annual_report", page=1,
                        quote=f"{company}:{query}", chunk_id=f"{company}|{query[:8]}")
        ]


class _MockReport:
    def build(self, state):
        evs = state.evidence_pool or []
        return ResearchReport(
            title="t", summary="s",
            advantages=[ReportClaim(claim=e.quote, evidence=[e]) for e in evs],
            risks=[], uncertainties=[], evidence=evs,
        )


# ── store 工厂 ─────────────────────────────────────────────────

def test_store_factory_memory():
    """backend='memory' → MemorySaver。"""
    from langgraph.checkpoint.memory import MemorySaver

    assert isinstance(get_checkpointer("memory"), MemorySaver)


def test_store_factory_sqlite(tmp_path):
    """backend='sqlite' → SqliteSaver。"""
    from langgraph.checkpoint.sqlite import SqliteSaver

    saver = get_checkpointer("sqlite", db_path=str(tmp_path / "c.db"))
    assert isinstance(saver, SqliteSaver)
    saver.conn.close()


# ── Memory Boundary（to_record）───────────────────────────────

def test_to_record_boundary():
    """ResearchRecord 只含 Memory Boundary 字段，不含 embedding/LLM 产物。"""
    from app.rag.agent import run_adaptive_research

    state = run_adaptive_research(
        "分析小米汽车竞争力",
        _tools=_MockTools(),
        _report_builder=_MockReport(),
    )
    rec = to_record(state, "t1")
    assert rec.query == "分析小米汽车竞争力"
    assert rec.company == "小米"
    assert rec.status == RecordStatus.COMPLETED
    assert rec.completed_steps           # 进度非空
    assert rec.iteration >= 1
    assert rec.evidence_count == len(state.evidence_pool)
    assert rec.finding_count == len(state.findings)
    assert rec.coverage is not None
    # 边界：记录里不出现 embedding / LLM 原始产物字段
    dump = rec.model_dump()
    for forbidden in ("embedding", "llm_response", "cache", "vector"):
        assert not any(forbidden in k for k in dump), f"边界被破坏: {forbidden}"


# ── Case 1：服务重启恢复 ──────────────────────────────────────

def test_case1_restart_resume(tmp_path):
    """run → 新实例（同 SqliteSaver db）→ resume 幂等返回相同终态（模拟服务重启）。"""
    db = str(tmp_path / "ckpt.db")
    cp1 = ResearchCheckpointer(backend="sqlite", db_path=db,
                               tools=_MockTools(), report_builder=_MockReport())
    st1 = cp1.run("分析小米汽车未来竞争力", thread_id="xiaomi_001")
    cp1.close()  # "下线"

    # "重启"：新 checkpointer 实例指向同一 db 文件
    cp2 = ResearchCheckpointer(backend="sqlite", db_path=db)
    st2 = cp2.resume("xiaomi_001")
    assert st2.iteration == st1.iteration
    assert st2.target.company == st1.target.company == "小米"
    assert st2.evidence_pool == st1.evidence_pool
    rec = cp2.record("xiaomi_001")
    assert rec.status == RecordStatus.COMPLETED
    cp2.close()


# ── Case 2：人工暂停 → 审核 → 继续 ────────────────────────────

def test_case2_interrupt_resume(tmp_path):
    """interrupt 节点中断 → record=PAUSED → resume 携带 approve → COMPLETED。"""
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    saver = get_checkpointer("sqlite", db_path=str(tmp_path / "ckpt.db"))

    def execute(s: AgentState) -> dict:
        return {"iteration": 1, "current_step": "risk_analysis", "next_action": "continue"}

    def review(s: AgentState) -> dict:
        decision = interrupt({"step": "risk_analysis", "need": "approval"})
        approved = isinstance(decision, dict) and decision.get("approved") is True
        return {"next_action": "end", "current_step": "risk_analysis",
                "iteration": 1, "missing_dimensions": [] if approved else ["risk"]}

    g = StateGraph(AgentState)
    g.add_node("execute", execute)
    g.add_node("review", review)
    g.add_edge(START, "execute")
    g.add_edge("execute", "review")
    g.add_edge("review", END)
    graph = g.compile(checkpointer=saver)

    cp = ResearchCheckpointer(checkpointer=saver)

    # 第 1 次执行：在 review 中断（暂停）
    r1 = graph.invoke(AgentState.from_request("分析小米汽车风险"), config={"thread_id": "t1"})
    assert "__interrupt__" in r1
    rec = cp.record("t1")
    assert rec.status == RecordStatus.PAUSED
    assert rec.current_step == "risk_analysis"
    assert rec.iteration == 1

    # 人工审核 approve → 继续到 END
    r2 = graph.invoke(Command(resume={"approved": True}), config={"thread_id": "t1"})
    assert r2.get("next_action") == "end"
    rec2 = cp.record("t1")
    assert rec2.status == RecordStatus.COMPLETED


# ── Case 3：多任务线程隔离 ────────────────────────────────────

def test_case3_thread_isolation(tmp_path):
    """thread_A（小米）/ thread_B（宁德时代）checkpoint 互不污染。"""
    cp = ResearchCheckpointer(backend="sqlite", db_path=str(tmp_path / "ckpt.db"),
                              tools=_MockTools(), report_builder=_MockReport())
    st_a = cp.run("分析小米汽车未来竞争力", thread_id="thread_A")
    st_b = cp.run("分析宁德时代储能竞争力", thread_id="thread_B")

    assert st_a.target.company == "小米"
    assert st_b.target.company == "宁德时代"
    assert st_a.target.company != st_b.target.company

    rec_a, rec_b = cp.record("thread_A"), cp.record("thread_B")
    assert rec_a.company == "小米"
    assert rec_b.company == "宁德时代"

    # resume A：仍是小米状态，证据不混入 B
    st_a2 = cp.resume("thread_A")
    assert st_a2.target.company == "小米"
    assert all("宁德时代" not in e.quote for e in st_a2.evidence_pool)
    assert all("小米" in e.quote for e in st_a2.evidence_pool)
    cp.close()


# ── 未知线程 / 边界 ───────────────────────────────────────────

def test_resume_unknown_thread(tmp_path):
    """resume 不存在的 thread → None。"""
    cp = ResearchCheckpointer(backend="sqlite", db_path=str(tmp_path / "c.db"))
    assert cp.resume("nope") is None


def test_record_unknown_thread(tmp_path):
    """record 不存在的 thread → None。"""
    cp = ResearchCheckpointer(backend="sqlite", db_path=str(tmp_path / "c.db"))
    assert cp.record("nope") is None


# ── resume action 构造（PR41 Human-in-the-loop 接口就绪）─────

def test_resume_action_plumbing(monkeypatch):
    """ResearchCheckpointer.resume(action=...) 构造 Command(resume=action) 并带 thread_id。"""
    import app.rag.memory.checkpoint as checkpoint_mod

    captured: dict = {}

    class _FakeGraph:
        def invoke(self, input_val, config):
            captured["input"] = input_val
            captured["config"] = config
            return AgentState.from_request("测试").model_dump()

    monkeypatch.setattr(checkpoint_mod, "build_graph", lambda **kw: _FakeGraph())

    class _StubCp:
        def get_tuple(self, config):
            return object()  # 非 None：thread 存在

    cp = ResearchCheckpointer(checkpointer=_StubCp())
    cp.resume("t1", action=ResumeAction(decision="approve", note="证据充分"))
    assert captured["config"] == {"thread_id": "t1"}
    # input 是 Command(resume=action)，携带审核决策
    assert captured["input"].resume == {"decision": "approve", "note": "证据充分"}
    # 无 action → 纯恢复（input=None）
    captured.clear()
    cp.resume("t1")
    assert captured["input"] is None
