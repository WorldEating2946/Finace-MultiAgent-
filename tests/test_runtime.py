"""Production Runtime 测试（PR41）。

覆盖：
- arun_adaptive_research / ResearchCheckpointer.async（arun/aresume/aget_state）
- RecordStatus 扩展（queued/cancelled）
- EventBus 发布/订阅/退订
- SSE 端点（stream）+ TaskManager 生命周期视图
- postgres 后端 sync 方法报错（必须走 async）

策略：async 测试用 pytest-asyncio；DB 层用 sqlite/mock，不依赖真实 postgres。
SSE/TaskManager 用 sqlite service 的 sync 兼容路径 + EventBus 直测。
"""

import pytest

from app.rag.memory import ResearchCheckpointer
from app.rag.memory.schema import RecordStatus


# ── Mock 依赖（与 test_memory/test_api 一致）───────────────────

class _MockTools:
    def profile_lookup(self, company):
        from app.rag.profile.schema import CompanyProfile
        return CompanyProfile(company_name=company, industry="智能硬件")

    def evidence_search(self, query, company, source_types=None, top_k=5):
        from app.rag.profile.schema import EvidenceRef
        return [
            EvidenceRef(source="x.pdf", source_type="annual_report", page=1,
                        quote=f"{company}:{query}", chunk_id=f"{company}|{query[:8]}")
        ]


class _MockReport:
    def build(self, state):
        from app.rag.research import ResearchReport, ReportClaim
        evs = state.evidence_pool or []
        return ResearchReport(
            title="t", summary="s",
            advantages=[ReportClaim(claim=e.quote, evidence=[e]) for e in evs],
            risks=[], uncertainties=[], evidence=evs,
        )


# ── RecordStatus 扩展 ──────────────────────────────────────────

def test_record_status_new_states():
    """PR41 状态扩展：queued / cancelled 存在。"""
    assert RecordStatus.QUEUED.value == "queued"
    assert RecordStatus.CANCELLED.value == "cancelled"
    # 原状态兼容
    assert RecordStatus.PAUSED.value == "paused"
    assert RecordStatus.COMPLETED.value == "completed"


# ── async ResearchCheckpointer（sqlite 后端）───────────────────

@pytest.mark.asyncio
async def test_arun_sqlite(tmp_path):
    """arun 用 sqlite checkpointer 完成研究（async 图执行）。"""
    cp = ResearchCheckpointer(backend="sqlite", db_path=str(tmp_path / "c.db"),
                              tools=_MockTools(), report_builder=_MockReport())
    try:
        state = await cp.arun("分析小米汽车竞争力", thread_id="t_async1")
        assert state.iteration >= 1
        assert state.target.company == "小米"
        # aget_state 能还原
        st2 = await cp.aget_state("t_async1")
        assert st2 is not None
        assert st2.iteration == state.iteration
        # arecord 推导 completed
        rec = await cp.arecord("t_async1")
        assert rec.status == RecordStatus.COMPLETED
    finally:
        cp.close()


@pytest.mark.asyncio
async def test_aresume_interrupt(tmp_path):
    """interrupt 暂停 → aresume(approve) 继续到 end。"""
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    from app.rag.agent.state import AgentState

    conn = await aiosqlite.connect(str(tmp_path / "ckpt.db"))
    saver = AsyncSqliteSaver(conn)

    def review(s: AgentState) -> dict:
        decision = interrupt({"step": "risk", "need": "approval"})
        ok_ = isinstance(decision, dict) and decision.get("approved") is True
        return {"next_action": "end", "current_step": "risk",
                "iteration": 1, "missing_dimensions": [] if ok_ else ["risk"]}

    g = StateGraph(AgentState)
    g.add_node("execute", lambda s: {"iteration": 1, "next_action": "continue"})
    g.add_node("review", review)
    g.add_edge(START, "execute")
    g.add_edge("execute", "review")
    g.add_edge("review", END)
    graph = g.compile(checkpointer=saver)

    cp = ResearchCheckpointer(checkpointer=saver)

    try:
        # 首次执行：在 review 暂停
        r1 = await graph.ainvoke(AgentState.from_request("分析小米风险"),
                                  config={"thread_id": "t2"})
        assert "__interrupt__" in r1
        rec = await cp.arecord("t2")
        assert rec.status == RecordStatus.PAUSED

        # 人工 approve → 继续到 end
        st = await cp.aresume("t2", action={"approved": True, "action": "approve"})
        assert st.next_action == "end"
        rec2 = await cp.arecord("t2")
        assert rec2.status == RecordStatus.COMPLETED
    finally:
        await conn.close()


# ── arun_adaptive_research event_sink ──────────────────────────

@pytest.mark.asyncio
async def test_arun_event_sink(tmp_path):
    """arun_adaptive_research 的 event_sink 收到节点事件。"""
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from app.rag.agent import arun_adaptive_research

    conn = await aiosqlite.connect(str(tmp_path / "c.db"))
    saver = AsyncSqliteSaver(conn)
    events: list[dict] = []

    async def sink(event: dict):
        events.append(event)

    try:
        await arun_adaptive_research(
            "分析小米汽车竞争力",
            thread_id="t_evt",
            checkpointer=saver,
            _tools=_MockTools(),
            _report_builder=_MockReport(),
            event_sink=sink,
        )
        assert events, "event_sink 应收到节点事件"
        kinds = {e.get("type") for e in events}
        assert "node_start" in kinds
        assert "node_end" in kinds
    finally:
        await conn.close()


# ── EventBus ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_event_bus_pubsub():
    """EventBus：订阅 → 发布 → 收到；退订后不再收。"""
    import asyncio

    from app.runtime import EventBus
    from app.runtime.events import NodeEvent

    bus = EventBus()
    q = await bus.subscribe("r1")
    assert bus.has_subscribers("r1")

    await bus.publish(NodeEvent.make("r1", "node_start", node="execute", message="x"))
    event = await asyncio.wait_for(q.get(), timeout=2)
    assert event["type"] == "node_start"
    assert event["node"] == "execute"

    await bus.unsubscribe("r1", q)
    assert not bus.has_subscribers("r1")
    # 无订阅者时 publish 不报错
    await bus.publish(NodeEvent.make("r1", "done"))


# ── SSE 端点 ───────────────────────────────────────────────────

def test_stream_endpoint_completed(tmp_path):
    """已完成任务的 SSE：立即返回 done 事件（不阻塞）。"""
    from fastapi.testclient import TestClient

    from app.api import research as research_api
    from app.api.app import create_app
    from app.services.research_service import ResearchService

    cp = ResearchCheckpointer(backend="sqlite", db_path=str(tmp_path / "c.db"),
                              tools=_MockTools(), report_builder=_MockReport())
    service = ResearchService(checkpointer=cp)

    app = create_app()
    app.dependency_overrides[research_api.get_research_service] = lambda: service

    # 先创建任务（sync 路径，完成）
    task = service.create_task("分析小米汽车竞争力", company="小米")
    rid = task.research_id

    client = TestClient(app)
    with client.stream("GET", f"/api/v1/research/{rid}/stream") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        # 已完成任务 → 立即收到 done
        body = b""
        for line in resp.iter_lines():
            if line.startswith("data: "):
                body = line[6:]
                break
    assert "done" in body or "completed" in body


def test_stream_endpoint_not_found(tmp_path):
    """不存在的任务 SSE → 404。"""
    from fastapi.testclient import TestClient

    from app.api import research as research_api
    from app.api.app import create_app
    from app.services.research_service import ResearchService

    cp = ResearchCheckpointer(backend="sqlite", db_path=str(tmp_path / "c.db"))
    service = ResearchService(checkpointer=cp)

    app = create_app()
    app.dependency_overrides[research_api.get_research_service] = lambda: service

    client = TestClient(app)
    r = client.get("/api/v1/research/nonexistent/stream")
    assert r.status_code == 404
    assert r.json()["code"] == 40001


# ── postgres sync 防护 ─────────────────────────────────────────

def test_postgres_sync_method_raises():
    """postgres 后端 sync run() 报错（必须走 async）。"""
    cp = ResearchCheckpointer(backend="postgres")
    with pytest.raises(RuntimeError, match="async"):
        cp.get_state("x")


def test_postgres_flag():
    """is_postgres 标记识别 factory 后端。"""
    cp = ResearchCheckpointer(backend="postgres")
    assert cp.is_postgres is True
    cp2 = ResearchCheckpointer(backend="memory")
    assert cp2.is_postgres is False
