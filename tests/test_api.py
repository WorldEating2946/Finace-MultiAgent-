"""Research Agent Service Layer 测试（PR40）。

覆盖：health / 创建任务 / 状态查询 / 报告 / Human-in-the-loop 暂停→恢复 /
多任务隔离 / 异常信封（404/400/409）/ knowledge / profile。
全 mock（mock tools + temp sqlite），零真实 LLM / 向量库。
"""

from app.rag.profile.schema import CompanyProfile as _CP
from app.rag.document import DocumentChunk
from app.services.research_service import ResearchService


# ── Mock 依赖 ─────────────────────────────────────────────────

class _MockTools:
    """company 感知 mock：fail_keyword 首轮无证据（触发 replan/human_review）。"""

    def __init__(self, *, fail_keyword: str | None = None):
        self.fail_keyword = fail_keyword
        self._failed = False

    def profile_lookup(self, company):
        return _CP(company_name=company, industry="智能硬件")

    def evidence_search(self, query, company, source_types=None, top_k=5):
        from app.rag.profile.schema import EvidenceRef

        if self.fail_keyword and self.fail_keyword in query and not self._failed:
            self._failed = True
            return []
        return [
            EvidenceRef(source="x.pdf", source_type="annual_report", page=1,
                        quote=f"{company}:{query}", chunk_id=f"{company}|{query[:8]}")
        ]


class _MockReport:
    def build(self, state):
        from app.rag.research import ResearchReport, ReportClaim

        evs = state.evidence_pool or []
        return ResearchReport(
            title="小米汽车竞争力分析", summary="综合研判",
            advantages=[ReportClaim(claim=e.quote, evidence=[e]) for e in evs],
            risks=[], uncertainties=[], evidence=evs,
        )


class _FakeRetriever:
    def __call__(self, query, company, top_k=5):
        class _Result:
            chunks = [
                DocumentChunk(chunk_id="c1", company=company, doc_type="pdf",
                              source="x.pdf", source_name="x", page=1,
                              text=f"{company}业务快速发展", metadata={"chapter": "管理层讨论"})
            ]
        return _Result()


class _FakeLoader:
    def __call__(self, company):
        if company == "小米":
            return _CP(company_name="小米", industry="智能硬件")
        return None


class _NoReportService(ResearchService):
    """get_report 恒返回 None —— 触发报告未就绪异常路径。"""

    def get_report(self, research_id):
        return None


# ── helper ────────────────────────────────────────────────────

def _client(tmp_path, *, fail_keyword=None, service_cls=ResearchService):
    """构建 TestClient + 注入 mock service/retriever/loader。"""
    from fastapi.testclient import TestClient

    from app.api import knowledge as knowledge_api
    from app.api import profile as profile_api
    from app.api import research as research_api
    from app.api.app import create_app
    from app.rag.memory import ResearchCheckpointer

    cp = ResearchCheckpointer(
        backend="sqlite",
        db_path=str(tmp_path / "ckpt.db"),
        tools=_MockTools(fail_keyword=fail_keyword),
        report_builder=_MockReport(),
    )
    service = service_cls(checkpointer=cp)
    app = create_app()
    app.dependency_overrides[research_api.get_research_service] = lambda: service
    app.dependency_overrides[knowledge_api.get_knowledge_retriever] = lambda: _FakeRetriever()
    app.dependency_overrides[profile_api.get_profile_loader] = lambda: _FakeLoader()
    return TestClient(app), service


# ── health ─────────────────────────────────────────────────────

def test_health(tmp_path):
    """GET /health → 200 信封 {code:0, data:{status:ok}}。"""
    client, _ = _client(tmp_path)
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["code"] == 0
    assert body["data"]["status"] == "ok"


# ── Case 1：创建 → 状态 → 报告 ────────────────────────────────

def test_case1_start_status_report(tmp_path):
    """POST /start → completed；GET status / report 全链路。"""
    client, _ = _client(tmp_path)
    r = client.post("/api/v1/research/start",
                    json={"query": "分析小米汽车竞争力", "company": "小米"})
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["research_id"]
    assert data["thread_id"] == data["research_id"]
    assert data["status"] == "completed"
    rid = data["research_id"]

    r2 = client.get(f"/api/v1/research/{rid}")
    d2 = r2.json()["data"]
    assert d2["status"] == "completed"
    assert d2["iteration"] >= 1
    assert d2["current_step"]

    r3 = client.get(f"/api/v1/research/{rid}/report")
    d3 = r3.json()["data"]
    assert d3["title"] == "小米汽车竞争力分析"
    assert d3["advantages"]


# ── Case 2：Human-in-the-loop 暂停 → 审核 → 恢复 ───────────────

def test_case2_human_pause_approve_resume(tmp_path):
    """human_review + 风险缺失 → waiting_human；approve → completed。"""
    client, _ = _client(tmp_path, fail_keyword="风险")
    r = client.post("/api/v1/research/start",
                    json={"query": "分析小米汽车竞争力", "human_review": True})
    assert r.status_code == 200
    data = r.json()["data"]
    rid = data["research_id"]
    assert data["status"] == "waiting_human"

    # 状态：等待人工审核 + 缺失维度
    r2 = client.get(f"/api/v1/research/{rid}")
    d2 = r2.json()["data"]
    assert d2["status"] == "waiting_human"
    assert "risk" in d2["missing_dimensions"]

    # 审核 approve → 继续补步 → 完成
    r3 = client.post(f"/api/v1/research/{rid}/resume",
                     json={"action": "approve", "feedback": "证据充足，可继续补充研究"})
    d3 = r3.json()["data"]
    assert d3["status"] == "completed"
    # 报告已生成（补步后）
    r4 = client.get(f"/api/v1/research/{rid}/report")
    assert r4.json()["data"]["advantages"]


def test_case2_human_reject_stops(tmp_path):
    """human_review 拒绝 → 停止补步 → rejected（PR42a：不再伪装成 completed）。"""
    client, _ = _client(tmp_path, fail_keyword="风险")
    r = client.post("/api/v1/research/start",
                    json={"query": "分析小米汽车竞争力", "human_review": True})
    rid = r.json()["data"]["research_id"]

    r2 = client.post(f"/api/v1/research/{rid}/resume", json={"action": "reject"})
    assert r2.status_code == 200
    assert r2.json()["data"]["status"] == "rejected"


# ── Case 3：多任务线程隔离 ────────────────────────────────────

def test_case3_thread_isolation(tmp_path):
    """thread_A（小米）/ thread_B（宁德时代）报告证据互不污染。"""
    client, _ = _client(tmp_path)
    rA = client.post("/api/v1/research/start",
                     json={"query": "分析小米汽车竞争力", "company": "小米"})
    rB = client.post("/api/v1/research/start",
                     json={"query": "分析宁德时代竞争力", "company": "宁德时代"})
    id_a, id_b = rA.json()["data"]["research_id"], rB.json()["data"]["research_id"]

    rep_a = client.get(f"/api/v1/research/{id_a}/report").json()["data"]
    rep_b = client.get(f"/api/v1/research/{id_b}/report").json()["data"]
    claims_a = " ".join(a["claim"] for a in rep_a["advantages"])
    claims_b = " ".join(a["claim"] for a in rep_b["advantages"])
    assert "宁德时代" not in claims_a          # A 证据不混入 B
    assert "小米" not in claims_b              # B 证据不混入 A
    assert "小米" in claims_a and "宁德时代" in claims_b


# ── Case 4：异常信封 ───────────────────────────────────────────

def test_case4_research_not_found(tmp_path):
    """GET 不存在任务 → 404 {code:40001}。"""
    client, _ = _client(tmp_path)
    r = client.get("/api/v1/research/nonexistent")
    assert r.status_code == 404
    body = r.json()
    assert body["code"] == 40001
    assert body["data"] is None


def test_case4_invalid_decision(tmp_path):
    """resume 非法 action → 400 {code:40003}。"""
    client, _ = _client(tmp_path)
    r = client.post("/api/v1/research/start", json={"query": "分析小米汽车竞争力"})
    rid = r.json()["data"]["research_id"]
    r2 = client.post(f"/api/v1/research/{rid}/resume", json={"action": "invalid"})
    assert r2.status_code == 400
    assert r2.json()["code"] == 40003


def test_case4_report_not_ready(tmp_path):
    """报告未就绪 → 409 {code:40004}。"""
    client, _ = _client(tmp_path, service_cls=_NoReportService)
    r = client.post("/api/v1/research/start", json={"query": "分析小米汽车竞争力"})
    rid = r.json()["data"]["research_id"]
    r2 = client.get(f"/api/v1/research/{rid}/report")
    assert r2.status_code == 409
    assert r2.json()["code"] == 40004


# ── knowledge / profile 薄封装 ─────────────────────────────────

def test_knowledge_search(tmp_path):
    """GET /knowledge/search → 命中 chunk 列表。"""
    client, _ = _client(tmp_path)
    r = client.get("/api/v1/knowledge/search",
                   params={"query": "研发投入", "company": "小米", "top_k": 3})
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["chunks"]
    assert data["chunks"][0]["text"].startswith("小米")


def test_profile_get(tmp_path):
    """GET /profile/{company} → 画像 JSON；不存在 → 404。"""
    client, _ = _client(tmp_path)
    r = client.get("/api/v1/profile/小米")
    assert r.status_code == 200
    assert r.json()["data"]["company_name"] == "小米"

    r2 = client.get("/api/v1/profile/不存在的公司")
    assert r2.status_code == 404
    assert r2.json()["code"] == 40001
