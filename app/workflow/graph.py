"""
app/workflow/graph.py — LangGraph 工作流定义

本模块定义 FinanceAgent 的核心工作流状态图，包括:
    - 节点注册: Manager / Research / Financial / Sentiment / Risk / Report
    - 条件分支: intent_router 意图分流（full_research / clarify）
    - 并行路由: 使用 LangGraph Send API 实现真正的 fan-out → gather
    - 循环: 健康检查重试环（health_check → retry）+ Report 质量迭代环
      （evaluate_report → rework），双环均带轮次上限防死循环
    - 图编译: build_graph()

State 定义已独立至 app/workflow/state.py（避免循环导入）。
路由决策已独立至 app/workflow/routing.py（纯决策函数与图路由函数分离）。

工作流拓扑:
    START
      │
      ▼
    Manager (规划拆解)
      │
      ├── intent_router: "clarify" ──→ Clarify (追问) ──→ END
      │
      └── "full_research"
            │
            ▼
          fan_out (透传节点, 条件边挂载点)
            │
            └──→ Send("research")   ──┐
            ├──→ Send("financial") ──┤  并行执行
            └──→ Send("sentiment") ──┘
                    │          │
                    └────┬─────┘
                         ▼
                   health_check (健康检查, attempts+1)
                         │
            ┌────────────┴─────────────┐
            │ retry (attempts<上限)     │ risk (全健康/重试耗尽)
            ▼                          ▼
          retry (透传)               Risk (真实 RiskAgent, 降级容错)
            │                          │
            └──→ Send 仅失败节点 ─┐     ▼
                    ▲             │   Report (研报生成)
                    └── 环 ───────┘     │
                                        ▼
                                 evaluate_report (评估, iteration+1)
                                        │
                         ┌──────────────┴──────────────┐
                         │ rework (不达标且<上限)        │ end (达标/达上限)
                         ▼                             ▼
                       rework (追加修订) ─→ 环          END

Author: 工藤
Date: 2026-08-05
Version: 0.5.0 — 意图分流 + 健康检查重试环 + Report 质量迭代环 + 真实 Agent 接入
"""

import logging
from datetime import datetime
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from app.agents.risk_agent import RiskAgent
from app.agents.sentiment_agent import SentimentAgent
from app.models.sentiment_risk_models import (
    FinancialSummary,
    SentimentInput,
    SentimentResult,
)
from app.workflow.routing import (
    _MAX_HEALTH_RETRIES,
    assess_report_quality,
    check_node_health,
    health_router,
    intent_router,
    report_router,
    retry_fan_out,
)
from app.workflow.state import ResearchState

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Manager Agent — 意图理解与任务规划
# ============================================================================


async def manager_node(state: ResearchState) -> dict[str, Any]:
    """Manager Agent — 意图理解与任务规划

    职责:
        1. 解析用户 query，提取目标公司、分析维度
        2. 制定任务执行计划，决定并行/串行策略
        3. 返回规划结果，由 Send API 进行并行扇出

    TODO (Phase 2):
        - 接入 LLM 进行自然语言意图解析
        - 动态任务 DAG 生成
    """
    company = state.get("company", "未知")
    logger.info("[Manager] 开始规划: 公司=%s", company)

    plan = {
        "company": company,
        "ticker": state.get("ticker"),
        "tasks": [
            {
                "id": "research",
                "agent": "ResearchAgent",
                "description": "企业基本面与行业分析",
                "dependencies": [],
            },
            {
                "id": "financial",
                "agent": "FinancialAgent",
                "description": "财务指标计算与趋势分析",
                "dependencies": [],
            },
            {
                "id": "sentiment",
                "agent": "SentimentAgent",
                "description": "市场舆情与热点挖掘",
                "dependencies": [],
            },
            {
                "id": "risk",
                "agent": "RiskAgent",
                "description": "综合风险评估",
                "dependencies": ["research", "financial", "sentiment"],
            },
            {
                "id": "report",
                "agent": "ReportAgent",
                "description": "结构化研报生成",
                "dependencies": ["risk"],
            },
        ],
        "strategy": "parallel_fan_out",
        "reasoning": f"对 {company} 执行标准五阶段投研分析流程",
    }

    logger.info("[Manager] 规划完成: %d 个任务, 策略=%s", len(plan["tasks"]), plan["strategy"])

    return {
        "manager_plan": plan,
        "started_at": datetime.now().isoformat(),
    }


# ============================================================================
# 2. Research Agent — 企业基本面与行业分析
# ============================================================================


async def research_node(state: ResearchState) -> dict[str, Any]:
    """Research Agent — 企业基本面与行业分析（真实 RAG）

    调用 app.rag.agent.arun_adaptive_research（自适应最多 3 轮 Research Loop，
    内部 evidence_search → app.rag.retrieve 真实检索），把 AgentState.current_report
    (ResearchReport) 映射为 research_result dict，与 ReportAssembler / report_node 兼容。

    异常安全：RAG 不可用（无入库数据 / 模型加载失败 / 网络抽风）时降级返回占位，
    由主链健康检查环识别并重试，绝不让主链崩。
    """
    company = state.get("company", "未知企业")
    logger.info("[Research] 开始分析: %s", company)

    try:
        from app.rag.agent import arun_adaptive_research

        query = state.get("user_query") or f"分析{company}的基本面、行业地位、竞争优势与经营风险"
        agent_state = await arun_adaptive_research(query)
        report = agent_state.current_report
        summary = _safe_get(getattr(report, "summary", None))
        if not summary:
            logger.warning("[Research] RAG 输出空摘要，降级占位")
            return {"research_result": _degraded_research_result(company)}
        result = _map_research_report(company, report)
    except Exception:  # noqa: BLE001 —— RAG 异常降级，保证主链路不中断
        logger.exception("[Research] 自适应 RAG 研究异常，降级占位")
        result = _degraded_research_result(company)

    logger.info("[Research] 分析完成: %s", company)
    return {"research_result": result}


def _map_research_report(company: str, report) -> dict[str, Any]:
    """ResearchReport → research_result dict（claim 取字符串，evidence 汇总为 sources）。"""
    advantages = [c.claim for c in (getattr(report, "advantages", None) or []) if getattr(c, "claim", "")]
    risks = [c.claim for c in (getattr(report, "risks", None) or []) if getattr(c, "claim", "")]

    sources: list[dict[str, Any]] = []
    for e in getattr(report, "evidence", None) or []:
        src = getattr(e, "source_type", None) or getattr(e, "source", None) or getattr(e, "document_type", None)
        if src:
            sources.append({"source": str(src), "page": getattr(e, "page", None)})

    plan = getattr(report, "plan_summary", None) or []
    return {
        "company": company,
        "summary": _safe_get(getattr(report, "summary", None)),
        "business_model": "\n".join(str(x) for x in plan),
        "industry_position": "",
        "competitive_advantages": advantages,
        "key_risks_business": risks,
        "sources": sources,
        "generated_at": _safe_get(getattr(report, "generated_at", "")),
    }


def _degraded_research_result(company: str) -> dict[str, Any]:
    """RAG 不可用时的占位降级（字段结构与 research_result 对齐）。"""
    return {
        "company": company,
        "summary": f"（待实现）{company} 的企业基本面分析结果",
        "business_model": "",
        "competitive_advantages": [],
        "industry_position": "",
        "key_risks_business": [],
        "sources": [],
        "generated_at": datetime.now().isoformat(),
    }


# ============================================================================
# 3. Sentiment Agent — 市场舆情与热点分析（真实 Agent 接入）
# ============================================================================


async def sentiment_node(state: ResearchState) -> dict[str, Any]:
    """Sentiment Agent — 市场舆情与热点分析

    调用 app/agents/sentiment_agent.SentimentAgent 执行完整链路:
    新闻抓取 → FinBERT 情感评分 → 热点聚类 → LLM 摘要。

    异常安全: 任何环节失败均降级返回 searched_news_count=0 的空产出，
    由健康检查环 (check_node_health) 识别为不健康并触发重试。
    """
    company = state.get("company", "未知企业")
    ticker = state.get("ticker", "")
    logger.info("[Sentiment] 开始舆情分析: %s (ticker=%s)", company, ticker)

    try:
        agent = SentimentAgent(llm=_get_workflow_llm())
        result = await agent.run(SentimentInput(
            symbol=ticker,
            company_name=company,
            days=30,
        ))
        logger.info("[Sentiment] 舆情分析完成: %s, 新闻数=%d", company, result.searched_news_count)
        # mode="json": 枚举转字符串、datetime 转 ISO，保证 state 可 JSON 序列化
        return {"sentiment_result": result.model_dump(mode="json")}

    except Exception:
        logger.exception("[Sentiment] 舆情分析异常，降级返回空产出（健康检查可识别）")
        return {"sentiment_result": {
            "symbol": ticker,
            "company_name": company,
            "searched_news_count": 0,
            "scored_news": [],
            "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
            "topics": [],
            "summary": "",
        }}


# ============================================================================
# 4. Risk Agent — 综合风险评估（真实 Agent 接入）
# ============================================================================


async def risk_node(state: ResearchState) -> dict[str, Any]:
    """Risk Agent — 综合风险评估

    调用 app/agents/risk_agent.RiskAgent 融合舆情 + 财务 + 行业三维度信号:
    纯 Python 结构化评分 (synthesize_risk) → LLM 润色风险总结。

    降级路径: 舆情数据缺失（重试耗尽）或 Agent 异常时，
    回退到 _degraded_risk_result 规则定级（保留 ROE 定级逻辑）。
    """
    company = state.get("company", "未知企业")
    logger.info("[Risk] 开始综合风险评估: %s", company)

    updates: dict[str, Any] = {}

    # 上游重试耗尽 → 记录 errors（整体写回，禁止原地 mutate）
    failed = state.get("failed_agents") or []
    if failed:
        updates["errors"] = (state.get("errors") or []) + [{
            "step": "risk",
            "error": f"上游 Agent 降级执行: {failed}",
            "timestamp": datetime.now().isoformat(),
        }]

    sentiment = state.get("sentiment_result")
    if not sentiment:
        logger.warning("[Risk] 舆情数据缺失，使用规则降级评估")
        updates["risk_result"] = _degraded_risk_result(state)
        return updates

    try:
        s_result = SentimentResult.model_validate(sentiment)
        financial = _to_financial_summary(state.get("financial_result") or {})
        agent = RiskAgent(llm=_get_workflow_llm())
        result = await agent.run(sentiment_result=s_result, financial=financial)
        # mode="json": 枚举转字符串、datetime 转 ISO，保证 state 可 JSON 序列化
        updates["risk_result"] = result.model_dump(mode="json")
        logger.info("[Risk] 风险评估完成: %s, 等级=%s", company, result.overall_risk_level.value)

    except Exception:
        logger.exception("[Risk] 风险评估异常，使用规则降级评估")
        updates["risk_result"] = _degraded_risk_result(state)

    return updates


def _to_financial_summary(financial_result: dict[str, Any]) -> FinancialSummary:
    """将 FinancialAgentOutput dict 映射为 RiskAgent 消费的 FinancialSummary。

    字段对应关系（FinancialSummary 全 Optional，缺失字段传 None）:
        revenue_growth     = key_metrics.revenue_yoy_pct / 100
        net_profit_margin  = dupont.net_profit_margin（已是小数）
        gross_margin / debt_ratio / free_cash_flow → None（财务节点未计算）
        anomalies          = [fetch_error]（数据源异常作为风险信号）
    """
    key_metrics = financial_result.get("key_metrics", {}) or {}
    dupont = financial_result.get("dupont", {}) or {}

    revenue_yoy = key_metrics.get("revenue_yoy_pct")
    try:
        revenue_growth = revenue_yoy / 100.0 if revenue_yoy is not None else None
    except (TypeError, ValueError):
        revenue_growth = None

    anomalies = []
    fetch_error = financial_result.get("fetch_error")
    if fetch_error:
        anomalies.append(str(fetch_error)[:200])

    return FinancialSummary(
        revenue_growth=revenue_growth,
        gross_margin=None,
        net_profit_margin=dupont.get("net_profit_margin"),
        debt_ratio=None,
        free_cash_flow=None,
        anomalies=anomalies,
    )


def _degraded_risk_result(state: ResearchState) -> dict[str, Any]:
    """规则降级风险评估 — 真实 RiskAgent 无法执行时的兜底。

    保留 ROE 定级逻辑（纯 Python 确定性计算），字段结构与
    RiskAssessment.model_dump() 对齐（key 用 risk_summary）。
    """
    company = state.get("company", "未知企业")
    fin = state.get("financial_result", {}) or {}
    key_metrics = fin.get("key_metrics", {}) or {}
    dupont_roe = key_metrics.get("roe_pct")

    financial_risk_level = "待评估"
    if dupont_roe is not None:
        if dupont_roe < 0:
            financial_risk_level = "high"
        elif dupont_roe < 5:
            financial_risk_level = "medium"
        else:
            financial_risk_level = "low"

    return {
        "symbol": state.get("ticker", ""),
        "company_name": company,
        "overall_risk_level": financial_risk_level,
        "overall_score": 0.5,
        "dimensions": [],
        "risk_summary": f"（降级评估）{company} 综合风险评估完成，财务风险等级={financial_risk_level}",
        "key_risks": [],
        "reasoning_chain": (
            f"舆情数据缺失/异常 → 仅基于财务指标规则定级 → {financial_risk_level}"
        ),
    }


# ============================================================================
# 5. Report Agent — 结构化研报生成
# ============================================================================


async def report_node(state: ResearchState) -> dict[str, Any]:
    """Report Agent — 结构化研报生成

    汇总所有前序 Agent 的输出，渲染 Markdown 研报。
    Financial 部分直接使用 FinancialAgentOutput 的结构化字段；
    Risk 部分使用 RiskAssessment 的 risk_summary + overall_risk_level。
    """
    company = state.get("company", "未知企业")
    logger.info("[Report] 开始生成研报: %s", company)

    # ── Financial 部分（来自 FinancialAgentOutput）─────────────────
    fin = state.get("financial_result", {}) or {}
    fin_summary = fin.get("key_metrics", {})
    fin_commentary = fin.get("commentary", "（待分析）")
    dupont = fin.get("dupont", {}) or {}

    # 财务指标摘要
    roe_pct = fin_summary.get("roe_pct")
    npm_pct = fin_summary.get("net_profit_margin_pct")
    rev_yoy = fin_summary.get("revenue_yoy_pct")
    prf_yoy = fin_summary.get("net_profit_yoy_pct")

    metrics_lines = []
    if roe_pct is not None:
        metrics_lines.append(f"- **ROE**: {roe_pct:.2f}%")
    if npm_pct is not None:
        metrics_lines.append(f"- **净利润率**: {npm_pct:.2f}%")
    if rev_yoy is not None:
        metrics_lines.append(f"- **营收同比**: {rev_yoy:+.2f}%")
    if prf_yoy is not None:
        metrics_lines.append(f"- **净利润同比**: {prf_yoy:+.2f}%")

    # 杜邦拆解
    dupont_lines = []
    if dupont:
        dupont_lines = [
            f"- 净利润率: {dupont.get('net_profit_margin', 0) * 100:.2f}%",
            f"- 资产周转率: {dupont.get('asset_turnover', 0):.4f}",
            f"- 权益乘数: {dupont.get('equity_multiplier', 0):.2f}",
        ]

    # ── 历年增速 ─────────────────────────────────────────────────
    yoy_section = ""
    yoy_history = fin.get("yoy_history", []) or []
    for item in yoy_history:
        rev = item.get("revenue_growth_pct")
        prf = item.get("net_profit_growth_pct")
        rev_str = f"{rev:+.2f}%" if rev is not None else "N/A"
        prf_str = f"{prf:+.2f}%" if prf is not None else "N/A"
        yoy_section += f"- {item.get('period', '')}: 营收 {rev_str}, 净利润 {prf_str}\n"

    sections = [
        f"# {company} 深度投研分析报告\n",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "---\n",
        "## 一、企业基本面与行业分析\n",
        f"{_safe_get(state.get('research_result'), 'summary', '（待分析）')}\n",
        "## 二、财务指标分析\n",
        f"### 核心指标\n{chr(10).join(metrics_lines)}\n" if metrics_lines else "（待分析）\n",
        f"### 杜邦分析\n{chr(10).join(dupont_lines)}\n" if dupont_lines else "",
        f"### 历年同比增速\n{yoy_section}",
        f"### CFO 专业点评\n{fin_commentary}\n",
        "## 三、市场舆情分析\n",
        f"{_safe_get(state.get('sentiment_result'), 'summary', '（待分析）')}\n",
        "## 四、综合风险评估\n",
        f"### 风险等级: {_safe_get(state.get('risk_result'), 'overall_risk_level', '待评估')}\n",
        f"{_safe_get(state.get('risk_result'), 'risk_summary', '（待分析）')}\n",
        "---\n",
        "*本报告由 FinanceAgent 多 Agent 系统自动生成，仅供参考，不构成投资建议。*\n",
    ]

    report_md = "\n".join(sections)
    logger.info("[Report] 研报生成完成: %s (%d 字符)", company, len(report_md))

    return {
        "report": report_md,
        "completed_at": datetime.now().isoformat(),
    }


# ============================================================================
# 5.5 CLARIFY 分支 — 意图信息不足时的追问
# ============================================================================


async def clarify_node(state: ResearchState) -> dict[str, Any]:
    """CLARIFY 分支 — 用户意图信息不足时返回追问文案。

    report 字段直接承载追问文本（AnalyzeResponse 公共 API 不变），
    跳过全部 Agent 扇出，直接 END。
    """
    company = state.get("company", "未知企业")
    logger.info("[Clarify] 意图信息不足，返回追问: %s", company)

    return {
        "intent": "clarify",
        "report": (
            "# 需要补充信息\n\n"
            "无法根据当前输入执行完整投研分析。\n\n"
            f"**请补充**: 具体分析目标与股票代码，例如: "
            f"\"分析{company}的财务健康状况与发展前景\"。"
        ),
        "completed_at": datetime.now().isoformat(),
    }


# ============================================================================
# 5.6 Report 质量迭代环 — evaluate_report → rework → 环
# ============================================================================


async def evaluate_report_node(state: ResearchState) -> dict[str, Any]:
    """Report 质量评估 — 判定研报是否达标，不达标则进入 rework 修订。

    循环防死: 每轮 iteration+1，decide_report_action 在轮次达
    _MAX_REPORT_ITERATIONS 后无条件返回 "end"（强制输出）。
    """
    iteration = state.get("iteration", 0) + 1
    quality = assess_report_quality(state.get("report"))

    logger.info(
        "[EvaluateReport] 第 %d 轮评估: score=%.2f, passed=%s, missing=%s",
        iteration, quality["score"], quality["passed"], quality["missing"],
    )

    return {
        "iteration": iteration,
        "report_quality": quality,
        "report_missing": quality["missing"],
    }


async def rework_node(state: ResearchState) -> dict[str, Any]:
    """Report 修订 — 基于质量评估结果向研报追加补充说明。

    append-only 确定性修订（不重跑 LLM），追加缺失章节清单。
    修订后直接回到 evaluate_report 再评估（不重跑 report_node，
    避免重新拼接覆盖修订内容），配合 iteration 上限收敛。
    """
    iteration = state.get("iteration", 0)
    missing = state.get("report_missing") or []
    report = state.get("report", "") or ""

    supplement = (
        f"\n\n## 补充说明（第 {iteration} 轮修订）\n\n"
        "> 质量评估发现以下待完善项:\n"
        + "\n".join(f"> - {m}" for m in missing)
        + "\n\n> 已记录缺失项，待对应 Agent 接入后自动补齐。\n"
    )

    logger.info("[Rework] 第 %d 轮修订: 追加 %d 项缺失说明", iteration, len(missing))
    return {"report": report + supplement}


# ============================================================================
# 6. 健康检查重试环 — health_check → retry(Send) → 环
# ============================================================================


async def health_check_node(state: ResearchState) -> dict[str, Any]:
    """健康检查 — 判定三个并行 Agent 的产出质量，决定是否重试。

    循环防死: 每轮 attempts+1，health_router 在轮次达 _MAX_HEALTH_RETRIES
    后无条件放行 risk（降级执行），上限即唯一强制出口。
    """
    attempts = state.get("attempts", 0) + 1
    health = check_node_health(state)
    failed = [name for name, ok in health.items() if not ok]

    degraded = bool(failed) and attempts >= _MAX_HEALTH_RETRIES

    logger.info("[HealthCheck] 第 %d 轮: %s | 失败=%s", attempts, health, failed or "无")
    if degraded:
        logger.warning("[HealthCheck] 重试耗尽，标记 degraded，进入降级执行")

    return {
        "attempts": attempts,
        "failed_agents": failed,
        "degraded": degraded,
    }


async def _passthrough(state: ResearchState) -> dict[str, Any]:
    """条件边挂载点 — 零逻辑透传节点（retry 使用）。"""
    return {}


async def fan_out_node(state: ResearchState) -> dict[str, Any]:
    """意图分流后的扇出挂载点 — 记录 intent 后由 Send API 并行扇出。"""
    return {"intent": "full_research"}


# ============================================================================
# 7. 并行路由 — LangGraph Send API 实现真正的 fan-out
# ============================================================================


def fan_out_to_agents(state: ResearchState) -> list[Send]:
    """Manager 之后的并行扇出路由。

    使用 LangGraph Send API 创建三个独立并行分支。
    每个 Send 对象携带完整 state 副本，分别进入 research/financial/sentiment 节点。

    "financial" 节点由 app.agents.financial_agent.node.financial_analysis_node 处理，
    该函数内部执行完整的: fetch → calculate → LLM commentary → output 链路。

    LangGraph 自动保证:
        - 三个节点并发执行（asyncio 层面真正的并行）
        - 所有分支完成后自动汇聚，合并 state 更新
        - 汇聚后才会执行后续节点（health_check → risk → report）
    """
    logger.info("[Router] 并行扇出 → research | financial | sentiment")
    return [
        Send("research", dict(state)),
        Send("financial", dict(state)),
        Send("sentiment", dict(state)),
    ]


# ============================================================================
# 8. LLM 获取 — 真实模型优先，Mock 兜底
# ============================================================================


class _MockLLM:
    """不调真实 API，返回固定文本。测试/无 Key 环境默认使用。"""

    async def ainvoke(self, msg: str):
        class _Resp:
            content = "该企业近期舆情总体平稳，未发现显著负面信号。"
        return _Resp()


def _get_workflow_llm():
    """获取 LLM 实例: DEEPSEEK_API_KEY 已配置 → 真实模型，否则 Mock 兜底。

    舆情/风险 Agent 的评分与工具链均为纯 Python 计算，LLM 只润色
    summary/risk_summary，因此 Mock 下功能依然完整。
    """
    try:
        from app.core.config import get_settings
        from app.core.llm_factory import get_llm

        if get_settings().deepseek_api_key:
            return get_llm("sentiment")
    except Exception:  # noqa: BLE001 —— 静默降级 Mock，保证不配 Key 可跑通
        logger.warning("[Workflow] 真实 LLM 不可用，使用 Mock LLM")
    return _MockLLM()


# ============================================================================
# 9. Graph 构建函数
# ============================================================================


def build_graph() -> CompiledStateGraph:
    """构建并编译 FinanceAgent 的 LangGraph 工作流。

    拓扑结构 (条件分支 + 双循环 + Send API 并行扇出):
        START → Manager → [intent_router] → clarify(追问) → END
                                    └→ fan_out → [Research ‖ Financial ‖ Sentiment]
                                          → health_check → [retry 环 | risk]
                                          → Report → [evaluate_report → rework 环] → END

    - Financial 节点由独立模块 app.agents.financial_agent 提供（三级降级容错）
    - Sentiment/Risk 节点接入真实 Agent 类（评分纯 Python，LLM 仅润色）
    - 健康检查重试环: health_check → retry(Send 仅重发失败节点)，attempts 上限 2
    - Report 质量迭代环: evaluate_report → rework(追加修订)，iteration 上限 3

    使用示例:
        graph = build_graph()
        result = await graph.ainvoke({
            "company": "宁德时代",
            "ticker": "300750",
            "user_query": "分析宁德时代的未来发展情况与风险",
            "current_step": "start",
            "errors": [],
        })
        print(result["report"])
    """
    # 懒加载 financial_analysis_node，避免模块级导入导致循环引用:
    #   node.py → workflow.__init__ → graph.py → node.py (❌)
    from app.agents.financial_agent.node import financial_analysis_node

    builder = StateGraph(ResearchState)

    # ---- 添加节点 ----
    # financial 节点使用独立 Agent 模块；sentiment/risk 使用真实 Agent 类
    builder.add_node("manager", manager_node)
    builder.add_node("fan_out", fan_out_node)
    builder.add_node("research", research_node)
    builder.add_node("financial", financial_analysis_node)
    builder.add_node("sentiment", sentiment_node)
    builder.add_node("health_check", health_check_node)
    builder.add_node("retry", _passthrough)
    builder.add_node("risk", risk_node)
    builder.add_node("report", report_node)
    builder.add_node("evaluate_report", evaluate_report_node)
    builder.add_node("rework", rework_node)
    builder.add_node("clarify", clarify_node)

    # ---- 设置入口 ----
    builder.set_entry_point("manager")

    # ---- 意图分流 ----
    # Manager → intent_router（条件边）: full_research → 扇出 / clarify → 追问 END
    builder.add_conditional_edges(
        "manager", intent_router, {"full_research": "fan_out", "clarify": "clarify"}
    )
    builder.add_edge("clarify", END)

    # ---- 并行扇出 ----
    # fan_out(透传) → Send API 创建三个并行分支
    builder.add_conditional_edges("fan_out", fan_out_to_agents)

    # ---- 健康检查汇聚 ----
    # 三个 Agent 各自完成后 → health_check（LangGraph 自动等待全部完成）
    builder.add_edge("research", "health_check")
    builder.add_edge("financial", "health_check")
    builder.add_edge("sentiment", "health_check")

    # ---- 健康检查重试环 ----
    # health_check → retry（有失败且未达上限）→ Send 仅重发失败节点 → 环
    # health_check → risk（全健康 或 重试耗尽，risk 节点内做降级处理）
    builder.add_conditional_edges(
        "health_check", health_router, {"retry": "retry", "risk": "risk"}
    )
    builder.add_conditional_edges("retry", retry_fan_out)

    # ---- 后续线性链路 ----
    builder.add_edge("risk", "report")

    # ---- Report 质量迭代环 ----
    # report → evaluate_report（评估）→ rework（不达标且未达上限）→ 环
    # evaluate_report → END（达标 或 iteration 达上限，强制输出）
    builder.add_edge("report", "evaluate_report")
    builder.add_conditional_edges(
        "evaluate_report", report_router, {"rework": "rework", "end": END}
    )
    builder.add_edge("rework", "evaluate_report")

    # ---- 编译 ----
    graph = builder.compile()
    logger.info("LangGraph 工作流编译完成 (Send API fan-out + 健康检查重试环)")

    return graph


# ============================================================================
# 10. 辅助函数
# ============================================================================


def _safe_get(mapping: dict | None, key: str, default: str = "") -> str:
    """安全地从可选 dict 中获取字符串值。"""
    if mapping is None:
        return default
    return mapping.get(key, default)
