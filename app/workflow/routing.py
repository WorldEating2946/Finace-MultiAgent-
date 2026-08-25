"""
app/workflow/routing.py — 编排路由: 纯决策函数 + 图路由函数

本模块承载主 workflow 的条件分支/循环路由逻辑，设计模式照搬
app/rag/agent/ 的 router.py + edges.py 分层:

    纯决策函数（无副作用、无 LangGraph 依赖，可直接单测）
        decide_intent / check_node_health / assess_report_quality /
        decide_report_action
    图路由函数（只读 state，返回条件边 key 或 Send 列表）
        intent_router / health_router / retry_fan_out / report_router

循环终结性保证（防死循环）:
    - 健康检查重试环: attempts 达 _MAX_HEALTH_RETRIES 时无条件走 "risk"
    - Report 质量迭代环: iteration 达 _MAX_REPORT_ITERATIONS 时无条件走 "end"
    上限即唯一强制出口，与质量判断函数的稳定性无关。

Author: 工藤
Date: 2026-08-19
Version: 0.1.0
"""

from typing import Any, Mapping

from langgraph.types import Send

from app.workflow.state import ResearchState

# ════════════════════════════════════════════════════════════
# 常量（循环上限）
# ════════════════════════════════════════════════════════════

_MAX_HEALTH_RETRIES = 2      # 健康检查环最多执行轮数（首轮 + 1 次重试）
_MAX_REPORT_ITERATIONS = 3   # Report 质量环最多迭代轮数（第 3 轮强制输出）


# ════════════════════════════════════════════════════════════
# 纯决策函数
# ════════════════════════════════════════════════════════════

def decide_intent(user_query: str | None) -> str:
    """意图分流决策。

    user_query 为空/过短 → "clarify"（信息不足，追问用户）
    否则 → "full_research"（标准五阶段投研流程）
    """
    q = (user_query or "").strip()
    return "full_research" if len(q) >= 4 else "clarify"


def check_node_health(state: Mapping[str, Any]) -> dict[str, bool]:
    """只读 state，判定三个并行 Agent 节点的产出健康度。

    判定规则（与各 Agent 的降级输出约定一致）:
        research  健康 = research_result 为 dict 且 summary 非空
        financial 健康 = financial_result.data_source 不在 ("none", "error")
        sentiment 健康 = sentiment_result.searched_news_count > 0
    """
    research = state.get("research_result") or {}
    financial = state.get("financial_result") or {}
    sentiment = state.get("sentiment_result") or {}

    research_ok = bool(isinstance(research, dict) and (research.get("summary") or "").strip())
    # None = 节点完全无产出，同样判为不健康
    financial_ok = financial.get("data_source") not in (None, "none", "error")
    sentiment_ok = bool((sentiment.get("searched_news_count") or 0) > 0)

    return {"research": research_ok, "financial": financial_ok, "sentiment": sentiment_ok}


def assess_report_quality(report: str | None) -> dict[str, Any]:
    """纯函数评估 Report 质量（对应 RAG 的 evaluate 环节）。

    检查项:
        1. 四个必需章节（## 一、## 二、## 三、## 四）是否齐全
        2. 是否残留占位符（"（待实现）" / "（待分析）"）
        3. 报告长度是否过短

    返回 {"score": float, "missing": list[str], "passed": bool}
    """
    text = report or ""
    missing: list[str] = []

    for section in ("## 一、企业基本面", "## 二、财务指标", "## 三、市场舆情", "## 四、综合风险"):
        if section not in text:
            missing.append(section)

    has_placeholder = ("（待实现）" in text) or ("（待分析）" in text)
    if has_placeholder:
        missing.append("（占位符残留）")

    if len(text) < 200:
        missing.append("（报告过短）")

    score = max(0.0, 1.0 - 0.25 * len(missing))
    return {"score": score, "missing": missing, "passed": not missing}


def decide_report_action(quality: dict[str, Any] | None, iteration: int) -> str:
    """决定 Report 质量环下一步: "rework" / "end"。

    防死循环: iteration 达上限时无条件 "end"（强制输出），
    镜像 app/rag/agent/router.py decide_next_action 的顶部强制出口。
    """
    if iteration >= _MAX_REPORT_ITERATIONS:
        return "end"
    if (quality or {}).get("passed"):
        return "end"
    return "rework"


# ════════════════════════════════════════════════════════════
# 图路由函数（只读 state，返回条件边 key / Send 列表）
# ════════════════════════════════════════════════════════════

def intent_router(state: ResearchState) -> str:
    """Manager 之后的意图分流（条件边，返回 path map 的 key）。"""
    return decide_intent(state.get("user_query"))


def health_router(state: ResearchState) -> str:
    """health_check 之后的分流。

    有失败节点且未达重试上限 → "retry"；
    全部健康或重试耗尽 → "risk"（进入风险评估，risk 节点内做降级处理）。
    """
    failed = state.get("failed_agents") or []
    attempts = state.get("attempts", 0)
    if failed and attempts < _MAX_HEALTH_RETRIES:
        return "retry"
    return "risk"


def retry_fan_out(state: ResearchState) -> list[Send]:
    """重试扇出 — 只 re-Send 健康检查判定失败的节点。

    重新计算健康度而非信任 state.failed_agents（两者永不互相矛盾），
    避免 financial 等有副作用的节点被重复执行真实 API 调用。
    """
    health = check_node_health(state)
    return [Send(name, dict(state)) for name, ok in health.items() if not ok]


def report_router(state: ResearchState) -> str:
    """evaluate_report 之后的分流: "rework" / "end"。"""
    return decide_report_action(
        state.get("report_quality"),
        state.get("iteration", 0),
    )
