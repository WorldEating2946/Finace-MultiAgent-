"""
tests/test_workflow_routing.py — app/workflow/routing.py 单元测试

覆盖 routing.py 全部决策函数与图路由函数:
    decide_intent / check_node_health / assess_report_quality / decide_report_action
    intent_router / health_router / retry_fan_out / report_router

纯函数测试——零 LLM、零网络、零 LangGraph 执行，直接断言返回值。
循环终止性（防死循环）是本文件的核心验证点:
    - 健康检查环: attempts 达 _MAX_HEALTH_RETRIES → 无条件 "risk"
    - Report 质量环: iteration 达 _MAX_REPORT_ITERATIONS → 无条件 "end"

Author: 工藤
Date: 2026-08-19
"""

import pytest
from langgraph.types import Send

from app.workflow.routing import (
    _MAX_HEALTH_RETRIES,
    _MAX_REPORT_ITERATIONS,
    assess_report_quality,
    check_node_health,
    decide_intent,
    decide_report_action,
    health_router,
    intent_router,
    report_router,
    retry_fan_out,
)


# ════════════════════════════════════════════════════════════
# 测试数据构建辅助
# ════════════════════════════════════════════════════════════

def _full_report() -> str:
    """四章节齐全、无占位符、足够长的合法报告。"""
    return (
        "# 测试公司 深度投研分析报告\n\n"
        "## 一、企业基本面与行业分析\n" + "行业背景说明。" * 10 + "\n"
        "## 二、财务指标分析\n" + "ROE 15%，营收同比增长 20%。" * 10 + "\n"
        "## 三、市场舆情分析\n" + "近期舆情平稳，未发现显著负面信号。" * 10 + "\n"
        "## 四、综合风险评估\n" + "综合风险等级低，主要风险可控。" * 10 + "\n"
    )


def _healthy_state() -> dict:
    """三个并行 Agent 产出全部健康的最小 state。"""
    return {
        "research_result": {"summary": "企业基本面分析结果"},
        "financial_result": {"data_source": "api"},
        "sentiment_result": {"searched_news_count": 9},
    }


# ════════════════════════════════════════════════════════════
# decide_intent — 意图分流决策
# ════════════════════════════════════════════════════════════

class TestDecideIntent:
    def test_empty_query(self):
        assert decide_intent("") == "clarify"

    def test_whitespace_query(self):
        assert decide_intent("   ") == "clarify"

    def test_none_query(self):
        assert decide_intent(None) == "clarify"

    def test_too_short_query(self):
        assert decide_intent("分析") == "clarify"

    @pytest.mark.parametrize("query", [
        "分析宁德时代",
        "宁德时代怎么样",
        "分析宁德时代的未来发展情况与风险",
    ])
    def test_normal_query(self, query):
        assert decide_intent(query) == "full_research"


# ════════════════════════════════════════════════════════════
# check_node_health — 三 Agent 产出健康度判定
# ════════════════════════════════════════════════════════════

class TestCheckNodeHealth:
    def test_all_healthy(self):
        health = check_node_health(_healthy_state())
        assert health == {"research": True, "financial": True, "sentiment": True}

    def test_empty_state_all_unhealthy(self):
        health = check_node_health({})
        assert health == {"research": False, "financial": False, "sentiment": False}

    def test_research_empty_summary(self):
        health = check_node_health({**_healthy_state(), "research_result": {"summary": ""}})
        assert health["research"] is False

    def test_research_missing(self):
        state = _healthy_state()
        state.pop("research_result")
        health = check_node_health(state)
        assert health["research"] is False

    @pytest.mark.parametrize("data_source", ["none", "error"])
    def test_financial_degraded_source(self, data_source):
        health = check_node_health(
            {**_healthy_state(), "financial_result": {"data_source": data_source}}
        )
        assert health["financial"] is False

    def test_financial_result_missing(self):
        """financial 节点完全无产出时不得误判为健康（回归: None 漏洞）。"""
        state = _healthy_state()
        state.pop("financial_result")
        health = check_node_health(state)
        assert health["financial"] is False

    def test_sentiment_zero_news(self):
        health = check_node_health(
            {**_healthy_state(), "sentiment_result": {"searched_news_count": 0}}
        )
        assert health["sentiment"] is False

    def test_sentiment_missing(self):
        state = _healthy_state()
        state.pop("sentiment_result")
        health = check_node_health(state)
        assert health["sentiment"] is False

    def test_partial_failure_only_failed_is_false(self):
        health = check_node_health(
            {**_healthy_state(), "sentiment_result": {"searched_news_count": 0}}
        )
        assert health["research"] is True
        assert health["financial"] is True
        assert health["sentiment"] is False


# ════════════════════════════════════════════════════════════
# assess_report_quality — Report 质量评估
# ════════════════════════════════════════════════════════════

class TestAssessReportQuality:
    def test_full_report_passed(self):
        quality = assess_report_quality(_full_report())
        assert quality["passed"] is True
        assert quality["missing"] == []
        assert quality["score"] == 1.0

    def test_none_report(self):
        quality = assess_report_quality(None)
        assert quality["passed"] is False

    def test_missing_one_section(self):
        report = _full_report().replace("## 三、市场舆情分析\n", "")
        quality = assess_report_quality(report)
        assert quality["passed"] is False
        assert "## 三、市场舆情" in quality["missing"]
        assert quality["score"] == pytest.approx(0.75)

    def test_placeholder_detected(self):
        report = _full_report().replace(
            "行业背景说明。行业背景说明。行业背景说明。",
            "（待实现）行业背景说明。",
        )
        quality = assess_report_quality(report)
        assert "（占位符残留）" in quality["missing"]

    def test_too_short_report(self):
        quality = assess_report_quality("太短")
        assert "（报告过短）" in quality["missing"]


# ════════════════════════════════════════════════════════════
# decide_report_action — 质量环下一步决策（防死循环核心）
# ════════════════════════════════════════════════════════════

class TestDecideReportAction:
    def test_passed_ends(self):
        assert decide_report_action({"passed": True}, 0) == "end"

    def test_not_passed_reworks(self):
        assert decide_report_action({"passed": False}, 0) == "rework"
        assert decide_report_action({"passed": False}, 1) == "rework"

    def test_limit_forces_end(self):
        """达上限无条件强制输出——即使质量不达标也终止循环。"""
        assert decide_report_action(
            {"passed": False}, _MAX_REPORT_ITERATIONS
        ) == "end"

    def test_over_limit_ends(self):
        assert decide_report_action({"passed": False}, 99) == "end"

    def test_none_quality_reworks(self):
        assert decide_report_action(None, 0) == "rework"


# ════════════════════════════════════════════════════════════
# intent_router — 图路由: Manager 后分流
# ════════════════════════════════════════════════════════════

class TestIntentRouter:
    def test_clarify(self):
        assert intent_router({"user_query": ""}) == "clarify"

    def test_missing_query_defaults_clarify(self):
        assert intent_router({}) == "clarify"

    def test_full_research(self):
        assert intent_router({"user_query": "分析宁德时代"}) == "full_research"


# ════════════════════════════════════════════════════════════
# health_router — 图路由: health_check 后分流（防死循环核心）
# ════════════════════════════════════════════════════════════

class TestHealthRouter:
    def test_failed_under_limit_retries(self):
        state = {"failed_agents": ["sentiment"], "attempts": 1}
        assert health_router(state) == "retry"

    def test_failed_at_limit_goes_risk(self):
        """重试耗尽——达上限无条件放行 risk（降级执行）。"""
        state = {"failed_agents": ["sentiment"], "attempts": _MAX_HEALTH_RETRIES}
        assert health_router(state) == "risk"

    def test_no_failure_goes_risk(self):
        state = {"failed_agents": [], "attempts": 1}
        assert health_router(state) == "risk"

    def test_missing_fields_goes_risk(self):
        assert health_router({}) == "risk"


# ════════════════════════════════════════════════════════════
# retry_fan_out — 图路由: 仅 re-Send 失败节点
# ════════════════════════════════════════════════════════════

class TestRetryFanOut:
    def test_all_healthy_no_send(self):
        sends = retry_fan_out(_healthy_state())
        assert sends == []

    def test_single_failure(self):
        state = {**_healthy_state(), "sentiment_result": {"searched_news_count": 0}}
        sends = retry_fan_out(state)
        assert len(sends) == 1
        assert isinstance(sends[0], Send)
        assert sends[0].node == "sentiment"

    def test_multi_failure(self):
        state = {
            "research_result": {"summary": ""},
            "financial_result": {"data_source": "none"},
            "sentiment_result": {"searched_news_count": 9},
        }
        sends = retry_fan_out(state)
        assert {s.node for s in sends} == {"research", "financial"}

    def test_all_failed(self):
        sends = retry_fan_out({})
        assert {s.node for s in sends} == {"research", "financial", "sentiment"}

    def test_send_carries_state_snapshot(self):
        state = _healthy_state() | {"company": "宁德时代"}
        sends = retry_fan_out(state)
        # 全健康无重发；改为单失败场景验证快照携带
        sends = retry_fan_out(state | {"sentiment_result": {"searched_news_count": 0}})
        assert sends[0].arg.get("company") == "宁德时代"


# ════════════════════════════════════════════════════════════
# report_router — 图路由: evaluate_report 后分流
# ════════════════════════════════════════════════════════════

class TestReportRouter:
    def test_passed_ends(self):
        state = {"report_quality": {"passed": True}, "iteration": 1}
        assert report_router(state) == "end"

    def test_not_passed_reworks(self):
        state = {"report_quality": {"passed": False}, "iteration": 1}
        assert report_router(state) == "rework"

    def test_limit_forces_end(self):
        state = {"report_quality": {"passed": False}, "iteration": _MAX_REPORT_ITERATIONS}
        assert report_router(state) == "end"

    def test_missing_fields_reworks(self):
        assert report_router({}) == "rework"
