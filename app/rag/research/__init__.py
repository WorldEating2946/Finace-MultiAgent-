"""研究执行引擎（PR #36+37+37.5）—— Research Intent Understanding + Execution + Evaluation。

PR #36 规划：NL 请求 → 意图分类 + 目标抽取 + 维度识别 → 有序研究步骤序列。
PR #37 执行：按 ResearchPlan.steps 顺序执行（工具抽象）→ 结构化研究报告（带证据链）。
PR37.5 评测：研究报告质量指标（证据覆盖率 / 引用准确度 / 完整度 / 步骤产出率）。

    from app.rag.research import run_research, build_research_plan, evaluate_report
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.rag.research.evaluate import ClaimEval, ResearchMetrics, evaluate_report
from app.rag.research.executor import ResearchExecutor
from app.rag.research.intent import IntentParser
from app.rag.research.planner import ResearchPlanner
from app.rag.research.report import ReportBuilder, ReportClaim, ResearchReport
from app.rag.research.schema import (
    ResearchDimension,
    ResearchIntent,
    ResearchPlan,
    ResearchStep,
    ResearchTarget,
)
from app.rag.research.state import Finding, ResearchState
from app.rag.research.tools import ResearchTools


def build_research_plan(
    request: str,
    *,
    _parser: IntentParser | None = None,
    _planner: ResearchPlanner | None = None,
) -> ResearchPlan:
    """一步构建研究计划：NL 请求 → 完整 ResearchPlan。

    Args:
        request:   自然语言研究请求（"分析小米汽车未来竞争力"）。
        _parser:   测试 seam —— 注入 IntentParser。
        _planner:  测试 seam —— 注入 ResearchPlanner。

    Returns:
        ResearchPlan（意图 + 目标 + 维度 + 有序步骤）。
    """
    parser = _parser or IntentParser()
    planner = _planner or ResearchPlanner()
    intent, target, dims = parser.parse(request)
    steps = planner.plan(intent, target, dims)
    # 置信度：规则分类，GENERIC 兜底为 0.3，否则按意图命中质量给分（占位可解释值）
    confidence = 0.3 if intent == ResearchIntent.GENERIC_RESEARCH else 0.85
    return ResearchPlan(
        request=request,
        intent=intent,
        target=target,
        dimensions=dims,
        steps=steps,
        confidence=confidence,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def run_research(
    request: str,
    *,
    _parser: IntentParser | None = None,
    _planner: ResearchPlanner | None = None,
    _tools: ResearchTools | None = None,
    _report_builder: ReportBuilder | None = None,
) -> ResearchReport:
    """NL 研究请求 → 结构化研究报告（端到端）。

    Args:
        request:   自然语言研究请求（"分析小米汽车未来竞争力"）。
        _parser:       测试 seam —— 注入 IntentParser。
        _planner:      测试 seam —— 注入 ResearchPlanner。
        _tools:        测试 seam —— 注入 ResearchTools。
        _report_builder: 测试 seam —— 注入 ReportBuilder。

    Returns:
        ResearchReport（title/summary/advantages/risks/uncertainties + 证据链）。
    """
    plan = build_research_plan(request, _parser=_parser, _planner=_planner)
    state = ResearchExecutor(tools=_tools).execute(plan)
    builder = _report_builder or ReportBuilder()
    return builder.build(state)


__all__ = [
    # schema
    "ResearchIntent",
    "ResearchTarget",
    "ResearchDimension",
    "ResearchStep",
    "ResearchPlan",
    # 规划
    "IntentParser",
    "ResearchPlanner",
    "build_research_plan",
    # 执行（PR #37）
    "ResearchState",
    "Finding",
    "ResearchTools",
    "ResearchExecutor",
    "ReportClaim",
    "ResearchReport",
    "ReportBuilder",
    "run_research",
    # 评测（PR37.5）
    "ClaimEval",
    "ResearchMetrics",
    "evaluate_report",
]
