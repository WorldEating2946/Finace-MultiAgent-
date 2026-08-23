"""自适应路由决策（PR38）。

决策基于 PR37.5 质量指标 + 迭代上限：
    - 迭代已满 → 强制 end（第 3 轮强制输出，防无限循环 / 成本失控）
    - 质量达标 → end（研究充分）
    - 有缺失维度 → replan（证据不足 → 补步重执行）
    - 否则 → end（无缺失但质量不足，避免盲目循环）

阈值可调（超参），不参与硬断言。
"""

from __future__ import annotations

from app.rag.research.evaluate import ResearchMetrics

# 最大迭代轮次（第 3 轮强制输出）
_MAX_ITERATIONS = 3

# 质量阈值：全部满足才判定"研究充分"
_EVIDENCE_COVERAGE_FLOOR = 0.5
_CITATION_ACCURACY_FLOOR = 0.5
_COMPLETENESS_FLOOR = 0.8


def quality_ok(m: ResearchMetrics | None) -> bool:
    """研究质量是否达标（None = 未评测 → 视为不达标）。"""
    if m is None:
        return False
    return (
        m.evidence_coverage >= _EVIDENCE_COVERAGE_FLOOR
        and m.citation_accuracy >= _CITATION_ACCURACY_FLOOR
        and m.completeness >= _COMPLETENESS_FLOOR
    )


def decide_next_action(
    m: ResearchMetrics | None,
    iteration: int,
    missing_dimensions: list[str],
    max_iterations: int = _MAX_ITERATIONS,
) -> str:
    """决定下一动作：replan（补步后重执行）或 end（终止）。

    Args:
        m:                 当前轮研究报告质量。
        iteration:         已执行轮次。
        missing_dimensions:证据缺失维度列表（空 = 无缺口）。
        max_iterations:    循环上限（快速模式=1 只用单轮，深模式=3）。

    Returns:
        "replan" 或 "end"。
    """
    if iteration >= max_iterations:
        return "end"              # 强制输出
    if missing_dimensions:
        return "replan"           # 有证据缺口 → 补步（即使整体质量尚可）
    if quality_ok(m):
        return "end"              # 研究充分
    return "end"                  # 无缺口但质量不足 → 停止
