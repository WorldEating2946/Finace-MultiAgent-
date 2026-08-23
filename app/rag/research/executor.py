"""研究执行引擎（PR #37）。

按 ResearchPlan.steps 顺序执行：每个步骤分派到对应工具 → 收集证据 → 更新 ResearchState。
确定性执行（无 while True LLM 循环）—— 每步可溯源，结果可审计。

Step dispatch:
    profile_lookup     → Step 名含"画像"（企业知识画像）
    evidence_search    → 其他步骤（query=step.retrieval_query + source_types）
    conflict_analysis  → 预留（跨源冲突由报告合成阶段用 evidence_pool 判断）
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.rag.research.schema import ResearchPlan, ResearchStep
from app.rag.research.state import Finding, ResearchState
from app.rag.research.tools import ResearchTools

# 画像步骤判定：步骤名含这些关键词 → 调 profile_lookup
_PROFILE_STEP_KEYWORDS = ("画像", "企业概况", "企业知识")
# 冲突分析触发：步骤名含竞争/冲突 → 预留（本期不自动触发）
_CONFLICT_STEP_KEYWORDS = ("竞争", "冲突", "格局")


class ResearchExecutor:
    """研究执行引擎：按计划顺序执行 → ResearchState。"""

    def __init__(self, tools: ResearchTools | None = None) -> None:
        """Args:
            tools: 工具集（测试可注入 mock）。
        """
        self._tools = tools or ResearchTools()

    # ── 主入口 ─────────────────────────────────────────────────
    def execute(self, plan: ResearchPlan) -> ResearchState:
        """执行 ResearchPlan → ResearchState（含 profile + findings + evidence_pool）。"""
        now = datetime.now(timezone.utc).isoformat()
        state = ResearchState(
            request=plan.request,
            intent=plan.intent.value,
            target=plan.target,
            plan=plan,
            started_at=now,
            updated_at=now,
        )
        return self.resume(state)

    def resume(self, state: ResearchState) -> ResearchState:
        """从已有 ResearchState 继续执行未完成步骤（PR38 自适应循环 / 断点恢复）。

        与 execute() 不同：不重建 state，只跑 completed_steps 之外的步骤，
        证据增量聚合——每次只检索补步步骤，已收集证据保留。
        execute() 等价于 resume(空 state)。
        """
        completed = set(state.completed_steps)
        # 快速模式 max_steps 截断执行步数，减少检索次数（提速）；默认 10 全跑
        max_steps = getattr(state, "max_steps", 10)
        for step in state.plan.steps[: max_steps or 10]:
            if step.order in completed:
                continue
            finding = self._execute_step(state, step)
            if finding is not None:
                state.findings.append(finding)
                self._accumulate_evidence(state, finding)
            state.completed_steps.append(step.order)
            state.updated_at = datetime.now(timezone.utc).isoformat()

        self._dedupe_evidence(state)
        return state

    # ── 单步执行 ───────────────────────────────────────────────
    def _execute_step(self, state: ResearchState, step: ResearchStep) -> Finding | None:
        """分派单步到工具；画像步骤返回 None（profile 直接入 state）。"""
        if self._is_profile_step(step):
            state.profile = self._tools.profile_lookup(state.target.company)
            return None
        if self._is_conflict_step(step):
            # 冲突步骤：本期先用证据检索填充（跨源冲突留报告合成/PR38 处理）
            pass
        refs = self._tools.evidence_search(
            step.retrieval_query,
            state.target.company,
            source_types=step.source_types or None,
        )
        return Finding(
            step_order=step.order,
            step_name=step.name,
            evidence=refs,
            source_types=step.source_types,
        )

    # ── 判定 ───────────────────────────────────────────────────
    @staticmethod
    def _is_profile_step(step: ResearchStep) -> bool:
        return any(kw in step.name for kw in _PROFILE_STEP_KEYWORDS)

    @staticmethod
    def _is_conflict_step(step: ResearchStep) -> bool:
        return any(kw in step.name for kw in _CONFLICT_STEP_KEYWORDS)

    # ── 证据聚合 ───────────────────────────────────────────────
    @staticmethod
    def _accumulate_evidence(state: ResearchState, finding: Finding) -> None:
        """把 finding 的证据追加到 evidence_pool（去重在 execute 末尾做）。"""
        state.evidence_pool.extend(finding.evidence)

    @staticmethod
    def _dedupe_evidence(state: ResearchState) -> None:
        """按 chunk_id 去重（保持首现顺序）。"""
        seen: set[str] = set()
        deduped: list = []
        for e in state.evidence_pool:
            if e.chunk_id in seen:
                continue
            seen.add(e.chunk_id)
            deduped.append(e)
        state.evidence_pool = deduped
