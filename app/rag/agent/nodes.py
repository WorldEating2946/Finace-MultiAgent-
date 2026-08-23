"""自适应研究节点（PR38 LangGraph nodes）。

build_nodes(tools, report_builder) 返回 6 个闭包节点，签名统一为
`fn(state: AgentState) -> dict | AgentState`：
    - 返回 dict = LangGraph 按字段合并（field update）；
    - 返回 AgentState 实例 = 整体替换状态（execute_node 用，resume 原地变异）。
依赖注入：tools / report_builder 在构建时绑定，测试可传 mock。

节点流：intent → planning(占位) → execute → report → evaluate → [router] → replan → execute ...
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.rag.agent.router import decide_next_action
from app.rag.agent.state import AgentState
from app.rag.research import ResearchExecutor, ResearchPlanner, evaluate_report
from app.rag.research.intent import IntentParser
from app.rag.research.planner import add_replan_step
from app.rag.research.report import ReportBuilder
from app.rag.research.schema import ResearchIntent, ResearchPlan


def build_nodes(*, tools=None, report_builder=None) -> dict[str, callable]:
    """构建 6 个节点（闭包绑定依赖，测试可注入 mock）。"""
    executor = ResearchExecutor(tools=tools)
    builder = report_builder or ReportBuilder()

    def intent_node(state: AgentState) -> dict:
        """意图理解 + 规划：NL 请求 → ResearchPlan（复用 PR36 parser + planner）。"""
        parser = IntentParser()
        planner = ResearchPlanner()
        intent, target, dims = parser.parse(state.request)
        plan = ResearchPlan(
            request=state.request,
            intent=intent,
            target=target,
            dimensions=dims,
            steps=planner.plan(intent, target, dims),
            confidence=0.3 if intent == ResearchIntent.GENERIC_RESEARCH else 0.85,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        return {"intent": intent.value, "target": target, "plan": plan}

    def planning_node(state: AgentState) -> dict:
        """规划节点（占位）：plan 已在 intent_node 生成，保留图结构。"""
        return {}

    def execute_node(state: AgentState) -> AgentState:
        """执行研究步骤：每轮 +1 迭代计数，只跑未完成步骤（增量收集证据）。"""
        new = state.model_copy(update={"iteration": state.iteration + 1})
        new = executor.resume(new)
        # 执行进度：指向本轮最后一个已完成步骤名（供 checkpoint record / 人工审核用）
        if new.plan.steps and new.completed_steps:
            last_order = new.completed_steps[-1]
            last_step = next((s for s in new.plan.steps if s.order == last_order), None)
            if last_step is not None:
                new.current_step = last_step.name
        return new

    def report_node(state: AgentState) -> dict:
        """合成研究报告（PR37 ReportBuilder，1 次 LLM / 轮）。"""
        report = builder.build(state)
        return {"current_report": report}

    def evaluate_node(state: AgentState) -> dict:
        """质量反馈（PR37.5）：metrics + 缺失维度 + 决策。"""
        m = evaluate_report(state.current_report, state=state)
        missing = _missing_dimensions(state)
        action = decide_next_action(m, state.iteration, missing, state.max_iterations)
        return {
            "evaluation": m,
            "missing_dimensions": missing,
            "next_action": action,
        }

    def replan_node(state: AgentState) -> dict:
        """动态补步：为每个缺失维度生成补充研究步骤（replanned 去重防膨胀）。"""
        plan = state.plan
        replanned = set(state.replanned_dimensions)
        for dim in state.missing_dimensions:
            if dim in replanned:
                continue
            new_step = add_replan_step(plan, dim)
            if new_step is None or new_step.order in {s.order for s in plan.steps}:
                continue
            replanned.add(dim)
            plan = plan.model_copy(update={"steps": [*plan.steps, new_step]})
        return {
            "plan": plan,
            "replanned_dimensions": sorted(replanned),
            "next_action": "continue",
        }

    def review_node(state: AgentState) -> dict:
        """人工审核闸口（PR40）：证据不足需补步时，暂停展示缺失维度等人工决策。

        首次 invoke：pause() 触发 interrupt → 图暂停，等待 resume；
        resume 后：pause() 返回人工决策 dict → 存入 human_decision（human_router 消费）。
        reject 决策 → next_action 置 end（derive_status 判 completed），避免停留 replan。
        """
        from app.rag.agent.interrupt import pause
        from app.rag.agent.review import build_review_payload

        decision = pause(build_review_payload(state).model_dump()) or {}
        action = decision.get("action", "approve")
        next_action = "end" if action == "reject" else state.next_action
        return {"human_decision": decision, "next_action": next_action}

    return {
        "intent": intent_node,
        "planning": planning_node,
        "execute": execute_node,
        "report": report_node,
        "evaluate": evaluate_node,
        "replan": replan_node,
        "review": review_node,
    }


def _missing_dimensions(state: AgentState) -> list[str]:
    """推导证据缺失维度：某维度下所有步骤均无证据产出 → 视为缺失。

    只考虑研究计划中出现过的维度；比 low_yield_steps 反查更稳健——
    补充步骤成功后，该维度即使原步骤空证据也视为已覆盖。
    """
    by_order = {s.order: s for s in state.plan.steps}
    dim_has_evidence: dict[str, bool] = {}
    for finding in state.findings:
        step = by_order.get(finding.step_order)
        if step is None:
            continue
        has = bool(finding.evidence)
        for dim in step.dimensions:
            dim_has_evidence[dim] = dim_has_evidence.get(dim, False) or has
    plan_dims = {d for s in state.plan.steps for d in s.dimensions}
    return sorted(d for d in plan_dims if not dim_has_evidence.get(d))
