"""
app/workflow/state.py — 全局工作流状态定义

将 ResearchState 独立于此文件，避免 graph.py 与 agents 子模块之间
产生循环导入。所有模块均可安全引用此 TypedDict。

Author: 工藤
Date: 2026-08-05
Version: 0.2.0
"""

from typing import Annotated, Any, TypedDict


class ResearchState(TypedDict, total=False):
    """FinanceAgent 工作流全局状态

    所有节点通过此 TypedDict 进行数据传递。
    每个节点返回部分状态更新（Partial Update），LangGraph 自动合并。

    设计约束:
        - 所有字段必须可 JSON 序列化（禁止放入 LLM Client、DB Session 等）
        - 大段文本用 str，结构化数据用 dict/list
        - 节点返回值 = 按字段覆盖合并（不是整体替换），计数字段必须在
          节点内读旧值写新值整体返回，禁止原地 mutate
    """

    # ---- 输入 ----
    company: Annotated[
        str,
        "目标公司名称，由用户输入",
    ]
    ticker: Annotated[
        str | None,
        "股票代码（可选），用于精确数据查询",
    ]
    user_query: Annotated[
        str | None,
        "用户原始提问文本",
    ]

    # ---- Manager 规划 ----
    manager_plan: Annotated[
        dict[str, Any] | None,
        "Manager Agent 生成的任务规划 {tasks: [...], reasoning: str}",
    ]
    current_step: Annotated[
        str,
        "当前执行步骤标识，用于状态追踪和调试",
    ]

    # ---- 各 Agent 输出 ----
    research_result: Annotated[
        dict[str, Any] | None,
        "Research Agent 输出: 企业背景、商业模式、行业环境等",
    ]
    financial_result: Annotated[
        dict[str, Any] | None,
        "Financial Agent 输出: 财务指标计算、趋势分析等",
    ]
    sentiment_result: Annotated[
        dict[str, Any] | None,
        "Sentiment Agent 输出: 舆情情感、热点聚类等",
    ]
    risk_result: Annotated[
        dict[str, Any] | None,
        "Risk Agent 输出: 风险矩阵、归因树、应对建议等",
    ]

    # ---- 最终输出 ----
    report: Annotated[
        str | None,
        "Report Agent 生成的最终研报 (Markdown)",
    ]

    # ---- 编排增强: 意图分流 ----
    intent: Annotated[
        str | None,
        "Manager 意图分流结果: full_research / clarify",
    ]

    # ---- 编排增强: 健康检查重试环 ----
    attempts: Annotated[
        int,
        "健康检查已执行轮次（上限 _MAX_HEALTH_RETRIES，防死循环）",
    ]
    failed_agents: Annotated[
        list[str],
        "最近一轮健康检查判定失败的节点名（重试成功后置空）",
    ]
    degraded: Annotated[
        bool,
        "重试耗尽仍失败时置 True（降级执行标记）",
    ]

    # ---- 编排增强: Report 质量迭代环 ----
    iteration: Annotated[
        int,
        "Report 质量评估轮次（上限 _MAX_REPORT_ITERATIONS，防死循环）",
    ]
    report_quality: Annotated[
        dict[str, Any] | None,
        "Report 质量评估结果 {score, missing, passed}",
    ]
    report_missing: Annotated[
        list[str],
        "Report 缺失章节（rework 依据）",
    ]

    # ---- 元信息 ----
    errors: Annotated[
        list[dict[str, Any]],
        "执行过程中捕获的错误列表 [{step, error, timestamp}]",
    ]
    started_at: Annotated[
        str | None,
        "工作流启动时间 (ISO 8601)",
    ]
    completed_at: Annotated[
        str | None,
        "工作流完成时间 (ISO 8601)",
    ]
