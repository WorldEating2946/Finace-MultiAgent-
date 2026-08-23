"""Research Memory Schema（PR39）。

ResearchRecord 是研究任务的压缩记录（human view / 任务状态）：
    任务信息 + 执行进度 + Agent 决策 + 中间产物摘要。
衍生自 AgentState（serializer.to_record），不重复持久化——避免与 LangGraph checkpoint 不同步。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RecordStatus(str, Enum):
    """任务生命周期状态（PR41 扩展：queued / cancelled）。"""

    QUEUED = "queued"          # 已创建，等待 worker 拾取
    RUNNING = "running"        # 执行中
    PAUSED = "paused"          # 中断 / 等待人工审核（pending interrupt）
    COMPLETED = "completed"    # 已完成（到达 END）
    REJECTED = "rejected"      # 人工审核拒绝（PR42a：不再伪装成 completed）
    FAILED = "failed"          # 执行失败
    CANCELLED = "cancelled"    # 用户取消（排队/运行中）


class ResumeAction(BaseModel):
    """人工审核决策载荷（PR41 Human-in-the-loop 接入时的 resume 输入）。"""

    decision: str = ""         # "approve" / "reject"
    note: str = ""             # 审核备注


class ResearchRecord(BaseModel):
    """研究任务的压缩记录（Memory Boundary 内的持久化字段）。"""

    # ── 任务信息 ──
    research_id: str = ""      # = thread_id（LangGraph 线程唯一标识）
    query: str = ""            # 原始研究请求
    company: str = ""          # 目标公司
    intent: str = ""           # 研究意图
    created_at: str = ""       # started_at
    updated_at: str = ""       # 最近 checkpoint 时间
    # ── 执行进度 ──
    status: RecordStatus = RecordStatus.RUNNING
    current_step: str = ""     # 当前/最后执行步骤名
    completed_steps: list[str] = Field(default_factory=list)  # 已完成步骤名列表
    # ── Agent 决策 ──
    iteration: int = 0
    next_action: str = ""
    missing_dimensions: list[str] = Field(default_factory=list)
    # ── 中间产物摘要（不存全量证据，全量在 checkpoint）──
    evidence_count: int = 0    # evidence_pool 条数
    finding_count: int = 0     # findings 条数
    coverage: float | None = None  # 报告证据覆盖率（evaluation 存在时）
