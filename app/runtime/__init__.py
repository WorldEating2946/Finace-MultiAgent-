"""Production Runtime（PR41）—— 异步 Agent 运行层。

把 PR40 的"同步 Service"升级为"异步 Runtime"：
    - TaskManager   任务编排（提交 / 后台 worker / 查询 / 干预 / 取消）
    - EventBus      SSE 事件总线（节点事件 → 订阅者流）
    - NodeEvent     SSE 事件载体

架构：API → ResearchService → TaskManager → ResearchCheckpointer → Agent
"""

from __future__ import annotations

from app.runtime.events import EventBus, NodeEvent
from app.runtime.task_manager import TaskManager
from app.runtime.worker_pool import RedisQueue, WorkerPool

__all__ = ["TaskManager", "EventBus", "NodeEvent", "WorkerPool", "RedisQueue"]
