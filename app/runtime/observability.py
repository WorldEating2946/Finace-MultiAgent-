"""Runtime 可观测性词表（PR42b）—— Worker 生命周期事件 + 任务结局分类 + 结构化日志。

设计原则（见 docs 与 PR42b 复盘）：
    - 不新增审计表：Worker 生命周期追溯先走结构化日志（key=value），
      未来接 ELK / OpenTelemetry 时直接消费。
    - 结局分类（TaskOutcome）持久化到 research_tasks.terminal_reason，
      供 Evaluation / Monitoring 直接 SQL 查询"任务为什么这样结束"。

结构化日志示例：
    event=worker_reclaim research_id=r... worker_id=PC-01:123 attempt=2 reason=heartbeat_expired recovery_duration=351.2

事件（Worker 生命周期）与结局（任务终态原因）职责分离：
    event     —— 发生了什么（claim / heartbeat / stale / reclaim / timeout / fenced / completed）
    outcome   —— 任务为什么到达终态（completed / crash_recovered / runtime_timeout / ...）
"""

from __future__ import annotations

import logging

# 模块日志：`logger = logging.getLogger(__name__)` 标准约定，应用层统一配置 handler。
logger = logging.getLogger(__name__)


class WorkerEvent:
    """Worker 生命周期事件（一次状态变迁，进日志 + 指标）。"""

    WORKER_CLAIM = "worker_claim"          # CAS 认领成功（新任务 / 接管 / resume）
    WORKER_HEARTBEAT = "worker_heartbeat"  # 心跳续约（成功或失败）
    WORKER_STALE = "worker_stale"          # 检测到孤儿 / 过期心跳（reap/sweep 巡检发现）
    WORKER_RECLAIM = "worker_reclaim"      # 接管 stale running 孤儿任务（崩溃恢复）
    WORKER_TIMEOUT = "worker_timeout"      # watchdog：本代运行超 max_run → failed
    WORKER_FENCED = "worker_fenced"        # 租约已被接管，本 worker 写被拒（旧 worker 复活）
    WORKER_COMPLETED = "worker_completed"  # worker 到达终态（含 duration / outcome）


class TaskOutcome:
    """任务结局分类（持久化到 research_tasks.terminal_reason）。

    区分"崩溃恢复"与"运行超时"——本质不是一回事：
        CRASH_RECOVERED   进程崩溃 → 接管 → 最终完成
        RUNTIME_TIMEOUT   worker 没死，只是跑超 max_run → watchdog 判 failed
        USER_CANCELLED    用户主动取消
        WORKER_FENCED     本 worker 的租约已被接管（旧 worker 复活写被拒）
        MAX_ATTEMPTS_EXCEEDED  恢复尝试达到上限，不再无限重试
    常规终态：COMPLETED / REJECTED / FAILED。
    """

    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    USER_CANCELLED = "USER_CANCELLED"
    RUNTIME_TIMEOUT = "RUNTIME_TIMEOUT"
    MAX_ATTEMPTS_EXCEEDED = "MAX_ATTEMPTS_EXCEEDED"
    WORKER_FENCED = "WORKER_FENCED"
    CRASH_RECOVERED = "CRASH_RECOVERED"


def log_worker_event(
    event: str,
    *,
    research_id: str,
    worker_id: str,
    attempt: int,
    level: int = logging.INFO,
    **fields,
) -> None:
    """发射一条结构化 Worker 事件日志（key=value 单行，便于 grep / ELK 解析）。

    Args:
        event:      WorkerEvent 之一。
        research_id / worker_id / attempt: 必填定位字段。
        level:      日志级别（默认 INFO）。
        fields:     附加字段（reason / duration / status / outcome / error ...），
                    按 key=value 追加，None 值跳过。
    """
    parts = [f"event={event}", f"research_id={research_id}",
             f"worker_id={worker_id}", f"attempt={attempt}"]
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            value = f"{value:.3f}"
        parts.append(f"{key}={value}")
    logger.log(level, " ".join(parts))
