# Runtime v1 架构

> 对应代码：`app/runtime/`（task_manager / worker_pool / observability / metrics）
> 覆盖 PR41 → PR43.6，稳定基线 Tag `v0.43-runtime-stable`（develop @1d1a6df）。

## 1. Runtime 职责

一个 Research 任务的生命周期由四个组件分工，各司其职、**互不为对方兜底**：

| 组件 | 职责 | 一句话 |
|------|------|--------|
| **Redis** | **任务调度** | "这个任务**需要被执行**"——list 队列（RPUSH/BLPOP），Worker 只从队列取 research_id，取到后**必须再走 PostgreSQL CAS 认领**才能执行 |
| **PostgreSQL** | **状态事实** | `research_tasks` 是唯一状态真相（queued/running/paused/completed/failed/rejected/cancelled + worker_id + attempts 租约列）。Redis 只是提示，PG 说了算 |
| **Checkpoint** | **Agent 恢复** | LangGraph checkpoint（thread_id）存 Agent 执行中间态；崩溃/重启后 `aresume` 幂等续跑，不从头执行 |
| **Report** | **业务产物** | `research_reports` 落最终报告快照（upsert），与终态**原子提交**（PR43.6）——运行时状态 ≠ 业务产物 |

```
用户 POST /research/start
        │
        ▼
 submit(): INSERT queued ──► Redis 入队
        │                        │
        ▼                        ▼
   PostgreSQL(真相) ◄── Worker BLPOP 取到 rid
                             │
                             ▼
                  _claim(): CAS 认领（queued/paused/stale→running）
                             │
                             ▼
                  _run_worker → Agent arun/aresume（checkpoint 续跑）
                             │
                             ▼
                  _finalize(): 终态 + 报告快照 单事务原子提交
```

**核心原则：**
- Redis = 调度，PG Lease = 所有权，Checkpoint = 恢复，Report = 产物。
- 四条路径共享同一条 CAS 认领（`_claim`）：**新任务入队 / 暂停后 resume / 崩溃接管续跑**。

## 2. Failure Matrix

| 故障 | 现象 | 恢复方式 | 关键机制 |
|------|------|---------|----------|
| **Worker crash** | 任务卡在 running、心跳停更 | Lease + Reaper | Reaper 看门狗发现 heartbeat 远超 `lease_ttl×3` → CAS 接管（attempts+1）→ 重置 queued + 重新入 Redis → 另一 Worker 认领续跑 |
| **Redis crash** | 队列丢失、enqueue 失败 | PG queued + re-enqueue | 任务已先落 PG queued 不丢；submit/worker 的 enqueue 失败被捕获；reaper 周期补入队。AOF（`docker/redis/redis.conf` appendonly）把重启丢队列窗口从 ≤1h 降到 ≤1s |
| **旧 Worker 复活** | 已交接的任务旧进程还在写 | **Fencing** | 所有写带 `WHERE worker_id=... AND attempts=...` 校验；新 Worker 已用新一代租约覆盖 → 旧 Worker 写 rowcount=0 被拒（`worker_fenced_total`） |
| **Agent 中断** | 进程崩溃 / 长任务被打断 | Checkpoint | 同一 thread_id + checkpoint 状态 → `aresume` 幂等续跑（attempts>1 → `CRASH_RECOVERED`）；SSE 可继续推送 |
| **服务关闭** | 正常下线 / 重启部署 | Graceful shutdown | 停收新任务 → 空闲 worker（阻塞 BRPOP）立即取消 → 忙碌 worker 等在途任务完成（≤ `runtime_shutdown_timeout`=30s）→ 超时强制取消；已出队任务归还队列 |
| **运行超时** | Agent 跑死 / 同步阻塞 | Watchdog | `claimed_at` 超 `max_run_seconds` → 判 `RUNTIME_TIMEOUT` failed（并取消本地卡死 worker） |
| **重试超限** | 反复崩溃 | Max attempts | attempts ≥ `max_attempts`（默认 3）→ 不再接管，判 failed（`MAX_ATTEMPTS_EXCEEDED`） |

> 结局分类统一落 `terminal_reason`：`COMPLETED / CRASH_RECOVERED / FAILED / RUNTIME_TIMEOUT /
> MAX_ATTEMPTS_EXCEEDED / WORKER_FENCED / USER_CANCELLED / REJECTED`。

## 3. 状态机

**主链（自动执行）：**

```
      queued
        │
        │ claim (CAS 认领)
        ▼
      running
        │
        ├──────────────────────────┐
        ▼                          ▼
   completed                  failed
        │
        │ 报告快照(原子提交)
        ▼
   research_reports
```

**HITL 暂停/恢复（人工审核闸口）：**

```
      running
        │
        │ 人工审核挂起（checkpoint pending → _derive_terminal）
        ▼
      paused
        │
        │ resume(approve/modify) → 同一 _claim 路径
        ▼
      running
```

**其它终态：**

```
 running/queued ── cancel ──► cancelled（USER_CANCELLED）
 paused ── resume(reject) ──► rejected（REJECTED）
 running ── watchdog 超时 ──► failed（RUNTIME_TIMEOUT）
 running ── attempts≥max ──► failed（MAX_ATTEMPTS_EXCEEDED）
 running ── stale 心跳过期 ──► (Reaper) queued 重新入队 → running（新一代租约）
```

**认领路径汇总（`_claim` 一条 SQL 三种来源）：**

```
status='queued'                          → running     （新任务）
status='running' AND 心跳过期(>lease_ttl) → running     （崩溃接管，attempts+1）
status='paused'                          → running     （人工审核后 resume）
```

> 注：`rejected` / `cancelled` 无报告快照；`completed` / `rejected` 走快照（`_finalize` 单事务原子写）。
> 状态字段：`queued / running / paused / completed / failed / rejected / cancelled`。

## 关联文档

- [PR42a Worker Recovery 设计](pr42a_worker_recovery_design.md)
- [架构决策记录](architecture_decisions.md)
- [RAG 架构](RAG_ARCHITECTURE.md)
