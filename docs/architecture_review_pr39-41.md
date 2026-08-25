# PR39 → PR41 架构复盘

> **文档版本**: v1.0
> **创建时间**: 2026-08-07
> **适用范围**: Research Agent 全链路（Memory / Checkpoint / Service / HITL / Async Runtime / PostgreSQL / SSE）
> **文档状态**: 已确认 —— 作为 **PR42a（Runtime Reliability）** 的设计依据
> **决策**（2026-08-07 项目组确认）:
>   - PR42a 先做 **Worker Crash Recovery 完整闭环**（启动恢复 + heartbeat + lease + stale detection + watchdog）
>   - PR43 再做 Multi-worker / Redis Queue；PR44 再做 Milvus 迁移
>   - SSE 竞态 / research_reports 死表 / reject 语义 / resume 锁 —— 顺手修，**不扩展成新模块**

---

## 1. 复盘范围与目的

PR39 → PR41 三次 PR 连起来后，Research Agent 拥有了完整生产链路：

```
Agent → Memory → Checkpoint → Service → HITL → Async Runtime → PostgreSQL → SSE
```

本文档复盘这条链路的**分层职责、数据流、状态模型、已实现能力与可靠性缺口**，
回答两个问题：

1. 当前架构哪里可靠、哪里是单进程假定下才能成立；
2. PR42 先做 Runtime Reliability 而不是 Milvus，依据是什么。

---

## 2. 分层全景与数据流

### 2.1 分层职责

| 层 | 位置 | PR | 职责 |
| -- | ---- | -- | ---- |
| Adaptive Agent | `app/rag/agent/` | PR38 | LangGraph 自适应研究循环（intent→plan→execute→report→evaluate→补步，≤3 轮） |
| Research Memory | `app/rag/memory/` | PR39 | ResearchRecord 压缩视图 + Memory Boundary（什么持久化/什么重新生成） |
| Durable Checkpoint | `app/rag/memory/store.py` | PR39/PR41 | memory / sqlite / postgres 三后端抽象，AsyncPostgresSaver 生产化 |
| Service 编排 | `app/services/research_service.py` | PR40/PR41 | API → Agent 的中间层，sync/async 双接口 |
| HITL 闸口 | `app/rag/agent/{human,review,interrupt}.py` | PR40 | 证据不足时 interrupt 暂停，人工 approve/reject/modify 续传 |
| Async Runtime | `app/runtime/task_manager.py` | PR41 | 后台 worker 生命周期：submit→queued→running→done |
| 事件流 | `app/runtime/events.py` | PR41 | NodeEvent + EventBus（进程内 pub/sub）→ SSE |
| API | `app/api/` | PR40/PR41 | FastAPI 5 端点 + 统一异常信封 {code, message, data} |
| 存储 | `public.research_tasks` + `langgraph.checkpoints` | PR41 | 业务表管生命周期，checkpoint 表管执行状态 |

### 2.2 请求数据流（postgres 生产路径）

```
POST /start
  → ResearchService.acreate_task
    → TaskManager.submit
      → INSERT research_tasks(status=queued)          [业务表]
      → asyncio.create_task(_run_worker)
      → 立即返回 {research_id, status: queued}
  → _run_worker
    → UPDATE research_tasks(status=running)
    → ResearchCheckpointer.arun (AsyncPostgresSaver)
      → graph.ainvoke / astream_events
      → 每节点 event_sink → EventBus.publish          [SSE 数据源]
    → 从 checkpoint 推导终态：
        pending interrupt → paused
        next_action == "end" → completed
    → UPDATE research_tasks(status=终态)
    → EventBus.publish(done)
```

**双库 1:1 关联**：`research_id == thread_id`，业务表与 checkpoint 表通过它互查。
这是整个架构的基石，PR42 的 recovery 设计也建立在它之上。

### 2.3 状态模型

`RecordStatus`: `queued → running → (paused|completed|failed|cancelled)`
（PR41 扩展 queued/cancelled；API 层 paused 对外映射为 `waiting_human`，兼容 PR40 语义）

`derive_status` 判定规则（`app/rag/memory/serializer.py`）：
- checkpoint 有 pending interrupt → `paused`
- `next_action == "end"` → `completed`
- 否则 → `running`

---

## 3. 分层质量评估

### 3.1 PR39 Memory + Checkpoint —— 设计最好的一层

- **Memory Boundary 清晰**：`to_record()` 只持久化「任务信息 + 进度 + Agent 决策 + 产物摘要」；
  embedding / LLM raw 天然不在 AgentState 里（PR37 可序列化 IR），边界零额外维护成本。
- **三后端抽象到位**：`get_checkpointer()` 调用方零改动切换 memory/sqlite/postgres，
  这是 PR41 无缝切生产的直接原因。
- **postgres 返回 factory 而非实例**（AsyncPostgresSaver 需 running event loop）——
  取舍正确，但把"实例化时机"的复杂度下沉到了下游（`_aresolve_cp` 分支 + `is_postgres` 判定）。

### 3.2 PR40 Service + HITL + API —— 结构正确，双接口偏重

- API→Service→Agent→Memory 分层到位，API 不直接耦合 LangGraph。
- HITL 用 `interrupt()` + `Command(resume=)` 语义正确，`human_router_fn` 正确路由 reject→end。
- **过渡痕迹**：sync + async 双接口并存（5 组方法靠 `is_postgres` 分支）。
  这是 PR40→PR41 的演进成本，生产路径已收敛到 async；sync 目前仅服务测试/脚本，建议后续标记 deprecated。

### 3.3 PR41 Runtime —— 能跑，但可靠性建立在"单进程假定"

- TaskManager 状态机完整：`submit → queued → running → done`，`except → FAILED` 兜底。
- SSE 事件协议设计良好：`astream_events` 逐节点推送 + 30s 心跳保活 + done/error 断流。
- **核心软肋：`_active` 是纯进程内存 dict** —— 这是下述所有可靠性问题的根。

---

## 4. 已验证的可靠性缺口（按严重度排序）

> 每条均已对代码确认，是 PR42a 设计输入。

### 🔴 #1 进程崩溃 → 任务永久卡死（无恢复机制）

`_run_worker` 的 worker 是 `asyncio.create_task`，只存在于本进程。**进程重启后**：
- `research_tasks` 里 `running`/`queued` 的任务永远无人再拾取（`_active` 随进程销毁）；
- LangGraph checkpoint 在 Postgres 完好（PR39 的价值），但**无任何代码扫描并接管孤儿任务**。

grep `recover|sweep|orphan|heartbeat|lease|reaper|stale` → **零命中**。
这是「任务可能中途凭空消失」的硬伤，PR42a 主战场。

### 🔴 #2 单进程 worker，无法横向扩展

所有 worker 在一个 uvicorn 进程内。多实例各自维护自己的 `_active`，不共享队列，
任务只被提交它的那个进程执行。→ PR43（Multi-worker / Queue）。

### 🟠 #3 无超时/看门狗

`_run_worker` 无 max-run 时限。LLM 调用（deepseek 长 prompt）或检索挂起时，
worker 永远占用，任务永远 `running`。→ PR42a watchdog。

### 🟠 #4 SSE 竞态窗口（PR41 遗留）

现有兜底只覆盖「订阅时任务**已完成**」；还存在一条更窄的窗口：
- 订阅瞬间任务仍是 `running` → `aget_task` 返回 running
- 但 worker 恰好此刻跑完并 publish `done` → **事件已发出，新订阅者错过**（EventBus 无历史）
- 流进入 30s keep-alive 死循环，永不结束。

对应 memory 记录「EventBus 无历史缓存（PR42 如需 replay）」。→ PR42a 顺手修。

### 🟠 #5 `research_reports` 表是死表

`init_db.py` 建了 `public.research_reports`，但全仓**无代码写它** ——
`get_report` 读的是 checkpoint 的 `current_report`。表建了不用 = Schema 漂移隐患。
→ PR42a 二选一：删表，或 worker 完成时落一份报告快照（对报表/审计更有意义）。

### 🟡 #6 其他小项

- **reject 语义丢失**：reject 后 `next_action="end"` → `derive_status` 判 COMPLETED，
  被拒绝的任务对外显示"完成"。审计痕迹只埋在 checkpoint 的 `human_decision`。
- **resume 双重提交无锁**：并发两次 resume 会同时 spawn `_resume_worker`，无 claim 保护。
- **checkpoint 全量快照**：每节点序列化整个 AgentState（含 evidence_pool），3 轮迭代攒大量
  全量副本，无 compaction —— 规模上来是存储隐患，暂不阻塞。
- **postgres 状态漂移风险**：`_has_pending` 吞所有异常返回 False，异常时可能把 PAUSED 误判为 RUNNING。

---

## 5. PR42 方向决策

**结论：先 Runtime Reliability（PR42a），再 Multi-worker（PR43），最后 Milvus（PR44）。**

依据：

1. **失败模式严重度不对等**：#1 意味着「进程重启一次，运行中的研究就永久消失」——这是
   正确性问题。Milvus 提升检索规模/延迟，完全碰不到这个坑。在会丢任务的生产系统上做
   向量库迁移 = 在流沙上盖楼。
2. **技术路线更完整**：`Agent→Memory→Checkpoint→Service→HITL→Async→Postgres→SSE`
   这条链唯一缺的是**可靠性闭环**。持久化已就绪，但恢复缺失。先把 Runtime 钉死，
   Milvus 迁移（替换 `vector_store.py` 全链路 + 重跑小米/CATL 回归门禁）才有安全网。
3. **改动规模符合 CLAUDE.md 约束**：Worker Crash Recovery 是**增量**（新模块 + 状态列 +
   接管扫描）；Milvus 是**重写一个工作正常的子系统**。按「Bug修复 > 功能增加 > 局部优化
   > 重构」优先级，可靠性修复在前，Milvus 重构在后。
4. **回归门禁可用性**：Milvus 迁移必须守住小米 R@5=80% / CATL R@5=100% 回归门，
   在 Runtime 稳定前动它 = 两线作战。

### 演进路线（已确认）

```
PR38  Adaptive Agent
  ↓
PR39  Memory / Checkpoint
  ↓
PR40  Service / HITL
  ↓
PR41  Async Runtime / PostgreSQL / SSE
  ↓
PR42a Runtime Reliability（Worker Recovery 闭环）  ← 下一步
  ↓
PR43  Multi-worker / Redis Queue
  ↓
PR44  Milvus
```

---

## 6. PR42a 范围界定

### ✅ 做：Worker Crash Recovery 完整闭环

- 启动恢复（startup sweep）：接管 stale 的 `running`/`queued` 孤儿任务，从 checkpoint
  `aresume(research_id, action=None)` 续跑（幂等，PR39 已备好）；
- heartbeat：worker 定期上报，证明自己存活；
- lease：任务被某 worker 认领的租约，防止多 worker 重复执行；
- stale detection：心跳超时 → 标记孤儿 → 重新调度；
- watchdog：max-run 时限 → FAILED + error_message。

### ⚠️ 顺手修（不扩展成新大模块）

- SSE 竞态窗口（订阅后二次确认 / EventBus 补终态）；
- `research_reports` 死表去留；
- reject 语义（被拒绝任务对外状态）；
- resume 并发锁。

### ❌ 后移

- Redis / Multi-worker 队列（→ PR43）；
- Milvus 迁移（→ PR44）；
- Multi-tenant。

---

## 7. 相关文档

- [docs/architecture_decisions.md](architecture_decisions.md) — ADR 记录（本复盘若产生新架构决策，追加 ADR）
- [docs/known_issues.md](known_issues.md) — 已知问题清单（可靠性缺口 #1–#6 可同步登记）
- [docs/RAG_ARCHITECTURE.md](RAG_ARCHITECTURE.md) — RAG 检索侧架构（PR44 Milvus 相关）
