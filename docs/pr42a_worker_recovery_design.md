# PR42a Worker Crash Recovery 设计

> **文档版本**: v1.0（终版，2026-08-07 项目组确认）
> **设计依据**: [docs/architecture_review_pr39-41.md](architecture_review_pr39-41.md)
> **范围锁定**: P0 Worker Recovery 闭环 + P1 顺手修。**不新增** Redis / Multi-worker / Milvus / Multi-tenant。

---

## 1. 目标

把 PR39 的「持久化」真正变成「可恢复」。

当前痛点：进程崩溃 → `research_tasks` 里 `running`/`queued` 任务永久卡死，
LangGraph checkpoint 完好却无人接管。

本 PR 建立 Worker 崩溃后的**自动发现 → 判定过期 → CAS 接管 → 从 checkpoint 续跑 → 心跳 → 正常完成**闭环。

```
发现 Worker 死了
   ↓
判断 lease 过期（heartbeat_at）
   ↓
CAS 接管（worker_id + attempts fencing）
   ↓
从 Checkpoint 恢复（aresume，幂等）
   ↓
继续执行 + heartbeat
   ↓
正常完成
```

---

## 2. DDL 变更（`public.research_tasks` 增量，需审批后执行）

四个时间戳/标识职责**正交**：

| 列 | 职责 | 用于 |
| -- | ---- | ---- |
| `worker_id` | 谁在执行 | fencing（配合 attempts） |
| `claimed_at` | 什么时候开始执行 | **watchdog**（max_run 判定） |
| `heartbeat_at` | 最近什么时候还活着 | **lease 过期**（stale 判定） |
| `attempts` | 第几次执行（第几代租约） | fencing + max_attempts 上限 |

```sql
ALTER TABLE research_tasks ADD COLUMN IF NOT EXISTS worker_id      TEXT;
ALTER TABLE research_tasks ADD COLUMN IF NOT EXISTS claimed_at     TIMESTAMPTZ;
ALTER TABLE research_tasks ADD COLUMN IF NOT EXISTS heartbeat_at   TIMESTAMPTZ;
ALTER TABLE research_tasks ADD COLUMN IF NOT EXISTS attempts       INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_research_tasks_lease
    ON research_tasks(status, heartbeat_at);

-- research_reports 落最终快照（P1）需要唯一约束 + 业务字段
ALTER TABLE research_reports ADD COLUMN IF NOT EXISTS company      TEXT NOT NULL DEFAULT '';
ALTER TABLE research_reports ADD COLUMN IF NOT EXISTS query        TEXT NOT NULL DEFAULT '';
ALTER TABLE research_reports ADD COLUMN IF NOT EXISTS summary      TEXT NOT NULL DEFAULT '';
ALTER TABLE research_reports ADD COLUMN IF NOT EXISTS final_status TEXT NOT NULL DEFAULT 'completed';
ALTER TABLE research_reports ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
CREATE UNIQUE INDEX IF NOT EXISTS uq_research_reports_rid
    ON research_reports(research_id);
```

**为什么 watchdog 用 `claimed_at` 而不是 `heartbeat_at`**：
heartbeat_at 一直在跳（10:00 claim → 10:01/10:02/... 心跳），它只回答「worker 最近活着吗」，
回答不了「这个任务跑了多久」。`claimed_at` 是当前租约代次的启动时刻，
`claimed_at < NOW() - max_run` 才正确表达「一次执行超时」。

---

## 3. Lease 语义与 fencing（PR42a 核心）

### 3.1 租约 = `worker_id` + `attempts` 代次

每次 CAS 认领产生**新一代租约**（attempts+1）。worker 后续**所有写操作**必须携带
租约代次，数据库侧用复合条件拒绝旧 worker：

```sql
-- 认领（claim）：queued / 过期 running / paused → running，新一代租约
UPDATE research_tasks
SET    worker_id = :me, claimed_at = NOW(), heartbeat_at = NOW(),
       status = 'running', attempts = attempts + 1
WHERE  research_id = :rid
  AND  (
        status = 'queued'
        OR (status = 'running' AND heartbeat_at < NOW() - :lease_ttl)  -- stale，接管
        OR (status = 'paused')                                          -- 人工审核恢复
       )
RETURNING research_id, attempts, query, company, human_review;

-- 心跳（heartbeat）：只有当前租约持有者能续约
UPDATE research_tasks SET heartbeat_at = NOW()
WHERE research_id = :rid AND worker_id = :me AND attempts = :attempt;

-- 完成/失败（terminal）：同样带 fencing
UPDATE research_tasks SET status = :status, error_message = :err, updated_at = NOW()
WHERE research_id = :rid AND worker_id = :me AND attempts = :attempt;
```

### 3.2 为什么需要 fencing（旧 Worker 复活防护）

```
Worker A: attempt=1, lease_token=A → claim 成功 → running
Worker A 崩溃
Worker B: attempt=2 → CAS 接管 → running（A 的租约已失效）
Worker A 网络延迟"死而复生" → 继续 heartbeat/写状态
  ❌ 无 fencing：A 的写会污染 B 的状态
  ✅ 有 fencing：A 的写 WHERE attempts=1 匹配不到（B 已是 attempt=2）→ 0 行，拒绝
```

**单进程内 `worker_id` 唯一**，`worker_id + attempts` 即有效的代次令牌。
`lease_token UUID` 列留作 PR43 多进程时的强化选项（进程间 worker_id 可能碰撞场景），**本期不做**。

### 3.3 单一认领路径（三条路径统一）

```
新任务     queued  ──► claim() ──► running ──► Agent Runtime
恢复任务   paused  ──► claim() ──► running ──► Agent Runtime
崩溃恢复   stale   ──► claim() ──► running ──► Agent Runtime
```

- 天然解决原 `resume 双重提交无锁`：两个 resume 并发，CAS 只让一个成功。
- paused → resume 也消耗一次 attempt（新一代租约），防止病态 resume 循环。

---

## 4. Recovery 流程

### 4a. 启动扫描（startup sweep）—— `TaskManager.start()`，进程启动时跑一次

1. `ensure_started()`（幂等，首次 async 调用触发）：建池 → 启动扫描 → 启动 reaper 后台任务；
2. 扫描 `status IN ('queued','running')`：
   - `queued` → CAS 认领 + 调度（process 崩于 submit 后 worker 未跑的场景）；
   - `running` 且 `heartbeat_at < NOW() - lease_ttl` → 孤儿 → CAS 接管 + `aresume(action=None)` 续跑；
   - `attempts >= max_attempts` → 标记 `failed`（不再无限重试）。

进程重启 → 新 TaskManager → `ensure_started` → 自动接管上一进程的孤儿。这是崩溃恢复的主场景。

### 4b. 运行期 reaper（后台周期任务）

每个 `reaper_interval` 扫一次所有 `running` 任务：

| 判定 | 条件 | 动作 |
| ---- | ---- | ---- |
| **watchdog** | `claimed_at < NOW() - max_run_seconds` | `failed` + error_message（治 LLM 挂死 / 死循环） |
| **stale** | `heartbeat_at < NOW() - lease_ttl` | 重新 CAS 认领 + 续跑（同一接管函数） |
| **attempts 超限** | `attempts >= max_attempts` | `failed` + error_message |

**顺序**：watchdog 先于 stale。正常崩溃 → lease_ttl（30s）远小于 max_run（600s）→ 先被 stale 接管；
watchdog 只作用于「活着但跑太久」的任务。两者互不冲突。

### 4c. 心跳实现位置

`_run_worker` 内**并行协程**与 `arun` 同跑，`finally` 取消：

```python
async def _run_worker(self, research_id, attempt):
    beat = asyncio.create_task(self._heartbeat_loop(research_id, attempt))
    try:
        await self._cp.arun(...)   # 或 resume 续跑
        # 推导终态（pending→paused / reject→rejected / next_action=end→completed / watchdog→failed）
    finally:
        beat.cancel()
```

**已知约束（设计内声明）**：heartbeat 与 arun 共享单事件循环。若某节点的同步阻塞（如
`evidence_search` 的 FAISS/CrossEncoder）超过 `lease_ttl`，心跳会停跳导致误判接管。
PR42a 缓解：`lease_ttl` 必须大于最坏单节点阻塞时长（默认 30s，经验值足够）；根治（worker
线程/独立循环隔离）留给 PR43 Multi-worker。reaper 重复接管由 `attempts` 上限兜底，不会无限炸。

---

## 5. P1 顺手修（锁在现有模块内，不新建模块）

### 5a. SSE 竞态修复（`app/api/stream.py`）

现有兜底只覆盖「订阅时已终态」。修复：**先 subscribe 再二次确认**，并在 keep-alive 超时时复查：

```
subscribe(research_id) 后：
  - 再查一次 status：若已终态 → 立即补发 done → 退出（关闭订阅与发布之间的窗口）
  - 循环内 wait_for 超时（keep-alive）时：复查 status，终态则补发 done 退出
```

治「worker 恰好订阅瞬间跑完、done 已发出」导致 30s keep-alive 死循环的窄竞态。

### 5b. research_reports 落最终报告快照

**职责分离**：

| 存储 | 内容 | 用途 |
| ---- | ---- | ---- |
| LangGraph checkpoint | 当前 State / progress / decisions / 中间产物 / 恢复信息 | **运行时状态** |
| `public.research_reports` | research_id / company / query / final_report / summary / completed_at / final_status | **业务最终产物** |

worker 到达 `completed`/`rejected` 时 INSERT（`research_id` 唯一，upsert）最终报告快照。
以后做历史研究记录 / 报告列表 / 审计 / 导出，直接读业务表，不从 checkpoint 挖。

### 5c. reject → `rejected` 独立状态

`RecordStatus.REJECTED = "rejected"`：
- `derive_status`：`human_decision.action == "reject"` → REJECTED（先于 `next_action=="end"` 判定）；
- TaskManager 终态推导同样处理；`_STATUS_MAP` 增加 `rejected`。

被拒绝的任务不再伪装成 completed。

---

## 6. 配置项（`app/core/config.py` 新增）

| 配置 | 默认 | 说明 |
| ---- | ---- | ---- |
| `RUNTIME_HEARTBEAT_INTERVAL` | 10 | 心跳周期（秒） |
| `RUNTIME_LEASE_TTL` | 30 | 心跳过期判 stale（秒；须 > 最坏单节点阻塞） |
| `RUNTIME_REAPER_INTERVAL` | 15 | reaper 扫描周期（秒） |
| `RUNTIME_MAX_RUN_SECONDS` | 600 | watchdog 单代次上限（秒） |
| `RUNTIME_MAX_ATTEMPTS` | 3 | 单任务最大认领代次 |
| `RUNTIME_WORKER_ID` | auto | worker 标识（默认 `hostname:pid`） |

---

## 7. 测试计划

**单测（sqlite 后端 + mock 工具，不依赖真实 postgres）**
1. CAS 并发认领：两个 claim 抢同一任务 → 仅一个成功（RETURNING 行数）；
2. 心跳 fencing：attempt=1 的 heartbeat 在 attempt=2 接管后 → 0 行（旧 worker 被拒）；
3. stale 判定 + 接管：heartbeat 过期 → reaper 重新认领 → `aresume` 续跑；
4. watchdog：claimed_at 超 max_run → failed；
5. attempts 上限：超过 max_attempts → failed 不再调度；
6. paused → resume 走 CAS：并发两 resume 仅一个成功；
7. reject → RecordStatus.REJECTED。

**集成（postgres 后端，真实容器）**
- 模拟 worker 崩溃：submit → 杀死 worker task → 重启 TaskManager → startup sweep 接管 → 续跑完成；
- queued→running→completed→cancel 全生命周期回归。

**回归**
- `tests/test_runtime.py` 现有 8 项 + `tests/test_api.py` 不破。

---

## 8. 风险与回滚

| 风险 | 缓解 |
| ---- | ---- |
| 心跳停跳误判接管（同步阻塞节点） | lease_ttl > 最坏单节点阻塞；attempts 上限兜底 |
| 重复执行（接管后旧 worker 未死） | fencing 拒绝旧代次写；aresume 幂等 |
| DDL 变更 | 全 `IF NOT EXISTS` 幂等；独立 `ALTER` 语句可单列回滚 |
| 无限 crash 循环 | attempts 上限 → failed |

**回滚**：代码回退到 PR41 提交即可（新列对旧代码无害）；新列可留（幂等 DDL 不阻塞）。
