# 01 · 系统架构设计

> Research Agent V1 · 2026-08-07 · 对应代码基线 `v1.0-research-agent`（develop @ PR44.4）

---

## 1. 总体分层

FinanceAgent 采用**六层架构**：API 层 → Runtime 层 → Agent 层 → RAG 层 → 向量存储层 → 数据/基础设施层。

```
┌───────────────────────────────────────────────────────────────┐
│ ① API 层          app/api/                                     │
│     HTTP + SSE · 路由 · 请求校验 · 人工审核闸口                │
└──────────────────────────┬────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│ ② Runtime 层       app/runtime/                                │
│     任务调度 · Worker 池 · 恢复 · 可观测性 · 指标              │
└──────────────────────────┬────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│ ③ Agent 层         app/rag/agent/ + app/rag/research/          │
│     意图理解 · 计划 · 执行 · 报告 · 评测 · 画像               │
└──────────────────────────┬────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│ ④ RAG 层           app/rag/                                    │
│     加载/解析 · 切片 · Dense+Sparse · RRF · 精排 · Query改写   │
└──────────────────────────┬────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│ ⑤ 向量存储层       app/rag/vectorstore/                        │
│     VectorStore ABC + FAISS + Milvus + Factory + 健康检查      │
└──────────────────────────┬────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│ ⑥ 基础设施         PostgreSQL · Redis · Milvus · LLM API       │
└───────────────────────────────────────────────────────────────┘
```

**依赖方向严格单向**：上层调用下层，下层不感知上层。各层通过 `services/` 业务编排串接，模块边界清晰。

---

## 2. 各层职责与关键模块

| 层 | 目录 | 职责 | 关键模块 |
|----|------|------|---------|
| API | `app/api/` | HTTP 路由、SSE 事件流、参数校验、健康检查 | `research.py` / `knowledge.py` / `profile.py` / `stream.py` / `health.py` |
| Runtime | `app/runtime/` | 异步任务调度、Worker 并发、故障恢复、审计日志、指标 | `task_manager.py` / `worker_pool.py` / `observability.py` / `metrics.py` / `events.py` |
| Agent | `app/rag/research/` | 研究意图理解→计划→执行→报告合成→质量评测 | `intent.py` / `planner.py` / `executor.py` / `report.py` / `evaluate.py` |
| 画像 | `app/rag/profile/` | 企业知识画像（9 字段 LLM 抽取 + 证据归因） | `extractor.py` / `storage.py` |
| RAG | `app/rag/` | 文档加载解析、智能切片、混合检索、精排 | `ingestion.py` / `pipeline.py` / `retriever.py` / `reranker/` / `fusion.py` |
| 向量存储 | `app/rag/vectorstore/` | 统一存储契约 + FAISS/Milvus 双实现 + 配置切换 | `base.py` / `faiss.py` / `milvus.py` / `factory.py` / `health.py` |
| 评测 | `app/rag/evaluation/` | RAG 检索质量评测（Recall/MRR/NDCG） | `evaluator.py` / `datasets/` |

---

## 3. 核心数据流（一个研究请求的完整旅程）

```
用户: "分析小鹏汽车2025年业务发展情况"
  │
  ▼
POST /research/start ──→ TaskManager 入 Redis 队列（立即返回 task_id）
  │
  ▼
Worker 认领 (PG CAS 租约) ──→ ResearchExecutor
  │  ├── ① 意图理解       → business_overview
  │  ├── ② 计划生成       → 企业知识画像 + 业务发展(财务/产品/市场)
  │  ├── ③ 画像查找/构建   → 9 字段 ProfileExtractor
  │  ├── ④ 逐步证据检索    → Hybrid RAG (BGE-M3 + BM25 → RRF → Rerank)
  │  ├── ⑤ 报告合成       → LLM 只输出证据索引，后端补全来源/页码/原文
  │  └── ⑥ 质量评测       → coverage/citation/alignment/yield
  │
  ▼
Atomic Finalize (单事务: 终态 + 报告快照) ──→ GET /research/{id} 可取结果
  │
  ▼
GET /research/{id}/stream ←── SSE 实时推送每个步骤状态
```

全程可观测：每一步状态/耗时/产出通过 `Observability` 结构化审计日志记录，`Metrics` 输出内置指标。

---

## 4. 模块边界规则（架构不可变）

1. **Agent 不直接触碰基础设施**：Agent 中禁止写 SQL、调外部接口、管理数据库连接——一律经 services/tools 封装（CLAUDE.md §4）。
2. **LLM 不负责精确计算**：结构化数据走 Tool + Python 计算；LLM 只做理解、规划、文本生成（CLAUDE.md §2.2）。
3. **State 必须是序列化 TypedDict**：可追踪、可序列化、可恢复（CLAUDE.md §2.3）。
4. **Agent 职责单一**：Research Agent 只做企业知识研究与证据检索，不越界做财务计算/舆情抓取。

---

## 5. 部署拓扑（生产）

```
[Client] ── HTTPS ──> [FastAPI 网关]
                        │
   ┌────────────────────┼─────────────────────┐
   ▼                    ▼                     ▼
[Worker Pool ×N]   [Redis(队列/AOF)]   [PostgreSQL(状态/租约)]
   │                                         │
   └──────────► [Milvus]  [LLM API]          └── 存 checkpoints + 报告
```

- **无状态 API**：可水平扩展，任务状态全在 PG/Redis。
- **Worker 可多实例**：CAS 租约保证一任务同一时刻只被一个 Worker 执行。
- **崩溃可恢复**：Checkpoint 持久化 + Reaper 看门狗，进程重启后任务续跑。

---

## 6. 演进预留

| 预留方向 | 现状 | 扩展点 |
|---------|------|--------|
| 多 Agent 协作 | 单 Research Agent | `app/agents/` + `app/workflow/` LangGraph 编排（已预留目录） |
| 多数据源 | 本地 PDF 年报/研报 | `app/rag/source/` 接入器扩展 |
| 多租户 | 单租户 | Runtime 层 task 归属字段 + API 鉴权 |
| 向量后端 | FAISS/Milvus | `vectorstore/factory.py` 注册新实现即可 |
