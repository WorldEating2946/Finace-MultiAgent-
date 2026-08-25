# Changelog

> 版本节奏：`vX.Y` = 产品里程碑；`PR#` 见 docs/ 设计记录。稳定基线 Tag 记录在下方。

## [修正] README 定位还原 — 2026-08-07

**问题**：交付 PR（#56）误将 README 从**整体项目文档**重写为单一 Research Agent V1 视角，覆盖了 Mini FinAgent 多 Agent 平台（Manager / Research / Financial / Sentiment / Risk / Report）的整体设计蓝图。

**修正**（后续 PR）：README 恢复为整体项目文档，Research Agent V1 作为**已交付模块**标注（顶部状态块 + 6.2 节 + 路线图 Phase 1 完成区 + 团队分工 Member 2 ✅）。同步校准 README 中与现状不符的表述：技术栈（Python 3.11+ / BGE-M3 / VectorStore 抽象 / SSE）、目录结构（`app/agents`、`app/rag` 等实际布局）、快速开始（Docker PG/Redis + conda + `app.api.app` 入口）、输出格式（HTML 可视化报告）。

## [v1.0-research-agent] — 2026-08-07

**Research Agent V1 交付里程碑。** 完整链路：企业年报/研报入库 → Hybrid RAG 检索 → 多步骤 Research Agent → Evidence 归因报告 → 生产级任务运行。

### 新增

- **RAG 知识库**（PR30-35）
  - OCR 结构适配（PDF outline 优先 TOC，小米 Recall@5 10%→70%）
  - Query Rewrite（规则同义词 + LLM 改写，小米 Recall 70%→80%）
  - 评测体系（Recall@K / MRR / NDCG + 回归门禁）
  - Metadata-aware Rerank（小米 MRR 0.26→0.42，CATL 零回归）
  - 企业知识画像（9 字段 LLM 抽取 + Evidence 归因）
  - 多源融合（年报/研报/政策 + 跨源冲突检测）
- **Research Agent**（PR36-37.5）
  - 9 类研究意图理解 + 模板化研究计划
  - 逐步执行引擎（ResearchTools 抽象 + EvidenceRef 证据链）
  - LLM 报告合成（抗幻觉：LLM 只返回证据索引，来源由后端补全）
  - 5 项质量评测（证据覆盖/引用准确/论点对齐/完整度/步骤产出率）
- **自适应 + 记忆**（PR38-39）
  - 质量反馈路由自动决定继续查/补什么/何时结束
  - SqliteSaver 磁盘持久化，任务可跨进程重启恢复
- **服务层**（PR40-41）
  - FastAPI 4 端点 + Human-in-the-loop 审核闸口
  - 异步 TaskManager + AsyncPostgresSaver + SSE 事件流
- **生产级 Runtime**（PR42-43.6）
  - Worker Recovery（启动恢复 + CAS 租约 + Fencing + Reaper 看门狗）
  - Observability（结构化审计日志 + 内置 Metrics + 结局分类）
  - Redis 队列 + Worker Pool（多任务并发，CAS 保证一任务一 worker）
  - Redis AOF 持久化 / 优雅关闭 / 重连退避
- **企业级向量存储层**（PR44.1-44.4）
  - VectorStore 抽象层（add/search/delete/update/count 契约）
  - FAISS Adapter（本地开发/离线评测/紧急回滚）
  - Milvus Adapter（生产：finance_agent 库，FLAT/COSINE，多公司 scalar filter）
  - Migration Tool（FAISS→Milvus，幂等 + 分批 + 对账校验 + 回滚）
  - Benchmark（FAISS == Milvus 精度逐项一致，延迟仅高 6~9%）
  - 配置化后端切换 `RAG_VECTOR_BACKEND` + 启动健康检查 fail-fast

### 修复

- Hybrid 检索 sparse 缓存 id 复用漂移（WeakKeyDictionary，回归门禁根因）
- 画像/报告 LLM 空 content 自动重试（deepseek-v4-flash 推理模型）
- Atomic Finalize（状态终态与报告快照单事务，消除"completed 无报告"竞态）

### 质量基线

| 数据集 | Recall@5 | MRR | NDCG@5 | Top1 |
| ------ | -------- | ---- | ------ | ---- |
| Xiaomi（OCR 年报） | 80% | 0.423 | 0.474 | 30% |
| CATL（干净年报） | 100% | 0.950 | 0.957 | 90% |

**真实生产流程验证**（小鹏汽车 2025 年报，351 页）：772 chunks 入库 + 3 研究问题意图全命中、证据覆盖 100%、报告带章节/页码/原文证据链。

### 稳定基线 Tag

- `v0.43-runtime-stable` @ `1d1a6df`（PR41-43.6，全量 316 passed）
- `v1.0-research-agent` @ 本次交付（含 PR44 矢量存储层，407 passed）

---

## 历史 PR 索引

- PR30-35：RAG 检索链路（OCR 结构 / Query Rewrite / 评测 / Rerank / 画像 / 多源融合）
- PR36-37.5：Research Agent（意图理解 / 执行引擎 / 报告合成 / 质量评测）
- PR38-39：自适应循环 + Memory + Checkpoint
- PR40-41：Service Layer + 异步 Runtime
- PR42-43.6：Worker 加固 / 可观测性 / 队列与并发
- PR44.1-44.4：企业级向量存储层（抽象 / FAISS / Milvus / 迁移 / 基准 / 切换）
