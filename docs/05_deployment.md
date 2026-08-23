# 05 · 部署与运维指南

> Research Agent V1 · 2026-08-07 · 基于 `docs/DOCKER_ARCHITECTURE.md` 整合升级

---

## 1. 部署策略一句话

> **Conda 管理 Python 环境 + Docker 管理基础服务，应用代码保持灵活开发，基础服务保持统一稳定。**

```
开发环境
    │
    ▼
Conda Environment
    │
    ▼
FastAPI Application
    │
    ▼
Docker Compose ──────────────┐
    │                        │
    ▼                        ▼
PostgreSQL                 Redis          （生产 + Milvus）
```

| 层级 | 工具 | 职责 |
|------|------|------|
| Python 运行时 | Conda | 管理 Python 3.11 版本与依赖 |
| 应用代码 | FastAPI | 业务逻辑、Research Agent、RAG |
| 基础服务 | Docker Compose | PostgreSQL / Redis / Milvus |

**设计理由**：Python 代码频繁修改调试，Conda 更灵活；数据库/中间件版本固定，Docker 统一管理。

---

## 2. 服务规划

| 服务 | 版本 | 用途 | 宿主机端口 |
|------|------|------|-----------|
| PostgreSQL | 16-alpine | 业务状态事实源 + Checkpoint | `5433` |
| Redis | 7-alpine | 任务调度队列（AOF 持久化） | `16379` |
| Milvus | 2.4.4 | 生产向量库（PR44） | `19531` |
| Attu | v2.4.12 | Milvus Web 控制台 | `30000` |

> 端口刻意避开官方默认端口，避免与宿主机已有服务冲突。容器内部保持默认端口。

---

## 3. 启动步骤

### 3.1 前置检查

| 步骤 | 说明 |
|------|------|
| 1 | Docker Desktop 已启动（`docker ps` 确认） |
| 2 | 当前目录为项目根目录 |
| 3 | 已创建 `.env`（`cp .env.example .env`） |
| 4 | 已准备 conda 环境 `finance-agent`（Python 3.11 + CUDA torch） |

### 3.2 启动基础服务

```bash
docker compose up -d
```

预期容器：`finance-postgres` Up、`finance-redis` Up。

### 3.3 初始化数据库（PR41）

```bash
# 幂等：建库 + business 业务表 + langgraph checkpoint schema
bash scripts/setup_postgres.sh
PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe scripts/init_db.py
```

### 3.4 向量后端准备（PR44）

- **FAISS（默认/开发/离线评测）**：零部署，索引落盘 `data/vector_store/`。
- **Milvus（生产）**：独立部署（Docker 或裸机）→ `scripts/migrate_faiss_to_milvus.py` 迁移（幂等 + 分批 + 对账 + 回滚）→ `scripts/verify_milvus.py` 验证。

### 3.5 启动服务

```bash
PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe -c "
from app.api.app import create_app
import uvicorn
uvicorn.run(create_app(), host='0.0.0.0', port=8000)"
```

> 启动时对所选向量后端做**健康检查 fail-fast**：后端不可用则拒绝启动（PR44.4），不产生"假健康"。

---

## 4. 环境变量清单

`.env.example` 已覆盖 PostgreSQL / Redis / LLM / RAG 基础键。PR41/PR44 新增键如下（按需在 `.env` 补充）：

| 变量 | 默认 | 说明 |
|------|------|------|
| `DATABASE_URL` | 空 | 业务库 DSN（PG），TaskManager 用 |
| `CHECKPOINT_DB_URL` | 空 | LangGraph checkpoint DSN（AsyncPostgresSaver） |
| `RAG_CHECKPOINT_STORE` | `memory` | checkpoint 后端：`memory` / `sqlite` / `postgres` |
| `RAG_CHECKPOINT_DB_PATH` | `data/checkpoints.db` | sqlite 后端时用 |
| `RAG_VECTOR_BACKEND` | `faiss` | 向量后端：`faiss` / `milvus` |
| `MILVUS_URI` | `http://localhost:19530` | Milvus 服务地址 |
| `MILVUS_DB_NAME` | `finance_agent` | **冻结**：共享环境只新建不碰 default 库 |
| `MILVUS_COLLECTION_NAME` | `finance_knowledge` | collection 名 |
| `MILVUS_DIM` | `1024` | BGE-M3 向量维度 |
| `RAG_QUERY_REWRITER` | `rule` | 查询改写：`rule` / `llm` / `off` |
| `RAG_RERANKER_MODEL` | 本地路径 | 精排模型：`metadata` 或 bge-reranker 路径 |
| `RAG_METADATA_COMPANY_WEIGHTS` | 空 | 各公司 metadata 精排权重（如 `{"小米":"0.90,0.08,0.02"}`） |

**敏感信息管理**：

- API Key / 数据库密码只进 `.env`，禁止出现在代码中（CLAUDE.md §12）。
- `.env` 已 `.gitignore`；只提交 `.env.example` 模板。
- `JWT_SECRET_KEY` 生产环境必须修改。

---

## 5. 常用运维命令

```bash
# 停止基础服务
docker compose down

# 查看日志
docker logs finance-postgres
docker logs finance-redis

# 向量后端切换（faiss → milvus）
# 1. .env 设 RAG_VECTOR_BACKEND=milvus + MILVUS_* 
# 2. migrate_faiss_to_milvus.py 迁移 → verify_milvus.py 验证
# 3. 重启 API（fail-fast 健康检查通过即就绪）

# 回归门禁（任何 RAG/Agent 改动必须跑）
PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe -m pytest \
  --run-real tests/test_regression.py -s

# 全量单测
PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe -m pytest
```

---

## 6. 团队协作规范

- 新增服务流程：说明用途 → 更新 `docker-compose.yml` → 更新本文档 → 通知团队。
- 禁止：私自启动不同版本服务、本地安装替代 Docker 管理的服务、提交含密码的配置。
- 端口冲突规避：新服务一律映射项目专属端口，容器内保持默认。

---

## 7. 故障排查

| 症状 | 排查 |
|------|------|
| API 启动即退出 | 向量后端健康检查 fail-fast（Milvus 未起 / 地址错） |
| 任务卡 queued | Redis 未起 / Worker 未启动 / PG 连接失败 |
| 任务卡 running | 检查 `lease_ttl`、Reaper 是否在跑 |
| 检索全空 | 向量库未入库或后端切换后未迁移（见 §5） |
| SSE 无事件 | 确认事件总线连接、task_id 正确 |

详见 `docs/04_runtime_architecture.md` Failure Matrix。
