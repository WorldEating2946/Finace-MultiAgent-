# FinaceAgent Docker 基础设施设计

> **文档版本**: v2.0
> **适用阶段**: Phase 0.3+ — 基础服务容器化（PostgreSQL + Redis + Milvus + Attu）
> **最后更新**: 2026-08-10
> **搭建步骤**: 见 [DOCKER_SETUP.md](DOCKER_SETUP.md)（本文件讲"为什么"，那份讲"怎么做"）

---

## 1. 文档目的

本文档规范 FinaceAgent 项目 Docker 基础设施设计，主要目标：

- **统一开发环境**：避免团队成员本地安装不同版本的数据库、中间件，消除"我机器上能跑"类问题；
- **保证环境一致性**：开发、测试使用相同的服务版本和配置；
- **为容器化部署打底**：为后续生产环境容器化提供基础设施。

---

## 2. 架构总览

FinaceAgent 采用 **Conda 管理 Python 开发环境 + Docker 管理基础服务** 的混合架构：

```
开发环境
    │
    ▼
Conda Environment (finance-agent)
    │
    ▼
FastAPI Application
    │
    ├──────────────┬───────────────┬───────────────┐
    ▼              ▼               ▼               ▼
PostgreSQL      Redis         Milvus Stack       Attu
(5433)         (16379)      (19531 + internal)   (30000)
                              ├── etcd   (元数据)
                              └── MinIO  (对象存储)
```

| 层级 | 工具 | 职责 |
|------|------|------|
| Python 运行时 | Conda | 管理 Python 版本与依赖 |
| 应用代码 | FastAPI | 业务逻辑、Agent 工作流 |
| 基础服务 | Docker Compose | 数据库、缓存、向量库、中间件 |

### Docker 负责

- 数据库服务（PostgreSQL）
- 缓存服务（Redis）
- 向量检索（Milvus Standalone）
- Milvus 配套（etcd 元数据 / MinIO 对象存储）
- 可视化控制台（Attu）

### Docker 不负责

- Python 环境管理
- 项目代码运行
- 本地代码调试

> **设计理由**：Python 项目开发中需要频繁修改和调试代码，Conda 更灵活；
> 数据库、中间件版本固定、不常变动，Docker 统一管理更合适。

---

## 3. 服务规划

### 3.1 当前阶段（Phase 0.3+）

| 服务 | 版本 | 用途 | 宿主机端口 | 状态 |
|------|------|------|-----------|------|
| PostgreSQL | 16-alpine | 主业务库 + LangGraph checkpoint | 5433 | 启用 |
| Redis | 7-alpine | 缓存、Session、任务调度队列（AOF） | 16379 | 启用 |
| Milvus Standalone | v2.4.4 | 生产向量检索后端 | 19531 | 启用 |
| etcd | v3.5.14 | Milvus 元数据存储（仅 internal） | — | 启用 |
| MinIO | RELEASE.2023-03-20 | Milvus 对象存储（仅 internal） | — | 启用 |
| Attu | v2.4.12 | Milvus Web 控制台 | 30000 | 启用 |

> **端口策略**：容器内部保持官方默认端口，宿主机使用项目专属端口，避免与本地已装服务冲突。

### 3.2 暂不加入

| 服务 | 原因 |
|------|------|
| Elasticsearch | 当前无独立全文检索需求（Milvus 标量过滤 + BM25 Hybrid 已覆盖） |
| MongoDB | 项目已采用 PostgreSQL JSONB 设计 |
| Celery / RabbitMQ | PR43 确认 Redis 队列够用（RUNTIME_REDIS_URL） |
| 应用容器化 | 属 Phase 3 平台化；当前应用走 Conda，保持开发灵活性 |

> **原则**：基础设施保持最小化，根据业务需求逐步增加，不提前引入。

---

## 4. 服务设计

### 4.1 PostgreSQL

**用途覆盖**：

- Agent 业务数据（`public.research_tasks` / `research_reports`）
- LangGraph Workflow 状态（`langgraph.checkpoints` 系列表）
- RAG 元数据

**Schema 隔离**：

```
public  ── 业务表（research_tasks / research_reports，scripts/init_db.py 幂等创建）
langgraph ── checkpoint 表（AsyncPostgresSaver.setup() 创建，search_path=langgraph）
```

`langgraph` schema 由容器首次启动时的 `docker/postgres/init.sql` 预建——
若缺省，checkpoint 表会落入 public，破坏状态隔离。

**数据持久化**：`postgres_data` named volume，容器删除后数据不丢失。

### 4.2 Redis

**用途覆盖**：

- 任务调度队列（PR43，`finance:research:queue`）
- 缓存热点数据
- Session 管理

**数据持久化**：AOF 开启（`docker/redis/redis.conf`，appendfsync everysec）。
任务状态以 PostgreSQL 为准，AOF 只把"重启丢队列"窗口从 ≤1h 降到 ≤1s。

### 4.3 Milvus Standalone（etcd + MinIO + Milvus）

Milvus 是分布式向量数据库，standalone 模式由三个进程组成：

```
Milvus Standalone
    │  存储数据 / 向量索引
    ▼
MinIO（对象存储，默认 minioadmin/minioadmin）
    │  保存 collection 元数据 / 索引描述
    ▼
etcd（元数据注册中心）
```

- **etcd**：Milvus 的元数据存储，不暴露宿主机端口（仅容器内部）；
- **MinIO**：Milvus 的对象存储，不暴露宿主机端口（仅容器内部）；
- **Milvus**：对外暴露 19531（gRPC/RESTful）与 9091（healthz / metrics）。

**业务库与 Collection**（与 `docs/pr44_milvus_design.md` AD-1/AD-2 一致）：

| 项 | 值 | 说明 |
|----|----|------|
| 业务库 | `finance_agent` | 需在 4.2 步骤手动建库；应用健康检查 fail-fast 依赖它存在 |
| Collection | `finance_knowledge` | dim=1024，FLAT/COSINE；首次写入时自动创建 |
| 访问凭证 | minioadmin/minioadmin | Milvus 镜像内建默认，与 MinIO root 凭证一致 |

### 4.4 Attu

Milvus 官方 Web 控制台，通过环境变量 `MILVUS_URL` 预连 `milvus:19530`，
供团队成员可视化查看 collection 数据与检索调试。

---

## 5. 网络设计

统一网络 `finance_network`（bridge，`internal: false`——宿主机需访问容器端口）。
容器之间通过 **service name** 通信，不依赖 IP。

```
                    finance_network (bridge)
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │  FastAPI(Conda) ──→ postgres:5432   (PostgreSQL)        │
    │     │   │   │                                             │
    │     │   │   └──→ milvus:19530     (Milvus)              │
    │     │   │            │                │                  │
    │     │   │            ▼                ▼                  │
    │     │   │      etcd:2379          minio:9000            │
    │     │   │            │                │                  │
    │     │   └──→ redis:6379      (Redis)                     │
    │     │                                                     │
    │     └──────→ attu:3000       (Attu ──→ milvus:19530)     │
    │                                                          │
    └──────────────────────────────────────────────────────────┘

    宿主机 (localhost)
        ├── localhost:5433  ──→ PostgreSQL
        ├── localhost:16379 ──→ Redis
        ├── localhost:19531 ──→ Milvus
        ├── localhost:9091  ──→ Milvus healthz
        └── localhost:30000 ──→ Attu
```

| 访问方式 | PostgreSQL | Redis | Milvus |
|----------|------------|-------|--------|
| 容器内部 | `postgres:5432` | `redis:6379` | `milvus:19530` |
| 宿主机 | `localhost:5433` | `localhost:16379` | `localhost:19531` |

> ⚠️ 容器内部不使用 `localhost`，必须使用 service name。

---

## 6. 项目目录结构

```
FinaceAgent/
├── docker-compose.yml          # 服务定义、网络、数据卷、环境变量
├── docker/
│   ├── postgres/
│   │   └── init.sql            # PostgreSQL 首启建 langgraph schema
│   └── redis/
│       └── redis.conf          # Redis AOF 持久化配置
├── scripts/
│   ├── init_db.py              # 建业务表 + LangGraph checkpoint 表
│   ├── migrate_faiss_to_milvus.py  # FAISS → Milvus 一次性迁移
│   └── verify_milvus.py        # Milvus 一致性校验
├── app/                        # 应用代码
├── docs/
│   ├── DOCKER_SETUP.md         # 搭建指南（快速开始/初始化/故障排查）
│   └── DOCKER_ARCHITECTURE.md  # 本文档（设计说明）
├── requirements.txt
├── .env                        # 环境变量（不入库）
└── .env.example                # 环境变量模板
```

| 文件/目录 | 职责 |
|-----------|------|
| `docker-compose.yml` | 服务定义、网络配置、数据卷管理、环境变量注入 |
| `docker/` | 数据库初始化脚本、服务配置文件 |
| `scripts/init_db.py` | 幂等建业务表与 checkpoint 表（Python DDL，避免 SQL 与代码脱节） |

---

## 7. 服务启动规范

### 7.1 前置检查

| 步骤 | 说明 |
|------|------|
| 1. Docker Desktop 已启动 | `docker ps` 确认可用 |
| 2. 当前目录为项目根目录 | 确保读取到 `docker-compose.yml` |
| 3. 已创建 `.env` 文件 | `cp .env.example .env`，并填好 `DATABASE_URL` / `CHECKPOINT_DB_URL` |

### 7.2 启动 / 停止

```bash
docker compose up -d       # 启动（含依赖顺序 + 健康检查）
docker compose ps          # 预期 6 个容器 healthy
docker compose down        # 停止（保留数据卷）
docker compose down -v     # ⚠️ 彻底清理（删除全部数据卷）
```

### 7.3 初始化

启动后按 [DOCKER_SETUP.md](DOCKER_SETUP.md) 第 4 节执行：
1. `scripts/init_db.py`（PostgreSQL 业务表 + checkpoint 表）
2. pymilvus 一行建 `finance_agent` 库（Milvus，首次必做）
3. 可选：`migrate_faiss_to_milvus.py` + `verify_milvus.py`

---

## 8. 环境变量管理

### 8.1 端口规范

| 服务 | 容器端口 | 宿主机端口 |
|------|----------|------------|
| PostgreSQL | 5432 | 5433 |
| Redis | 6379 | 16379 |
| Milvus | 19530 | 19531 |
| Attu | 3000 | 30000 |

**连接字符串示例**：

```
# PostgreSQL（与 .env 的 POSTGRES_* 一致）
postgresql://finance:finance123@localhost:5433/finance_agent

# Redis（与 .env 的 REDIS_PORT 一致）
redis://localhost:16379

# Milvus
http://localhost:19531
```

### 8.2 敏感信息管理

| 规则 | 说明 |
|------|------|
| 禁止直接提交 | 数据库密码、API Key、Token 等不得出现在代码中 |
| `.env` 加入 `.gitignore` | 确保敏感信息不入库 |
| 提交 `.env.example` | 提供配置模板，不含真实密码 |

**配置流向**：

```
.env  ──→  docker-compose.yml（postgres env_file）  ──→  service container
.env  ──→  应用 pydantic-settings                 ──→  连接串 / Milvus URI / Redis URL
```

---

## 9. 扩展规划

### 9.1 RAG 阶段

已拍板：**Milvus 为生产向量后端**（PR44），FAISS 保留用于本地开发 / 离线评测 / 紧急回滚。
向量后端通过 `RAG_VECTOR_BACKEND=faiss|milvus` 一键切换，与应用代码解耦。

### 9.2 Agent 生产化阶段

可能引入：

| 组件 | 用途 |
|------|------|
| 应用容器化（Dockerfile + Compose 应用服务） | Phase 3 平台化部署 |
| Monitoring | 服务监控与告警 |
| 对象存储（MinIO 直连应用） | 文档 / 报告持久化 |

---

## 10. 团队协作规范

### 新增服务流程

1. 说明服务用途与必要性
2. 更新 `docker-compose.yml`
3. 更新本文档与 `DOCKER_SETUP.md`
4. 通知团队成员同步

### 禁止事项

- ❌ 私自启动不同版本的服务
- ❌ 本地安装替代 Docker 管理的服务
- ❌ 提交包含密码的配置文件

---

## 11. 总结

FinaceAgent Docker 策略一句话：

> **应用代码保持灵活开发，基础服务保持统一稳定。**

```
Conda                            ← 管理 Python 环境
    │
    ▼
FastAPI + Agent 代码             ← 灵活开发、频繁修改
    │
    ▼
Docker Compose                   ← 统一基础服务
    ├── PostgreSQL 16            ← 主业务库 + LangGraph checkpoint
    ├── Redis 7                  ← 缓存 / 任务队列（AOF）
    ├── Milvus 2.4 (etcd+MinIO)  ← 向量检索后端
    └── Attu                     ← Milvus Web 控制台
```
