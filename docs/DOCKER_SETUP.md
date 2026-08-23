# FinaceAgent Docker 基础设施搭建指南

> **文档版本**: v2.0
> **适用阶段**: Phase 0.3+ — 基础服务容器化（PostgreSQL + Redis + Milvus + Attu）
> **最后更新**: 2026-08-10
> **架构设计**: 详见 [DOCKER_ARCHITECTURE.md](DOCKER_ARCHITECTURE.md)

---

## 1. 这是什么

FinaceAgent 采用 **Conda 管理应用 + Docker 管理基础设施** 的混合架构。
本文件指导团队成员在**自己的电脑**上，用一条命令拉起整套基础服务：

```
PostgreSQL（业务库 + LangGraph checkpoint）
Redis（缓存 / 任务调度队列）
Milvus Standalone（向量检索）+ etcd（元数据）+ MinIO（对象存储）
Attu（Milvus Web 控制台）
```

> ⚠️ **应用（FastAPI / Agent）不在 Docker 中运行**，请用 Conda 环境 `finance-agent` 跑应用。
> 本编排只解决「数据库 / 缓存 / 向量库」的安装与统一，与 `docs/pr44_milvus_design.md` 的
> 生产设计一致。

---

## 2. 前置要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Docker Desktop | 4.x 及以上 | Windows / macOS；Linux 用 Docker Engine + compose v2 插件 |
| Python | 3.11+ | 运行初始化脚本（用项目 Conda 环境 `finance-agent`） |
| Git | 任意 | 克隆仓库 |

检查 Docker 可用：

```bash
docker --version        # Docker 29.x 或更高
docker compose version  # Compose v2（输出含 "v2"）
```

---

## 3. 快速开始（5 步）

> 以下命令均在项目根目录（含 `docker-compose.yml` 的目录）执行。

### 第 1 步：克隆仓库

```bash
git clone <仓库地址> FinaceAgent
cd FinaceAgent
```

### 第 2 步：准备环境变量

```bash
# Linux / macOS
cp .env.example .env

# Windows (cmd / PowerShell)
copy .env.example .env
```

**关键：在 `.env` 中填写 PostgreSQL 连接串**（应用与初始化脚本读取，非必填默认值）

```bash
# 模板默认 POSTGRES_USER=finance / POSTGRES_PASSWORD=CHANGE_ME
# 若你改过这些，下面两行 DSN 要同步改
DATABASE_URL=postgresql://finance:CHANGE_ME@localhost:5433/finance_agent
CHECKPOINT_DB_URL=postgresql://finance:CHANGE_ME@localhost:5433/finance_agent

# LLM 密钥（需要时填，可后补）
DEEPSEEK_API_KEY=sk-xxxx
```

> `.env` 已加入 `.gitignore`，**禁止提交**。

### 第 3 步：启动全部基础服务

```bash
docker compose up -d
```

首次会拉取镜像（约 1~2 GB），随后自动按依赖顺序启动并等待健康检查。

### 第 4 步：验证容器状态

```bash
docker compose ps
```

预期 **6 个容器全部 healthy**：

```
NAME              IMAGE                                     STATUS
finance-postgres  postgres:16-alpine                        Up (healthy)
finance-redis     redis:7-alpine                            Up (healthy)
finance-etcd      quay.io/coreos/etcd:v3.5.14               Up (healthy)
finance-minio     minio/minio:RELEASE.2023-03-20T20-16-18Z  Up (healthy)
finance-milvus    milvusdb/milvus:v2.4.4                     Up (healthy)
finance-attu      zilliz/attu:v2.4.12                       Up (healthy)
```

快速连通性检查：

```bash
# PostgreSQL（密码按 .env 填写）
docker exec finance-postgres pg_isready -U finance -d finance_agent   # → accepting connections

# Redis
docker exec finance-redis redis-cli ping                              # → PONG

# Milvus（宿主侧，需在 .env 设 RAG_VECTOR_BACKEND=milvus 后由应用使用）
curl http://localhost:9091/healthz                                    # → OK
```

### 第 5 步：初始化数据库（见第 4 节）

```bash
# 用项目 Conda 环境执行（Windows 示例；Linux 用 python）
PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe scripts/init_db.py
```

---

## 4. 构建数据库（初始化步骤）

基础设施启动后，数据库还是"空壳"，按需执行以下初始化。

### 4.1 PostgreSQL — 业务表 + LangGraph checkpoint 表

容器首次启动时已通过 `docker/postgres/init.sql` 自动创建 `langgraph` schema。
业务表与 checkpoint 表由脚本创建（幂等，可重复执行）：

```bash
PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe scripts/init_db.py
```

成功输出：

```
[OK] public.research_tasks / research_reports 已就绪（含 PR42a 租约列）
[OK] langgraph.checkpoints / checkpoint_writes / checkpoint_blobs 已就绪
数据库初始化完成：business + langgraph schema
```

验证：

```bash
docker exec finance-postgres psql -U finance -d finance_agent -c "\dt"
# public.research_tasks / research_reports
docker exec finance-postgres psql -U finance -d finance_agent -c "\dt langgraph.*"
# langgraph.checkpoint_blobs / checkpoint_migrations / checkpoint_writes / checkpoints
```

> **注意**：`init_db.py` 读取 `.env` 的 `DATABASE_URL`；若为空，脚本回退到
> 硬编码默认 DSN。**务必先在第 3 步填好 `DATABASE_URL`**，否则会连错库。

### 4.2 Milvus — 创建业务库 + 灌入向量数据

Milvus 容器只提供引擎，**业务库 `finance_agent` 需要手动创建**。
应用健康检查（`app/api/app.py` 启动时）在库不存在时会直接报错 fail-fast，
所以首次使用 Milvus 前必须建库：

```bash
# 用项目 Conda 环境，一行建库（pymilvus 3.x：数据库管理在 MilvusClient；
# 库已存在会报错，可忽略——说明已建过）
PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe -c "\
from pymilvus import MilvusClient; \
c = MilvusClient(uri='http://localhost:19531'); \
c.create_database('finance_agent'); \
print('Milvus 业务库 finance_agent 已创建')"
```

验证库存在：

```bash
PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe -c "\
from pymilvus import MilvusClient; \
c = MilvusClient(uri='http://localhost:19531'); \
print(c.list_databases())"
# → ['default', 'finance_agent']
```

> Collection `finance_knowledge`（dim=1024, FLAT/COSINE）由应用在首次写入时自动创建，
> 无需手动建。Attu 控制台（见 4.3）可查看。

**灌入向量数据（可选，二选一）**：

- **从 FAISS 迁移**（你已有 `data/vector_store/<公司>/index.faiss` 本地索引）：
  ```bash
  PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe \
      scripts/migrate_faiss_to_milvus.py
  ```
- **从零灌入**：通过应用的文档入库接口（Research Agent 上传 / RAG ingest）写入，
  写入时自动创建 collection。

> 切到 Milvus 后端：`.env` 设 `RAG_VECTOR_BACKEND=milvus`，再重启应用。
> 详见 `docs/pr44_milvus_design.md`。

### 4.3 Attu — Milvus Web 控制台

浏览器打开 <http://localhost:30000>。Attu 已通过环境变量 `MILVUS_URL` 预连到
`milvus:19530`（容器内部 service name），登录即可浏览 `finance_agent.finance_knowledge`
的 collection 数据。

---

## 5. 端口与连接串速查

| 服务 | 宿主机端口 | 容器端口 | 容器内服务名 | 应用连接串 |
|------|-----------|----------|-------------|-----------|
| PostgreSQL | 5433 | 5432 | `postgres` | `postgresql://finance:CHANGE_ME@localhost:5433/finance_agent` |
| Redis | 16379 | 6379 | `redis` | `redis://localhost:16379` |
| Milvus | 19531 | 19530 | `milvus` | `http://localhost:19531` |
| Milvus healthz | 9091 | 9091 | — | `curl http://localhost:9091/healthz` |
| Attu | 30000 | 3000 | `attu` | 浏览器 <http://localhost:30000> |
| etcd | 不暴露 | 2379 | `etcd` | Milvus 内部使用 |
| MinIO | 不暴露 | 9000/9001 | `minio` | Milvus 内部使用 |

---

## 6. 日常操作

```bash
# 启动 / 停止（保留数据卷）
docker compose start
docker compose stop

# 查看状态 / 日志
docker compose ps
docker compose logs -f finance-milvus      # 单服务日志
docker compose logs -f                     # 全部

# 重建单服务（改配置后）
docker compose up -d --force-recreate redis

# 彻底停机（保留数据卷）
docker compose down

# ⚠️ 重置全部数据（删除所有数据卷，不可恢复！）
docker compose down -v
```

**数据持久化**：所有数据在 named volume 中，`docker compose down` 不丢。

| Volume | 挂载点 | 内容 |
|--------|--------|------|
| `postgres_data` | `/var/lib/postgresql/data` | PG 数据 |
| `redis_data` | `/data` | Redis AOF 文件 |
| `etcd_data` | `/etcd` | Milvus 元数据 |
| `minio_data` | `/minio_data` | 向量/标量数据对象存储 |
| `milvus_data` | `/var/lib/milvus` | Milvus 本地配置与日志 |

---

## 7. 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `docker compose up` 报端口占用 | 宿主机 5433/16379/19531/30000 已被占用 | 改 `.env` 对应端口并同步改 compose `ports` 映射；或 `docker ps` 找出占用容器 |
| `finance-postgres` 一直 unhealthy | `.env` 的 POSTGRES_USER/PASSWORD 与连接串不一致 | 检查 `.env` 中 POSTGRES_* 与 `DATABASE_URL` 的账号密码一致 |
| 应用启动报「Milvus database 'finance_agent' 不存在」 | 未执行 4.2 建库步骤 | 执行 4.2 的 `utility.create_database('finance_agent')` |
| 应用连 Milvus 超时 | Milvus 启动慢（首次建索引） | `docker compose ps` 等 `finance-milvus` 变 healthy；`curl localhost:9091/healthz` |
| `init_db.py` 报认证失败 | `DATABASE_URL` 为空，回退到默认 DSN | 在 `.env` 填写 4.1 的 DSN 后重跑 |
| Redis 队列重启后丢失 | 未挂载 redis.conf（AOF 未开） | 确认 `docker/redis/redis.conf` 已挂载且容器 healthy |
| 端口通但 Attu 打不开 | Attu 依赖 milvus healthy | `docker compose ps` 确认 `finance-milvus` healthy 后再刷新 |

---

## 8. 我不用这套编排，用自己的基础设施？

可以。本项目 Docker 编排面向**团队成员一键起基础服务**；若你本地已有等价服务
（如 PostgreSQL 5433 / Redis 16379 / Milvus 19531 / Attu 30000），
只需保证 `.env` 里的连接串、端口与你的服务一致即可，应用不关心服务从哪来。
