"""PR41 数据库初始化脚本。

创建 finance_agent 数据库的业务表（public schema）+ LangGraph checkpoint 表（langgraph schema）。

用法（需先启动 PostgreSQL 容器）：
    PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe scripts/init_db.py

幂等：所有 CREATE ... IF NOT EXISTS，可重复执行。
"""

import asyncio
import os
import sys

# Windows 下 psycopg async 需要 SelectorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import psycopg

# 业务库 DSN（读 .env 的 DATABASE_URL；未配置时用本地默认）
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://eduagent_user:123456@localhost:5433/finance_agent",
)

# PR42a：research_tasks 租约列（幂等 ALTER，兼容 PR41 已建库）
#   worker_id    —— 当前认领的 worker（fencing 用）
#   claimed_at   —— 本代租约开始时刻（watchdog max_run 依据）
#   heartbeat_at —— 最近心跳（lease 过期 / stale 依据）
#   attempts     —— 认领代次（fencing + max_attempts 防无限 crash 循环）
LEASE_DDL = """
ALTER TABLE research_tasks ADD COLUMN IF NOT EXISTS worker_id      TEXT;
ALTER TABLE research_tasks ADD COLUMN IF NOT EXISTS claimed_at     TIMESTAMPTZ;
ALTER TABLE research_tasks ADD COLUMN IF NOT EXISTS heartbeat_at   TIMESTAMPTZ;
ALTER TABLE research_tasks ADD COLUMN IF NOT EXISTS attempts       INTEGER NOT NULL DEFAULT 0;
-- PR42b：任务终态结局分类（COMPLETED/CRASH_RECOVERED/RUNTIME_TIMEOUT/USER_CANCELLED/
-- WORKER_FENCED/MAX_ATTEMPTS_EXCEEDED/REJECTED/FAILED），供 Evaluation/Monitoring SQL 查询
ALTER TABLE research_tasks ADD COLUMN IF NOT EXISTS terminal_reason TEXT;
CREATE INDEX IF NOT EXISTS idx_research_tasks_lease ON research_tasks(status, heartbeat_at);

-- research_reports 落最终业务快照（PR42a）：唯一约束 + 业务字段
ALTER TABLE research_reports ADD COLUMN IF NOT EXISTS company      TEXT NOT NULL DEFAULT '';
ALTER TABLE research_reports ADD COLUMN IF NOT EXISTS query        TEXT NOT NULL DEFAULT '';
ALTER TABLE research_reports ADD COLUMN IF NOT EXISTS summary      TEXT NOT NULL DEFAULT '';
ALTER TABLE research_reports ADD COLUMN IF NOT EXISTS final_status TEXT NOT NULL DEFAULT 'completed';
ALTER TABLE research_reports ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
CREATE UNIQUE INDEX IF NOT EXISTS uq_research_reports_rid ON research_reports(research_id);

-- 租约列索引必须放 ALTER 之后（旧库先加列再建索引，否则 UndefinedColumn）
CREATE INDEX IF NOT EXISTS idx_research_tasks_lease ON research_tasks(status, heartbeat_at);
"""

BUSINESS_DDL = """
CREATE TABLE IF NOT EXISTS research_tasks (
    research_id  TEXT PRIMARY KEY,
    thread_id    TEXT NOT NULL UNIQUE,
    status       TEXT NOT NULL DEFAULT 'queued',
    query        TEXT NOT NULL,
    company      TEXT NOT NULL DEFAULT '',
    human_review BOOLEAN NOT NULL DEFAULT FALSE,
    current_step TEXT DEFAULT '',
    iteration    INTEGER DEFAULT 0,
    missing_dimensions JSONB DEFAULT '[]',
    error_message TEXT,
    worker_id    TEXT,
    claimed_at   TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,
    attempts     INTEGER NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_research_tasks_status ON research_tasks(status);
CREATE INDEX IF NOT EXISTS idx_research_tasks_created ON research_tasks(created_at DESC);

CREATE TABLE IF NOT EXISTS research_reports (
    id           SERIAL PRIMARY KEY,
    research_id  TEXT NOT NULL UNIQUE REFERENCES research_tasks(research_id) ON DELETE CASCADE,
    report       JSONB NOT NULL,
    company      TEXT NOT NULL DEFAULT '',
    query        TEXT NOT NULL DEFAULT '',
    summary      TEXT NOT NULL DEFAULT '',
    final_status TEXT NOT NULL DEFAULT 'completed',
    completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_research_reports_rid ON research_reports(research_id);
"""


async def init_business(conn) -> None:
    """创建 public schema 业务表（research_tasks / research_reports）。

    顺序：BUSINESS_DDL（新建表，含 PR42a 租约列）→ LEASE_DDL（幂等 ALTER，兼容 PR41 已建库）。
    """
    async with conn.cursor() as cur:
        await cur.execute(BUSINESS_DDL)
        await cur.execute(LEASE_DDL)
    await conn.commit()
    print("[OK] public.research_tasks / research_reports 已就绪（含 PR42a 租约列）")


async def init_checkpoint() -> None:
    """创建 langgraph schema checkpoint 表（AsyncPostgresSaver.setup）。

    LangGraph 迁移含 `CREATE INDEX CONCURRENTLY`，必须 autocommit（不能进事务块）。
    直接用一个 autocommit 连接构造 saver，避开连接池事务上下文。
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    # autocommit 连接（CREATE INDEX CONCURRENTLY 需要）
    sep = "&" if "?" in DATABASE_URL else "?"
    pool_url = f"{DATABASE_URL}{sep}options=-c%20search_path%3Dlanggraph"
    conn = await psycopg.AsyncConnection.connect(pool_url, autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
    finally:
        await conn.close()
    print("[OK] langgraph.checkpoints / checkpoint_writes / checkpoint_blobs 已就绪")


async def main() -> None:
    conn = await psycopg.AsyncConnection.connect(DATABASE_URL, connect_timeout=10)
    try:
        await init_business(conn)
    finally:
        await conn.close()
    await init_checkpoint()
    print("\n数据库初始化完成：business + langgraph schema")


if __name__ == "__main__":
    asyncio.run(main())
