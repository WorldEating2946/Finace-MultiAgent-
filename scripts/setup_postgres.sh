#!/bin/bash
# PR41 PostgreSQL 容器启动脚本。
#
# 复用已有的 postgres 容器（edu_agent_postgres / finance_agent_postgres 均可），
# 若不存在则创建新容器。确保 finance_agent 数据库可用（通过 psycopg 创建）。
#
# 用法：
#   bash scripts/setup_postgres.sh
#
# 幂等：容器已存在则直接启动；finance_agent 库不存在则自动创建。

set -e

# ── 配置（与 .env 一致）──────────────────────────────────────
DB_NAME=finance_agent
DB_USER=eduagent_user
DB_PASS=CHANGE_ME
HOST_PORT=5433
CONTAINER_NAME=finance_agent_postgres

# 已有任意 postgres 容器 → 复用它（优先已有，不重复创建）
EXISTING=$(docker ps -a --format '{{.Names}}' | grep -E 'postgres' | head -1 || true)
if [ -n "$EXISTING" ]; then
    echo "[*] 复用已有 postgres 容器: $EXISTING"
    CONTAINER_NAME=$EXISTING
    docker start "$CONTAINER_NAME" 2>/dev/null || true
else
    echo "[*] 创建新容器 $CONTAINER_NAME（$DB_NAME / $DB_USER / :$HOST_PORT）"
    docker run -d --name "$CONTAINER_NAME" \
        -e POSTGRES_DB=postgres \
        -e POSTGRES_USER="$DB_USER" \
        -e POSTGRES_PASSWORD="$DB_PASS" \
        -p "$HOST_PORT:5432" \
        -v finance_agent_pgdata:/var/lib/postgresql/data \
        postgres:15-alpine
fi

# 等待就绪
echo "[*] 等待 PostgreSQL 就绪 ..."
for i in $(seq 1 30); do
    if docker exec "$CONTAINER_NAME" pg_isready -U "$DB_USER" -d postgres >/dev/null 2>&1; then
        echo "[OK] PostgreSQL 就绪：localhost:$HOST_PORT"
        break
    fi
    [ "$i" = 30 ] && { echo "[!] 启动超时" >&2; exit 1; }
    sleep 1
done

# 创建 finance_agent 数据库（幂等）
echo "[*] 确保 $DB_NAME 数据库存在 ..."
docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d postgres -tc \
    "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1 \
    && echo "[OK] $DB_NAME 已存在" \
    || docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d postgres -c \
        "CREATE DATABASE $DB_NAME" >/dev/null

# 初始化表结构
echo "[*] 初始化表结构（business + langgraph schema）..."
PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe scripts/init_db.py

echo ""
echo "✅ PostgreSQL 就绪：postgresql://$DB_USER:***@localhost:$HOST_PORT/$DB_NAME"
