-- ==================================================
-- FinaceAgent PostgreSQL 初始化脚本
-- （postgres 容器首次启动、数据卷为空时由 docker-entrypoint 自动执行）
--
-- 职责：
--   1. 创建 langgraph schema —— 供 LangGraph AsyncPostgresSaver
--      checkpoint 表隔离使用（init_db.py 以 search_path=langgraph 连接）。
--      必须在此建 schema：若缺省，checkpoint 表会落入 public，
--      破坏 LangGraph 状态与业务表隔离。
--
-- 说明：
--   - 业务表（research_tasks / research_reports）由 scripts/init_db.py
--     幂等创建（IF NOT EXISTS），不在此 SQL 重复，避免 DDL 与 Python 代码脱节；
--   - langgraph.checkpoints 系列表由 AsyncPostgresSaver.setup() 创建；
--   - 本脚本仅在数据卷首次初始化时执行一次（后续重启不会重跑）。
-- ==================================================

CREATE SCHEMA IF NOT EXISTS langgraph;
