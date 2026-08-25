"""
app/services/agent_history.py — Agent 运行历史存取（供综合报告复用）

AgentRun 记录一次运行结果；综合报告生成时用 latest_run 按 company+agent_type
取最新历史，无需重跑全部 Agent。
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_run import AgentRun


async def save_run(
    db: AsyncSession, company: str, agent_type: str, result: dict, ticker: str | None = None
) -> AgentRun:
    """保存一次 Agent 运行结果。"""
    run = AgentRun(company=company, ticker=ticker, agent_type=agent_type, result=result)
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return run


async def latest_run(db: AsyncSession, company: str, agent_type: str) -> AgentRun | None:
    """该公司该 Agent 的最新一次运行。"""
    stmt = (
        select(AgentRun)
        .where(AgentRun.company == company, AgentRun.agent_type == agent_type)
        .order_by(AgentRun.created_at.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_runs(
    db: AsyncSession, company: str | None = None, agent_type: str | None = None, limit: int = 50
) -> list[AgentRun]:
    """按 company/agent_type 过滤列出历史（默认最新 50 条）。"""
    stmt = select(AgentRun)
    if company:
        stmt = stmt.where(AgentRun.company == company)
    if agent_type:
        stmt = stmt.where(AgentRun.agent_type == agent_type)
    stmt = stmt.order_by(AgentRun.created_at.desc()).limit(limit)
    return list((await db.execute(stmt)).scalars().all())
