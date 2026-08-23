"""
app/models/agent_run.py — Agent 运行历史（结果持久化，供综合报告复用）

记录每个 Agent（research/financial/sentiment/risk/report）的一次运行结果，
综合报告生成时可按 company + agent_type 取最新历史复用，无需重跑全部 Agent。
纯新增表（agent_runs），不动既有 research_tasks 等。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    ticker: Mapped[str | None] = mapped_column(String(16), nullable=True)
    agent_type: Mapped[str] = mapped_column(String(32), index=True, nullable=False)  # research/financial/sentiment/risk/report
    result: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
