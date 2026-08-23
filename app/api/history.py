"""Agent 运行历史 API —— 保存 / 列出，供综合报告复用。

    POST /api/v1/history     保存一次 Agent 运行结果
    GET  /api/v1/history     列出历史（company/agent_type 过滤）

前端在每个 Agent 运行成功后调用 POST 保存；综合报告生成可读最新历史复用。
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, ConfigDict, Field

from app.core.response import ok
from app.database.session import get_db
from app.models.user import User
from app.services import agent_history
from app.services.auth import get_current_user

router = APIRouter(prefix="/api/v1/history", tags=["history"])

AgentType = Literal["research", "financial", "sentiment", "risk", "report"]


class SaveRunRequest(BaseModel):
    company: str = Field(..., min_length=1, description="公司名")
    agent_type: AgentType = Field(..., description="Agent 类型")
    result: dict = Field(..., description="该 Agent 的结构化结果")
    ticker: str | None = Field(default=None, description="股票代码（可选）")

    model_config = ConfigDict(extra="forbid")


@router.post("")
async def save_history(
    req: SaveRunRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
) -> dict:
    """保存一次 Agent 运行结果。"""
    run = await agent_history.save_run(
        db, company=req.company, agent_type=req.agent_type, result=req.result, ticker=req.ticker
    )
    return ok({"id": run.id, "agent_type": run.agent_type, "company": run.company})


@router.get("")
async def list_history(
    db: Annotated[AsyncSession, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
    company: str | None = Query(None, description="按公司过滤"),
    agent_type: str | None = Query(None, description="按 Agent 过滤"),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    """列出历史运行（最新在前）。"""
    runs = await agent_history.list_runs(
        db, company=company, agent_type=agent_type, limit=limit
    )
    return ok({
        "runs": [
            {
                "id": r.id,
                "company": r.company,
                "ticker": r.ticker,
                "agent_type": r.agent_type,
                "created_at": r.created_at.isoformat(),
                "result": r.result,
            }
            for r in runs
        ]
    })
