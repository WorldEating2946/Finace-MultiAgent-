"""
统一研报生成 API —— 四 Agent 输出 dict → 六章研报（Markdown + 自包含 HTML）落盘。

    POST /api/v1/report/generate

输入 ReportGenerateRequest（company/ticker/user_query + research/financial/
sentiment/risk 各 dict），由 ReportAssembler 确定性组装为六章研报，
export_report 落盘到 data/reports/{report_id}/，返回 ReportOutput。
"""

from __future__ import annotations

from typing import Annotated

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_config import get_logger
from app.core.response import ok
from app.database.session import get_db
from app.models.user import User
from app.report import ReportAssembler, export_report
from app.report.schemas import ReportGenerateRequest
from app.services import agent_history
from app.services.auth import get_current_user

router = APIRouter(prefix="/api/v1/report", tags=["report"])

logger = get_logger(__name__)


@router.post("/generate")
async def generate_report(
    req: ReportGenerateRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
) -> dict:
    """四 Agent 输出 → 组装六章研报 → 导出落盘。返回 ReportOutput。

    任一 Agent 输出未提供 → 自动复用该公司该 Agent 的**最新历史运行**，无需重跑。
    """
    logger.info("收到研报生成请求: company=%s, ticker=%s", req.company, req.ticker)

    async def _resolve(agent_type: str, provided: dict | None) -> dict | None:
        if provided is not None:
            return provided
        run = await agent_history.latest_run(db, req.company, agent_type)
        if run:
            logger.info("研报复用历史 %s (%s, id=%s)", agent_type, req.company, run.id)
            return run.result
        return None

    research = await _resolve("research", req.research)
    financial = await _resolve("financial", req.financial)
    sentiment = await _resolve("sentiment", req.sentiment)
    risk = await _resolve("risk", req.risk)

    try:
        content = ReportAssembler().assemble(
            company=req.company,
            ticker=req.ticker,
            user_query=req.user_query,
            research=research,
            financial=financial,
            sentiment=sentiment,
            risk=risk,
        )
        output = export_report(content)
    except Exception as exc:
        logger.exception("研报生成失败")
        raise HTTPException(status_code=500, detail=f"研报生成失败: {exc}") from exc
    return ok(output.model_dump())
