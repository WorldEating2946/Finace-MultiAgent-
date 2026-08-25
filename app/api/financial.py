"""Financial Agent 独立入口 —— 单 Agent 财务分析，不依赖整条主链。

    POST /api/v1/financial

内部复用 app.agents.financial_agent.node.financial_analysis_node（完整 fetch →
计算 → 杜邦 → 点评链路），以最小 state 调用，返回 financial_result dict。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.core.response import ok
from app.models.user import User
from app.services.auth import get_current_user

router = APIRouter(prefix="/api/v1/financial", tags=["financial"])


class FinancialRequest(BaseModel):
    ticker: str = Field(..., min_length=1, description="股票代码，如 300750")
    company: str = Field(default="", description="企业名称（可选）")

    model_config = ConfigDict(extra="forbid")


@router.post("")
async def financial_analyze(
    req: FinancialRequest,
    _: Annotated[User, Depends(get_current_user)],
) -> dict:
    """单跑 Financial Agent，返回财务指标 + 杜邦 + 点评。"""
    # 懒加载避免潜在循环导入（node → app.workflow.state 链路）
    from app.agents.financial_agent.node import financial_analysis_node

    state = {"company": req.company or req.ticker, "ticker": req.ticker}
    result = await financial_analysis_node(state)
    return ok(result.get("financial_result", {}))
