"""
投研分析 SSE 全链路流 —— 多 Agent 主图执行进度实时可视化。

    POST /api/v1/analyze/stream

基于 graph.astream(stream_mode="updates")，请求内直接 yield SSE 事件，
无需 EventBus。事件序列：

    run_start                    — 任务开始（company/ticker）
    node_end  × N                — 每个节点执行完成（manager/research/financial/sentiment/risk/report）
    report_generated             — 结构化研报落盘完成（report_id/html_path/markdown_path）
    done / error                 — 结束或失败

前端用 EventSource 或 fetch + ReadableStream 消费，addEventListener 分类处理。
"""

from __future__ import annotations

import json

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.core.logging_config import get_logger
from app.models.user import User
from app.report import ReportAssembler, export_report
from app.services.auth import get_current_user
from app.workflow.graph import build_graph

router = APIRouter(prefix="/api/v1/analyze", tags=["analyze"])

logger = get_logger(__name__)

# 模块级图单例（编译一次复用；不与 main.py 的 get_graph 冲突，避免循环导入）
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        logger.info("[SSE] 编译 LangGraph 工作流...")
        _graph = build_graph()
    return _graph


class AnalyzeStreamRequest(BaseModel):
    company: str = Field(..., min_length=1, description="目标公司名称，如 '宁德时代'")
    ticker: str = Field(default="", description="股票代码（可选），如 '300750'")
    user_query: str = Field(default="", description="用户提问（可选）")

    model_config = ConfigDict(extra="forbid")


def _sse(event: dict) -> str:
    """dict → SSE 帧（带 event 类型，前端可用 addEventListener 分类监听）。"""
    etype = event.get("type", "message")
    payload = json.dumps(event, ensure_ascii=False)
    return f"event: {etype}\ndata: {payload}\n\n"


@router.post("/stream")
async def analyze_stream(
    req: AnalyzeStreamRequest,
    _: Annotated[User, Depends(get_current_user)],
) -> StreamingResponse:
    """执行完整多 Agent 分析链路，SSE 实时推送节点进度与最终研报。

    鉴权依赖在进入 SSE 生成器前完成校验；流内不再做二次鉴权。
    """

    async def _stream():
        graph = _get_graph()
        initial_state = {
            "company": req.company,
            "ticker": req.ticker,
            "user_query": req.user_query or f"分析{req.company}的财务健康状况与发展前景",
            "current_step": "start",
            "errors": [],
        }
        merged: dict = dict(initial_state)

        yield _sse({"type": "run_start", "company": req.company, "ticker": req.ticker})

        try:
            async for chunk in graph.astream(initial_state, stream_mode="updates"):
                for node, update in (chunk or {}).items():
                    logger.info("[SSE] 节点完成: %s", node)
                    yield _sse({
                        "type": "node_end",
                        "node": node,
                        "status": "ok",
                        "summary": f"{node} 节点执行完成",
                    })
                    if isinstance(update, dict):
                        merged.update(update)

            # 全链路结束 → 用四 Agent 输出组装结构化研报并落盘
            content = ReportAssembler().assemble(
                company=req.company,
                ticker=req.ticker,
                user_query=req.user_query,
                research=merged.get("research_result"),
                financial=merged.get("financial_result"),
                sentiment=merged.get("sentiment_result"),
                risk=merged.get("risk_result"),
            )
            out = export_report(content)
            yield _sse({
                "type": "report_generated",
                "report_id": out.report_id,
                "html_path": out.html_path,
                "markdown_path": out.markdown_path,
            })
            yield _sse({
                "type": "done",
                "report_id": out.report_id,
                "markdown": out.markdown,
                "html_path": out.html_path,
                "markdown_path": out.markdown_path,
            })
        except Exception as exc:  # SSE 流内兜底，推 error 事件而非中断
            logger.exception("[SSE] 分析链路异常")
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
