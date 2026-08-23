"""Research Agent 接口（PR40 sync → PR41 async）。

    POST /api/v1/research/start         创建研究任务（异步，立即返回 queued）
    GET  /api/v1/research/{id}          查询任务状态
    GET  /api/v1/research/{id}/report   获取研究报告
    POST /api/v1/research/{id}/resume   恢复任务（Human-in-the-loop）
    POST /api/v1/research/{id}/cancel   取消任务（PR41）

依赖 get_research_service() —— API 不直接耦合 LangGraph/Agent（经 Service 层）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.exceptions import (
    InvalidDecision,
    ResearchNotFound,
    ResearchReportNotReady,
)
from app.core.response import ok
from app.rag.agent.human import validate_decision
from app.services.research_service import ResearchService

router = APIRouter(prefix="/research", tags=["research"])


# ── 请求体 ────────────────────────────────────────────────────

class StartResearchRequest(BaseModel):
    """创建研究任务请求。"""

    query: str
    company: str = ""
    human_review: bool = False


class ResumeResearchRequest(BaseModel):
    """人工审核决策请求（Human-in-the-loop）。"""

    action: str = "approve"     # approve | reject | modify
    feedback: str = ""


# ── 依赖 ──────────────────────────────────────────────────────

_default_service: ResearchService | None = None


def get_research_service() -> ResearchService:
    """依赖：ResearchService（进程内单例，持有 checkpointer + TaskManager）。

    测试通过 app.dependency_overrides[get_research_service] 注入 mock service。
    """
    global _default_service
    if _default_service is None:
        _default_service = ResearchService()
    return _default_service


def _require_task(service: ResearchService, research_id: str):
    """查任务；不存在抛 ResearchNotFound。"""
    task = service.get_task(research_id)
    if task is None:
        raise ResearchNotFound(f"research task not found: {research_id}")
    return task


async def _require_task_async(service: ResearchService, research_id: str):
    """查任务（async）；不存在抛 ResearchNotFound。"""
    task = await service.aget_task(research_id)
    if task is None:
        raise ResearchNotFound(f"research task not found: {research_id}")
    return task


# ── 端点 ──────────────────────────────────────────────────────

@router.post("/start")
async def start_research(
    payload: StartResearchRequest,
    service: ResearchService = Depends(get_research_service),
) -> dict:
    """创建研究任务（异步：立即返回 queued，后台 worker 执行）。"""
    task = await service.acreate_task(
        payload.query,
        company=payload.company,
        human_review=payload.human_review,
    )
    return ok({
        "research_id": task.research_id,
        "thread_id": task.thread_id,
        "status": task.status,
    })


@router.get("/{research_id}")
async def get_research_status(
    research_id: str,
    service: ResearchService = Depends(get_research_service),
) -> dict:
    """查询任务状态。"""
    task = await _require_task_async(service, research_id)
    return ok({
        "research_id": task.research_id,
        "status": task.status,
        "current_step": task.current_step,
        "iteration": task.iteration,
        "missing_dimensions": task.missing_dimensions,
    })


@router.get("/{research_id}/report")
async def get_research_report(
    research_id: str,
    service: ResearchService = Depends(get_research_service),
) -> dict:
    """获取研究报告；未生成 → ResearchReportNotReady。"""
    await _require_task_async(service, research_id)
    report = await service.aget_report(research_id)
    if report is None:
        raise ResearchReportNotReady(f"report not ready for: {research_id}")
    return ok(report)


@router.post("/{research_id}/resume")
async def resume_research(
    research_id: str,
    payload: ResumeResearchRequest,
    service: ResearchService = Depends(get_research_service),
) -> dict:
    """人工审核后恢复任务（approve 继续补步 / reject 停止）。"""
    decision = validate_decision(payload.model_dump())
    if decision is None:
        raise InvalidDecision(f"invalid action: {payload.action!r}")
    task = await service.aresume(research_id, action=decision.model_dump())
    return ok({
        "research_id": task.research_id,
        "status": task.status,
        "current_step": task.current_step,
        "iteration": task.iteration,
    })


@router.post("/{research_id}/cancel")
async def cancel_research(
    research_id: str,
    service: ResearchService = Depends(get_research_service),
) -> dict:
    """取消排队/运行中任务（PR41）。"""
    task = await service.acancel(research_id)
    return ok({
        "research_id": task.research_id,
        "status": task.status,
    })
