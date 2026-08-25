"""研究任务 SSE 事件流（PR41.3）。

    GET /api/v1/research/{id}/stream

实时推送任务节点事件（node_start / node_end / progress / done / error），
用户无需轮询即可观察 Agent 研究进度：

    data: {"research_id":"...","type":"node_start","node":"execute","message":"开始节点：execute",...}

通过 ResearchService.event_bus（进程内 pub/sub）订阅 TaskManager worker 发布的事件。
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.research import get_research_service
from app.core.exceptions import ResearchNotFound
from app.services.research_service import ResearchService

router = APIRouter(prefix="/research", tags=["research"])


@router.get("/{research_id}/stream")
async def stream_research(
    research_id: str,
    service: ResearchService = Depends(get_research_service),
) -> StreamingResponse:
    """SSE 实时流：订阅任务事件，持续推送直到 done / error / 连接断开。

    PR42a 竞态修复：subscribe **之后**再二次确认终态，并 keep-alive 时复查 ——
    治「订阅瞬间 worker 恰好跑完、done 已发出」导致 30s keep-alive 死循环的窄竞态。
    """
    # 任务必须存在
    task = await service.aget_task(research_id)
    if task is None:
        raise ResearchNotFound(f"research task not found: {research_id}")

    bus = service.event_bus
    queue = await bus.subscribe(research_id)

    # 终态判定：completed / failed / cancelled / rejected（PR42a 新增 rejected）
    _TERMINAL = ("completed", "failed", "cancelled", "rejected")

    async def _terminal_status() -> str | None:
        """复查任务是否已终态；是 → 返回终态名，否则 None。"""
        t = await service.aget_task(research_id)
        if t is not None and t.status in _TERMINAL:
            return t.status
        return None

    async def _sse_stream():
        try:
            # subscribe 后二次确认（关闭订阅与发布之间的事件窗口）
            final = await _terminal_status()
            if final is not None:
                yield _sse({"research_id": research_id, "type": "done",
                            "message": f"任务已{final}"})
                return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # 心跳保活（SSE 代理超时防护）+ 复查终态（治窄竞态）
                    final = await _terminal_status()
                    if final is not None:
                        yield _sse({"research_id": research_id, "type": "done",
                                    "message": f"任务已{final}"})
                        return
                    yield ": keep-alive\n\n"
                    continue
                yield _sse(event)
                if event.get("type") in ("done", "error"):
                    break
        finally:
            await bus.unsubscribe(research_id, queue)

    return StreamingResponse(
        _sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _sse(event: dict) -> str:
    """dict → SSE 帧（带 event 类型，前端可用 addEventListener 分类监听）。

    格式：
        event: <type>
        data: {json}

    """
    etype = event.get("type", "message")
    payload = json.dumps(event, ensure_ascii=False)
    return f"event: {etype}\ndata: {payload}\n\n"
