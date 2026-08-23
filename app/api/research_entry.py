"""Research Agent 独立入口 —— 单 Agent 自适应 RAG 研究（SSE 流式进度）。

    POST /api/v1/research/analyze

直接调 app.rag.agent.arun_adaptive_research(query: str)（自适应最多 3 轮
Research Loop），经其 event_sink 逐节点推送进度，SSE 事件序列：
    agent_start → step×N（意图识别/研究规划/检索证据/生成报告/质量评估…）→ done/error

done.data 为 AgentState.current_report 映射的 research_result dict，与主链
research_node 的产出形态一致（ReportAssembler / report_node 兼容）。
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.models.user import User
from app.services.auth import get_current_user

router = APIRouter(prefix="/api/v1/research", tags=["research"])

_NODE_FRIENDLY = {
    "intent": "意图识别",
    "planning": "研究规划",
    "execute": "检索证据",
    "report": "生成报告",
    "evaluate": "质量评估",
    "replan": "补充研究",
    "review": "人工审核",
    "LangGraph": "研究流程",
}


class ResearchAnalyzeRequest(BaseModel):
    query: str = Field("", description="自然语言研究请求（留空则用 company 派生）")
    company: str = Field(..., min_length=1, description="目标公司，如 宁德时代")
    fast: bool = Field(True, description="快速模式=True 单轮(秒级) / False 深度(≤3轮,慢但更全面)")

    model_config = ConfigDict(extra="forbid")


def _safe_str(v) -> str:
    return "" if v is None else str(v)


def _map_current_report(company: str, report) -> dict:
    """AgentState.current_report(ResearchReport) → research_result dict。"""
    evidence = getattr(report, "evidence", None) or []
    sources: list[dict] = []
    for e in evidence:
        src = getattr(e, "source", None)
        if src:
            sources.append({"source": _safe_str(src), "page": getattr(e, "page", None)})

    # advantages/risks 是 ReportClaim 对象，须取 .claim 字符串（否则 JSON 序列化崩）
    advantages = [c.claim for c in (getattr(report, "advantages", None) or []) if getattr(c, "claim", "")]
    risks = [c.claim for c in (getattr(report, "risks", None) or []) if getattr(c, "claim", "")]

    return {
        "company": company,
        "summary": _safe_str(getattr(report, "summary", "")),
        "business_model": _safe_str(getattr(report, "plan_summary", "")),
        "industry_position": "",
        "competitive_advantages": advantages,
        "key_risks_business": risks,
        "sources": sources,
        "generated_at": _safe_str(getattr(report, "generated_at", "")),
    }


def _sse(event: dict) -> str:
    etype = event.get("type", "message")
    return f"event: {etype}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


# 进程级标志：BGE-M3(2.2GB) 只预热一次，之后 research 不再等待
_models_warmed = False


def _warm_embedding() -> None:
    """触发 BGE-M3 加载（进程单例缓存），供前端显示「模型准备就绪」。"""
    from app.rag.embedding import get_embedding_model

    get_embedding_model().embed(["模型预热"])


@router.post("/analyze")
async def research_analyze(
    req: ResearchAnalyzeRequest,
    _: Annotated[User, Depends(get_current_user)],
) -> StreamingResponse:
    """单跑 Research Agent（自适应 RAG），SSE 实时推送各节点进度，结束返回结论+证据链。"""
    from app.rag.agent import arun_adaptive_research

    query = req.query or f"分析{req.company}的基本面、行业地位与竞争优势与经营风险"

    async def _stream():
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        held: dict = {}

        async def sink(evt: dict) -> None:
            q.put_nowait(evt)

        async def run() -> None:
            try:
                state = await arun_adaptive_research(
                    query, event_sink=sink, max_iterations=1 if req.fast else 3
                )
                held["state"] = state
            except Exception as exc:  # noqa: BLE001 —— RAG 失败推 error 帧
                held["error"] = exc

        def drain() -> list[str]:
            frames: list[str] = []
            while not q.empty():
                evt = q.get_nowait()
                node = evt.get("node", "")
                frames.append(
                    _sse({
                        "type": "step",
                        "node": node,
                        "message": _NODE_FRIENDLY.get(node, evt.get("message", "")),
                    })
                )
            return frames

        yield _sse({"type": "agent_start", "agent": "research", "message": "Research 研究开始"})

        global _models_warmed  # noqa: PLW0603 —— 进程级预热标志
        if not _models_warmed:
            yield _sse({"type": "step", "node": "model", "message": "正在准备模型（首次加载 BGE-M3 + reranker，约 1–2 分钟）…"})
            try:
                await asyncio.to_thread(_warm_embedding)
                yield _sse({"type": "step", "node": "model", "message": "模型就绪，开始研究"})
            except Exception:  # noqa: BLE001 —— 预热失败不致命，研究中会惰性加载
                yield _sse({"type": "step", "node": "model", "message": "模型加载失败，将回退建设中加载"})
            finally:
                _models_warmed = True

        task = asyncio.create_task(run())
        last = time.time()
        while not task.done():
            frames = drain()
            if frames:
                last = time.time()
                for frame in frames:
                    yield frame
            elif time.time() - last > 10:
                # 模型加载/长节点期间无 step 事件 → 发保活帧防连接空闲断开
                yield ": keepalive\n\n"
                last = time.time()
            await asyncio.sleep(0.08)
        for frame in drain():
            yield frame

        if "error" in held:
            yield _sse({"type": "error", "message": str(held["error"])})
        else:
            yield _sse({
                "type": "done",
                "data": _map_current_report(req.company, held["state"].current_report),
            })

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
