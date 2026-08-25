"""Runtime 事件模型（PR41.3）。

SSE 事件流的统一载体：
    NodeEvent —— 单个节点事件（node_start / node_end / progress / error / done）
    EventBus  —— 进程内 pub/sub（research_id → subscriber queues），SSE 端点订阅。

事件格式（SSE data 字段）：
    {"research_id": "...", "type": "node_start", "node": "execute",
     "message": "开始节点：execute", "timestamp": "..."}
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any


class NodeEvent(dict):
    """研究节点事件（dict 便于 JSON 序列化）。"""

    @staticmethod
    def make(
        research_id: str,
        event_type: str,
        *,
        node: str = "",
        message: str = "",
        data: dict | None = None,
    ) -> "NodeEvent":
        return NodeEvent({
            "research_id": research_id,
            "type": event_type,          # node_start | node_end | progress | error | done
            "node": node,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {},
        })

    def sse(self) -> str:
        """序列化为 SSE 帧：data: {json}\n\n。"""
        return f"data: {json.dumps(self, ensure_ascii=False)}\n\n"


class EventBus:
    """进程内事件总线：research_id → set[asyncio.Queue]。

    生产 Runtime 的 SSE 数据源。TaskManager 在节点流转时 publish；
    SSE 端点 subscribe 后持续 yield，done/error 事件后断开。
    """

    def __init__(self) -> None:
        self._subs: dict[str, set[asyncio.Queue]] = {}

    async def subscribe(self, research_id: str) -> asyncio.Queue:
        """订阅某任务的事件流，返回专属 queue。"""
        queue: asyncio.Queue = asyncio.Queue()
        self._subs.setdefault(research_id, set()).add(queue)
        return queue

    async def unsubscribe(self, research_id: str, queue: asyncio.Queue) -> None:
        """退订（SSE 连接断开时清理）。"""
        subs = self._subs.get(research_id)
        if subs is None:
            return
        subs.discard(queue)
        if not subs:
            self._subs.pop(research_id, None)

    async def publish(self, event: NodeEvent) -> None:
        """广播事件到该 research_id 的所有订阅者。"""
        research_id = event.get("research_id", "")
        subs = list(self._subs.get(research_id, ()))
        for queue in subs:
            await queue.put(event)

    def has_subscribers(self, research_id: str) -> bool:
        """是否有人正在监听（供 worker 决定是否推送事件）。"""
        subs = self._subs.get(research_id)
        return bool(subs)
