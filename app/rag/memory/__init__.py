"""Research Memory + Durable Checkpoint（PR39）。

企业级 Research Agent 的生命力：任务可暂停、可恢复、可跨进程重启。

    from app.rag.memory import ResearchCheckpointer, ResumeAction

    cp = ResearchCheckpointer(backend="sqlite")
    state = cp.run("分析小米汽车未来竞争力", thread_id="xiaomi_auto_001")
    state = cp.resume("xiaomi_auto_001")                  # 幂等恢复
    state = cp.resume("xiaomi_auto_001",
                      action=ResumeAction(decision="approve"))  # 人工审核续传
    record = cp.record("xiaomi_auto_001")                 # 任务压缩记录
"""

from __future__ import annotations

from app.rag.memory.checkpoint import ResearchCheckpointer
from app.rag.memory.schema import RecordStatus, ResearchRecord, ResumeAction
from app.rag.memory.serializer import to_record
from app.rag.memory.store import get_checkpointer

__all__ = [
    "ResearchCheckpointer",
    "ResumeAction",
    "ResearchRecord",
    "RecordStatus",
    "to_record",
    "get_checkpointer",
]
