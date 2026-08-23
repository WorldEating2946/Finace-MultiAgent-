"""多源知识融合（PR #35）—— Multi-source Knowledge Fusion。

把 Enterprise Profile（单源年报）升级为 Enterprise Knowledge Base（多源）：
    年报（公司自述） + 研报（市场观点） + 政策（外部环境） + 新闻（近期变化）
不同来源各自独立构建画像，再跨源融合 + 冲突检测。

    from app.rag.source import SourceFusion, ConflictDetector, SourceType, \
        EnterpriseKnowledgePackage, SourceConflict
"""

from app.rag.source.conflict import ConflictDetector
from app.rag.source.fusion import SourceFusion
from app.rag.source.schema import (
    EnterpriseKnowledgePackage,
    SourceConflict,
    SourceType,
)

__all__ = [
    "SourceType",
    "SourceConflict",
    "EnterpriseKnowledgePackage",
    "SourceFusion",
    "ConflictDetector",
]
