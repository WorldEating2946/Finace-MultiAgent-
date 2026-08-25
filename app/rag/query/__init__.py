"""Query Rewrite 模块（PR #31）。

    from app.rag.query import get_query_rewriter, QueryRewriter, RuleBasedQueryRewriter, LLMQueryRewriter

模式（settings.rag_query_rewriter）：
    - rule：规则同义词扩展（默认，零成本，可离线测试）
    - llm ：LLM 生成检索变体（DeepSeek，需 DEEPSEEK_API_KEY）
    - off ：直通（不改写）
"""

from __future__ import annotations

from app.core.config import settings
from app.rag.query.llm_rewriter import LLMQueryRewriter
from app.rag.query.rewriter import QueryRewriter, RuleBasedQueryRewriter

__all__ = ["QueryRewriter", "RuleBasedQueryRewriter", "LLMQueryRewriter", "get_query_rewriter"]


def get_query_rewriter() -> QueryRewriter:
    """按 settings.rag_query_rewriter 返回改写器（retriever 默认调用）。"""
    mode = (settings.rag_query_rewriter or "rule").lower()
    if mode == "llm":
        return LLMQueryRewriter()
    if mode == "off":
        # 直通改写器：任何查询都原样返回
        return RuleBasedQueryRewriter(synonym_map={})
    return RuleBasedQueryRewriter()
