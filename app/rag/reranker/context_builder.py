"""Rerank Context Builder（PR #33）。

将 chunk 的 metadata（公司/章节/小节/页码）格式化为结构化上下文，
供 CrossEncoder 感知"这个 chunk 在企业文档中的位置"——
从 Text Rerank 升级为 Document Intelligence Rerank。

格式：
    [Company] 小米集团
    [Chapter] 管理层讨论及分析
    [Section] 5.1 气候战略与治理
    [Page] 68
    [Content] 正文...

空字段跳过（不生成空标签），避免噪声干扰模型。
"""

from __future__ import annotations

from app.rag.document import DocumentChunk


class RerankContextBuilder:
    """将 chunk metadata + text 格式化为排序模型的结构化上下文。"""

    def build(self, chunk: DocumentChunk) -> str:
        """构造上下文文本。

        Args:
            chunk: 候选 chunk（含 text + metadata）。

        Returns:
            "[Company]...\\n[Chapter]...\\n[Section]...\\n[Page]...\\n[Content]..."
            空字段的标签行省略。
        """
        meta = chunk.metadata or {}
        parts: list[str] = []

        company = meta.get("company") or ""
        if company:
            parts.append(f"[Company] {company}")

        chapter = meta.get("chapter") or ""
        if chapter:
            parts.append(f"[Chapter] {chapter}")

        section = meta.get("section") or ""
        if section:
            parts.append(f"[Section] {section}")

        if chunk.page is not None:
            parts.append(f"[Page] {chunk.page}")

        content = (chunk.text or "").strip()
        if content:
            parts.append(f"[Content] {content}")

        return "\n".join(parts)
