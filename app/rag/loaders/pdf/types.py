"""PDF 解析器通用类型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    """单页抽取文本（page 为 1-based 页码）。"""

    page: int
    text: str
