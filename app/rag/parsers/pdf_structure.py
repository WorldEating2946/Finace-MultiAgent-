"""PDF 章节结构识别（heading detection）。

将 PDF 页文本中的标题行识别为层级结构（第X章 / X.Y / X.Y.Z），
供 hierarchical_splitter 做 Structure-aware 切分。

v1 启发式（中文年报/研报常见约定）：
    - ``第X章``（或 部分/篇）       → 层级 1
    - ``X.Y``（如 4.1）            → 层级 2
    - ``X.Y.Z``（如 4.1.1）        → 层级 3
    - 过滤：标题须为短行（≤40 字）且不以句末标点结尾，降低正文误判。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.document import Document

# 章节编号启发式（按优先级从高到低判断层级）
# `(?!\s*[\d.,亿元万%％])` 负向断言：X.Y 后不能紧跟（允许空格）数字/小数点/财务单位。
# OCR 常在数字内插入空格（"1.237.567.000元" → "1.2 37.567.000元"），
# 若只看紧邻字符会漏判；故 \s* 越过空格后再断言。
# 原则：宁可漏掉标题，也不把财务数据当章节标题（PR #30）。
_NO_FINANCIAL = r"(?!\s*[\d.,亿元万%％])"
_LEVEL3 = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{1,2}" + _NO_FINANCIAL)
_LEVEL2 = re.compile(r"^\d{1,2}\.\d{1,2}" + _NO_FINANCIAL)
_LEVEL1 = re.compile(r"^第[一二三四五六七八九十百千零两0-9]+[章节部分篇]")

# 标题行长度的启发式上限（避免把正文段落误判为标题）
_MAX_HEADING_LEN = 40

# 句末标点：标题通常不以这些结尾
_SENTENCE_ENDINGS = ("。", "！", "？", "；", "，")


@dataclass(frozen=True)
class Heading:
    """识别出的章节标题。"""

    title: str  # 标题文本
    level: int  # 1=章 / 2=节 / 3=小节
    page: int   # 所在页码


def heading_level(line: str) -> int | None:
    """判断一行文本是否为标题，返回层级（1/2/3）；非标题返回 None。

    Args:
        line: 单行文本。

    Returns:
        层级 1/2/3；非标题返回 None。
    """
    line = line.strip()
    if not line or len(line) > _MAX_HEADING_LEN:
        return None
    if line.endswith(_SENTENCE_ENDINGS):
        return None
    # 纯数字/百分比/标点行（如 "22.3%"、"76.8%"）不是标题
    if not re.search(r"[一-鿿A-Za-z]", line):
        return None
    if _LEVEL3.match(line):
        return 3
    if _LEVEL2.match(line):
        return 2
    if _LEVEL1.match(line):
        return 1
    return None


def detect_headings(
    docs: list[Document],
    toc_titles: list[str] | None = None,
) -> list[Heading]:
    """在 PDF 页 Documents 的文本行中识别标题层级。

    Args:
        docs: 按页码顺序的 PDF Document 列表。
        toc_titles: 可选，OCR 容错目录标题列表。OCR 后章节标题可能
            无 "第X章" 编号，提供目录标题后，正文中模糊匹配到该标题
            的行视为 level-1 章节（如 "公司简介 / 董事会报告"）。

    Returns:
        Heading 列表（按出现顺序；相邻重复标题去重，过滤页眉）。
    """
    headings: list[Heading] = []
    seen_toc_titles: set[str] = set()  # OCR 目录标题：每标题只出一次（防运行页眉重复）
    for doc in docs:
        page = doc.metadata.page or 1
        for line in doc.text.splitlines():
            level = heading_level(line)
            is_toc = False
            if level is None and toc_titles and _matches_toc_title(line, toc_titles):
                level = 1  # OCR 目录标题 → 章
                is_toc = True
            if level is None:
                continue
            title = line.strip()
            if is_toc:
                norm = re.sub(r"[\s　]", "", title)
                if norm in seen_toc_titles:
                    continue  # 该标题已作为章节出现过（运行页眉等重复）
                seen_toc_titles.add(norm)
            # 相邻重复标题（如页眉）去重
            if headings and headings[-1].title == title:
                continue
            headings.append(Heading(title=title, level=level, page=page))
    return headings


def _matches_toc_title(line: str, toc_titles: list[str]) -> bool:
    """匹配目录标题：归一化（去空格）后整行等于某标题。

    仅当该行本身就是章节标题（短行、与目录标题一致）才判为章，
    避免正文句子包含标题词导致误判。容忍 OCR 空格噪声。
    """
    norm = re.sub(r"[\s　]", "", line)
    if not norm or len(norm) > _MAX_HEADING_LEN:
        return False
    for title in toc_titles:
        t = re.sub(r"[\s　]", "", title)
        if t and t == norm:
            return True
    return False
