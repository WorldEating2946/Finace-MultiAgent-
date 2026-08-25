"""PDF 目录（TOC）解析：章节名称 → page range 映射。

企业年报天然有目录页，解析后建立章节 page-range，比纯正则标题识别更可靠：
    chunk 直接继承 canonical 章节名 + page range（如 "管理层讨论及分析" [14-41]）。

目录行格式（点线引导）：标题 ....... 页码
    e.g.  管理层讨论及分析 ........................ 14

解析结果同时标记目录页（excluded_pages），切分时跳过（导航页不是内容）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.rag.document import Document
from app.rag.parsers.text_normalizer import normalize_text

# 目录条目：标题 + 点线引导 + 页码
_TOC_ENTRY = re.compile(r"^(?P<title>.+?)\s*\.{2,}\s*(?P<page>\d{1,3})$")

# 目录页判据：页面前几行含"目录"，且点线引导条目数
_MIN_TOC_ENTRIES = 3

# OCR 容错目录：短标题行的最大长度
_TITLE_LIKE_MAX = 16
# OCR 目录页判据：至少的标题行数（真实目录通常含 ≥6 个章标题）
_MIN_TOC_TITLES = 6
# 目录页判据：标题行占比（目录页几乎全是短标题行）
_TOC_DENSITY = 0.7
# 目录通常位于文档前部（封面之后）；只在此范围内搜索
_TOC_SEARCH_PAGES = 10
# 句末标点（标题通常不以这些结尾）
_SENTENCE_ENDINGS = ("。", "！", "？", "；", "，")

# 目录页自身标题（非章节名），提取时排除
_TOC_HEADER_WORDS = {"目录", "contents", "Contents", "CONTENTS", "目錄"}

# PDF outline 导航页标题（封面/目录/封底，非内容章节）；归一化（简体）+ 原始繁体双匹配
_NAV_TITLES = {
    "封面", "内封面", "目录", "封底", "目錄", "內封面",
    "contents", "Contents", "CONTENTS",
    "cover", "Cover", "back cover",
}


def _is_title_like(line: str) -> bool:
    """是否为"标题候选"行：短行、无句末标点、非纯数字、含中文/英文。"""
    if len(line) > _TITLE_LIKE_MAX:
        return False
    if line.endswith(_SENTENCE_ENDINGS):
        return False
    if re.fullmatch(r"[\d\s.,%()\-–—月日年]+", line):
        return False
    return re.search(r"[一-鿿A-Za-z]", line) is not None


def extract_toc_titles(
    docs: list[Document],
    min_titles: int = _MIN_TOC_TITLES,
) -> tuple[list[str], int]:
    """OCR 容错：从目录页提取章节标题列表（标题可能无页码/点线）。

    OCR 后目录常变成纯标题行（如 "公司简介 / 主要业务 / 主席报告..."），
    无 "标题.......页码" 点线格式。此处提取标题列表，供后续在正文中
    匹配出章节起点（level-1 heading）。

    目录页判据：≥min_titles 个标题候选行，且标题行占比 ≥60%
    （避免把封面等短行多的页面误判为目录）。

    Args:
        docs: PDF 页 Documents。
        min_titles: 目录页判据（至少的标题行数）。

    Returns:
        (章节标题列表, 目录页页码)；未找到返回 ([], 0)。
    """
    # 目录在文档前部；释义/索引等短行页在末尾，限制扫描范围
    # 选页策略：取前部「第一个」达标的目录页（而非标题行最多），
    # 避免公司资料/地址页（短行多但非目录，如 "香港/湾仔/皇后大道东183号"）
    # 被误判为目录。真实目录在封面之后、正文之前。
    front_pages = [d for d in docs if (d.metadata.page or 1) <= _TOC_SEARCH_PAGES]
    best_titles: list[str] = []
    best_page = 0
    for doc in front_pages:
        page = doc.metadata.page or 1
        lines = [l.strip() for l in doc.text.splitlines() if l.strip()]
        if not lines:
            continue
        titles = [
            l for l in lines if _is_title_like(l) and l not in _TOC_HEADER_WORDS
        ]
        if len(titles) >= min_titles and len(titles) / len(lines) >= _TOC_DENSITY:
            best_titles, best_page = titles, page
            break
    return best_titles, best_page


@dataclass(frozen=True)
class TocEntry:
    """目录条目（标题 + 起始页）。"""

    title: str
    page: int  # 1-based 起始页


class PageRangeMap:
    """章节 → (start, end) 有序映射，按页码查找。"""

    def __init__(self, entries: list[TocEntry], total_pages: int):
        # 按页码排序 + 去重（多个目录页时避免重复条目）
        ordered = sorted(set(entries), key=lambda e: e.page)
        self._ranges: list[tuple[str, int, int]] = []
        for i, e in enumerate(ordered):
            end = ordered[i + 1].page - 1 if i + 1 < len(ordered) else total_pages
            self._ranges.append((e.title, e.page, end))

    @property
    def items(self) -> list[tuple[str, int, int]]:
        """[(章节名, start, end), ...]，按起始页升序。"""
        return list(self._ranges)

    def lookup(self, page: int) -> tuple[str, int, int] | None:
        """返回 page 所属的 (章节名, start, end)；找不到返回 None。"""
        for title, start, end in self._ranges:
            if start <= page <= end:
                return title, start, end
        return None


@dataclass
class TocResult:
    """TOC 解析结果。

    - 标准路径：ranges（章节 → page-range）+ excluded_pages（目录页）；
    - OCR 容错路径：titles（目录标题列表，供 detect_headings 模糊匹配出章节）。
    """

    ranges: PageRangeMap
    excluded_pages: frozenset[int] = field(default_factory=frozenset)
    titles: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.excluded_pages or self.ranges.items or self.titles)


def _toc_entry(line: str) -> TocEntry | None:
    m = _TOC_ENTRY.match(line.strip())
    if m is None:
        return None
    return TocEntry(title=m.group("title").strip(), page=int(m.group("page")))


def _find_toc_pages(docs: list[Document]) -> list[int]:
    """识别目录页：含"目录"标题 + 至少 3 条点线引导条目。"""
    toc_pages: list[int] = []
    for doc in docs:
        lines = [l for l in doc.text.splitlines() if l.strip()]
        has_header = any("目录" in l for l in lines[:5])
        entries = sum(1 for l in lines if _toc_entry(l) is not None)
        if has_header and entries >= _MIN_TOC_ENTRIES:
            toc_pages.append(doc.metadata.page or 1)
    return toc_pages


def _total_pages(docs: list[Document]) -> int:
    return max((d.metadata.page or 1) for d in docs) if docs else 0


def _extract_outline_toc(docs: list[Document]) -> TocResult | None:
    """PDF outline（书签）目录：最可靠的 TOC 源。

    损坏文本层的 PDF（如小米年报）OCR 读不出图形分隔页的章节标题，
    但 PDF 内置书签含**干净章节名 + 真实页码**（元数据，独立于文本层）。
    标题经繁→简归一化后与评测期望对齐。

    Returns:
        TocResult（章节 page-range + 导航页 excluded）；无 outline / 失败返回 None。
    """
    if not docs:
        return None
    src = docs[0].metadata.source
    if not src or not Path(src).exists():
        return None
    try:
        import fitz  # 惰性导入（仅 PDF outline 路径需要）

        pdf = fitz.open(src)
        try:
            outline = pdf.get_toc(simple=True)
        finally:
            pdf.close()
    except Exception:  # noqa: BLE001 无 outline / 文件异常 → 回退其他 TOC 路径
        return None
    if not outline:
        return None

    entries: list[TocEntry] = []
    excluded: set[int] = set()
    for level, title, page in outline:
        if level != 1:
            continue  # 只要一级章；二级条目会切碎 PageRangeMap
        norm = re.sub(r"[\s　]", "", normalize_text(title.strip()))
        if not norm:
            continue
        if norm in _NAV_TITLES:
            excluded.add(page)
            continue
        entries.append(TocEntry(title=norm, page=page))
    if not entries:
        return None
    return TocResult(
        ranges=PageRangeMap(entries, total_pages=_total_pages(docs)),
        excluded_pages=frozenset(excluded),
    )


def parse_toc(docs: list[Document]) -> TocResult:
    """从 PDF 页 Documents 解析目录 → 章节 page-range 映射 / 标题列表。

    路径优先级（由最可靠到启发式）：
        1. PDF outline（书签）：干净章节名 + 真实页码；
        2. 标准路径：点线引导目录（"标题 .... 页码"）；
        3. OCR 容错路径：无页码目录 → 标题列表。
    均未找到时返回空结果。

    Args:
        docs: 按页码顺序的 PDF Document 列表（doc_type="pdf"）。

    Returns:
        TocResult：
            - outline/标准路径：ranges（章节映射）+ excluded_pages（导航页）；
            - OCR 容错路径（无页码目录）：titles（目录标题列表）。
    """
    total = _total_pages(docs)
    outline = _extract_outline_toc(docs)
    if outline is not None:
        return outline
    toc_pages = _find_toc_pages(docs)
    if toc_pages:
        entries: list[TocEntry] = []
        for doc in docs:
            if doc.metadata.page not in toc_pages:
                continue
            for line in doc.text.splitlines():
                entry = _toc_entry(line)
                if entry is not None:
                    entries.append(entry)
        return TocResult(
            ranges=PageRangeMap(entries, total_pages=total),
            excluded_pages=frozenset(toc_pages),
        )

    # OCR 容错路径：目录标题无页码/点线 → 提取标题列表（供章节识别）
    titles, toc_page = extract_toc_titles(docs)
    if titles:
        return TocResult(
            ranges=PageRangeMap([], total_pages=total),
            titles=titles,
            excluded_pages=frozenset({toc_page}) if toc_page else frozenset(),
        )

    return TocResult(ranges=PageRangeMap([], total_pages=total))
