"""层级化切分：按章节边界切块，chunk 携带 section 路径。

将 PDF 页 Documents 按章节结构（heading detection）跨页切分：
    - 每个 chunk 的 metadata["section"] 为章节路径（如 "第四章 业务回顾 > 4.1 智能手机业务"）；
    - metadata["page"] 为 chunk 起始页码；
    - source_name 附加 section 路径，检索结果可溯源。

与 parsers/pdf_structure 分工：
    parser   → 识别标题层级（Heading 列表）
    splitter → 按标题边界 + 尺寸切块

限制（v1）：运行页眉（每页重复的章节名）可能使章节路径退化为章级，后续可优化。
"""

from __future__ import annotations

from hashlib import md5

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.document import Document, DocumentChunk
from app.rag.parsers.pdf_structure import Heading, detect_headings
from app.rag.parsers.toc_parser import TocResult

# 中文优先的分隔符（与 app/rag/splitter.py 保持一致）
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

# 缓冲阈值：超过 4×chunk_size 先切一刀，避免超长节撑爆内存
_BUFFER_FACTOR = 4


def _make_recursive_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
    )


def _section_path(stack: list[Heading]) -> str:
    """章节栈 → 完整路径（如 "第四章 业务回顾 > 4.1 智能手机业务"）。"""
    return " > ".join(h.title for h in stack)


def _make_chunk(
    doc: Document,
    text: str,
    section: str,
    chapter: str,
    page: int,
    index: int,
) -> DocumentChunk:
    """构建带 section 路径 / 章标题 / 页码的 DocumentChunk。"""
    source = doc.metadata.source
    base = doc.metadata.source_name
    source_name = f"{base} > {section}" if section else base
    return DocumentChunk(
        chunk_id=md5(f"{source}_{index}_{text[:50]}".encode()).hexdigest(),
        company=doc.metadata.company,
        doc_type="pdf",
        source=source,
        source_name=source_name,
        page=page,
        text=text,
        metadata={
            **doc.metadata.model_dump(),
            "section": section,
            "chapter": chapter,
            "page": page,
            "chunk_index": index,
        },
    )


def _enrich_with_toc(chunk: DocumentChunk, toc: TocResult | None) -> None:
    """用 TOC 映射补全 chunk 的 chapter / page_range / section_level。

    TOC 提供 canonical 章节名与 page range，比正则标题识别更可靠；
    未命中 TOC 时保留正则识别结果。
    """
    if toc is None or chunk.page is None:
        return
    hit = toc.ranges.lookup(chunk.page)
    if hit is None:
        return
    title, start, end = hit
    chunk.metadata["chapter"] = title
    chunk.metadata["page_range"] = f"{start}-{end}"
    chunk.metadata["section_level"] = 1


def split_pdf_hierarchical(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    toc: TocResult | None = None,
) -> list[DocumentChunk]:
    """按章节边界切分 PDF 页 Documents（单文件）。

    Args:
        docs:         按页码顺序的 PDF Document 列表。
        chunk_size:   目标 chunk 大小（字符数）。
        chunk_overlap: 相邻 chunk 重叠字符数。
        toc:          目录解析结果（可选）。提供时跳过目录页，并用
                      TOC 章节映射补全 chunk 的 chapter / page_range。

    Returns:
        DocumentChunk 列表；每个 chunk 带 section 路径与页码。
        若无识别到标题，退化为跨页纯文本尺寸切分。
    """
    # 目录页是导航而非内容，切分时跳过
    if toc is not None and toc.excluded_pages:
        docs = [d for d in docs if (d.metadata.page or 1) not in toc.excluded_pages]

    # OCR 容错：TOC 无页码时用目录标题列表驱动章节识别（level-1）
    headings = detect_headings(docs, toc.titles if toc else None)
    rs = _make_recursive_splitter(chunk_size, chunk_overlap)

    stack: list[Heading] = []
    hi = 0  # headings 指针
    result: list[DocumentChunk] = []
    index = 0

    buf_text = ""
    buf_page: int | None = None
    buf_section = ""
    buf_chapter = ""
    buf_has_content = False  # 当前 buffer 是否含正文（非标题）

    def flush() -> None:
        nonlocal buf_text, buf_page, buf_chapter, buf_has_content, index
        if not buf_text.strip():
            buf_page = None
            return
        for part in rs.split_text(buf_text):
            if not part.strip():
                continue
            chunk = _make_chunk(docs[0], part, buf_section, buf_chapter, buf_page or 1, index)
            _enrich_with_toc(chunk, toc)
            result.append(chunk)
            index += 1
        buf_text = ""
        buf_page = None
        buf_chapter = ""
        buf_has_content = False

    for doc in docs:
        page = doc.metadata.page or 1
        for raw_line in doc.text.splitlines():
            line = raw_line.strip()

            # 当前行是否为下一个识别出的标题
            if hi < len(headings) and line == headings[hi].title:
                heading = headings[hi]
                hi += 1
                # 有正文才切分当前节；仅含标题的 buffer 续接标题（不单独成空块）
                if buf_has_content:
                    flush()
                # 章节栈回溯：弹出 >= 本层级的标题，再入栈
                while stack and stack[-1].level >= heading.level:
                    stack.pop()
                stack.append(heading)
                buf_section = _section_path(stack)
                # 章标题：栈顶的第一个 1 级标题
                buf_chapter = stack[0].title if stack and stack[0].level == 1 else ""
                # 标题续接：章标题 + 节标题链进同一节首块（避免只有标题的空块）
                buf_text = buf_text + heading.title + "\n"
                buf_has_content = False
                buf_page = page
            else:
                if buf_page is None:
                    buf_page = page
                buf_text += raw_line + "\n"
                buf_has_content = True
                # 超大缓冲先切一刀（同一 section 内继续累积）
                if len(buf_text) > chunk_size * _BUFFER_FACTOR:
                    flush()

    flush()
    return result
