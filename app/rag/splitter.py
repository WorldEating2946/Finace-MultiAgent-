"""文本切片策略（Phase 1：RecursiveCharacter + Markdown 标题感知）。

将 loader 输出的 Document（原始文档）按语义边界切分为细粒度 DocumentChunk。
根据 doc_type 选择切分策略：
    - markdown → 标题层级感知切分（source_name 附加标题路径）；
    - 其他     → 纯文本按 chunk_size 递归切分。

参数对齐 docs/RAG_ARCHITECTURE.md §5.1：
    chunk_size≈512，chunk_overlap≈100，separators 含中文标点。

对外接口（经 app.rag 包暴露）：
    from app.rag import split_documents
"""

from hashlib import md5

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.rag.document import Document, DocumentChunk
from app.rag.splitters.hierarchical_splitter import (
    _enrich_with_toc,
    split_pdf_hierarchical,
)

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 100

# 中文优先的分隔符顺序：段 > 行 > 句 > 短句 > 逗号 > 空格 > 字符
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

_MD_HEADERS = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
    ("####", "H4"),
]

# Markdown 标题切片器（模块级单例，无状态）
_md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=_MD_HEADERS,
    strip_headers=False,
)


def _make_recursive_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """构建递归字符切片器（按分隔符优先级递归切分）。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
    )


def split_documents(
    docs: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """将 loader 输出的 Document 切分为细粒度 DocumentChunk。

    Args:
        docs:          loader 输出的原始文档列表。
        chunk_size:    目标 chunk 大小（字符数），默认 512。
        chunk_overlap: 相邻 chunk 重叠字符数，默认 100。

    Returns:
        切分后的 DocumentChunk 列表。

    说明：
        - PDF：走 hierarchical_splitter（章节识别，chunk 带 section 路径 + 页码）；
        - markdown → 标题感知；其他 → 纯文本递归切分；
        - 从 DocumentMetadata 提取 company / source / source_name / doc_type / page；
        - markdown 章节路径写入 chunk.metadata["section"]；
        - chunk_id 为 MD5(source + 序号 + 内容前缀)，内容不变则 ID 稳定。
    """
    rs = _make_recursive_splitter(chunk_size, chunk_overlap)
    result: list[DocumentChunk] = []

    # 表格 Document：已是结构化文本（Markdown 表格），直接成单 chunk，不切分
    table_docs = [d for d in docs if d.metadata.content_type == "table"]

    # PDF 正文：跨页结构化切分（章节识别 + section 路径 + TOC 章节映射）
    pdf_docs = [
        d for d in docs
        if d.metadata.doc_type == "pdf" and d.metadata.content_type != "table"
    ]
    toc = None
    if pdf_docs:
        from app.rag.parsers.toc_parser import parse_toc

        toc = parse_toc(pdf_docs)
        result.extend(split_pdf_hierarchical(pdf_docs, chunk_size, chunk_overlap, toc=toc))

    # 表格 chunk（补 TOC 章节）
    for doc in table_docs:
        chunk = _build_table_chunk(doc, len(result))
        if toc is not None:
            _enrich_with_toc(chunk, toc)
        result.append(chunk)

    # 非 PDF：逐 Document 按 doc_type 分发
    for doc in docs:
        if doc.metadata.doc_type == "pdf" or doc.metadata.content_type == "table":
            continue
        if doc.metadata.doc_type == "markdown":
            pieces = _split_markdown(doc, rs, chunk_size)
        else:
            pieces = _split_plain_text(doc, rs)

        for index, (text, source_name, section) in enumerate(pieces):
            result.append(
                DocumentChunk(
                    chunk_id=_generate_chunk_id(doc, index, text),
                    company=doc.metadata.company,
                    doc_type=doc.metadata.doc_type or "text",
                    source=doc.metadata.source,
                    source_name=source_name,
                    page=doc.metadata.page,
                    text=text,
                    metadata={
                        **doc.metadata.model_dump(),
                        "section": section,
                        "chunk_index": index,
                    },
                )
            )

    return result


def _build_table_chunk(doc: Document, index: int) -> DocumentChunk:
    """表格 Document → 单 chunk（不切分，保留结构化表格文本）。"""
    return DocumentChunk(
        chunk_id=md5(f"{doc.metadata.source}_{index}_{doc.text[:50]}".encode()).hexdigest(),
        company=doc.metadata.company,
        doc_type="pdf",
        source=doc.metadata.source,
        source_name=doc.metadata.source_name,
        page=doc.metadata.page,
        text=doc.text,
        metadata={**doc.metadata.model_dump(), "chunk_index": index},
    )


def _base_name(doc: Document) -> str:
    """Document 的可读溯源名（loader 填写的 source_name）。"""
    return doc.metadata.source_name


def _split_plain_text(doc: Document, rs: RecursiveCharacterTextSplitter) -> list[tuple[str, str, str]]:
    """纯文本：直接递归切分，source_name 沿用原值，section 为空。"""
    name = _base_name(doc)
    return [(part, name, "") for part in rs.split_text(doc.text) if part.strip()]


def _split_markdown(
    doc: Document,
    rs: RecursiveCharacterTextSplitter,
    chunk_size: int,
) -> list[tuple[str, str, str]]:
    """Markdown：先按标题层级切分，超大节再按 chunk_size 细分。

    返回 [(text, source_name, section)]；
    source_name 附加标题路径（如 "招股书 > 第一章 > 1.1 商业模式"）；
    section 为文档内章节路径（如 "第一章 > 1.1 商业模式"），写入 chunk.metadata。
    """
    pieces: list[tuple[str, str, str]] = []
    base = _base_name(doc)

    for section in _md_splitter.split_text(doc.text):
        header_path = " > ".join(
            value for key, value in section.metadata.items() if key in {"H1", "H2", "H3", "H4"}
        )
        section_name = f"{base} > {header_path}" if header_path else base

        content = section.page_content
        if len(content) <= chunk_size:
            if content.strip():
                pieces.append((content, section_name, header_path))
            continue

        for part in rs.split_text(content):
            if part.strip():
                pieces.append((part, section_name, header_path))

    return pieces


def _generate_chunk_id(doc: Document, index: int, text: str) -> str:
    """生成 chunk 唯一 ID：MD5(来源 + 序号 + 内容前缀)，内容不变则 ID 稳定。"""
    source = doc.metadata.source
    raw = f"{source}_{index}_{text[:50]}"
    return md5(raw.encode("utf-8")).hexdigest()
