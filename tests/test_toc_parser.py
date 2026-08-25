"""TOC 目录解析 + 章节 page-range 映射单元测试。"""

from app.rag.document import Document, DocumentMetadata
from app.rag.parsers.toc_parser import (
    PageRangeMap,
    TocEntry,
    _is_title_like,
    extract_toc_titles,
    parse_toc,
)


def _make_doc(page: int, text: str) -> Document:
    return Document(
        text=text,
        metadata=DocumentMetadata(
            doc_type="pdf", page=page, source="/tmp/x.pdf", source_name="x"
        ),
    )


def test_parse_toc_finds_entries_and_ranges():
    docs = [
        _make_doc(1, "小米集团年度报告（模拟）"),
        _make_doc(
            2,
            "目录\n"
            "第一章 公司概况 ................ 3\n"
            "第二章 经营情况讨论与分析 ...... 48\n"
            "第三章 未来战略规划 ............ 93\n",
        ),
        _make_doc(3, "第一章 公司概况\n公司成立。"),
        _make_doc(200, "正文"),
    ]

    toc = parse_toc(docs)

    assert toc.found
    assert toc.excluded_pages == {2}  # 目录页被标记排除
    items = toc.ranges.items
    assert items[0] == ("第一章 公司概况", 3, 47)
    assert items[1] == ("第二章 经营情况讨论与分析", 48, 92)
    assert items[2] == ("第三章 未来战略规划", 93, 200)  # 最后一章到文档末页


def test_parse_toc_no_toc_returns_empty():
    docs = [_make_doc(1, "没有目录的普通文本页"), _make_doc(2, "继续内容")]
    toc = parse_toc(docs)
    assert not toc.found
    assert toc.excluded_pages == set()
    assert toc.ranges.items == []


def test_page_range_lookup():
    rmap = PageRangeMap(
        [
            TocEntry("第一节", 5),
            TocEntry("第二节", 10),
            TocEntry("第三节", 14),
        ],
        total_pages=30,
    )
    assert rmap.lookup(7) == ("第一节", 5, 9)
    assert rmap.lookup(14) == ("第三节", 14, 30)
    assert rmap.lookup(3) is None  # 目录页之前的页无章节


def test_parse_toc_prefers_pdf_outline(tmp_path):
    """PDF outline（书签）是最高优先 TOC 源：干净章节名 + 真实页码（PR #30）。"""
    import fitz

    pdf = fitz.open()
    for _ in range(4):
        pdf.new_page()
    pdf.set_toc(
        [
            [1, "封面", 1],
            [1, "目錄", 2],
            [1, "管理層討論及分析", 3],
            [1, "董事會報告", 4],
        ]
    )
    path = tmp_path / "with_outline.pdf"
    pdf.save(str(path))
    pdf.close()

    docs = [_make_doc(i, f"page {i}") for i in range(1, 5)]
    for d in docs:
        d.metadata.source = str(path)
    toc = parse_toc(docs)

    assert toc.found
    assert toc.excluded_pages == {1, 2}  # 封面 + 目錄 导航页被排除
    items = toc.ranges.items
    # 标题转简体 → 与评测/正文一致
    assert items[0] == ("管理层讨论及分析", 3, 3)
    assert items[1] == ("董事会报告", 4, 4)
    assert toc.ranges.lookup(3) == ("管理层讨论及分析", 3, 3)


def test_parse_toc_outline_missing_falls_back(tmp_path):
    """无 outline 的 PDF → 回退到原有路径（不抛异常，走空结果）。"""
    import fitz

    pdf = fitz.open()
    pdf.new_page()
    pdf.new_page()
    path = tmp_path / "no_outline.pdf"
    pdf.save(str(path))
    pdf.close()

    docs = [_make_doc(1, "没有目录的普通文本页"), _make_doc(2, "继续内容")]
    for d in docs:
        d.metadata.source = str(path)
    toc = parse_toc(docs)
    assert not toc.found
    assert toc.ranges.items == []


def test_is_title_like():
    """OCR 目录标题候选判定（短行、无句末标点、非纯数字）。"""
    assert _is_title_like("公司简介") is True
    assert _is_title_like("主要业务") is True
    assert _is_title_like("释义") is True
    # 正文/数字/日期 → 非标题
    assert _is_title_like("收入增长4.1个百分点。") is False  # 句末标点
    assert _is_title_like("4.063,148,182") is False  # 纯数字
    assert _is_title_like("2026年8月5日") is False  # 纯日期
    assert _is_title_like("") is False
    assert _is_title_like("这一段是超过十六个字符长度的正文句子") is False  # 超长


def test_extract_toc_titles_finds_ocr_toc_page():
    """OCR 容错：无点线页码的目录页 → 提取标题列表（PR #30）。"""
    docs = [
        _make_doc(1, "小米集团年度报告"),
        _make_doc(
            2,
            "目录\n公司简介\n主要业务\n主席报告\n管理层讨论及分析\n董事会报告\n释义\n独立核数师报告",
        ),
        _make_doc(3, "正文开始"),
    ]
    titles, page = extract_toc_titles(docs)
    assert page == 2
    assert "公司简介" in titles
    assert "主要业务" in titles
    assert "董事会报告" in titles
    assert "目录" not in titles  # "目录" 非标题候选（短行但判据过滤）


def test_extract_toc_titles_skips_cover_page():
    """封面短行页（标题行占比不足 60%）不误判为目录页。"""
    docs = [
        _make_doc(1, "小米集团\n2025\n年度报告"),
        _make_doc(
            3,
            "公司简介\n主要业务\n主席报告\n管理层讨论及分析\n董事会报告\n释义",
        ),
    ]
    titles, page = extract_toc_titles(docs)
    assert page == 3  # 真实目录在第 3 页（首个达标页，非标题最多的资料页）
    assert "公司简介" in titles


def test_extract_toc_titles_first_qualified_page_beats_noise_page():
    """前部先出现真实目录页时，即使后续资料页标题行更多也不被选中。"""
    # 页3 为真实目录（6 标题），页5 为地址/资料页（10+ 短行但非目录）
    address_page = (
        "香港\n湾仔\n皇后大道东183号\n合和中心\n17楼1712至1716室\n"
        "香港证券登记处\n香港中央证券登记有限公司\n主要股份过户处\n"
        "Cricket Square\nCayman lslands\n主要往来银行\n公司网址"
    )
    docs = [
        _make_doc(1, "封面"),
        _make_doc(
            3,
            "公司简介\n主要业务\n主席报告\n管理层讨论及分析\n董事会报告\n释义",
        ),
        _make_doc(5, address_page),
    ]
    titles, page = extract_toc_titles(docs)
    assert page == 3  # 选首个达标页（真实目录），而非标题更多的地址页
    assert "香港" not in titles
    assert "公司简介" in titles


def test_toc_enrichment_in_splitter():
    """切分器应跳过目录页，并用 TOC 补全 chunk 的 chapter / page_range。"""
    from app.rag.splitters.hierarchical_splitter import split_pdf_hierarchical

    docs = [
        _make_doc(1, "封面"),
        _make_doc(
            2,
            "目录\n"
            "第一章 公司概况 ........ 3\n"
            "第二章 业务回顾 ........ 6\n"
            "第三章 未来规划 ........ 9\n",
        ),
        _make_doc(3, "第一章 公司概况\n公司成立于2010年，总部位于北京。"),
        _make_doc(4, "第一章 公司概况\n主营业务涵盖三大板块。"),
        _make_doc(6, "第二章 业务回顾\n智能手机业务全球出货量领先。"),
        _make_doc(9, "第三章 未来规划\n公司将持续加大研发投入。"),
    ]
    docs[0].metadata.company = "小米"

    toc = parse_toc(docs)
    chunks = split_pdf_hierarchical(docs, chunk_size=512, chunk_overlap=0, toc=toc)

    # 目录页（page 2）不应产生 chunk
    assert all(c.page != 2 for c in chunks)
    # 第一章页面的 chunk 补全 TOC 章节
    chapter1 = [c for c in chunks if c.page in (3, 4)]
    assert chapter1, "第一章页面应产生 chunk"
    assert all(c.metadata.get("chapter") == "第一章 公司概况" for c in chapter1)
    assert all(c.metadata.get("page_range") == "3-5" for c in chapter1)
    # 第二章 / 第三章页面的 chunk
    chapter2 = [c for c in chunks if c.page == 6]
    assert chapter2
    assert all(c.metadata.get("chapter") == "第二章 业务回顾" for c in chapter2)
    assert all(c.metadata.get("page_range") == "6-8" for c in chapter2)
    chapter3 = [c for c in chunks if c.page == 9]
    assert chapter3
    assert all(c.metadata.get("chapter") == "第三章 未来规划" for c in chapter3)
