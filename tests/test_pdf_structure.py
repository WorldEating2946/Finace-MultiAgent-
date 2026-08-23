"""PDF 章节结构识别（heading detection）单元测试。"""

from app.rag.document import Document, DocumentMetadata
from app.rag.loaders.pdf_loader import PDFLoader
from app.rag.parsers.pdf_structure import detect_headings, heading_level


def test_heading_level_detection():
    assert heading_level("第四章 业务回顾") == 1
    assert heading_level("第1章 公司概况") == 1
    assert heading_level("4.1 智能手机业务") == 2
    assert heading_level("4.1.1 市场表现") == 3
    # 正文/普通行
    assert heading_level("小米主要业务包括智能手机、IoT与生活消费产品。") is None
    assert heading_level("收入增长4.1个百分点。") is None  # 句末标点
    assert heading_level("") is None


def test_heading_level_rejects_financial_numbers():
    """OCR 后财务数据行不得被误判为标题（PR #30 数字误判过滤）。"""
    # 金额/百分比/财务表格数字 → 非标题
    assert heading_level("1.237.567.000元") is None
    assert heading_level("4.063,148,182") is None
    assert heading_level("22.3%") is None
    assert heading_level("76.8") is None
    assert heading_level("1.25亿") is None
    # OCR 在数字中插入空格（"1.237.567.000" → "1.2 37.567.000"）也不得误判
    assert heading_level("1.2 37.567.000元") is None
    assert heading_level("1.70 7.042.853股") is None
    assert heading_level("1.7 07.042.853股") is None
    # 但真实小节编号仍识别
    assert heading_level("4.1 智能手机业务") == 2
    assert heading_level("10.2.1 投资策略") == 3


def _heading_doc(page: int, text: str) -> Document:
    return Document(
        text=text,
        metadata=DocumentMetadata(
            doc_type="pdf", page=page, source="/tmp/x.pdf", source_name="x"
        ),
    )


def test_detect_headings_with_toc_titles_marks_chapters():
    """OCR 容错：目录标题列表驱动正文标题识别为 level-1 章（PR #30）。"""
    docs = [
        _heading_doc(3, "公司简介\n小米成立于2010年。"),
        _heading_doc(4, "主要业务\n智能手机与IoT。"),
        _heading_doc(5, "正文内容\n收入稳步增长。"),
    ]
    toc_titles = ["公司简介", "主要业务", "董事会报告"]
    headings = detect_headings(docs, toc_titles=toc_titles)

    chapters = [h for h in headings if h.level == 1]
    titles = [h.title for h in chapters]
    assert titles == ["公司简介", "主要业务"]  # 无编号标题 → 章级
    assert [h.page for h in chapters] == [3, 4]


def test_detect_headings_toc_titles_dedup_running_header():
    """同一目录标题在正文重复出现（如页眉/多页）只记为一次章节。"""
    docs = [
        _heading_doc(3, "董事会报告\n董事会报告\n本年度董事会运作情况。"),
        _heading_doc(4, "董事会报告\n下一年度计划。"),
    ]
    headings = detect_headings(docs, toc_titles=["董事会报告"])
    chapters = [h for h in headings if h.level == 1]
    assert len(chapters) == 1  # 去重后仅一个章节起点


def test_detect_headings_finds_chapters_and_sections(xiaomi_pdf_path):
    docs = PDFLoader().load(str(xiaomi_pdf_path))
    headings = detect_headings(docs)

    chapters = [h for h in headings if h.level == 1]
    sections = [h for h in headings if h.level == 2]

    assert any("第一章" in h.title for h in chapters)
    assert any("第四章 业务回顾" in h.title for h in chapters)
    assert any("4.1" in s.title for s in sections)
    assert any("5.3 风险提示" in s.title for s in sections)


def test_detect_headings_page_assignment(xiaomi_pdf_path):
    docs = PDFLoader().load(str(xiaomi_pdf_path))
    headings = detect_headings(docs)

    # 第一个标题应在第 1 页
    assert headings[0].page == 1
    # 章节顺序按页码递增
    pages = [h.page for h in headings]
    assert pages == sorted(pages)
