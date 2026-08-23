"""PDF 表格解析单元测试。"""

from app.rag.document import Document, DocumentMetadata
from app.rag.parsers.table_parser import TableParser, _clean_grid, render_table


def test_clean_grid_removes_empty_and_leading_column():
    grid = [
        ["", "指标", "", "2025", "", "2024"],
        ["", "收入", "", "457286", "", "365906"],
        ["", "毛利", "", "101805", "", "76560"],
        ["", "", "", "", "", ""],  # 全空行
    ]
    cleaned = _clean_grid(grid)
    assert cleaned == [
        ["指标", "", "2025", "", "2024"],
        ["收入", "", "457286", "", "365906"],
        ["毛利", "", "101805", "", "76560"],
    ]


def test_render_table_markdown():
    text = render_table(
        "五年财务概要", ["指标", "2025", "2024"], [["收入", "457286", "365906"]], 14
    )
    assert "五年财务概要" in text
    assert "指标 | 2025 | 2024" in text
    assert "收入 | 457286 | 365906" in text


def _make_grid_pdf(tmp_path):
    """生成带网格线的表格 PDF（3 列 × 4 行）。"""
    import fitz

    path = tmp_path / "表格测试.pdf"
    doc = fitz.open()
    page = doc.new_page()
    x0, y0 = 72, 120
    col_w, row_h = 120, 30
    cols, rows = 3, 4
    data = [
        ["指标", "2025", "2024"],
        ["收入", "457286", "365906"],
        ["毛利", "101805", "76560"],
        ["研发", "24100", "19000"],
    ]
    # 网格线
    for i in range(cols + 1):
        page.draw_line((x0 + i * col_w, y0), (x0 + i * col_w, y0 + row_h * rows))
    for j in range(rows + 1):
        page.draw_line((x0, y0 + j * row_h), (x0 + col_w * cols, y0 + j * row_h))
    # 单元格文本
    for j in range(rows):
        for i in range(cols):
            page.insert_text(
                (x0 + i * col_w + 5, y0 + j * row_h + 20),
                data[j][i],
                fontname="china-s",
                fontsize=10,
            )
    doc.save(str(path))
    doc.close()
    return path


def test_table_parser_extracts_generated_table(tmp_path):
    """生成的网格表格 PDF 应被提取为结构化表格（含指标/年份/数据）。"""
    pdf = _make_grid_pdf(tmp_path)
    tables = TableParser().extract_pdf_tables(str(pdf))

    assert len(tables) >= 1
    t = tables[0]
    assert "收入" in t.text
    assert "457286" in t.text
    assert "2025" in t.text


def test_table_document_becomes_single_chunk(tmp_path):
    """表格 Document 经 split_documents 应为单 chunk，保留结构化表格文本。"""
    from app.rag.splitter import split_documents

    table_doc = Document(
        text="【五年财务概要】(第14页)\n指标 | 2025 | 2024\n收入 | 457286 | 365906",
        metadata=DocumentMetadata(
            doc_type="pdf", page=14, source="/tmp/x.pdf", source_name="x",
            company="宁德时代", content_type="table", table_title="五年财务概要",
            table_headers=["指标", "2025", "2024"],
        ),
    )
    chunks = split_documents([table_doc], chunk_size=100, chunk_overlap=0)

    assert len(chunks) == 1  # 表格不切分
    assert chunks[0].metadata["content_type"] == "table"
    assert "457286" in chunks[0].text
