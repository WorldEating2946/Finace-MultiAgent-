"""PDF 表格解析：检测 + 提取 + 结构化为 Markdown 表格。

企业年报大量关键数据（收入/毛利/研发/出货量）在表格中，纯文本抽取会丢失语义关系
（如 "收入 457286 365906" 无法知道哪一年）。本模块将表格转为结构化文本，
供 BGE-M3 语义理解。

分层：
    第一层：PyMuPDF ``page.find_tables()``（快、已依赖、年报适配好）
    第二层：pdfplumber ``extract_tables()``（预留 fallback）
    第三层：视觉模型（预留，处理图片表格/复杂合并单元格）

输出格式（供 embedding）：
    【五年财务概要】(第114页)
    指标 | 2025 | 2024
    收入 | 457,286,687 | 365,906,350
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TableData:
    """单张表格数据。"""

    page: int
    title: str
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    text: str = ""  # 渲染后的 Markdown 表格


def _clean_cell(cell) -> str:
    """清洗单元格：None → ""，去空白。"""
    if cell is None:
        return ""
    return str(cell).strip()


def _clean_grid(grid: list[list]) -> list[list[str]]:
    """清洗网格：清理单元格、去全空行、去全空前列。"""
    rows = [[_clean_cell(c) for c in row] for row in grid]
    rows = [r for r in rows if any(c for c in r)]
    # 去掉所有行都为空的第一列（常见于财务表左侧空行标签列）
    while rows and all(not r[0] for r in rows) and len(rows[0]) > 1:
        rows = [r[1:] for r in rows]
    return rows


def render_table(title: str, headers: list[str], rows: list[list[str]], page: int) -> str:
    """渲染为 Markdown 表格文本（供 embedding 理解语义关系）。"""
    lines = [f"【{title or '表格'}】(第{page}页)"]
    # 用第一个非空行作为表头
    header = headers or (rows[0] if rows else [])
    data = rows[1:] if not headers and rows else rows
    if header:
        header_line = "指标 | " + " | ".join(header[1:]) if len(header) > 1 else header[0]
        lines.append(header_line)
    for row in data:
        lines.append(" | ".join(row))
    return "\n".join(lines)


def _detect_title(page, table, max_dist: float = 80.0) -> str:
    """取表格上方最近的短文本行作为标题（如 "合并资产负债表"）。"""
    top = table.bbox[1]
    best, best_dist = "", float("inf")
    try:
        for block in page.get_text("blocks"):
            _x0, _y0, _x1, y1, text, *_ = block
            t = text.strip().replace("\n", " ")
            if y1 <= top and 0 < (top - y1) < best_dist and t and len(t) <= 30:
                best_dist = top - y1
                best = t
    except Exception as exc:  # noqa: BLE001 文本块异常不影响表格提取
        logger.warning("表格标题检测失败: %s", exc)
    if best_dist < max_dist:
        return best
    return ""


class TableParser:
    """基于 PyMuPDF 的表格提取（第一层）。"""

    def extract_pdf_tables(self, file_path: str) -> list[TableData]:
        """从 PDF 逐页提取表格。

        Args:
            file_path: PDF 路径。

        Returns:
            TableData 列表（已渲染 Markdown 文本）。
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("表格解析需要 pymupdf，请执行：uv pip install -r requirements.txt")

        tables: list[TableData] = []
        pdf = fitz.open(file_path)
        try:
            for page in pdf:
                found = page.find_tables().tables
                for t in found:
                    grid = _clean_grid(t.extract())
                    if not grid:
                        continue
                    headers, rows = grid[0], grid[1:]
                    title = _detect_title(page, t)
                    tables.append(
                        TableData(
                            page=page.number + 1,
                            title=title,
                            headers=headers,
                            rows=rows,
                            text=render_table(title, headers, rows, page.number + 1),
                        )
                    )
        finally:
            pdf.close()

        return tables
