"""
报告生成模块 —— 统一研报合成与导出。

- schemas:    输入/输出 Pydantic 模型
- assembler:  四 Agent 输出 → 六章结构化研报（纯规则）
- exporter:   结构化研报 → Markdown + 自包含打印 HTML

用法:
    from app.report import ReportAssembler, export_report
    content = ReportAssembler().assemble(company, research=..., financial=...)
    output = export_report(content, out_dir)
"""

from .assembler import ReportAssembler
from .exporter import export_report
from .schemas import (
    ReportBlock,
    ReportContent,
    ReportGenerateRequest,
    ReportOutput,
    ReportSection,
)

__all__ = [
    "ReportAssembler",
    "ReportBlock",
    "ReportContent",
    "ReportGenerateRequest",
    "ReportOutput",
    "ReportSection",
    "export_report",
]
