"""
ReportExporter 单元测试

覆盖:
  1. HTML 含 DOCTYPE / charset / @media print / 章节卡片
  2. Markdown 含标题层级 / 表格 / 引用
  3. 动态值 HTML 转义（< > &）
  4. 落盘两文件存在且可读
  5. ReportOutput 字段完整
"""

from __future__ import annotations

from pathlib import Path

from app.report.assembler import ReportAssembler
from app.report.exporter import export_report
from app.report.schemas import ReportContent

DANGEROUS = {
    "company": "A&B<C>",
    "summary": "包含 <script>alert(1)</script> 的文本",
    "business_model": "模型 & 数据",
    "industry_position": "第一",
}


def _content(company: str = "测试公司", **kw) -> ReportContent:
    return ReportAssembler().assemble(company=company, **kw)


def test_html_structure(tmp_path: Path):
    content = _content()
    out = export_report(content, out_dir=tmp_path)
    html_doc = Path(out.html_path).read_text(encoding="utf-8")
    assert html_doc.startswith("<!DOCTYPE html>")
    assert 'lang="zh-CN"' in html_doc
    assert '<meta charset="utf-8">' in html_doc
    assert "@media print" in html_doc
    assert "六、投资建议与风险提示" in html_doc
    assert 'class="card"' in html_doc


def test_markdown_structure(tmp_path: Path):
    content = _content(
        financial={
            "analysis_period": "2021年报",
            "key_metrics": {"roe_pct": 21.5},
            "data_source": "akshare",
        }
    )
    out = export_report(content, out_dir=tmp_path)
    md = Path(out.markdown_path).read_text(encoding="utf-8")
    assert md.startswith("# 测试公司 深度投研分析报告")
    assert "## 一、企业概况" in md
    assert "## 二、财务分析" in md
    assert "|" in md, "markdown 应含表格语法"
    assert "21.5%" in md
    assert "免责声明" in md


def test_html_escapes_dynamic_values(tmp_path: Path):
    content = _content(company=DANGEROUS["company"], research=DANGEROUS)
    out = export_report(content, out_dir=tmp_path)
    html_doc = Path(out.html_path).read_text(encoding="utf-8")
    assert "&lt;script&gt;" in html_doc
    assert "<script>" not in html_doc
    assert "&amp;" in html_doc
    # 标题里的 company 也被转义
    assert "A&amp;B&lt;C&gt;" in html_doc


def test_markdown_escapes_pipe_in_table(tmp_path: Path):
    content = _content(
        company="测试公司",
        risk={
            "overall_risk_level": "MEDIUM",
            "overall_score": 0.5,
            "dimensions": [{"dimension": "合规 | 政策", "score": 0.6, "reasoning": "r", "evidence": []}],
            "key_risks": [],
        },
    )
    out = export_report(content, out_dir=tmp_path)
    md = Path(out.markdown_path).read_text(encoding="utf-8")
    assert "合规 \\| 政策" in md, "markdown 表格内管道应转义"


def test_files_written_and_output_fields(tmp_path: Path):
    content = _content(company="CATL", ticker="300750")
    out = export_report(content, out_dir=tmp_path)
    assert out.report_id.startswith("catl_")
    assert Path(out.markdown_path).exists()
    assert Path(out.html_path).exists()
    assert out.title == "CATL 深度投研分析报告"
    assert out.generated_at == content.generated_at
    assert out.markdown  # 非空


def test_chinese_company_slug_fallback(tmp_path: Path):
    content = _content(company="宁德时代")
    out = export_report(content, out_dir=tmp_path)
    assert out.report_id.startswith("report_")
    assert Path(out.markdown_path).exists()
