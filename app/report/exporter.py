"""
研报导出器 —— ReportContent（结构化）→ Markdown + 自包含打印 HTML。

设计要点：
  - 零第三方依赖、零 markdown 解析：exporter 从 ReportBlock 直接派生两种格式。
  - 所有动态值经 html.escape 转义；行内 **加粗** 标记 → <strong>（确定性小变换）。
  - HTML 复用 scripts/render_financial_report.py 的 CSS 设计语言，
    无 CDN、离线可打开、浏览器「另存为 PDF」即可打印。

落盘：{out_dir}/{report_id}/report.md 与 report.html
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from pathlib import Path

from .schemas import ReportBlock, ReportContent, ReportOutput

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT = _ROOT / "data" / "reports"

# 行内加粗 **x** → <strong>x</strong>（内容已转义）
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")

_CSS = """
:root{
  --bg:#f4f6fa; --card:#ffffff; --ink:#1a2233; --sub:#5b6472; --line:#e4e8f0;
  --accent:#2563eb; --green:#16a34a; --amber:#d97706; --red:#dc2626; --chip:#eef2ff;
}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--ink)}
.wrap{max-width:1000px;margin:0 auto;padding:24px 20px 80px}
header.hero{background:linear-gradient(135deg,#1e3a8a,#2563eb 55%,#0d9488);border-radius:18px;color:#fff;padding:30px 28px;margin-bottom:22px;box-shadow:0 10px 30px rgba(37,99,235,.22)}
.hero h1{margin:0 0 6px;font-size:24px;line-height:1.4}
.hero .sub{opacity:.92;font-size:13.5px}
.hero .tag{display:inline-block;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.35);padding:3px 12px;border-radius:999px;font-size:12px;margin:10px 6px 0 0}
section.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 22px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,.04);page-break-inside:avoid}
section.card h2{font-size:17px;margin:0 0 12px;display:flex;align-items:center;gap:8px}
section.card h2 .num{background:var(--accent);color:#fff;border-radius:8px;padding:2px 10px;font-size:13px;white-space:nowrap}
p{font-size:13.5px;line-height:1.9;margin:6px 0}
p strong{color:var(--accent)}
ul{margin:6px 0 6px 2px;padding-left:18px}
li{font-size:13.5px;line-height:1.9}
li strong{color:var(--accent)}
table{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--sub);font-weight:600;background:#f8fafc}
.tbl-wrap{overflow-x:auto}
.commentary{background:#f8fafc;border-left:3px solid var(--accent);border-radius:0 8px 8px 0;padding:12px 16px;margin:8px 0;font-size:13px;line-height:1.9;white-space:pre-wrap;color:var(--sub)}
.footer{margin-top:34px;text-align:center;color:var(--sub);font-size:12px;line-height:1.8}
@media print{
  body{background:#fff}
  .wrap{max-width:100%;padding:0}
  header.hero{box-shadow:none;border-radius:0;background:linear-gradient(135deg,#1e3a8a,#2563eb)}
  section.card{box-shadow:none;break-inside:avoid}
  .footer{page-break-after:auto}
}
"""


def _slug(text: str) -> str:
    """ASCII 安全 slug（中文 → 空，由调用方兜底）。"""
    return re.sub(r"[^0-9A-Za-z-]", "", text.lower())


def _inline_md(text: str) -> str:
    """行内加粗 → <strong>（先转义再替换，内容安全）。"""
    return _BOLD_RE.sub(r"<strong>\1</strong>", html.escape(text))


def _render_block_md(block: ReportBlock) -> str:
    if block.kind == "para":
        return f"{block.text}\n\n"
    if block.kind == "quote":
        return f"> {block.text}\n\n"
    if block.kind == "bullets":
        return "".join(f"- {i}\n" for i in block.items) + "\n"
    if block.kind == "table":
        esc = lambda x: str(x).replace("|", "\\|")
        head = "| " + " | ".join(esc(h) for h in block.headers) + " |"
        sep = "|" + "---|" * len(block.headers)
        rows = "".join(
            "| " + " | ".join(esc(c) for c in row) + " |\n"
            for row in block.rows
        )
        return f"{head}\n{sep}\n{rows}\n\n"
    return ""


def _render_block_html(block: ReportBlock) -> str:
    if block.kind == "para":
        return f"<p>{_inline_md(block.text)}</p>"
    if block.kind == "quote":
        return f'<div class="commentary">{_inline_md(block.text)}</div>'
    if block.kind == "bullets":
        lis = "".join(f"<li>{_inline_md(i)}</li>" for i in block.items)
        return f"<ul>{lis}</ul>"
    if block.kind == "table":
        heads = "".join(f"<th>{html.escape(str(h))}</th>" for h in block.headers)
        tbody = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
            for row in block.rows
        )
        return (
            '<div class="tbl-wrap"><table>'
            f"<thead><tr>{heads}</tr></thead>"
            f"<tbody>{tbody}</tbody></table></div>"
        )
    return ""


def _content_to_markdown(content: ReportContent) -> str:
    lines: list[str] = [f"# {content.title}", "", f"> 生成时间：{content.generated_at}", ""]
    for sec in content.sections:
        lines.append(f"## {sec.title}")
        lines.append("")
        for block in sec.blocks:
            lines.append(_render_block_md(block).rstrip())
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _content_to_html(content: ReportContent) -> str:
    tag = f'<span class="tag">🔖 {html.escape(content.company)}</span>' if content.company else ""
    if content.ticker:
        tag += f'<span class="tag">📈 {html.escape(content.ticker)}</span>'
    sections = "".join(
        '<section class="card"><h2><span class="num">{}</span>{}</h2>{}</section>'.format(
            html.escape(str(i + 1)),
            html.escape(sec.title),
            "".join(_render_block_html(b) for b in sec.blocks),
        )
        for i, sec in enumerate(content.sections)
    )
    return (
        "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(content.title)}</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n<body>\n"
        '<div class="wrap">\n'
        f'<header class="hero"><h1>{html.escape(content.title)}</h1>'
        f'<div class="sub">FinaceAgent 多 Agent 智能投研平台 · 自动生成 {html.escape(content.generated_at)}</div>{tag}</header>\n'
        f"{sections}\n"
        '<div class="footer">本报告由 FinaceAgent 系统自动生成，仅供研究参考，不构成投资建议。<br>'
        f"报告 ID：{html.escape(content.company)}-{html.escape(content.generated_at)}</div>\n"
        "</div>\n</body>\n</html>\n"
    )


def export_report(
    content: ReportContent,
    out_dir: str | Path | None = None,
) -> ReportOutput:
    """结构化研报 → Markdown + HTML 落盘。返回 ReportOutput。"""
    out_root = Path(out_dir) if out_dir else _DEFAULT_OUT
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe = _slug(content.company) or "report"
    report_id = f"{safe}_{ts}"
    out = out_root / report_id
    out.mkdir(parents=True, exist_ok=True)

    markdown = _content_to_markdown(content)
    html_doc = _content_to_html(content)

    md_path = out / "report.md"
    html_path = out / "report.html"
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html_doc, encoding="utf-8")

    return ReportOutput(
        report_id=report_id,
        title=content.title,
        markdown=markdown,
        markdown_path=str(md_path),
        html_path=str(html_path),
        generated_at=content.generated_at,
    )
