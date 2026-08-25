"""
统一研报生成模块 —— 端到端验证脚本（离线，无网络依赖）。

读取 app/data/300750_宁德时代.json 的真实财务数据，纯计算派生
FinancialAgentOutput 结构 → 结合代表性 research / sentiment / risk 输出
→ ReportAssembler 组装六章研报 → export_report 导出 Markdown + 自包含 HTML。

用法:
    python scripts/run_report_generate_demo.py

输出:
    data/reports/{report_id}/report.md 与 report.html（打印绝对路径）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.report import ReportAssembler, export_report

_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE = _ROOT / "app" / "data" / "300750_宁德时代.json"


def _derive_financial(fixture: dict) -> dict:
    """从 fixture fiscal_data 派生 FinancialAgentOutput 结构（纯计算）。"""
    fiscal = fixture["fiscal_data"]
    years = sorted(fiscal.keys())
    key_metrics = None
    dupont = None
    yoy_history: list[dict] = []
    for i, yr in enumerate(years):
        f = fiscal[yr]
        revenue = f["revenue"]
        net_profit = f["net_profit"]
        total_assets = f["total_assets"]
        equity = f["shareholders_equity"]
        if i == len(years) - 1:
            prev = fiscal[years[i - 1]]
            rev_g = (revenue / prev["revenue"] - 1) * 100 if prev["revenue"] else None
            prf_g = (net_profit / prev["net_profit"] - 1) * 100 if prev["net_profit"] else None
            key_metrics = {
                "roe_pct": round(net_profit / equity * 100, 2),
                "net_profit_margin_pct": round(net_profit / revenue * 100, 2),
                "revenue_yoy_pct": round(rev_g, 2) if rev_g is not None else None,
                "net_profit_yoy_pct": round(prf_g, 2) if prf_g is not None else None,
                "equity_multiplier": round(total_assets / equity, 2),
                "asset_turnover": round(revenue / total_assets, 4),
            }
            dupont = {
                "net_profit_margin": round(net_profit / revenue * 100, 2),
                "asset_turnover": round(revenue / total_assets, 4),
                "equity_multiplier": round(total_assets / equity, 2),
                "roe_computed": round(net_profit / revenue * revenue / total_assets * total_assets / equity * 100, 2),
                "roe_direct": round(net_profit / equity * 100, 2),
            }
        if i >= 1:
            prev = fiscal[years[i - 1]]
            rev_g = (revenue / prev["revenue"] - 1) * 100 if prev["revenue"] else None
            prf_g = (net_profit / prev["net_profit"] - 1) * 100 if prev["net_profit"] else None
            yoy_history.append({
                "period": yr,
                "revenue_growth_pct": round(rev_g, 2) if rev_g is not None else None,
                "revenue_trend": "增长" if (rev_g or 0) > 0 else "下降",
                "net_profit_growth_pct": round(prf_g, 2) if prf_g is not None else None,
                "profit_trend": "增长" if (prf_g or 0) > 0 else "下降",
            })
    assert key_metrics and dupont
    return {
        "company": fixture["company_name"],
        "ticker": fixture["ticker"],
        "analysis_period": f"{years[-1]}年报（离线验证数据）",
        "key_metrics": key_metrics,
        "dupont": dupont,
        "yoy_history": yoy_history,
        "commentary": (
            f"{fixture['company_name']}（{fixture['ticker']}）近年营收与净利润趋势分化，"
            f"最新 ROE {key_metrics['roe_pct']:.1f}%，盈利质量尚可，需关注营收增速与费用变化。"
        ),
        "data_source": "fixture",
        "fetch_error": None,
    }


def _research(fixture: dict) -> dict:
    """代表性 Research 输出（占位形态，演示字段兼容）。"""
    return {
        "company": fixture["company_name"],
        "summary": f"{fixture['company_name']} 是国内动力电池行业龙头，技术与规模全球领先。",
        "business_model": "研发-生产-销售一体化，覆盖电池材料、电芯与系统集成。",
        "industry_position": f"所属行业：{fixture['industry']}；全球动力电池装机量第一梯队。",
        "competitive_advantages": ["技术壁垒高", "规模效应显著", "客户结构优质"],
        "key_risks_business": ["原材料价格波动", "行业竞争加剧"],
    }


def _sentiment(fixture: dict) -> dict:
    """代表性 Sentiment 输出（演示四章舆情渲染）。"""
    return {
        "symbol": fixture["ticker"],
        "company_name": fixture["company_name"],
        "searched_news_count": 16,
        "sentiment_distribution": {"positive": 8, "negative": 5, "neutral": 3},
        "topics": [
            {"topic_id": 0, "label": "产品与技术", "keywords": ["麒麟电池", "快充"], "news_count": 7, "representative_news": []},
            {"topic_id": 1, "label": "海外市场", "keywords": ["欧洲", "产能"], "news_count": 5, "representative_news": []},
        ],
        "summary": "整体情绪中性偏多，技术与出海话题关注度最高。",
    }


def _risk(fixture: dict, financial: dict) -> dict:
    """代表性 Risk 输出（演示五章风险评估渲染）。"""
    roe = (financial["key_metrics"]["roe_pct"] or 0)
    level = "low" if roe >= 10 else "medium"
    return {
        "symbol": fixture["ticker"],
        "company_name": fixture["company_name"],
        "overall_risk_level": level,
        "overall_score": 0.35 if level == "low" else 0.55,
        "dimensions": [
            {"dimension": "市场风险", "score": 0.4, "evidence": ["行业竞争加剧"], "reasoning": "市场份额受挑战"},
            {"dimension": "经营风险", "score": 0.3, "evidence": [], "reasoning": "ROE 表现稳健"},
            {"dimension": "财务风险", "score": 0.35, "evidence": [], "reasoning": "资产负债结构健康"},
        ],
        "key_risks": ["原材料价格波动", "技术路线迭代"],
        "reasoning_chain": "各维度均处中低位 → 综合风险可控",
        "risk_summary": "综合风险可控，主要关注原材料价格与竞争格局变化。",
    }


def main() -> int:
    if not _FIXTURE.exists():
        print(f"[ERR] 数据文件不存在: {_FIXTURE}")
        return 1
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))

    financial = _derive_financial(fixture)
    content = ReportAssembler().assemble(
        company=fixture["company_name"],
        ticker=fixture["ticker"],
        user_query=f"分析{fixture['company_name']}的财务状况与发展前景",
        research=_research(fixture),
        financial=financial,
        sentiment=_sentiment(fixture),
        risk=_risk(fixture, financial),
    )
    out = export_report(content)

    print("=" * 60)
    print("[OK] 统一研报生成完成")
    print(f"  报告标题 : {out.title}")
    print(f"  report_id: {out.report_id}")
    print(f"  Markdown : {out.markdown_path}")
    print(f"  HTML     : {out.html_path}")
    print(f"  章节数   : {len(content.sections)}")
    print("=" * 60)
    print("Markdown 预览（前 500 字符）:")
    print("-" * 60)
    print(out.markdown[:500])
    print("-" * 60)
    print("浏览器打开 HTML 后可按 Ctrl+P 另存为 PDF。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
