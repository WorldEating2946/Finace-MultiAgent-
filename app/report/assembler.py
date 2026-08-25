"""
统一研报组装器 —— 四 Agent 输出 → 结构化六章研报。

纯 Python 确定性组装（不调 LLM）：
  1. 企业概况       ← research
  2. 财务分析       ← financial
  3. 行业与竞争力   ← research
  4. 舆情风向       ← sentiment
  5. 风险评估       ← risk
  6. 投资建议       ← 纯规则综合（风险等级 × ROE × 正负舆情比）

任一 Agent 输出缺失 → 章节保留、正文「（待补充）」，不崩。

输入端兼容两种 research 形态：
  - ResearchReport（真实 Research Agent）：title/summary/advantages/uncertainties
  - research_result（主图占位）：company/summary/business_model/industry_position/competitive_advantages
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .schemas import ReportBlock, ReportContent, ReportSection

_PLACEHOLDER = "（数据待补充）"


# ════════════════════════════════════════════════════════════
# 通用取值/格式化防御
# ════════════════════════════════════════════════════════════


def _s(v: Any, default: str = "") -> str:
    """任意值 → 去除首尾空白的字符串。"""
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _pct(v: Any, d: int = 1) -> str:
    """数值 → 百分比字符串；非数值返回 '—'。"""
    if v is None:
        return "—"
    try:
        return f"{float(v):.{d}f}%"
    except (TypeError, ValueError):
        return "—"


def _num(v: Any, d: int = 2) -> str:
    """数值 → 定点字符串；非数值返回 '—'。"""
    if v is None:
        return "—"
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return "—"


def _risk_cn(level: Any) -> str:
    """风险等级 → 中文标签。"""
    return {
        "high": "高风险",
        "medium": "中风险",
        "low": "低风险",
    }.get(_s(level).lower(), _s(level) or "待评估")


def _d(mapping: dict | None, *keys: str) -> Any:
    """多级安全取值：逐级 get，任何一层缺失返回 None。"""
    cur: Any = mapping
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


# ════════════════════════════════════════════════════════════
# 章节构造辅助
# ════════════════════════════════════════════════════════════


def _para(text: str) -> ReportBlock:
    return ReportBlock(kind="para", text=text)


def _quote(text: str) -> ReportBlock:
    return ReportBlock(kind="quote", text=text)


def _bullets(items: list[str]) -> ReportBlock:
    return ReportBlock(kind="bullets", items=items)


def _table(headers: list[str], rows: list[list[str]]) -> ReportBlock:
    return ReportBlock(kind="table", headers=headers, rows=rows)


# ════════════════════════════════════════════════════════════
# 组装器
# ════════════════════════════════════════════════════════════


class ReportAssembler:
    """四 Agent 输出 → ReportContent（六章结构化研报）。"""

    def assemble(
        self,
        company: str,
        ticker: str = "",
        user_query: str = "",
        research: dict | None = None,
        financial: dict | None = None,
        sentiment: dict | None = None,
        risk: dict | None = None,
    ) -> ReportContent:
        """组装六章研报。任一输入为 None → 对应章节显示待补充。"""
        title = f"{company} 深度投研分析报告"
        return ReportContent(
            title=title,
            company=company,
            ticker=ticker,
            generated_at=datetime.now(timezone.utc).isoformat(),
            sections=[
                self._ch_overview(company, ticker, research),
                self._ch_financial(financial),
                self._ch_industry(research),
                self._ch_sentiment(sentiment),
                self._ch_risk(risk),
                self._ch_advice(risk, financial, sentiment),
            ],
        )

    # ── 第一章：企业概况 ────────────────────────────────────
    def _ch_overview(self, company: str, ticker: str, research: dict | None) -> ReportSection:
        blocks: list[ReportBlock] = []
        if research:
            # ResearchReport 形态
            summary = _s(_d(research, "summary"))
            if summary:
                blocks.append(_para(summary))
            # research_result 占位形态
            biz = _s(_d(research, "business_model"))
            if biz:
                blocks.append(_para(f"**业务模式**：{biz}"))
            industry = _s(_d(research, "industry_position"))
            if industry:
                blocks.append(_para(f"**行业地位**：{industry}"))
            # 竞争优势（两种形态兼容）
            advs: list[str] = []
            for item in _d(research, "advantages") or []:
                claim = _s(item.get("claim") if isinstance(item, dict) else None)
                if claim:
                    advs.append(claim)
            advs.extend(str(x) for x in (_d(research, "competitive_advantages") or []) if _s(x))
            if advs:
                blocks.append(_bullets([f"**亮点**：{a}" for a in advs]))
        if not blocks:
            blocks.append(_para(_PLACEHOLDER))
        tag = f"（{ticker}）" if ticker else ""
        return ReportSection(
            key="chapter_1", title=f"一、企业概况{tag}", blocks=blocks
        )

    # ── 第二章：财务分析 ────────────────────────────────────
    def _ch_financial(self, financial: dict | None) -> ReportSection:
        blocks: list[ReportBlock] = []
        if financial:
            period = _s(_d(financial, "analysis_period"))
            source = _s(_d(financial, "data_source"), "unknown")
            header = f"分析周期：{period}；数据来源：{source}" if period else f"数据来源：{source}"
            blocks.append(_para(header))

            km = _d(financial, "key_metrics") or {}
            if km:
                rows = [
                    ["ROE（净资产收益率）", _pct(_d(km, "roe_pct"))],
                    ["净利润率", _pct(_d(km, "net_profit_margin_pct"))],
                    ["营收同比增速", _pct(_d(km, "revenue_yoy_pct"))],
                    ["净利润同比增速", _pct(_d(km, "net_profit_yoy_pct"))],
                    ["权益乘数（财务杠杆）", _num(_d(km, "equity_multiplier"), 2)],
                    ["资产周转率", _num(_d(km, "asset_turnover"), 4)],
                ]
                blocks.append(_table(["核心指标", "数值"], rows))

            dupont = _d(financial, "dupont") or {}
            if dupont:
                rows = [
                    ["净利润率（净利 / 营收）", _pct(_d(dupont, "net_profit_margin"), 2)],
                    ["资产周转率（营收 / 总资产）", _num(_d(dupont, "asset_turnover"), 4)],
                    ["权益乘数（总资产 / 股东权益）", _num(_d(dupont, "equity_multiplier"), 2)],
                    ["ROE（三因子乘积）", _pct(_d(dupont, "roe_computed"), 2)],
                    ["ROE（净利 / 权益 交叉验证）", _pct(_d(dupont, "roe_direct"), 2)],
                ]
                blocks.append(_table(["杜邦分析", "数值"], rows))

            yoy = _d(financial, "yoy_history") or []
            if yoy:
                rows = [
                    [
                        _s(x.get("period")),
                        _pct(x.get("revenue_growth_pct")) if x.get("revenue_growth_pct") is not None else "—",
                        _s(x.get("revenue_trend"), "—"),
                        _pct(x.get("net_profit_growth_pct")) if x.get("net_profit_growth_pct") is not None else "—",
                        _s(x.get("profit_trend"), "—"),
                    ]
                    for x in yoy
                    if isinstance(x, dict)
                ]
                blocks.append(_table(["周期", "营收同比", "营收趋势", "净利润同比", "利润趋势"], rows))

            commentary = _s(_d(financial, "commentary"))
            if commentary:
                blocks.append(_quote(commentary))
        if not blocks:
            blocks.append(_para(_PLACEHOLDER))
        return ReportSection(key="chapter_2", title="二、财务分析", blocks=blocks)

    # ── 第三章：行业与竞争力 ────────────────────────────────
    def _ch_industry(self, research: dict | None) -> ReportSection:
        blocks: list[ReportBlock] = []
        if research:
            industry = _s(_d(research, "industry_position"))
            if industry:
                blocks.append(_para(f"**行业地位**：{industry}"))
            advs: list[str] = []
            for item in _d(research, "advantages") or []:
                claim = _s(item.get("claim") if isinstance(item, dict) else None)
                if claim:
                    advs.append(claim)
            advs.extend(str(x) for x in (_d(research, "competitive_advantages") or []) if _s(x))
            if advs:
                blocks.append(_bullets(advs))
            biz_risks = [str(x) for x in (_d(research, "key_risks_business") or []) if _s(x)]
            if biz_risks:
                blocks.append(_bullets([f"**经营风险**：{x}" for x in biz_risks]))
        if not blocks:
            blocks.append(_para(_PLACEHOLDER))
        return ReportSection(key="chapter_3", title="三、行业与竞争力", blocks=blocks)

    # ── 第四章：舆情风向 ────────────────────────────────────
    def _ch_sentiment(self, sentiment: dict | None) -> ReportSection:
        blocks: list[ReportBlock] = []
        if sentiment:
            count = _d(sentiment, "searched_news_count")
            dist = _d(sentiment, "sentiment_distribution") or {}
            if count is not None or dist:
                blocks.append(
                    _para(
                        f"共抓取 {_s(count, '0')} 条相关新闻，"
                        f"情感分布：看多 {dist.get('positive', 0)}"
                        f" / 看空 {dist.get('negative', 0)} / 中立 {dist.get('neutral', 0)}。"
                    )
                )
            if dist:
                rows = [
                    ["看多（positive）", str(dist.get("positive", 0))],
                    ["看空（negative）", str(dist.get("negative", 0))],
                    ["中立（neutral）", str(dist.get("neutral", 0))],
                ]
                blocks.append(_table(["情感倾向", "新闻数"], rows))
            # 热点主题（兼容占位形态 hot_topics）
            topics = _d(sentiment, "topics") or []
            topic_items = [
                f"**{_s(t.get('label'))}**（{_s(t.get('news_count'), '0')} 条，关键词 {_s('、'.join(t.get('keywords') or []))}）"
                for t in topics
                if isinstance(t, dict) and _s(t.get("label"))
            ]
            if not topic_items and _d(sentiment, "hot_topics"):
                topic_items = [f"**{_s(h)}**" for h in _d(sentiment, "hot_topics")]
            if topic_items:
                blocks.append(_bullets(topic_items))
            summary = _s(_d(sentiment, "summary"))
            if summary:
                blocks.append(_quote(summary))
        if not blocks:
            blocks.append(_para(_PLACEHOLDER))
        return ReportSection(key="chapter_4", title="四、舆情风向", blocks=blocks)

    # ── 第五章：风险评估 ────────────────────────────────────
    def _ch_risk(self, risk: dict | None) -> ReportSection:
        blocks: list[ReportBlock] = []
        if risk:
            level = _risk_cn(_d(risk, "overall_risk_level"))
            score = _d(risk, "overall_score")
            score_txt = _num(score, 3) if score is not None else "—"
            blocks.append(_para(f"综合风险等级：**{level}**（评分 {score_txt} / 1.0）"))

            dims = _d(risk, "dimensions") or []
            if dims:
                rows = [
                    [
                        _s(d.get("dimension")),
                        _num(d.get("score"), 3) if d.get("score") is not None else "—",
                        _s(d.get("reasoning"), "—"),
                        "；".join(str(e) for e in (d.get("evidence") or []))[:200] or "—",
                    ]
                    for d in dims
                    if isinstance(d, dict)
                ]
                blocks.append(_table(["风险维度", "评分(0-1)", "推理", "证据"], rows))

            key_risks = [str(x) for x in (_d(risk, "key_risks") or []) if _s(x)]
            if key_risks:
                blocks.append(_bullets([f"**关键风险**：{x}" for x in key_risks]))

            chain = _s(_d(risk, "reasoning_chain"))
            if chain:
                blocks.append(_quote(f"推理链条：{chain}"))
            summary = _s(_d(risk, "risk_summary"))
            if summary:
                blocks.append(_quote(summary))
        if not blocks:
            blocks.append(_para(_PLACEHOLDER))
        return ReportSection(key="chapter_5", title="五、风险评估", blocks=blocks)

    # ── 第六章：投资建议（纯规则，恒含免责声明）────────────
    def _ch_advice(
        self, risk: dict | None, financial: dict | None, sentiment: dict | None
    ) -> ReportSection:
        blocks: list[ReportBlock] = []

        level = _s(_d(risk, "overall_risk_level")).lower()
        if level == "high":
            advice = "综合风险较高，建议谨慎对待，暂缓加仓并持续跟踪风险信号演变。"
        elif level == "medium":
            advice = "存在一定风险因素，建议保持关注，结合后续数据验证后再作决策。"
        elif level == "low":
            advice = "综合风险较低，基本面与舆情信号相对稳健，可作为重点研究对象。"
        else:
            advice = "风险评估数据不足，建议补充舆情与财务数据后再作判断。"
        blocks.append(_para(f"**投资建议**：{advice}"))

        watch: list[str] = []
        key_risks = [str(x) for x in (_d(risk, "key_risks") or []) if _s(x)]
        watch.extend(key_risks[:3])
        for dim in _d(risk, "dimensions") or []:
            if isinstance(dim, dict) and dim.get("score") is not None and float(dim["score"]) >= 0.7:
                watch.append(f"{_s(dim.get('dimension'), '该')}维度风险评分偏高（{_num(dim['score'], 3)}）")
        dist = _d(sentiment, "sentiment_distribution") or {}
        pos, neg, neu = (
            int(dist.get("positive", 0)),
            int(dist.get("negative", 0)),
            int(dist.get("neutral", 0)),
        )
        total = pos + neg + neu
        if total > 0 and neg / total >= 0.4:
            watch.append(f"负面舆情占比偏高（{neg / total:.0%}），需警惕市场情绪拖累")
        if not watch:
            watch.append("暂未识别显著风险关注点")
        blocks.append(_bullets([f"**关注点**：{x}" for x in watch]))

        blocks.append(
            _para(
                "**免责声明**：本报告由 FinaceAgent 多 Agent 系统自动生成，仅供研究参考，"
                "不构成任何投资建议。投资者据此操作，风险自担。"
            )
        )
        return ReportSection(key="chapter_6", title="六、投资建议与风险提示", blocks=blocks)
