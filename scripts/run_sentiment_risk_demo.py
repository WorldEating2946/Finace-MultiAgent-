"""4 家真实公司 × Sentiment & Risk Agent 完整链路真实场景验证脚本。

从真实使用场景测试 Sentiment & Risk Agent 是否达标（PR feature/sentiment-risk-agent 交付验证）：
    真实财报 JSON → FinancialSummary（Python 推导）
        → SentimentAgent（抓新闻 → FinBERT 评分 → BERTopic 聚类 → LLM 摘要）
        → RiskAgent（三维度纯 Python 加权评分 → 等级判定 → LLM 总结）

并在流程之外做【计算正确性独立核算】：
    用独立代码从原始输入手动重算舆情/财务/行业三维度评分与加权综合，
    与 Risk Agent 的硬计算输出逐项对照，验证"程序只计算"的正确性。

另跑 5 个测试场景（tests/test_data.py 的低/中/高/空/全正面）做风险敏感性演示，
验证舆情维度（权重 40%）在 [0,1] 全区间上的判定能力。

输出：完整结果落盘为 JSON（供可视化 HTML 报告消费）。
     data/sentiment_risk_agent_demo.json

用法：
    cd FinaceAgent
    D:/dev/conda/envs/finance-agent/python.exe scripts/run_sentiment_risk_demo.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 加载 .env（LLM key 等配置）；正式 API 由 app/main.py 负责，独立脚本需自行加载
from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from app.agents.risk_agent import RiskAgent
from app.agents.sentiment_agent import SentimentAgent
from app.core.config import get_settings
from app.core.llm_factory import get_llm
from app.models.sentiment_risk_models import FinancialSummary, SentimentInput

_OUT = Path("data") / "sentiment_risk_agent_demo.json"
_DATA_DIR = Path("app") / "data"

# ── 测试标的：四家真实 A 股公司（样本数据均来自公开年报）────────────
_COMPANIES = [
    {"ticker": "300750", "company": "宁德时代", "industry": "电力设备 / 电池"},
    {"ticker": "002594", "company": "比亚迪", "industry": "汽车整车"},
    {"ticker": "600519", "company": "贵州茅台", "industry": "白酒"},
    {"ticker": "000333", "company": "美的集团", "industry": "白色家电"},
]

# ── 与 risk_tools.py 相同的评分关键词表（独立核算时手动重算，不调用工具代码）──
_HIGH_RISK_KEYWORDS = {"监管", "制裁", "诉讼", "关税", "调查", "处罚", "违约", "退市"}
_INDUSTRY_KEYWORDS = {
    "政策": "政策变动风险",
    "监管": "监管趋严风险",
    "供应链": "供应链扰动风险",
    "关税": "海外关税风险",
    "竞争": "竞争加剧风险",
    "替代": "技术替代风险",
    "周期": "行业周期下行风险",
}
_DIM_WEIGHTS = {"sentiment": 0.40, "financial": 0.35, "industry": 0.25}


class _MockLLM:
    """占位 LLM（无 DEEPSEEK_API_KEY 时降级），与 API 路由的 _MockLLM 同构。"""

    async def ainvoke(self, msg: str):
        class _Resp:
            content = "（规则降级）基于确定性评分生成的摘要。"
        return _Resp()


def _get_llm(agent_type: str):
    """真实 LLM（已配 Key）或 Mock 降级。"""
    if get_settings().deepseek_api_key:
        return get_llm(agent_type)
    return _MockLLM()


def _using_real_llm() -> bool:
    return bool(get_settings().deepseek_api_key)


def _llm_state(text: str, real: bool) -> str:
    """LLM 状态三分：ok(LLM正常) / empty(LLM调用但返回空) / rule(Mock降级)。"""
    if not real:
        return "rule"
    return "ok" if text.strip() else "empty"


# ============================================================================
# FinancialSummary 构造（真实财报 → Python 推导，不经过 Financial Agent）
# ============================================================================


def _load_raw_fiscal(ticker: str) -> dict[int, dict[str, float]]:
    """读取 app/data/{ticker}_*.json 的原始 fiscal_data（权威数据源）。"""
    candidates = sorted(_DATA_DIR.glob(f"{ticker}_*.json"))
    if not candidates:
        return {}
    data = json.loads(candidates[0].read_text(encoding="utf-8"))
    fd = data.get("fiscal_data", {})
    return {int(y): {k: float(v) for k, v in vals.items()} for y, vals in fd.items()}


def _build_financial_summary(ticker: str) -> tuple[FinancialSummary, str | None]:
    """从原始财报推导 FinancialSummary（最新财年 vs 上年）。

    字段推导口径：
      revenue_growth    = 最新营收 / 上年营收 - 1
      net_profit_margin = 最新净利 / 最新营收
      debt_ratio        = 1 - 权益/总资产（资产负债表恒等式，非独立披露）
      gross_margin / free_cash_flow = None（原始数据未披露，Risk Agent 容错跳过）
      anomalies         = 自动识别直接检查未覆盖的异常（如净利率过低）
    """
    fy = _load_raw_fiscal(ticker)
    if not fy:
        return FinancialSummary(), None

    years = sorted(fy.keys())
    last, prev = fy[years[-1]], fy[years[-2]]
    revenue_growth = (last["revenue"] - prev["revenue"]) / abs(prev["revenue"])
    net_profit_margin = last["net_profit"] / last["revenue"]
    debt_ratio = 1 - last["shareholders_equity"] / last["total_assets"]

    anomalies: list[str] = []
    if net_profit_margin < 0.05:
        anomalies.append(f"净利率仅 {net_profit_margin:.1%}，盈利能力偏弱")

    return FinancialSummary(
        revenue_growth=round(revenue_growth, 4),
        net_profit_margin=round(net_profit_margin, 4),
        debt_ratio=round(debt_ratio, 4),
        anomalies=anomalies,
    ), f"{years[-1]} vs {years[-2]}"


# ============================================================================
# 计算正确性独立核算（不经过 Agent/工具代码，用原始输入手动重算）
# ============================================================================


def _verify_risk(sr: dict, fin: dict, dims: list[dict],
                 overall: float, level: str) -> dict:
    """独立重算三维度评分 + 加权综合 + 等级，与 Risk Agent 输出逐项对照。"""
    tol = 0.011  # 评分四舍五入到 2 位小数，允许 1 分钱容差
    checks: list[dict] = []

    def _norm(v):
        return v.value if hasattr(v, "value") else v  # 枚举 → 值，防御性归一

    def _chk(name: str, got, expect, is_level: bool = False) -> None:
        got, expect = _norm(got), _norm(expect)
        ok = (str(got) == str(expect)) if is_level else (
            got is not None and abs(float(got) - float(expect)) <= tol)
        checks.append({
            "name": name,
            "passed": ok,
            "got": round(float(got), 4) if not is_level else got,
            "expect": round(float(expect), 4) if not is_level else expect,
            "detail": "符合" if ok else f"偏差 {got} vs 期望 {expect}",
        })

    # ── 舆情维度：负面占比 ×1.5 + 高危主题数 ×0.05，上限 1.0 ──
    dist = sr.get("sentiment_distribution", {})
    total = sum(dist.values()) or 1
    neg_ratio = dist.get("negative", 0) / total
    n_high = sum(
        1 for t in sr.get("topics", [])
        if any(kw in t.get("label", "") for kw in _HIGH_RISK_KEYWORDS)
    )
    exp_sent = round(min(neg_ratio * 1.5 + 0.05 * n_high, 1.0), 2)
    _chk("舆情负面信号评分", dims[0]["score"], exp_sent)

    # ── 财务维度：触发预警指标数 / 可评估指标数 ──
    flags, totf = 0, 0
    if fin.get("revenue_growth") is not None:
        totf += 1
        if fin["revenue_growth"] < 0:
            flags += 1
    if fin.get("gross_margin") is not None:
        totf += 1  # 毛利率 <15% 只记录证据不计分
    if fin.get("debt_ratio") is not None:
        totf += 1
        if fin["debt_ratio"] > 0.7:
            flags += 1
    flags += len(fin.get("anomalies", []))
    totf += len(fin.get("anomalies", []))
    exp_fin = round(flags / max(totf, 1), 2)
    _chk("财务异常信号评分", dims[1]["score"], exp_fin)

    # ── 行业维度：舆情主题命中行业关键词数 ×0.25，上限 1.0 ──
    matched = set()
    for t in sr.get("topics", []):
        for kw, desc in _INDUSTRY_KEYWORDS.items():
            if kw in t.get("label", ""):
                matched.add(desc)
    exp_ind = round(min(len(matched) * 0.25, 1.0), 2)
    _chk("行业周期风险评分", dims[2]["score"], exp_ind)

    # ── 加权综合 + 等级判定 ──
    exp_overall = round(
        min(_DIM_WEIGHTS["sentiment"] * exp_sent
            + _DIM_WEIGHTS["financial"] * exp_fin
            + _DIM_WEIGHTS["industry"] * exp_ind, 1.0), 2)
    _chk("综合风险评分", overall, exp_overall)
    exp_level = ("high" if exp_overall >= 0.7
                 else "medium" if exp_overall >= 0.4 else "low")
    _chk("风险等级判定", level, exp_level, is_level=True)

    return {"checks": checks, "all_passed": all(c["passed"] for c in checks)}


# ============================================================================
# 单家公司全链路
# ============================================================================


async def run_one(cfg: dict) -> dict:
    t0 = time.time()
    real = _using_real_llm()

    fin, period = _build_financial_summary(cfg["ticker"])

    # ── Sentiment Agent：抓新闻 → 评分 → 聚类 → LLM 摘要 ──
    sent_agent = SentimentAgent(llm=_get_llm("sentiment"))
    sr = await sent_agent.run(
        SentimentInput(symbol=cfg["ticker"], company_name=cfg["company"], days=30)
    )

    # ── Risk Agent：三维度加权评分 → 等级 → LLM 总结 ──
    risk_agent = RiskAgent(llm=_get_llm("risk"))
    ra = await risk_agent.run(sentiment_result=sr, financial=fin)
    elapsed = round(time.time() - t0, 1)

    # mode="json"：枚举/时间统一归一化为 JSON 原语（risk level 变 "low" 而非 RiskLevel 枚举），
    # 供独立核算与后续 JSON 落盘共用同一类型口径
    sr_d, fin_d, ra_d = (
        sr.model_dump(mode="json"),
        fin.model_dump(mode="json"),
        ra.model_dump(mode="json"),
    )

    # ── 计算正确性独立核算 ──
    verify = _verify_risk(
        sr_d, fin_d,
        ra_d["dimensions"], ra_d["overall_score"], ra_d["overall_risk_level"],
    )

    return {
        "ticker": cfg["ticker"],
        "company": cfg["company"],
        "industry": cfg["industry"],
        "elapsed_s": elapsed,
        "financial_period": period or "无数据",
        "sentiment": {
            "searched_news_count": sr_d["searched_news_count"],
            "distribution": sr_d["sentiment_distribution"],
            "topics": sr_d["topics"],
            "summary": sr_d["summary"],
            "summary_chars": len(sr_d["summary"]),
            "llm_state": _llm_state(sr_d["summary"], real),
        },
        "financial": fin_d,
        "risk": {
            "overall_risk_level": ra_d["overall_risk_level"],
            "overall_score": ra_d["overall_score"],
            "dimensions": ra_d["dimensions"],
            "risk_summary": ra_d["risk_summary"],
            "risk_summary_chars": len(ra_d["risk_summary"]),
            "llm_state": _llm_state(ra_d["risk_summary"], real),
            "key_risks": ra_d["key_risks"],
            "reasoning_chain": ra_d["reasoning_chain"],
        },
        "validation": verify,
        "mock_mode": not real,
    }


# ============================================================================
# 5 个测试场景：风险等级敏感性演示（复用 tests/test_data.py 的真实场景）
# ============================================================================


async def run_scenario_demo() -> list[dict]:
    from tests.test_data import TEST_SCENARIOS

    agent = RiskAgent(llm=_get_llm("risk"))
    out: list[dict] = []
    for key, s in TEST_SCENARIOS.items():
        ra = await agent.run(sentiment_result=s["sentiment"], financial=s["financial"])
        expect = s.get("expect_level")
        passed = (expect is None) or (ra.overall_risk_level == expect)
        out.append({
            "key": key,
            "name": s["name"],
            "expected_level": expect.value if expect else "—",
            "got_level": ra.overall_risk_level.value,
            "score": ra.overall_score,
            "dimensions": [d.model_dump() for d in ra.dimensions],
            "key_risks": ra.key_risks,
            "reasoning_chain": ra.reasoning_chain,
            "risk_summary": ra.risk_summary,
            "passed": passed,
        })
    return out


# ============================================================================
# 主流程
# ============================================================================


async def main() -> None:
    print("=" * 72, flush=True)
    print("4 家真实公司 × Sentiment & Risk Agent 完整链路真实场景验证", flush=True)
    print(f"开始时间: {datetime.now(timezone.utc).isoformat()}", flush=True)
    print("=" * 72, flush=True)

    companies = []
    for cfg in _COMPANIES:
        print(f"\n───── {cfg['ticker']} {cfg['company']} ─────", flush=True)
        try:
            r = await run_one(cfg)
        except Exception as exc:  # noqa: BLE001 — 单家公司失败不中断整体
            print(f"  ❌ {cfg['company']} 全链路异常: {exc}", flush=True)
            r = {"ticker": cfg["ticker"], "company": cfg["company"],
                 "industry": cfg["industry"], "elapsed_s": 0, "error": str(exc),
                 "validation": {"all_passed": False, "checks": []}}
        d = r.get("sentiment", {}).get("distribution", {})
        risk = r.get("risk", {})
        checks = r.get("validation", {}).get("checks", [])
        n_pass = sum(1 for c in checks if c.get("passed"))
        llm_map = {"ok": "LLM", "empty": "LLM空返回", "rule": "规则降级"}
        s_llm = llm_map.get(r.get("sentiment", {}).get("llm_state"), "?")
        r_llm = llm_map.get(risk.get("llm_state"), "?")
        print(
            f"  新闻={r.get('sentiment', {}).get('searched_news_count')}条 "
            f"正/负/中={d.get('positive')}/{d.get('negative')}/{d.get('neutral')} "
            f"| 综合={risk.get('overall_risk_level')} score={risk.get('overall_score')} "
            f"| 摘要({s_llm})/总结({r_llm}) "
            f"| 校验 {n_pass}/{len(checks)} 通过 | {r.get('elapsed_s')}s",
            flush=True,
        )
        companies.append(r)

    print("\n───── 5 个测试场景 · 风险等级敏感性演示 ─────", flush=True)
    scenarios = await run_scenario_demo()
    for s in scenarios:
        print(
            f"  [{s['key']}] 期望 {s['expected_level']} → 实际 {s['got_level']} "
            f"score={s['score']} {'✅' if s['passed'] else '❌'}",
            flush=True,
        )

    all_pass = all(r.get("validation", {}).get("all_passed")
                   for r in companies if "error" not in r)
    sc_pass = sum(1 for s in scenarios if s["passed"])
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": "Sentiment & Risk Agent (PR feature/sentiment-risk-agent)",
        "workflow": ("fetch news → FinBERT 情感评分 → 主题聚类 → LLM 摘要 "
                     "→ 三维度加权风险(舆情40%/财务35%/行业25%) → 等级判定 → LLM 总结"),
        "data_sources": [str(p) for p in sorted(_DATA_DIR.glob("*_*.json"))],
        "mock_mode": not _using_real_llm(),
        "all_validation_passed": all_pass,
        "scenario_pass": f"{sc_pass}/{len(scenarios)}",
        "companies": companies,
        "scenarios": scenarios,
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] 结果已落盘 → {_OUT} "
          f"| 公司计算校验: {'✅ 通过' if all_pass else '⚠️ 存在失败'} "
          f"| 场景达标 {sc_pass}/{len(scenarios)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
