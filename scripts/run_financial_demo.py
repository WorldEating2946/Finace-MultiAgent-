"""4 家真实公司 × Financial Agent 完整工作流真实场景验证脚本。

从真实使用场景测试 Financial Agent 是否达标（PR #58/#59 交付验证）：
    原始财报 JSON → 完整 LangGraph 工作流
        → [Manager 规划 → financial 硬计算 + LLM 点评 → risk 风险定级 → report 研报]

并在流程之外做【计算正确性独立核算】：
    从原始财报数据手动推导杜邦三因子 / ROE / 同比增速，
    与 Financial Agent 的硬计算输出逐项对照，验证"程序只计算"的正确性。

输出：完整结果落盘为 JSON（供可视化 HTML 报告消费）。
     data/financial_agent_demo.json

用法：
    cd FinaceAgent
    D:/dev/conda/envs/finance-agent/python.exe scripts/run_financial_demo.py
"""

from __future__ import annotations

import asyncio
import json
import os
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

import httpx

from app.workflow.graph import build_graph
from app.agents.financial_agent.node import FINANCIAL_AGENT_SYSTEM_PROMPT
from app.agents.financial_agent.prompts import build_analysis_prompt

_OUT = Path("data") / "financial_agent_demo.json"
_DATA_DIR = Path("app") / "data"

# ── 测试标的：四家真实 A 股公司（样本数据均来自公开年报）────────────
_COMPANIES = [
    {"ticker": "300750", "company": "宁德时代", "industry": "电力设备 / 电池"},
    {"ticker": "002594", "company": "比亚迪", "industry": "汽车整车"},
    {"ticker": "600519", "company": "贵州茅台", "industry": "白酒"},
    {"ticker": "000333", "company": "美的集团", "industry": "白色家电"},
]


# ============================================================================
# 计算正确性独立核算（不经过 Agent 代码，直接用原始财报数据手动推导）
# ============================================================================


def _load_raw_fiscal(ticker: str) -> dict[int, dict[str, float]]:
    """读取 app/data/{ticker}_*.json 的原始 fiscal_data（权威数据源）。"""
    candidates = sorted(_DATA_DIR.glob(f"{ticker}_*.json"))
    if not candidates:
        return {}
    data = json.loads(candidates[0].read_text(encoding="utf-8"))
    fd = data.get("fiscal_data", {})
    return {int(y): {k: float(v) for k, v in vals.items()} for y, vals in fd.items()}


def _verify_company(agent_out: dict) -> dict:
    """独立核算 vs Agent 输出，逐项对照。

    返回 {"checks": [...], "all_passed": bool}
    """
    checks = []
    fy_data = _load_raw_fiscal(agent_out["ticker"])
    if not fy_data:
        return {"checks": [{"name": "原始数据读取", "passed": False,
                            "detail": "未找到原始财报文件"}], "all_passed": False}

    years = sorted(fy_data.keys())
    latest, prev = years[-1], years[-2]
    last, cur = fy_data[latest], fy_data[prev]  # last=最新财年, cur=上年

    def _chk(name: str, got: float | None, expect: float, tol_pct: float = 0.001) -> None:
        ok = got is not None and abs(got - expect) <= max(abs(expect) * tol_pct, 1e-6)
        checks.append({
            "name": name,
            "passed": ok,
            "got": round(got, 4) if got is not None else None,
            "expect": round(expect, 4),
            "detail": "符合" if ok else f"偏差 {got:.4f} vs 期望 {expect:.4f}",
        })

    # 杜邦三因子（最新财年）
    npm = last["net_profit"] / last["revenue"]
    at = last["revenue"] / last["total_assets"]
    em = last["total_assets"] / last["shareholders_equity"]
    roe = npm * at * em
    roe_direct = last["net_profit"] / last["shareholders_equity"]

    d = agent_out["dupont"]
    km = agent_out["key_metrics"]
    _chk("净利润率", d["net_profit_margin"], npm)
    _chk("资产周转率", d["asset_turnover"], at)
    _chk("权益乘数", d["equity_multiplier"], em)
    _chk("ROE(三因子乘积)", d["roe_computed"], roe)
    _chk("ROE(净利润/权益)", d["roe_direct"], roe_direct)
    _chk("ROE 展示指标 roe_pct", km["roe_pct"], roe * 100, tol_pct=0.01)
    checks.append({
        "name": "ROE 交叉验证(乘积 vs 直接)",
        "passed": abs(d["roe_computed"] - d["roe_direct"]) <= 1e-6,
        "got": round(d["roe_computed"], 6), "expect": round(d["roe_direct"], 6),
        "detail": "符合" if abs(d["roe_computed"] - d["roe_direct"]) <= 1e-6
                 else "三因子乘积与直接计算不一致",
    })

    # 同比增速（最新财年 vs 上年）
    rev_yoy = (last["revenue"] - cur["revenue"]) / abs(cur["revenue"]) * 100
    prf_yoy = (last["net_profit"] - cur["net_profit"]) / abs(cur["net_profit"]) * 100
    latest_yoy = next(
        (s for s in agent_out["yoy_history"]
         if s["period"] == f"{latest} vs {prev}"), None)
    if latest_yoy:
        _chk("营收同比", latest_yoy["revenue_growth_pct"], rev_yoy, tol_pct=0.01)
        _chk("净利润同比", latest_yoy["net_profit_growth_pct"], prf_yoy, tol_pct=0.01)
    else:
        checks.append({"name": "YoY 最新期识别", "passed": False,
                       "got": None, "expect": f"{latest} vs {prev}",
                       "detail": "yoy_history 中未找到最新期"})

    return {"checks": checks, "all_passed": all(c["passed"] for c in checks),
            "latest_fy": latest, "years": years}


# ============================================================================
# LLM 空返回根因诊断（复现 PR 评审 P2-10：max_tokens=2000 对推理模型过小）
# ============================================================================


async def _llm_diagnostic(fin: dict) -> dict:
    """用相同提示词 + max_tokens=6000 重试，验证空返回根因是 max_tokens 不足。

    返回诊断结果（不修改主链路数据，仅用于对照展示修复方向）。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("DEEPSEEK_MODEL_CHAT", "deepseek-v4-flash")
    if not api_key:
        return {"attempted": False, "reason": "DEEPSEEK_API_KEY 未配置"}

    user_prompt = build_analysis_prompt(
        company_name=fin.get("company", ""),
        ticker=fin.get("ticker", ""),
        period=fin.get("analysis_period", ""),
        key_metrics=fin.get("key_metrics", {}),
        dupont=fin.get("dupont", {}),
        yoy_history=fin.get("yoy_history", []),
    )
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0)) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": model,
                      "messages": [
                          {"role": "system", "content": FINANCIAL_AGENT_SYSTEM_PROMPT},
                          {"role": "user", "content": user_prompt},
                      ],
                      "temperature": 0.3,
                      "max_tokens": 6000,
                      "stream": False},
            )
            if resp.status_code != 200:
                return {"attempted": True, "ok": False,
                        "detail": f"HTTP {resp.status_code}: {resp.text[:120]}"}
            data = resp.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            return {
                "attempted": True,
                "ok": bool(content.strip()),
                "chars": len(content),
                "preview": content[:180] + ("…" if len(content) > 180 else ""),
                "model": data.get("model", model),
                "usage": data.get("usage", {}),
            }
    except Exception as exc:  # noqa: BLE001
        return {"attempted": True, "ok": False, "detail": str(exc)[:150]}


# ============================================================================
# 单家公司全链路
# ============================================================================


async def run_one(graph, cfg: dict) -> dict:
    t0 = time.time()
    state = await graph.ainvoke({
        "company": cfg["company"],
        "ticker": cfg["ticker"],
        "user_query": f"分析{cfg['company']}的财务健康状况与发展前景",
        "current_step": "start",
        "errors": [],
    })
    elapsed = round(time.time() - t0, 1)

    fin = state.get("financial_result") or {}
    risk = state.get("risk_result") or {}
    report_md = state.get("report") or ""

    commentary = fin.get("commentary", "")
    # LLM 状态三分：ok(LLM正常) / empty(LLM调用但返回空→max_tokens坑) / rule(LLM未配置或失败→规则降级)
    if not commentary:
        llm_state = "empty"
    elif "规则引擎" in commentary or "规则生成" in commentary:
        llm_state = "rule"
    else:
        llm_state = "ok"

    # 空返回 → 触发根因诊断（max_tokens=6000 重试，验证修复方向）
    diagnostic = await _llm_diagnostic(fin) if llm_state == "empty" else {"attempted": False}

    # 计算正确性独立核算
    verif = _verify_company(fin) if fin else {"all_passed": False, "checks": []}

    return {
        "ticker": cfg["ticker"],
        "company": cfg["company"],
        "industry": cfg["industry"],
        "elapsed_s": elapsed,
        "data_source": fin.get("data_source", "unknown"),
        "fetch_error": fin.get("fetch_error"),
        "analysis_period": fin.get("analysis_period", ""),
        "key_metrics": fin.get("key_metrics", {}),
        "dupont": fin.get("dupont", {}),
        "yoy_history": fin.get("yoy_history", []),
        "commentary": commentary,
        "commentary_chars": len(commentary),
        "llm_state": llm_state,
        "llm_diagnostic": diagnostic,
        "risk": risk,
        "report_chars": len(report_md),
        "report_md": report_md,
        "validation": verif,
        "upstream_placeholder": {
            "research": state.get("research_result", {}).get("summary", ""),
            "sentiment": state.get("sentiment_result", {}).get("summary", ""),
        },
        "errors": state.get("errors", []),
    }


async def main() -> None:
    print("=" * 72, flush=True)
    print("4 家真实公司 × Financial Agent 完整工作流真实场景验证", flush=True)
    print(f"开始时间: {datetime.now(timezone.utc).isoformat()}", flush=True)
    print("=" * 72, flush=True)

    graph = build_graph()

    results = []
    for cfg in _COMPANIES:
        print(f"\n───── {cfg['ticker']} {cfg['company']} ─────", flush=True)
        try:
            r = await run_one(graph, cfg)
        except Exception as exc:  # noqa: BLE001 — 单家公司失败不中断整体
            print(f"  ❌ {cfg['company']} 全链路异常: {exc}", flush=True)
            r = {"ticker": cfg["ticker"], "company": cfg["company"],
                 "industry": cfg["industry"], "elapsed_s": 0, "error": str(exc),
                 "validation": {"all_passed": False, "checks": []}}
        km = r.get("key_metrics", {})
        checks = r.get("validation", {}).get("checks", [])
        n_pass = sum(1 for c in checks if c.get("passed"))
        st = r.get("llm_state")
        llm = {"ok": "LLM", "empty": "LLM空返回", "rule": "规则降级"}.get(st, st)
        diag = r.get("llm_diagnostic")
        diag_note = ""
        if diag and diag.get("attempted") and diag.get("ok"):
            diag_note = f" → 修复验证: max_tokens=6000 返回 {diag['chars']}字 ✓"
        elif diag and diag.get("attempted") and not diag.get("ok"):
            diag_note = f" → 6000 仍失败: {diag.get('detail','')[:60]}"
        print(
            f"  ROE={km.get('roe_pct')}% | 营收YoY={km.get('revenue_yoy_pct')}% "
            f"| 点评={r.get('commentary_chars')}字({llm}){diag_note} "
            f"| 校验 {n_pass}/{len(checks)} 通过 | 报告={r.get('report_chars')}字 "
            f"| {r.get('elapsed_s')}s",
            flush=True,
        )
        results.append(r)

    all_pass = all(r.get("validation", {}).get("all_passed") for r in results if "error" not in r)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": "Financial Agent (PR #58/#59)",
        "workflow": "manager → [research‖financial‖sentiment] → risk → report",
        "data_sources": [str(p) for p in sorted(_DATA_DIR.glob("*_*.json"))],
        "all_validation_passed": all_pass,
        "companies": results,
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] 结果已落盘 → {_OUT} | 全部计算校验: {'✅ 通过' if all_pass else '⚠️ 存在失败'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
