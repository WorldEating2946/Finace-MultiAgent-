"""
Financial Agent 测试脚本

运行方式:
    cd <项目根目录>
    conda activate finance-agent
    python test_financial_agent.py        # 直接运行
    pytest test_financial_agent.py -v     # 或 pytest

测试范围:
    1. FinancialCalculator 硬计算 — 同比增速、杜邦分析
    2. Pydantic 模型校验 — FinancialAgentInput/Output
    3. Prompt 构建 — build_analysis_prompt
    4. financial_analysis_node — 完整异步链路 (data fetch → calc → LLM)
    5. LangGraph Workflow — 全图编译 + ainvoke (5.1-5.3)
    6. 编排增强 — 真实 Agent 节点 + 健康检查重试环收敛

Author: 工藤
Date: 2026-08-05
"""

import asyncio
import os
import sys
import time
import traceback
from datetime import date
from pathlib import Path

import pytest

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# 工具
# ============================================================================

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================================
# Test 1: FinancialCalculator 硬计算
# ============================================================================
@pytest.mark.asyncio
async def test_calculator():
    section("Test 1: FinancialCalculator 硬计算")

    from app.core.schemas import (
        DuPontAnalysisInput,
        YoYBatchInput,
        YoYGrowthInput,
    )
    from app.quant_engine.calculator import FinancialCalculator

    calc = FinancialCalculator()

    # 1a. 同比增速 — 正常情况
    result = calc.calculate_yoy_growth(
        YoYGrowthInput(current_value=1200.0, previous_value=1000.0, metric_name="营收")
    )
    check("1a. 同比增速计算", result.growth_rate_pct == 20.0,
          f"增速={result.growth_rate_pct}%, 趋势={result.trend}")
    check("1a. 绝对变动额", result.absolute_change == 200.0)

    # 1b. 同比增速 — 下降
    result = calc.calculate_yoy_growth(
        YoYGrowthInput(current_value=800.0, previous_value=1000.0, metric_name="净利润")
    )
    check("1b. 同比下滑", result.trend == "下降",
          f"增速={result.growth_rate_pct}%")

    # 1c. 同比增速 — 上年为 0 (除零保护)
    result = calc.calculate_yoy_growth(
        YoYGrowthInput(current_value=500.0, previous_value=0.0, metric_name="新业务")
    )
    check("1c. 除零保护", abs(result.growth_rate_pct) == float("inf"),
          "增速=inf (上年为0)")

    # 1d. 批量同比
    batch_out = calc.calculate_yoy_batch(
        YoYBatchInput(items=[
            YoYGrowthInput(current_value=1500, previous_value=1200, metric_name="营收"),
            YoYGrowthInput(current_value=300, previous_value=250, metric_name="净利润"),
        ])
    )
    check("1d. 批量同比", len(batch_out.results) == 2,
          batch_out.summary[:80])

    # 1e. 杜邦分析
    dupont = calc.calculate_dupont_analysis(
        DuPontAnalysisInput(
            net_income=100_000_000,
            revenue=1_000_000_000,
            total_assets=2_000_000_000,
            shareholders_equity=800_000_000,
            company_name="测试公司",
            period="2024FY",
        )
    )
    check("1e. 杜邦 ROE", abs(dupont.roe_pct - 12.5) < 0.01,
          f"ROE={dupont.roe_pct}%")
    check("1e. 杜邦验证", abs(dupont.roe - dupont.roe_check) < 1e-9,
          f"三因子乘积={dupont.roe:.6f}, 直接计算={dupont.roe_check:.6f}")
    check("1e. 杜邦解读", len(dupont.interpretation) > 0,
          dupont.interpretation[:100])

    # 1f. CAGR
    cagr = calc.compute_cagr(100.0, 200.0, 5)
    check("1f. CAGR", abs(cagr - 0.1487) < 0.001,
          f"CAGR(100→200, 5年)={cagr*100:.2f}%")


# ============================================================================
# Test 2: Pydantic 模型校验
# ============================================================================
@pytest.mark.asyncio
async def test_schemas():
    section("Test 2: Pydantic 模型校验")

    from app.agents.financial_agent.schemas import (
        DuPontBreakdown,
        FinancialAgentInput,
        FinancialAgentOutput,
        KeyMetrics,
        YoYSummary,
    )

    # 2a. FinancialAgentInput 正常
    inp = FinancialAgentInput(
        ticker="300750",
        company_name="宁德时代",
        start_date=date(2020, 1, 1),
        end_date=date(2025, 12, 31),
    )
    check("2a. AgentInput 构造", inp.ticker == "300750")

    # 2b. FinancialAgentInput 校验 — end < start
    try:
        FinancialAgentInput(
            ticker="300750",
            start_date=date(2025, 1, 1),
            end_date=date(2020, 1, 1),
        )
        check("2b. 日期校验应抛异常", False)
    except Exception:
        check("2b. 日期校验 (end<start)", True)

    # 2c. FinancialAgentOutput 完整构造
    output = FinancialAgentOutput(
        company="宁德时代",
        ticker="300750",
        analysis_period="2020~2025",
        key_metrics=KeyMetrics(
            roe_pct=17.95,
            net_profit_margin_pct=9.03,
            revenue_yoy_pct=15.13,
            net_profit_yoy_pct=43.64,
            equity_multiplier=2.5,
            asset_turnover=0.7955,
        ),
        dupont=DuPontBreakdown(
            net_profit_margin=0.0903,
            asset_turnover=0.7955,
            equity_multiplier=2.5,
            roe_computed=0.1795,
            roe_direct=0.1795,
        ),
        yoy_history=[
            YoYSummary(
                period="2024 vs 2023",
                revenue_growth_pct=15.13,
                net_profit_growth_pct=43.64,
                revenue_trend="上升",
                profit_trend="上升",
            ),
        ],
        commentary="公司盈利能力优秀，ROE 处于行业领先水平。",
    )
    check("2c. AgentOutput 构造", output.company == "宁德时代")
    check("2c. model_dump 可序列化", isinstance(output.model_dump(), dict))
    check("2c. model_dump_json", len(output.model_dump_json()) > 100)

    # 2d. 空 commentary 也允许
    output2 = FinancialAgentOutput(
        company="测试",
        ticker="000000",
        analysis_period="2024",
        key_metrics=KeyMetrics(
            roe_pct=10.0,
            net_profit_margin_pct=5.0,
            equity_multiplier=2.0,
            asset_turnover=1.0,
        ),
        dupont=DuPontBreakdown(
            net_profit_margin=0.05,
            asset_turnover=1.0,
            equity_multiplier=2.0,
            roe_computed=0.10,
            roe_direct=0.10,
        ),
    )
    check("2d. 最小构造 (无commentary)", output2.commentary == "")


# ============================================================================
# Test 3: Prompt 构建
# ============================================================================
@pytest.mark.asyncio
async def test_prompts():
    section("Test 3: Prompt 构建")

    from app.agents.financial_agent.prompts import (
        FINANCIAL_AGENT_SYSTEM_PROMPT,
        build_analysis_prompt,
    )

    # 3a. System Prompt 完整性
    check("3a. SystemPrompt 有 CFO 角色", "资深 CFO" in FINANCIAL_AGENT_SYSTEM_PROMPT)
    check("3a. SystemPrompt 禁止编造", "严禁编造" in FINANCIAL_AGENT_SYSTEM_PROMPT)

    # 3b. build_analysis_prompt — 正常数据
    prompt = build_analysis_prompt(
        company_name="测试公司",
        ticker="000001",
        period="2020-2024",
        key_metrics={
            "roe_pct": 18.5,
            "net_profit_margin_pct": 12.0,
            "revenue_yoy_pct": 20.0,
            "net_profit_yoy_pct": 25.0,
            "equity_multiplier": 2.3,
            "asset_turnover": 0.85,
        },
        dupont={
            "net_profit_margin": 0.12,
            "asset_turnover": 0.85,
            "equity_multiplier": 2.3,
            "roe_computed": 0.2346,
            "roe_direct": 0.2346,
        },
        yoy_history=[
            {
                "period": "2024 vs 2023",
                "revenue_growth_pct": 20.0,
                "revenue_trend": "上升",
                "net_profit_growth_pct": 25.0,
                "profit_trend": "上升",
            },
        ],
    )
    check("3b. Prompt 包含公司名", "测试公司" in prompt)
    check("3b. Prompt 包含 ROE", "18.50%" in prompt)
    check("3b. Prompt 包含增速", "+20.00%" in prompt)
    check("3b. Prompt 包含杜邦", "净利润率" in prompt)
    check("3b. Prompt 不编造", "绝不编造" in prompt)

    # 3c. build_analysis_prompt — None 值处理
    prompt2 = build_analysis_prompt(
        company_name="测试",
        ticker="000002",
        period="2024",
        key_metrics={
            "roe_pct": None,
            "net_profit_margin_pct": None,
            "revenue_yoy_pct": None,
            "net_profit_yoy_pct": None,
            "equity_multiplier": None,
            "asset_turnover": None,
        },
        dupont={},
        yoy_history=[],
    )
    check("3c. None 值 → N/A", "N/A" in prompt2)


# ============================================================================
# Test 4: financial_analysis_node (独立测试)
# ============================================================================
@pytest.mark.asyncio
async def test_node():
    section("Test 4: financial_analysis_node 完整链路")

    from app.agents.financial_agent.node import financial_analysis_node

    # 设置本地数据文件路径，模拟有数据可用的场景
    fixture_path = str(PROJECT_ROOT / "app" / "data" / "fixtures" / "sample_company.json")
    os.environ["FINANCE_AGENT_DATA_FILE"] = fixture_path

    # 用样本数据 (ticker="" 会触发本地文件降级)
    state = {
        "company": "宁德时代",
        "ticker": "",
        "user_query": "分析宁德时代",
        "current_step": "start",
        "errors": [],
    }

    t0 = time.perf_counter()
    result = await financial_analysis_node(state)
    elapsed = (time.perf_counter() - t0) * 1000

    check("4a. 返回 dict", isinstance(result, dict))
    check("4b. 有 financial_result", "financial_result" in result)

    fr = result.get("financial_result", {})

    # 顶层字段
    check("4c. company", fr.get("company") == "宁德时代")
    check("4d. data_source", fr.get("data_source") == "api")  # 本地文件加载视为 api

    # key_metrics
    km = fr.get("key_metrics", {})
    check("4e. ROE > 0", km.get("roe_pct", 0) > 0,
          f"ROE={km.get('roe_pct', 'N/A')}%")
    check("4f. 净利率 > 0", km.get("net_profit_margin_pct", 0) > 0,
          f"净利率={km.get('net_profit_margin_pct', 'N/A')}%")
    check("4g. 营收增速有值", km.get("revenue_yoy_pct") is not None,
          f"营收同比={km.get('revenue_yoy_pct', 'N/A')}")

    # dupont
    dp = fr.get("dupont", {})
    check("4h. 杜邦 ROE", dp.get("roe_computed", 0) > 0,
          f"ROE={dp.get('roe_computed', 0)*100:.2f}%")
    check("4i. 杜邦验证", abs(dp.get("roe_computed", 0) - dp.get("roe_direct", 0)) < 1e-9)

    # yoy_history
    yoy = fr.get("yoy_history", [])
    check("4j. 有历年数据", len(yoy) >= 4,
          f"{len(yoy)} 条记录")
    if len(yoy) > 1:
        check("4k. 最新年有增速", yoy[-1].get("revenue_growth_pct") is not None)

    # commentary (LLM 不可用时用规则降级)
    commentary = fr.get("commentary", "")
    check("4l. 有点评文本", len(commentary) > 0,
          f"点评长度={len(commentary)} 字")

    print(f"  ⏱ 耗时: {elapsed:.0f}ms")


# ============================================================================
# Test 5: LangGraph Workflow (依赖 langgraph 库)
# ============================================================================
@pytest.mark.asyncio
async def test_graph():
    section("Test 5: LangGraph Workflow")

    try:
        from app.workflow.graph import build_graph
        check("5a. 导入 build_graph", True)
    except Exception as e:
        check("5a. 导入 build_graph", False, str(e)[:100])
        print("  ⚠️ 跳过后续图测试 (导入失败)")
        return

    # 5b. 编译图
    try:
        graph = build_graph()
        check("5b. 编译 StateGraph", graph is not None)
    except Exception as e:
        check("5b. 编译 StateGraph", False, str(e)[:100])
        return

    # 5c. ainvoke 完整运行
    try:
        # 确保本地数据文件可用（可能已被 test_node 设置，此处做防御）
        fixture_path = str(PROJECT_ROOT / "app" / "data" / "fixtures" / "sample_company.json")
        os.environ.setdefault("FINANCE_AGENT_DATA_FILE", fixture_path)

        initial_state = {
            "company": "宁德时代",
            "ticker": "",
            "user_query": "分析宁德时代的未来发展情况与风险",
            "current_step": "start",
            "errors": [],
        }
        t0 = time.perf_counter()
        final_state = await graph.ainvoke(initial_state)
        elapsed = (time.perf_counter() - t0) * 1000

        check("5c. ainvoke 完成", True, f"耗时={elapsed:.0f}ms")

        # 检查各节点输出
        check("5d. manager_plan", final_state.get("manager_plan") is not None)
        check("5e. research_result", final_state.get("research_result") is not None)
        check("5f. financial_result", final_state.get("financial_result") is not None)

        fr = final_state.get("financial_result", {})
        km = fr.get("key_metrics", {})
        check("5g. financial ROE", km.get("roe_pct", 0) > 0,
              f"ROE={km.get('roe_pct', 'N/A')}%")

        check("5h. sentiment_result", final_state.get("sentiment_result") is not None)
        check("5i. risk_result", final_state.get("risk_result") is not None)

        risk = final_state.get("risk_result", {})
        check("5j. 风险定级", risk.get("overall_risk_level") != "待评估",
              f"风险等级={risk.get('overall_risk_level')}")

        report = final_state.get("report", "")
        check("5k. report 生成", len(report) > 200,
              f"报告长度={len(report)} 字")

        # 打印报告摘要
        print("\n  ── 研报摘要 ──")
        for line in report.split("\n")[:15]:
            print(f"  {line}")

    except Exception as e:
        check("5c. ainvoke", False, str(e)[:200])
        traceback.print_exc()


# ============================================================================
# Test 6: 健康检查重试环 + 真实 Agent 接入 (编排增强)
# ============================================================================
@pytest.mark.asyncio
async def test_sentiment_node_offline():
    section("Test 6a: sentiment_node 独立测试 (离线)")

    from app.workflow.graph import sentiment_node

    result = await sentiment_node({"company": "宁德时代", "ticker": ""})
    sr = result.get("sentiment_result", {})

    check("6a. 返回 dict", isinstance(result, dict))
    check("6a. 抓取新闻数", sr.get("searched_news_count", 0) == 9,
          f"新闻数={sr.get('searched_news_count')}")
    check("6a. 情感分布有值", sr.get("sentiment_distribution", {}).get("positive", 0) > 0)
    check("6a. 有舆情摘要", len(sr.get("summary", "")) > 0,
          f"摘要={sr.get('summary', '')[:40]}...")


@pytest.mark.asyncio
async def test_risk_node_offline():
    section("Test 6b: risk_node 独立测试 (离线)")

    from app.workflow.graph import risk_node

    state = {
        "company": "宁德时代",
        "ticker": "",
        "sentiment_result": {
            "symbol": "",
            "company_name": "宁德时代",
            "searched_news_count": 9,
            "sentiment_distribution": {"positive": 4, "negative": 2, "neutral": 3},
            "topics": [],
            "summary": "测试舆情摘要",
        },
        "financial_result": {
            "key_metrics": {"roe_pct": 17.95, "revenue_yoy_pct": 15.13},
            "dupont": {"net_profit_margin": 0.0903},
        },
    }
    result = await risk_node(state)
    risk = result.get("risk_result", {})

    check("6b. 风险等级合法", risk.get("overall_risk_level") in ("low", "medium", "high"),
          f"等级={risk.get('overall_risk_level')}")
    check("6b. 三维度评分", len(risk.get("dimensions", [])) == 3,
          f"维度数={len(risk.get('dimensions', []))}")
    check("6b. 有 risk_summary", len(risk.get("risk_summary", "")) > 0)
    check("6b. 推导链非空", len(risk.get("reasoning_chain", "")) > 0)


@pytest.mark.asyncio
async def test_health_retry_ring():
    """健康检查重试环端到端 — sentiment 首次失败 → 重试 1 次 → 成功。

    通过手动 patch 注入 flaky sentiment_node（首次返回空产出），验证:
        - 环收敛: 恰好重试 1 次（attempts 达 _MAX_HEALTH_RETRIES 上限）
        - 重试成功: 最终 sentiment_result 健康
        - degraded=False（未触发降级标记）
    """
    section("Test 6c: 健康检查重试环收敛")

    import app.workflow.graph as gm
    from app.workflow.routing import _MAX_HEALTH_RETRIES

    fixture_path = str(PROJECT_ROOT / "app" / "data" / "fixtures" / "sample_company.json")
    os.environ.setdefault("FINANCE_AGENT_DATA_FILE", fixture_path)

    real_sentiment = gm.sentiment_node
    calls = {"n": 0}

    async def _flaky(state):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"sentiment_result": {
                "symbol": "",
                "company_name": state.get("company", ""),
                "searched_news_count": 0,
                "summary": "",
            }}
        return await real_sentiment(state)

    # patch 模块级函数后重建图（build_graph 编译时取模块级全局名）
    gm.sentiment_node = _flaky
    try:
        graph = gm.build_graph()

        initial_state = {
            "company": "宁德时代",
            "ticker": "",
            "user_query": "分析宁德时代的未来发展情况与风险",
            "current_step": "start",
            "errors": [],
        }
        final_state = await graph.ainvoke(initial_state)
    finally:
        gm.sentiment_node = real_sentiment

    check("6c. 恰好重试 1 次", calls["n"] == 2, f"sentiment 调用次数={calls['n']}")
    check("6c. 重试后舆情健康",
          (final_state.get("sentiment_result") or {}).get("searched_news_count", 0) > 0,
          f"新闻数={(final_state.get('sentiment_result') or {}).get('searched_news_count')}")
    check("6c. 未降级", final_state.get("degraded") is False)
    check("6c. 轮次受控", final_state.get("attempts", 99) <= _MAX_HEALTH_RETRIES,
          f"attempts={final_state.get('attempts')}")


# ============================================================================
# Test 7: 意图分流 + Report 质量迭代环 (编排增强 Round 2)
# ============================================================================
@pytest.mark.asyncio
async def test_clarify_routing():
    """意图分流 — user_query 为空时走 CLARIFY 分支，跳过全部 Agent 扇出。"""
    section("Test 7a: 意图分流 (clarify)")

    from app.workflow.graph import build_graph

    graph = build_graph()
    initial_state = {
        "company": "宁德时代",
        "ticker": "",
        "user_query": "",
        "current_step": "start",
        "errors": [],
    }
    final_state = await graph.ainvoke(initial_state)

    check("7a. intent=clarify", final_state.get("intent") == "clarify",
          f"intent={final_state.get('intent')}")
    check("7a. 扇出被跳过", final_state.get("research_result") is None)
    check("7a. financial 未执行", final_state.get("financial_result") is None)
    check("7a. 返回追问文案", "请补充" in (final_state.get("report") or ""),
          f"report={final_state.get('report', '')[:50]}...")


@pytest.mark.asyncio
async def test_report_quality_loop_terminates():
    """Report 质量迭代环 — research 占位导致报告含占位符 → 走满上限强制结束。

    确定性验证防死循环上限（零随机因素）:
        - iteration 不超过 _MAX_REPORT_ITERATIONS
        - report 含 rework 追加的"补充说明"
    """
    section("Test 7b: Report 质量迭代环收敛")

    from app.workflow.graph import build_graph
    from app.workflow.routing import _MAX_REPORT_ITERATIONS

    fixture_path = str(PROJECT_ROOT / "app" / "data" / "fixtures" / "sample_company.json")
    os.environ.setdefault("FINANCE_AGENT_DATA_FILE", fixture_path)

    graph = build_graph()
    initial_state = {
        "company": "宁德时代",
        "ticker": "",
        "user_query": "分析宁德时代的未来发展情况与风险",
        "current_step": "start",
        "errors": [],
    }
    final_state = await graph.ainvoke(initial_state)

    check("7b. 迭代轮次受控",
          final_state.get("iteration", 99) <= _MAX_REPORT_ITERATIONS,
          f"iteration={final_state.get('iteration')}")
    check("7b. 评估结果存在", final_state.get("report_quality") is not None)
    check("7b. rework 修订生效", "补充说明" in (final_state.get("report") or ""))
    check("7b. 报告仍完整", len(final_state.get("report", "")) > 200,
          f"报告长度={len(final_state.get('report', ''))}")


@pytest.mark.asyncio
async def test_decision_functions():
    """纯决策函数单元测试 — 循环终止性保证（对应 RAG router.py 可单测模式）。"""
    section("Test 7c: 路由决策函数")

    from app.workflow.routing import (
        _MAX_REPORT_ITERATIONS,
        decide_intent,
        decide_report_action,
    )

    # decide_intent
    check("7c. 空 query → clarify", decide_intent("") == "clarify")
    check("7c. 空白 query → clarify", decide_intent("   ") == "clarify")
    check("7c. 正常 query → full_research", decide_intent("分析宁德时代风险") == "full_research")
    check("7c. None → clarify", decide_intent(None) == "clarify")

    # decide_report_action — 防死循环: 达上限强制 end
    check("7c. 达标 → end", decide_report_action({"passed": True}, 1) == "end")
    check("7c. 不达标且未达上限 → rework",
          decide_report_action({"passed": False}, 1) == "rework")
    check("7c. 不达标但达上限 → end (强制输出)",
          decide_report_action({"passed": False}, _MAX_REPORT_ITERATIONS) == "end")
    check("7c. 质量为空 → rework", decide_report_action(None, 0) == "rework")


# ============================================================================
# Main
# ============================================================================
async def main():
    print("=" * 60)
    print("  FinanceAgent — Financial Agent 测试套件")
    print("=" * 60)

    await test_calculator()
    await test_schemas()
    await test_prompts()
    await test_node()
    await test_graph()
    await test_sentiment_node_offline()
    await test_risk_node_offline()
    await test_health_retry_ring()
    await test_clarify_routing()
    await test_report_quality_loop_terminates()
    await test_decision_functions()

    total = PASS + FAIL
    print(f"\n{'='*60}")
    print(f"  结果: {PASS}/{total} 通过" + (", " + f"{FAIL} 失败" if FAIL else ", 全部通过 🎉"))
    print(f"{'='*60}")

    return FAIL == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
