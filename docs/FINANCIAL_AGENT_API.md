# Financial Agent 接口文档

> **负责成员**: Member 3（工藤）
> **模块路径**: `app/agents/financial_agent/`
> **版本**: v0.2.0
> **最后更新**: 2026-08-05（下午更新：数据源重构 + 异常兜底）

---

## 1. 模块概述

Financial Agent 是多 Agent 投研系统中负责**财务指标计算与分析**的专业智能体。它被 LangGraph 主图注册为 `"financial"` 节点，与 Research Agent、Sentiment Agent **并行执行**，输出结构化数据供下游 Risk Agent 和 Report Agent 消费。

### 核心链路

```
ResearchState (ticker, company)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│              financial_analysis_node()                   │
│                                                          │
│  ① _fetch_financial_data()                              │
│     ├── AkShare/Tushare API  (异步 I/O)                  │
│     ├── 本地 JSON 文件      (ticker 自动匹配)            │
│     └── 无数据降级          (_build_no_data_output)      │
│  ② FinancialCalculator.calculate_yoy()    ← CPU 硬计算   │
│  ③ FinancialCalculator.calculate_dupont() ← CPU 硬计算   │
│  ④ build_analysis_prompt() + LLM 调用     ← 异步 I/O     │
│  ⑤ FinancialAgentOutput.model_dump()      ← 结构化输出   │
│                                                          │
│  异常兜底: try/except → _build_error_output()            │
└──────────────────────────────────────────────────────────┘
       │
       ▼
ResearchState.financial_result  (dict)
```

### 设计原则

- **计算隔离**: 所有数值由 `FinancialCalculator` 确定性计算，LLM 只负责文本解读
- **全异步**: I/O 操作使用 `async/await`，兼容 LangGraph 的异步调度
- **多层容错**:
  - 数据获取: API → 本地 JSON（ticker 自动匹配）→ 无数据降级
  - LLM: DeepSeek API → 规则引擎兜底
  - 异常: 顶层 try/except → `_build_error_output`

---

## 2. 给 Member 1（Manager/Workflow）— 如何集成

### 2.1 注册节点

```python
# app/workflow/graph.py — build_graph()
from app.agents.financial_agent.node import financial_analysis_node

builder.add_node("financial", financial_analysis_node)
```

### 2.2 并行扇出

你的 Send API 已经正确配置了金融节点：

```python
def fan_out_to_agents(state: ResearchState) -> list[Send]:
    return [
        Send("research", dict(state)),
        Send("financial", dict(state)),   # ← 自动执行 financial_analysis_node
        Send("sentiment", dict(state)),
    ]
```

### 2.3 依赖的 State 字段（输入）

`financial_analysis_node` 从 `ResearchState` 中读取以下字段：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `company` | `str` | ✅ | 公司全称，用于报告标识和 LLM Prompt |
| `ticker` | `str \| None` | 否 | 股票代码，用于 API 查询和本地文件自动匹配 `app/data/{ticker}_*.json` |
| `user_query` | `str \| None` | 否 | 用户原始提问（当前未使用，Phase 2 预留） |

> 💡 `ticker` 有值时，`_load_local_fixture` 会在 `app/data/` 下自动匹配 `{ticker}_*.json` 文件，无需手动设环境变量。若 ticker 为空，可通过 `FINANCE_AGENT_DATA_FILE` 环境变量指定数据文件，或使用 `app/data/fixtures/sample_company.json`。

### 2.4 返回的 State 更新（输出）

Agent 返回 `{"financial_result": FinancialAgentOutput.model_dump()}`，详细结构见第 3 节。

---

## 3. 给 Member 4（Risk Agent）和 Report Agent — 接口契约

### 3.1 顶层结构

```
financial_result: dict
├── company: str                     # 公司名称
├── ticker: str                      # 股票代码
├── analysis_period: str             # 分析周期，如 "2020-01-01 ~ 2026-08-05"
├── key_metrics: dict               # 核心指标（见 3.2）
├── dupont: dict                    # 杜邦三因子（见 3.3）
├── yoy_history: list[dict]         # 历年同比（见 3.4）
├── commentary: str                 # LLM 生成的 CFO 点评（Markdown）
├── raw_calculations: dict          # 计算器原始输出（调试用）
├── data_source: str                # "api" | "none" | "error"
├── fetch_error: str | None         # 错误信息（异常时含完整堆栈）
└── generated_at: str               # ISO 8601 时间戳
```

### 3.2 key_metrics — 核心财务指标

```python
{
    "roe_pct": 17.95,                    # float — 净资产收益率 (%)
    "net_profit_margin_pct": 9.03,       # float — 净利润率 (%)
    "revenue_yoy_pct": 15.13,            # float | None — 营收同比增速 (%)，基准年为 None
    "net_profit_yoy_pct": 43.64,         # float | None — 净利润同比增速 (%)
    "equity_multiplier": 2.5,            # float — 权益乘数
    "asset_turnover": 0.7955,            # float — 资产周转率
}
```

**Risk Agent 使用示例**:

```python
fin = state.get("financial_result", {}) or {}
km = fin.get("key_metrics", {}) or {}
roe = km.get("roe_pct")

if roe is not None:
    if roe < 0:
        risk_level = "high"      # 亏损
    elif roe < 5:
        risk_level = "medium"    # 盈利偏弱
    else:
        risk_level = "low"       # 盈利能力良好
```

### 3.3 dupont — 杜邦分析三因子

```python
{
    "net_profit_margin": 0.0903,    # float — 净利润率 = 净利润 / 营收
    "asset_turnover": 0.7955,       # float — 资产周转率 = 营收 / 总资产
    "equity_multiplier": 2.5,       # float — 权益乘数 = 总资产 / 股东权益
    "roe_computed": 0.1795,         # float — ROE = 三因子乘积
    "roe_direct": 0.1795,           # float — ROE 交叉验证 = 净利润 / 股东权益
}
```

> `roe_computed` 和 `roe_direct` 应一致（误差 < 1e-9），不一致说明输入数据有问题。

### 3.4 yoy_history — 历年同比增速

```python
[
    {
        "period": "2020 (基准年)",
        "revenue_growth_pct": None,         # None — 基准年无上年对比
        "net_profit_growth_pct": None,
        "revenue_trend": "持平",
        "profit_trend": "持平",
    },
    {
        "period": "2024 vs 2023",
        "revenue_growth_pct": 15.13,        # float — 营收同比增速 (%)
        "net_profit_growth_pct": 43.64,     # float — 净利润同比增速 (%)
        "revenue_trend": "上升",             # "上升" | "下降" | "持平"
        "profit_trend": "上升",
    },
    # ... 共 5 年（含基准年）
]
```

**Report Agent 渲染示例**:

```python
for item in fin.get("yoy_history", []):
    rev = item.get("revenue_growth_pct")
    rev_str = f"{rev:+.2f}%" if rev is not None else "N/A"
    print(f"- {item['period']}: 营收 {rev_str}, 趋势 {item['revenue_trend']}")
```

### 3.5 commentary — CFO 点评

`str` — LLM 生成的专业财务分析文本（Markdown 格式），结构如下：

```markdown
## 财务健康度总评
[1-2 句整体判断]

## 盈利能力分析
[基于 ROE、净利润率分析]

## 成长性分析
[基于营收同比、净利润同比趋势]

## 资产运营效率
[基于资产周转率分析]

## 财务杠杆与偿债风险
[基于权益乘数分析]

## 综合风险提示
[2-3 个关键风险点或亮点]
```

> LLM 不可用时自动降级为规则引擎生成的简要分析，格式为 `（LLM 服务不可用...）` 前缀 + 结构化文本。

---

## 4. 给 Member 5（后端/部署）— 配置与环境变量

### 4.1 必需配置

`.env` 文件中需要配置：

```bash
# LLM（用于生成 CFO 点评）
DEEPSEEK_API_KEY=sk-xxx          # 必填，不填则用规则降级
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL_CHAT=deepseek-v4-flash
```

### 4.2 依赖

`requirements.txt` 中已有：

```
pydantic          # Pydantic V2 数据模型
httpx             # 异步 HTTP（LLM 调用 + 数据获取）
numpy             # 数值计算
pandas            # 数据处理
langgraph         # 工作流编排
```

`FinancialCalculator` 的硬计算仅依赖 `numpy`，可在无 LLM 环境下独立运行。

### 4.3 模块边界

```
app/agents/financial_agent/     ← Member 3 维护
    ├── schemas.py              # FinancialAgentInput / Output / KeyMetrics 等
    ├── prompts.py              # CFO System Prompt + build_analysis_prompt()
    ├── node.py                 # financial_analysis_node() 入口
    └── __init__.py

app/quant_engine/               ← Member 3 维护（硬计算引擎，全局共享）
    └── calculator.py           # FinancialCalculator 类

app/services/                   ← Member 3 维护（可被其他 Agent 复用）
    └── data_fetcher.py         # MarketDataService 类

app/core/schemas.py             ← 全局基础模型（跨模块共享）
```

---

## 5. 错误处理与降级策略

### 5.1 数据获取降级链

```
AkShare/Tushare API
  ├── 成功 → 使用真实数据 (data_source="api")
  ├── 失败/空 → 本地 JSON 文件
  │     ├── FINANCE_AGENT_DATA_FILE 环境变量指定路径
  │     ├── ticker 自动匹配: app/data/{ticker}_*.json
  │     └── 未找到 → 无数据降级 (data_source="none")
  └── 无数据 → _build_no_data_output()
               → commentary 显示 "⚠️ 无可用财务数据"
```

### 5.2 各环节降级表

| 环节 | 错误场景 | 降级行为 | 对下游影响 |
|------|----------|----------|-----------|
| 数据获取 | API 不可用 | → 本地 JSON (ticker 匹配) | `data_source="api"`，指标有效 |
| 数据获取 | 所有源无数据 | `_build_no_data_output()` | 指标全 0，commentary 含原因和建议 |
| 硬计算 | 输入数据为 0（除零） | 增速返回 `inf`，趋势判"持平" | 部分指标为 `None` |
| 硬计算 | fy_data 为空 | 提前返回无数据输出 | 不会崩 |
| LLM 调用 | API Key 未配置 | 规则引擎生成结构化点评 | `commentary` 有内容但较简短 |
| LLM 调用 | 超时 (60s) | 规则引擎降级 | 同上 |
| 任意环节 | **未预期异常** | `_build_error_output()` | 指标全 0，commentary 含完整堆栈 |

> **顶层兜底**: `financial_analysis_node` 整个 try/except 包裹，任何未预期异常都不会中断 Workflow。`fetch_error` 字段含完整堆栈。

### 5.3 data_source 取值

| 值 | 含义 |
|----|------|
| `"api"` | 成功获取数据（真实 API 或本地 JSON） |
| `"none"` | 所有数据源均无数据 |
| `"error"` | 运行时未预期异常 |

---

## 6. 测试

```bash
cd F:\Final\FinaceAgent
conda activate finance-agent
pytest test_financial_agent.py -v
```

测试覆盖：Calculator 硬计算 / Pydantic 模型 / Prompt 构建 / Agent 节点 / LangGraph 全图。

---

## 7. Phase 2 规划（Member 3 后续交付）

| 功能 | 说明 | 对下游影响 |
|------|------|-----------|
| 现金流分析 | 新增经营性/投资性/筹资性现金流指标 | `key_metrics` 新增字段 |
| 偿债能力 | 流动比率、速动比率、利息保障倍数 | 同上 |
| 估值模型 | DCF / PE / PB 估值 | `raw_calculations` 新增 |
| 行业对比 | 接入同行业均值进行横向比较 | `key_metrics` 新增 `industry_avg_*` |
| 多 LLM 支持 | 抽象 LLM Client，支持 GPT-4o / Claude 切换 | 无接口变化 |
| 数据缓存 | Redis/TTL 缓存，减少重复请求 | `data_source` 新增 `"cache"` |
| 数据库接入 | 历史查询持久化，`_fetch_financial_data` 优先读 DB | 数据获取新增一层 |

> 所有新增字段均为**向后兼容**，不会破坏现有 `key_metrics` / `dupont` / `yoy_history` 的结构。

---

## 8. 本地数据文件

### 8.1 目录结构

```
app/data/
├── fixtures/
│   └── sample_company.json       ← 演示用样本数据
├── 300750_宁德时代.json           ← 真实财务数据（2020-2024）
├── 600519_贵州茅台.json
├── 002594_比亚迪.json
└── 000333_美的集团.json
```

### 8.2 JSON 格式

```json
{
  "_description": "公司描述（可选）",
  "company_name": "宁德时代新能源科技股份有限公司",
  "ticker": "300750",
  "industry": "电力设备 / 电池",
  "fiscal_data": {
    "2024": {
      "revenue": 362013150000,
      "net_profit": 50744620000,
      "total_assets": 812345000000,
      "shareholders_equity": 257012000000
    }
  }
}
```

### 8.3 添加新公司

只需将财报 JSON 放入 `app/data/`，按 `{ticker}_{公司名}.json` 命名，无需改代码。

---

## 9. 联系

- **Member 3**：工藤新一
- **模块路径**：`app/agents/financial_agent/`
- **计算引擎**：`app/quant_engine/calculator.py`
- **数据服务**：`app/services/data_fetcher.py`
- **问题/Bug**：请在 Git Issue 中 @ 工藤新一，附上 `fetch_error` 字段内容
