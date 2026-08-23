# Sentiment & Risk Agent 开发文档

> 模块负责人：任成宇  
> 所属项目：Mini FinAgent — 多 Agent 金融智能分析平台  
> 文档版本：v1.0  
> 最后更新：2026-08-09

---

## 目录

- [1. 模块定位](#1-模块定位)
- [2. 数据流全景](#2-数据流全景)
- [3. 项目结构](#3-项目结构)
- [4. 数据模型层](#4-数据模型层)
- [5. 工具层](#5-工具层)
- [6. Agent 层](#6-agent-层)
- [7. 基础设施层](#7-基础设施层)
- [8. 评分模型详解](#8-评分模型详解)
- [9. 测试](#9-测试)
- [10. 快速开始](#10-快速开始)
- [11. Phase 2 升级路线](#11-phase-2-升级路线)

---

## 1. 模块定位

Mini FinAgent 模拟真实金融投研团队的岗位分工：

```
用户请求
    ↓
Manager Agent（总调度）         ← 队友做
    ↓
┌───────────┬──────────┬──────────┐
│ Research  │Financial │Sentiment │  ← 队友+你
│ Agent     │Agent     │Agent     │
│ (RAG)     │(API计算) │(舆情分析) │  ← 你负责
└───────────┴──────────┴──────────┘
    ↓
Risk Agent（风险评估）            ← 你负责
    ↓
Report Agent（报告生成）          ← 队友做
```

**你负责的部分**：Sentiment Agent（舆情分析）+ Risk Agent（风险评估），两个 Agent 独立开发、串行协作。

---

## 2. 数据流全景

```
SentimentInput
  {symbol: "300750", company_name: "宁德时代", days: 30}
        ↓
  SentimentAgent.run()
        │
        ├─ Step 1: fetch_recent_news()      → list[NewsItem]      抓新闻
        ├─ Step 2: 空值检查                  → 提前返回或继续
        ├─ Step 3: batch_score_news()        → list[ScoredNews]   并行情感评分
        ├─ Step 4: 统计情感分布              → {pos:0, neg:4, neu:2}
        ├─ Step 5: cluster_topics()          → list[TopicCluster] 热点聚类
        └─ Step 6: LLM 生成摘要              → str                200字总结
        ↓
  SentimentResult
        ↓
        ├──────── FinancialSummary（来自 Financial Agent）────────┐
        ↓                                                         ↓
  RiskAgent.run(sentiment_result, financial)
        │
        ├─ Step 1: synthesize_risk()         → RiskAssessment    纯Python评分
        │   ├─ assess_sentiment_risk()       舆情维度（权重40%）
        │   ├─ assess_financial_risk()       财务维度（权重35%）
        │   └─ assess_industry_risk()        行业维度（权重25%）
        │
        └─ Step 2: LLM 润色总结              → str               自然语言总结
        ↓
  RiskAssessment
        ↓
  SentimentRiskJointOutput  → 交给 Report Agent 写研报
```

---

## 3. 项目结构

```
FinaceAgent/
├── app/
│   ├── models/
│   │   ├── sentiment_risk_models.py    # Sentiment+Risk 的 Pydantic Schema
│   │   └── orchestrator_models.py      # Manager Agent 统一编排 Schema
│   │
│   ├── tools/
│   │   ├── news_tools.py               # 新闻抓取 @tool
│   │   ├── sentiment_tools.py          # FinBERT 评分 + BERTopic 聚类 @tool
│   │   └── risk_tools.py               # 三维度评估 + 综合判定 @tool
│   │
│   ├── agents/
│   │   ├── sentiment_agent.py          # 舆情分析 Agent
│   │   └── risk_agent.py               # 风险评估 Agent
│   │
│   ├── core/
│   │   ├── config.py                   # pydantic-settings 配置中心
│   │   ├── llm_factory.py              # 统一 LLM 工厂
│   │   └── retry.py                    # 三层兜底 + 重试装饰器
│   │
│   ├── workflow/                       # LangGraph 工作流（Phase 2）
│   ├── database/                       # 数据库连接
│   └── api/                            # FastAPI 接口
│
├── tests/
│   ├── test_data.py                    # 5个场景的测试数据工厂
│   ├── test_sentiment_risk.py          # 8个自动化测试
│   └── conftest.py                     # pytest 配置
│
├── requirements.txt                    # Python 依赖
├── environment.yml                     # Conda 环境
└── .env                                # 环境变量（API Key 等，不提交 Git）
```

---

## 4. 数据模型层

### 4.1 设计原则

1. **数据模型先行**：先定义 Input/Output Schema，再写业务逻辑
2. **`Field(description=...)` 不只是注释**：Phase 2 接入 `with_structured_output` 后，description 会直接发给 LLM 作为提取指引
3. **`default_factory` 防共享**：列表/字典必须用工厂函数，杜绝 `default=[]` 导致的所有实例共享同一对象的经典坑
4. **`Optional` 容错**：财务指标全部可选——数据源常缺失，Agent 不会因为缺字段崩溃

### 4.2 Sentiment Agent 模型

| 模型 | 角色 | 核心字段 |
|------|------|---------|
| `SentimentInput` | 输入 | `symbol`, `company_name`, `days`(默认30) |
| `NewsItem` | 单条新闻 | `title`, `source`, `url`, `published_at`, `summary` |
| `SentimentScore` | 情感评分 | `label`(positive/negative/neutral), `confidence`[0,1], `explanation` |
| `ScoredNews` | 新闻+评分 | `news: NewsItem`, `sentiment: SentimentScore` |
| `TopicCluster` | 热点主题 | `topic_id`, `label`, `keywords`, `news_count`, `representative_news` |
| `SentimentResult` | 最终输出 | `scored_news`, `sentiment_distribution`, `topics`, `summary` |

### 4.3 Risk Agent 模型

| 模型 | 角色 | 核心字段 |
|------|------|---------|
| `FinancialSummary` | 财务数据（来自 Financial Agent） | 5 个财务指标（全 Optional）+ `anomalies` 异常列表 |
| `RiskDimension` | 单维度评分 | `dimension`, `score`[0,1], `evidence`, `reasoning` |
| `RiskAssessment` | 最终输出 | `overall_risk_level`, `overall_score`, `dimensions`(3个), `key_risks`, `reasoning_chain` |

### 4.4 枚举定义

```python
class SentimentLabel(str, Enum):
    POSITIVE = "positive"   # 看多
    NEGATIVE = "negative"   # 看空
    NEUTRAL  = "neutral"    # 中立

class RiskLevel(str, Enum):
    HIGH   = "high"    # score ≥ 0.7
    MEDIUM = "medium"  # score ≥ 0.4
    LOW    = "low"     # score < 0.4
```

---

## 5. 工具层

所有外部能力封装为 LangChain `@tool`，好处：

- LLM 可通过 Function Calling 自主调用（Phase 2）
- 输入/输出类型明确，自动生成 JSON Schema
- Phase 1 返回占位数据，Phase 2 替换实现——接口不变

### 5.1 `news_tools.py` — 新闻抓取

```
fetch_recent_news(symbol, company_name, days) → list[NewsItem]
```

| 阶段 | 实现 |
|------|------|
| Phase 1 | 返回 9 条模拟新闻（3 来源 × 3 条），验证管道通畅 |
| Phase 2 | 替换为 httpx 异步请求真实新闻 API |

### 5.2 `sentiment_tools.py` — 情感分析

```
score_sentiment(news_text) → SentimentScore      单条 FinBERT 评分
batch_score_news(news_list) → list[ScoredNews]   批量并行评分
cluster_topics(scored_news) → list[TopicCluster]  BERTopic 聚类
```

**关键优化：`batch_score_news` 使用 `asyncio.gather`**

```
串行: 评1 → 评2 → ... → 评9   总耗时 = 9 × 单次耗时
并行: 评1┐
      评2┤
       ...┤ asyncio.gather     总耗时 ≈ 最慢的 1 次
      评9┘
```

### 5.3 `risk_tools.py` — 风险评估

```
assess_sentiment_risk(sentiment) → RiskDimension    舆情维度（权重 40%）
assess_financial_risk(financial) → RiskDimension    财务维度（权重 35%）
assess_industry_risk(sentiment, financial) → RiskDimension  行业维度（权重 25%）
synthesize_risk(...) → RiskAssessment               综合：三维度加权 + 等级判定
```

**表驱动设计**：三维度不写死在代码里，而是配置表 `RISK_DIMENSIONS`：

```python
RISK_DIMENSIONS = [
    {"key": "sentiment", "name": "舆情负面信号",     "weight": 0.40, ...},
    {"key": "financial", "name": "财务异常信号",     "weight": 0.35, ...},
    {"key": "industry",  "name": "行业周期风险",     "weight": 0.25, ...},
]
```

扩维度只需：① 写评估函数 → ② 表中加一行 → ③ 调权重（总和不超 1.0），不改主逻辑。

---

## 6. Agent 层

### 6.1 SentimentAgent — 舆情分析

```python
class SentimentAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.tools = [fetch_recent_news, batch_score_news, cluster_topics]

    async def run(self, params: SentimentInput) -> SentimentResult:
        # 6 步串行：抓 → 空值检查 → 评 → 统计 → 聚类 → LLM 写摘要
```

**设计要点**：
- 独立可测试：`llm` 可以是真实 DeepSeek，也可以是 `_MockLLM()`
- 空值保护：0 条新闻时直接返回，不崩
- Phase 2 可切换为 LangGraph ReAct Agent（LLM 自主编排工具）

### 6.2 RiskAgent — 风险评估

```python
class RiskAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.tools = [synthesize_risk]

    async def run(self, sentiment_result, financial) -> RiskAssessment:
        # Step 1: synthesize_risk.invoke() — 纯 Python 评分（不调 LLM）
        # Step 2: self._enhance_with_llm()  — LLM 润色总结（叙事）
```

**评分和叙事分离**：
- 评分 → 代码：确定性、可复现，同一数据永远同一分数
- 总结 → LLM：把数字写成人类能读的自然语言

---

## 7. 基础设施层

### 7.1 `config.py` — 配置中心

```python
from app.core.config import get_settings
settings = get_settings()           # 全局单例
api_key = settings.deepseek_api_key # 从 .env 文件读取
```

- `@lru_cache()` 单例：全项目只读一次 `.env`
- 必填字段无默认值 → 缺失时启动即报错
- `extra="ignore"`：`.env` 多余字段不报错

### 7.2 `llm_factory.py` — LLM 工厂

```python
llm = LLMFactory.get_llm("sentiment")                      # 普通模型
struct = LLMFactory.get_structured_llm("risk", Schema)     # 结构化输出模型
```

- **路由表**：`agent_type → 模型名`，换模型只改一行
- **缓存**：相同参数的模型只创建一次
- **绕过代理**：`trust_env=False`，避免 Windows 代理导致 TLS 失败
- **硬规矩**：禁止 Agent 直接调 `init_chat_model`，一律走工厂

### 7.3 `retry.py` — 三层兜底

```
第一层：自动重试 2 次（间隔 1s → 3s）    ← 消化 90% 偶发故障
    ↓ 失败
第二层：Agent 级降级                      ← 返回备用文案
    ↓ 失败
第三层：系统级兜底                        ← 友好提示，永远不崩
```

用法：

```python
@with_retry(agent_type="sentiment")
async def _invoke():
    return await graph.ainvoke(state, config=config)
```

---

## 8. 评分模型详解

### 8.1 舆情维度评分（权重 40%）

```
负面占比 = negative_count / total_count
高危加分 = 0.05 × 命中高危关键词的主题数

高危关键词: 监管、制裁、诉讼、关税、调查、处罚、违约、退市

score = min(负面占比 × 1.5 + 高危加分, 1.0)
```

### 8.2 财务维度评分（权重 35%）

逐项检查，触发一条记一次预警：

| 指标 | 触发条件 |
|------|---------|
| 营收增长率 | < 0 → risk_flag；< 5% → 记录 evidence |
| 毛利率 | < 15% → 记录 evidence |
| 资产负债率 | > 70% → risk_flag |
| 财务异常项 | 每项 → risk_flag |

```
score = 触发预警数 / 可评估指标总数
```

### 8.3 行业维度评分（权重 25%）

扫描舆情主题标签，匹配行业风险关键词：

```
关键词 → 风险描述
"政策" → "政策变动风险"
"监管" → "监管趋严风险"
"供应链" → "供应链扰动风险"
"关税" → "海外关税风险"
"竞争" → "竞争加剧风险"
"替代" → "技术替代风险"
"周期" → "行业周期下行风险"

score = 命中数 × 0.25（上限 1.0）
```

### 8.4 综合判定

```
综合分 = Σ(维度评分 × 权重)
等级 = HIGH(≥0.7) / MEDIUM(≥0.4) / LOW(<0.4)
```

---

## 9. 测试

### 9.1 测试数据

`tests/test_data.py` 提供 5 个场景：

| 场景 | 舆情 | 财务 | 预期等级 |
|------|------|------|:--------:|
| `low_risk` | 3 正面 + 3 中性，无负面 | 营收 +25%，毛利 28%，零异常 | LOW |
| `medium_risk` | 4 负面 + 2 中性，有"关税"主题 | 营收 +3%，毛利 18%，2 异常 | MEDIUM |
| `high_risk` | 6 负面，制裁+诉讼+调查主题 | 营收 -8%，负债 78%，4 高危异常 | HIGH |
| `empty_data` | 0 条新闻 | 全空 | LOW |
| `all_positive` | 6 正面，无负面 | 毛利 92%，负债 19% | LOW |

### 9.2 测试用例

`tests/test_sentiment_risk.py` 共 8 个用例：

```
test_imports              ← 模块导入验证
test_sentiment_agent      ← Sentiment Agent 完整流程
test_risk_scenarios[low_risk]       ← 低风险
test_risk_scenarios[medium_risk]    ← 中风险
test_risk_scenarios[high_risk]      ← 高风险
test_risk_scenarios[empty_data]     ← 边界：空数据
test_risk_scenarios[all_positive]   ← 边界：全正面
test_joint_output         ← JointOutput 打包验证
```

### 9.3 运行方式

```bash
# PyCharm: 右键 tests/test_sentiment_risk.py → Run
# 终端:
pytest tests/test_sentiment_risk.py -v
```

Mock LLM 不调真实 API，测试离线可跑。

---

## 10. 快速开始

### 10.1 环境准备

```bash
# 1. 创建 Conda 环境
conda env create -f environment.yml
conda activate finance-agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 .env（可选，测试不需要）
cp .env.example .env
```

### 10.2 最小可运行示例

```python
import asyncio
from app.agents.sentiment_agent import SentimentAgent
from app.agents.risk_agent import RiskAgent
from app.models.sentiment_risk_models import SentimentInput, FinancialSummary


class _MockLLM:
    async def ainvoke(self, msg):
        class R:
            content = "测试摘要"
        return R()


async def main():
    llm = _MockLLM()

    # 舆情分析
    s_agent = SentimentAgent(llm=llm)
    sentiment = await s_agent.run(
        SentimentInput(symbol="300750", company_name="宁德时代", days=30)
    )
    print(f"抓取 {sentiment.searched_news_count} 条新闻")
    print(f"情感分布: {sentiment.sentiment_distribution}")

    # 风险评估
    r_agent = RiskAgent(llm=llm)
    risk = await r_agent.run(
        sentiment_result=sentiment,
        financial=FinancialSummary(
            revenue_growth=0.15,
            gross_margin=0.22,
            debt_ratio=0.65,
            anomalies=["应收账款周转天数同比增加30%"],
        ),
    )
    print(f"风险等级: {risk.overall_risk_level.value.upper()}")
    print(f"综合评分: {risk.overall_score}")
    print(f"关键风险: {len(risk.key_risks)} 项")


asyncio.run(main())
```

### 10.3 接入真实 LLM

```python
from app.core.llm_factory import get_llm

# 替换 _MockLLM()
llm = get_llm("sentiment")        # 需要 .env 里配好 DEEPSEEK_API_KEY
agent = SentimentAgent(llm=llm)
```

---

## 11. Phase 2 升级路线

| 项目 | 当前状态 | Phase 2 |
|------|---------|---------|
| 新闻抓取 | Mock 数据 | 接入东方财富/财联社新闻 API |
| 情感评分 | 占位返回 NEUTRAL | 接入 ProsusAI/finbert 本地推理 |
| 主题聚类 | 单主题占位 | 接入 BERTopic 自动聚类 |
| 评分模型 | 固定 40/35/25 权重 | 支持可配置（ESG、汇率等新维度） |
| Agent 编排 | 手写串行工具链 | LangGraph ReAct Agent，LLM 自主编排 |
| 错误处理 | 基础 try/except | `@with_retry` 三层兜底全覆盖 |
| 结构化输出 | LLM 自由文本 | `with_structured_output` → Pydantic 对象 |
| Manager 集成 | 独立可测 | 接入 `Orchestrator` 的 `AgentRequest/AgentResponse` |
| LangGraph | 未接入 | `SentimentNode → RiskNode → ReportNode` 图编排 |
| API | 无 | FastAPI + SSE 流式推送 |

---

## 附录：架构决策记录

### ADR-001: Sentiment 和 Risk 分开为两个 Agent

**决策**：分开。  
**理由**：吻合 PDF 架构设计，符合 CLAUDE.md 的 Agent 职责隔离原则，各自独立测试。

### ADR-002: 评分用纯 Python 计算，不用 LLM

**决策**：评分逻辑放在 `@tool` 函数里，纯规则+加权，LLM 只润色总结。  
**理由**：评分需要确定性和可复现性，LLM 即使 temperature=0 也有随机性。

### ADR-003: 使用相对导入

**决策**：全项目使用 `from ..xxx import ...` 相对导入。  
**理由**：不依赖 IDE 的 source root 配置和 PYTHONPATH。

### ADR-004: 表驱动维度配置

**决策**：三维度用 `RISK_DIMENSIONS` 配置表管理。  
**理由**：将来加维度/改权重只需改表，不动主逻辑（参照 EduAgent 经验）。
