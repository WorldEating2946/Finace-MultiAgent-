"""
Sentiment & Risk Agent 共享数据模型

════════════════════ 设计原则 ════════════════════════
1. 数据模型先行：每个 Agent 的 Input/Output 先定义清楚，再写业务逻辑。
   所有 LLM 调用都返回 Pydantic 对象（Phase 2 用 with_structured_output），
   不解析自由文本，保证类型安全。

2. Field(description=...) 不只是注释：Phase 2 接入 with_structured_output 时，
   description 会被转成 Function Calling 的参数说明，告诉 LLM「这个字段该填什么」。
   描述越清楚，LLM 提取越准。

3. default_factory=list / default_factory=dict：
   列表和字典的默认值必须用工厂函数，绝不能写 default=[]。
   写 default=[] 会让所有实例共享同一个列表对象——Python 经典坑。

4. 数据流向：
   SentimentInput → SentimentAgent.run() → SentimentResult
                                            ↓
   FinancialSummary（来自 Financial Agent）→ RiskAgent.run() → RiskAssessment
                                            ↓
                 SentimentRiskJointOutput（汇总，供 Report Agent 消费）
═══════════════════════════════════════════════════════════
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# ════════════════════════════════════════════════════════════
# 枚举定义
# ════════════════════════════════════════════════════════════

class SentimentLabel(str, Enum):
    """
    FinBERT 金融情感标签。

    为什么继承 str, Enum？
      → 成员可以直接当字符串用（label == "positive" 成立），
        LangChain 工具调用时自动序列化为 "positive"/"negative"/"neutral"，
        不需要手动 .value 转换。
    """
    POSITIVE = "positive"   # 看多：利好信号，市场情绪乐观
    NEGATIVE = "negative"   # 看空：利空信号，市场情绪悲观
    NEUTRAL = "neutral"     # 中立：无明显倾向或信息量不足


class RiskLevel(str, Enum):
    """
    风险等级。

    判定阈值（在 risk_tools.py 的 _score_to_level 中定义）：
      score ≥ 0.7 → HIGH   ：需要立即关注，存在明确且严重的风险信号
      score ≥ 0.4 → MEDIUM ：存在风险因素，建议持续监控
      score < 0.4 → LOW    ：当前未发现显著风险
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ════════════════════════════════════════════════════════════
# Sentiment Agent 模型
# ════════════════════════════════════════════════════════════

class SentimentInput(BaseModel):
    """
    Sentiment Agent 输入。

    用户在 API 层传入股票代码 + 企业名，Agent 据此抓取新闻。
    days 有取值范围约束（ge=1, le=365），Pydantic 自动校验，超范围返回 422。
    """
    symbol:       str = Field(..., description="股票代码, 如 300750")
    company_name: str = Field(..., description="企业名称, 如 宁德时代")
    days:         int = Field(default=30, ge=1, le=365, description="新闻回溯天数，默认30天")


class NewsItem(BaseModel):
    """
    单条新闻。
    每个字段都有明确的数据源头——source 标记来源、url 可点击溯源、
    published_at 用于时间排序。Phase 1 是占位数据，Phase 2 接入新闻 API 后
    直接替换 news_tools.py 中的 fetch_recent_news 返回体即可，模型不变。
    """
    title:        str = Field(..., description="新闻标题")
    source:       str = Field(default="", description="来源，如'东方财富'/'财联社'")
    url:          str = Field(default="", description="原始链接，可点击跳转")
    published_at: datetime | None = Field(default=None, description="发布时间")
    summary:      str = Field(default="", description="摘要，2-3句话概括")


class SentimentScore(BaseModel):
    """
    单条新闻的 FinBERT 情感评分。

    confidence 取值 [0, 1]，由 ge=0.0, le=1.0 约束。
    explanation 记录判断依据，让下游 Risk Agent 能溯源推理——
    不是只给一个「看空」标签，而是说清为什么看空。
    """
    label:       SentimentLabel = Field(..., description="情感标签：看多/看空/中立")
    confidence:  float = Field(..., ge=0.0, le=1.0, description="置信度，越接近1越确定")
    explanation: str = Field(default="", description="判断依据简述，如'标题含'处罚'关键词'")


class ScoredNews(BaseModel):
    """
    附带情感评分的新闻。
    把 NewsItem + SentimentScore 打包装在一起，一条新闻一个对象。
    这样 downstream（Risk Agent / Report Agent）拿到的不是散落的两个列表，
    而是「每条新闻带了什么情感标签」的完整视图。
    """
    news:      NewsItem
    sentiment: SentimentScore


class TopicCluster(BaseModel):
    """
    BERTopic 热点主题聚类。

    Phase 1 返回占位聚类结果。Phase 2 接入 BERTopic 后，新闻按内容自动聚合，
    每个 Cluster 代表一个市场热点——如'欧美关税政策'、'固态电池突破'。

    representative_news 存代表性标题（最多 3 条），方便人类快速理解
    这个主题在讨论什么，不用点进每条新闻看。
    """
    topic_id:            int = Field(..., description="主题ID，从0开始编号")
    label:               str = Field(..., description="主题标签, 如'海外监管'/'供应链打压'")
    keywords:            list[str] = Field(default_factory=list, description="关键词列表，如['关税','欧盟','电动车']")
    news_count:          int = Field(default=0, description="该主题下新闻数量")
    representative_news: list[str] = Field(default_factory=list, description="代表性新闻标题（最多3条）")


class SentimentResult(BaseModel):
    """
    Sentiment Agent 完整输出。

    这是 Sentiment Agent 对外的「合同」——Risk Agent 和 Report Agent 都按这个
    结构消费数据。字段变更 = 合同变更，需要上下游同步。

    内容覆盖：
      - 有多少条新闻（searched_news_count）
      - 每条新闻的情感标签+置信度（scored_news）
      - 正/负/中立的数量统计（sentiment_distribution，方便画饼图）
      - 热点主题聚类（topics）
      - LLM 写的自然语言摘要（summary）
    """
    symbol:                  str
    company_name:            str
    searched_news_count:     int = Field(default=0, description="抓取新闻总数")
    scored_news:             list[ScoredNews] = Field(default_factory=list, description="带评分的新闻列表")
    sentiment_distribution:  dict[str, int] = Field(
        default_factory=lambda: {"positive": 0, "negative": 0, "neutral": 0},
        description="情感分布统计，key=positive/negative/neutral，value=条数"
    )
    topics:   list[TopicCluster] = Field(default_factory=list, description="BERTopic 热点主题聚类")
    summary:  str = Field(default="", description="LLM 生成的舆情分析摘要，200字以内")


# ════════════════════════════════════════════════════════════
# Risk Agent 模型
# ════════════════════════════════════════════════════════════

class FinancialSummary(BaseModel):
    """
    财务数据摘要（由 Financial Agent 输出，Risk Agent 消费）。

    所有财务指标都是 Optional——Financial Agent 可能因为数据源缺失
    （如某些企业不披露自由现金流）而返回 None。Risk Agent 在 assess_financial_risk
    中逐项检查 is not None 后才评估，不会因为缺字段而崩溃。

    anomalies 列表存异常项描述，如'应收账款周转天数同比增加30%'——
    这些由 Financial Agent 的 LLM 识别并写入，Risk Agent 逐条计入风险评分。
    """
    revenue_growth:    float | None = Field(default=None, description="营收同比增长率，如 0.15 = 15%")
    gross_margin:      float | None = Field(default=None, description="毛利率，如 0.22 = 22%")
    net_profit_margin: float | None = Field(default=None, description="净利率")
    debt_ratio:        float | None = Field(default=None, description="资产负债率")
    free_cash_flow:    float | None = Field(default=None, description="自由现金流（亿元）")
    anomalies:         list[str] = Field(default_factory=list, description="财务异常项描述列表")


class RiskDimension(BaseModel):
    """
    单一风险维度的评分。

    这是 Risk Agent 的「最小评估单元」——每个维度独立评分、独立举证：
      - dimension：维度名（舆情/财务/行业）
      - score：[0,1] 连续值
      - evidence：支撑该评分的具体证据列表
      - reasoning：人类可读的推理过程

    设计意图：评分不能是黑盒。下游（Report Agent / 人类分析师）需要看到
    每个评分的证据链，才能信任或质疑 AI 的判断。
    """
    dimension: str = Field(..., description="风险维度名称, 如'舆情负面信号'")
    score:     float = Field(..., ge=0.0, le=1.0, description="风险评分，0=无风险，1=极高风险")
    evidence:  list[str] = Field(default_factory=list, description="支撑证据，每条一个独立事实")
    reasoning: str = Field(default="", description="人类可读的推理过程")


class RiskAssessment(BaseModel):
    """
    Risk Agent 完整输出。

    核心字段：
      - overall_risk_level：最终结论（高/中/低），管理层一眼看懂
      - overall_score：[0,1] 连续值，用于排序/筛选/告警阈值
      - dimensions：三维度评分明细，可展开查看细节
      - reasoning_chain：完整推导链条，格式为 'A → B → C = 结论'，
        让分析师能逐环验证 AI 的推理是否符合逻辑

    key_risks 按严重程度排序，Report Agent 可以直接拿来生成「主要风险」段落。
    """
    symbol:             str
    company_name:       str
    overall_risk_level: RiskLevel = Field(..., description="综合风险等级（HIGH/MEDIUM/LOW）")
    overall_score:      float = Field(..., ge=0.0, le=1.0, description="综合风险评分")
    dimensions:         list[RiskDimension] = Field(default_factory=list, description="各维度风险评分明细")
    risk_summary:       str = Field(default="", description="LLM 润色的风险总结，200字以内")
    key_risks:          list[str] = Field(default_factory=list, description="关键风险项，按严重程度排列")
    reasoning_chain:    str = Field(default="", description="完整风险推导链条，箭头分隔每步推理")


# ════════════════════════════════════════════════════════════
# 联合输出（供 workflow 下游消费）
# ════════════════════════════════════════════════════════════

class SentimentRiskJointOutput(BaseModel):
    """
    Sentiment + Risk 联合输出。

    为什么需要这个？
      Report Agent 需要同时拿到舆情和风险两个结果才能生成完整研报。
      直接把两个 Agent 输出打一个包，一个对象传到底，避免散落参数。

    generated_at 用 default_factory=datetime.now（工厂函数），
    不用 default=datetime.now()（函数调用）——后者会让所有实例共享
    同一个时间戳（定义时求值一次），前者每次创建新实例才求值。
    """
    sentiment:    SentimentResult
    risk:         RiskAssessment
    generated_at: datetime = Field(default_factory=datetime.now, description="生成时间戳")
