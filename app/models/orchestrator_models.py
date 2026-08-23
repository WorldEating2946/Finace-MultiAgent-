"""
统一编排 Schema：Manager Agent 调度所有子 Agent 的入参/出参。

════════════════════ 为什么需要这一层？ ════════════════════════
五个 Agent（Research/Financial/Sentiment/Risk/Report）各自的 State 结构
千差万别——Sentiment 有关键词分布、Risk 有维度评分。Manager Agent 要统一
调度它们，就需要一套「与具体 Agent 无关」的通用入参/出参。

这就是本文件的三个 Schema 的职责：
  - AgentRequest  →  「无论调哪个 Agent，入参格式统一」
  - AgentResponse →  「无论哪个 Agent 返回，出参格式统一」
  - PipelineResult → 「无论串联多少 Agent，聚合结果格式统一」

参照 EduAgent backend/core/orchestrator.py 的 AgentRequest / AgentResponse /
PipelineResult 模式，适配金融投研场景（股票代码、企业名、投研全链路）。
═══════════════════════════════════════════════════════════════
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ════════════════════════════════════════════════════════════
# 枚举
# ════════════════════════════════════════════════════════════

class OrchestratorAgentType(str, Enum):
    """
    可被编排器调度的 Agent 类型。

    为什么继承 str, Enum 而不是单纯的 Enum？
      → 因为 str, Enum 的成员可以直接当字符串用，传给函数或拼接 URL
        都不需要 .value 转换。写 agent_type == "sentiment" 也能比较。
        单纯 Enum 必须 agent_type == AgentType.SENTIMENT 才行。
    """
    RESEARCH   = "research"     # 企业知识检索（RAG）：查年报、商业模式、行业政策
    FINANCIAL  = "financial"    # 财务数据分析：营收、利润、现金流、杜邦分析
    SENTIMENT  = "sentiment"    # 舆情情感分析：新闻抓取 + FinBERT 多空评分 + 热点聚类
    RISK       = "risk"         # 综合风险评估：融合舆情 + 财务 + 行业 三维度评分
    REPORT     = "report"       # 研报生成：汇总所有 Agent 输出 → Markdown/PDF


class ExecutionMode(str, Enum):
    """
    执行模式——决定一个请求「怎么跑」。

    Manager Agent 根据用户意图选择执行模式：
      - "分析 300750 的舆情"      → SINGLE（直接调 Sentiment Agent）
      - "全面分析宁德时代"         → PIPELINE（串联所有 Agent）
      - "宁德时代怎么样"           → PARALLEL（同时调多个 Agent 加速）
      - "分析一下"                → CLARIFY（信息不够，追问股票代码）
    """
    SINGLE   = "single"       # 单 Agent 直达：只调一个 Agent
    PARALLEL = "parallel"     # 多 Agent 并行：Research + Financial + Sentiment 同时跑
    PIPELINE = "pipeline"     # 多 Agent 串联：Parallel 结果 → Risk Agent → Report Agent
    CLARIFY  = "clarify"      # 意图不明：返回追问，让用户补充信息（如股票代码）


# ════════════════════════════════════════════════════════════
# 统一 Schema
# ════════════════════════════════════════════════════════════

class AgentRequest(BaseModel):
    """
    所有 Agent 请求的统一入参 Schema。

    设计思路：
      各个 Agent 内部有自己的 Input Schema（如 SentimentInput 需要 symbol/
      company_name/days），但 Manager Agent 不能耦合到每个 Agent 的具体字段。
      所以 AgentRequest 设一个 context 兜底字典——Agent 需要什么特殊字段，
      就往 context 里塞（如 {"days": 60, "peer_comparison": True}）。

    最终流程：
      API 层接收用户请求 → 构造 AgentRequest → Orchestrator.handle()
      → 把 context 平铺进该 Agent 的 initial_state → Agent 图执行
    """

    # ── 必填字段：没有默认值 = Field(...) = 缺了 Pydantic 自动 422 ──
    user_id:    str = Field(..., description="用户 ID")
    session_id: str = Field(..., description="会话 ID，用于拼接 thread_id（LangGraph 靠它断点续传）")
    agent_type: OrchestratorAgentType = Field(..., description="目标 Agent 类型，Orchestrator 据此路由")
    user_message: str = Field(..., description="用户输入的原始文本，如'分析宁德时代风险'")

    # ── 可选字段：金融场景特有，大部分请求会带上 ──
    symbol:       str = Field(default="", description="股票代码，如 300750。空串表示用户未指定")
    company_name: str = Field(default="", description="企业名称，如 宁德时代。空串 = 让 Agent 自己识别")

    # ── Pipeline 数据通道 ──
    context: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "附加上下文 / 前序 Agent 的输出。\n"
            "单 Agent 模式：放额外参数（如 days=60）。\n"
            "Pipeline 模式：每步 Agent 的 structured 结果存进来，"
            "下一步 Agent 从此读取上游数据。\n"
            "例：{'sentiment_result': {...}, 'financial_summary': {...}}"
        ),
    )
    pipeline_mode: bool = Field(
        default=False,
        description="是否强制走串联 Pipeline。True = 按 PIPELINE_REGISTRY 顺序执行",
    )

    # ── 计算属性 ──
    @property
    def thread_id(self) -> str:
        """
        LangGraph Checkpointer（MemorySaver）使用的线程 ID。

        为什么是 @property 而不直接用字段？
          → thread_id 的拼接规则必须和各个 Agent 内部的 build_thread_id
            格式严格一致，否则断点续传命中不了正确的检查点。
            做成只读属性，避免手拼时格式出错。
            规则：user_{user_id}_session_{session_id}
        """
        return f"user_{self.user_id}_session_{self.session_id}"


class AgentResponse(BaseModel):
    """
    所有 Agent 响应的统一出参 Schema。

    设计要点：
      - success + fallback_used：让上游知道结果是正常返回还是降级/兜底
      - content（文本）+ structured（结构化数据）双通道：研报走 content，
        评分/指标走 structured。前端可以精准渲染评分卡片，不是只能显示长文本
      - error_msg：失败时不抛异常，而是返回 AgentResponse(success=False, error_msg=...)
        保证前端始终能渲染，不会白屏
    """
    success:       bool = Field(..., description="执行是否成功")
    agent_type:    OrchestratorAgentType = Field(..., description="实际执行的 Agent 类型")
    content:       str = Field(default="", description="主要文本响应（摘要、解释、报告）")
    structured:    dict[str, Any] | None = Field(
        default=None,
        description="结构化数据。例如 Risk Agent 返回 {'overall_score': 0.65, 'risk_level': 'medium', ...}",
    )
    fallback_used: bool = Field(default=False, description="是否触发了降级处理（重试耗尽后的备选方案）")
    error_msg:     str | None = Field(default=None, description="失败时的错误信息，成功时为 None")
    metadata:      dict[str, Any] = Field(
        default_factory=dict,
        description="附加元数据，如 {'elapsed_seconds': 3.2, 'model': 'deepseek-chat'}",
    )


class PipelineResult(BaseModel):
    """
    多 Agent 串联 Pipeline 的聚合结果。

    使用场景——用户说「全面分析宁德时代」：
      1. PARALLEL：Research + Financial + Sentiment 同时跑 → 3 个 AgentResponse
      2. PIPELINE：上一步结果喂给 Risk Agent → 1 个 AgentResponse
      3. PIPELINE：Risk 结果喂给 Report Agent → 1 个 AgentResponse
      4. 5 个 AgentResponse 全部进 steps，structured 按 key 聚合进 combined

    all_success 让前端快速判断是否需要展示「部分结果可能不完整」的提示。
    """

    steps: list[AgentResponse] = Field(
        default_factory=list,
        description="各步骤的 AgentResponse 列表，按执行顺序排列",
    )
    combined: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "聚合后的最终数据。\n"
            "规则：每步的 structured 以 agent_type.value 为 key 存入 combined。\n"
            "例：{'research': {...}, 'financial': {...}, 'sentiment': {...}, "
            "'risk': {...}, 'report': '...markdown...'}"
        ),
    )
    all_success: bool = Field(
        default=False,
        description="所有步骤是否全部成功。任一 Agent 失败则为 False",
    )


# ════════════════════════════════════════════════════════════
# Pipeline 定义（Manager Agent 调度依据）
# ════════════════════════════════════════════════════════════

# 标准投研全链路：
#   Manager 先并行调 Research + Financial + Sentiment
#   → 三步结果汇总后交给 Risk 做综合推演
#   → Risk 输出交给 Report 合成最终研报
# 对应 PDF 设计文档的 Step 1~4 执行生命周期
INVESTMENT_RESEARCH_PIPELINE: list[OrchestratorAgentType] = [
    OrchestratorAgentType.RESEARCH,     # Step 2a：企业知识 + 行业背景（RAG）
    OrchestratorAgentType.FINANCIAL,    # Step 2b：财务数据 + 指标计算（API）
    OrchestratorAgentType.SENTIMENT,    # Step 2c：新闻舆情 + 情感评分（FinBERT）
    # ── 以上三个在 Manager Agent 中通过 asyncio.gather 并行执行 ──
    OrchestratorAgentType.RISK,         # Step 3：融合三维度信号，综合风险判定
    OrchestratorAgentType.REPORT,       # Step 4：汇总生成标准化投研报告
]

# 快速舆情查询：
#   用户只想看舆情风向 + 风险判断，不需要完整研报
#   Sentiment → Risk，两步直达
SENTIMENT_RISK_PIPELINE: list[OrchestratorAgentType] = [
    OrchestratorAgentType.SENTIMENT,
    OrchestratorAgentType.RISK,
]

# Pipeline 注册表：
#   key = 用户意图标签（由 Manager Agent 的意图路由产出）
#   value = 该意图对应的 Agent 执行顺序
# 扩展方式：加一行 key → list[AgentType] 即可，不动 Manager Agent 代码
PIPELINE_REGISTRY: dict[str, list[OrchestratorAgentType]] = {
    "full_research":    INVESTMENT_RESEARCH_PIPELINE,     # 完整投研分析
    "sentiment_risk":   SENTIMENT_RISK_PIPELINE,          # 舆情 + 风险（你负责的模块！）
}
