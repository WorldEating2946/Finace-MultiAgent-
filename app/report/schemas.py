"""
统一研报 Schema —— 四 Agent 输出 → 结构化报告内容 → 导出。

成员 5（平台工程与报告生成）：Report Agent 汇总 Research / Financial /
Sentiment / Risk 四方输出，合成标准化投研报告（企业概况 → 财务分析 →
行业竞争力 → 舆情风向 → 风险评估 → 投资建议）。

设计要点：
- 报告内容用类型化 block（para / bullets / table / quote）表达，
  exporter 从同一结构派生 Markdown 与自包含 HTML，零解析、零依赖、确定性。
- 输入端（ReportGenerateRequest）接收各 Agent 输出的 dict（model_dump），
  与具体 Agent 解耦 —— 无论数据来自主图 final state 还是 demo 脚本均可。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ReportGenerateRequest(BaseModel):
    """统一研报生成请求 —— 接收四 Agent 输出 dict。"""

    company: str = Field(..., min_length=1, description="公司名称")
    ticker: str = Field(default="", description="股票代码")
    user_query: str = Field(default="", description="用户原始提问")
    research: dict | None = Field(
        default=None,
        description="Research Agent 输出（ResearchReport 或 research_result dict）",
    )
    financial: dict | None = Field(
        default=None, description="Financial Agent 输出（FinancialAgentOutput dict）"
    )
    sentiment: dict | None = Field(
        default=None, description="Sentiment Agent 输出（SentimentResult dict）"
    )
    risk: dict | None = Field(
        default=None, description="Risk Agent 输出（RiskAssessment dict）"
    )

    model_config = ConfigDict(extra="ignore")


class ReportBlock(BaseModel):
    """报告中的单个内容块（类型化，渲染确定性）。"""

    kind: Literal["para", "bullets", "table", "quote"] = Field(..., description="块类型")
    text: str = Field(default="", description="para / quote 的正文")
    items: list[str] = Field(default_factory=list, description="bullets 的列表项")
    headers: list[str] = Field(default_factory=list, description="table 的表头")
    rows: list[list[str]] = Field(default_factory=list, description="table 的数据行")

    model_config = ConfigDict(extra="ignore")


class ReportSection(BaseModel):
    """报告章节。"""

    key: str = Field(..., description="章节标识（chapter_1 ... chapter_6）")
    title: str = Field(..., description="章节标题")
    blocks: list[ReportBlock] = Field(default_factory=list, description="内容块列表")

    model_config = ConfigDict(extra="ignore")


class ReportContent(BaseModel):
    """结构化研报内容（assembler 产物，exporter 输入）。"""

    title: str = Field(..., description="报告标题")
    company: str = Field(..., description="公司名称")
    ticker: str = Field(default="", description="股票代码")
    generated_at: str = Field(default="", description="生成时间 ISO 8601")
    sections: list[ReportSection] = Field(default_factory=list, description="章节列表")

    model_config = ConfigDict(extra="ignore")


class ReportOutput(BaseModel):
    """导出结果（API 返回 / 落盘信息）。"""

    report_id: str = Field(..., description="报告 ID（= 落盘子目录名）")
    title: str = Field(..., description="报告标题")
    markdown: str = Field(default="", description="Markdown 全文")
    markdown_path: str = Field(default="", description="markdown 落盘路径")
    html_path: str = Field(default="", description="html 落盘路径")
    generated_at: str = Field(default="", description="生成时间 ISO 8601")

    model_config = ConfigDict(extra="ignore")
