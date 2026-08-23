"""研究计划生成（PR #36）。

模板驱动：ResearchIntent → 有序 ResearchStep 序列。
{company}/{segment} 在 plan() 时替换；segment 为空自动省略相关关键词。

步骤只描述"要研究什么"，不执行——执行在 PR #37 Agent Workflow。
"""

from __future__ import annotations

from app.rag.research.schema import (
    ResearchDimension,
    ResearchIntent,
    ResearchPlan,
    ResearchStep,
    ResearchTarget,
)

# 步骤模板：{company}/{segment} 占位符在 plan() 时替换
# query 中的 " {segment}" 在 segment 为空时整体移除
_TEMPLATES: dict[ResearchIntent, list[dict]] = {
    ResearchIntent.COMPETITIVE_ANALYSIS: [
        dict(name="企业知识画像", query="{company} 企业概况 商业模式 主营",
             dims=["business"], src=["annual_report"], fields=[]),
        dict(name="{segment}业务深度分析", query="{company} {segment} 业务 战略 布局",
             dims=["business", "strategy"], src=["annual_report"],
             fields=["business_segments", "strategic_direction"]),
        dict(name="竞争格局与壁垒", query="{company} {segment} 竞争 壁垒 优势 份额",
             dims=["competition"], src=["annual_report", "research_report"],
             fields=["competitive_advantages"]),
        dict(name="市场环境与客户", query="{company} {segment} 市场 客户 需求 全球化",
             dims=["market"], src=["annual_report"],
             fields=["geographic_markets", "customers"]),
        dict(name="技术壁垒与研发", query="{company} {segment} 技术 研发 专利",
             dims=["technology"], src=["annual_report"],
             fields=["technologies"]),
        dict(name="政策与监管环境", query="{segment} 政策 补贴 监管 法规",
             dims=["policy"], src=["policy"], fields=[]),
        dict(name="风险因素", query="{company} {segment} 风险 不确定性",
             dims=["risk"], src=["annual_report", "research_report"],
             fields=["risks"]),
        dict(name="综合研判报告", query="{company} {segment} 综合 总结 前景 展望",
             dims=["strategy"], src=["annual_report", "research_report"], fields=[]),
    ],
    ResearchIntent.BUSINESS_OVERVIEW: [
        dict(name="企业知识画像", query="{company} 企业概况 商业模式 主营",
             dims=["business"], src=["annual_report"], fields=[]),
        dict(name="业务矩阵", query="{company} 业务 板块 分部",
             dims=["business"], src=["annual_report"], fields=["business_segments"]),
        dict(name="产品矩阵", query="{company} 产品 产品线 组合",
             dims=["product"], src=["annual_report"], fields=["products"]),
        dict(name="客户与市场定位", query="{company} 客户 用户 市场 定位",
             dims=["market"], src=["annual_report"],
             fields=["customers", "geographic_markets"]),
        dict(name="战略方向", query="{company} 战略 未来 规划 愿景",
             dims=["strategy"], src=["annual_report"], fields=["strategic_direction"]),
        dict(name="综合研判报告", query="{company} 业务 综合 总结 评价",
             dims=["business"], src=["annual_report", "research_report"], fields=[]),
    ],
    ResearchIntent.FINANCIAL_ANALYSIS: [
        dict(name="企业知识画像", query="{company} 企业概况 商业模式",
             dims=["business"], src=["annual_report"], fields=[]),
        dict(name="财务概览", query="{company} 财务 收入 利润 资产负债",
             dims=["financial"], src=["annual_report"], fields=[]),
        dict(name="盈利能力", query="{company} 毛利率 净利率 盈利能力 盈利质量",
             dims=["financial"], src=["annual_report"], fields=[]),
        dict(name="现金流与负债", query="{company} 现金流 负债 偿债能力",
             dims=["financial"], src=["annual_report"], fields=[]),
        dict(name="财务风险", query="{company} 财务风险 风险因素 减值",
             dims=["financial", "risk"], src=["annual_report"],
             fields=["risks"]),
        dict(name="综合研判报告", query="{company} 财务 总结 前景 质量",
             dims=["financial"], src=["annual_report", "research_report"], fields=[]),
    ],
    ResearchIntent.RISK_ANALYSIS: [
        dict(name="企业知识画像", query="{company} 企业概况 商业模式",
             dims=["business"], src=["annual_report"], fields=[]),
        dict(name="经营风险", query="{company} 经营风险 风险因素 不确定性",
             dims=["risk"], src=["annual_report", "research_report"], fields=["risks"]),
        dict(name="财务风险", query="{company} 财务风险 负债 现金流 减值",
             dims=["financial", "risk"], src=["annual_report"], fields=[]),
        dict(name="市场与竞争风险", query="{company} {segment} 市场风险 竞争 冲击",
             dims=["market", "competition"], src=["annual_report", "research_report"], fields=[]),
        dict(name="政策与监管风险", query="{segment} 政策 监管 法规 风险",
             dims=["policy", "risk"], src=["policy"], fields=[]),
        dict(name="风险综合研判", query="{company} 风险 综合 研判 应对",
             dims=["risk"], src=["annual_report", "research_report"], fields=[]),
    ],
    ResearchIntent.STRATEGY_ANALYSIS: [
        dict(name="企业知识画像", query="{company} 企业概况 商业模式",
             dims=["business"], src=["annual_report"], fields=[]),
        dict(name="战略方向与愿景", query="{company} 战略 愿景 未来规划 方向",
             dims=["strategy"], src=["annual_report"], fields=["strategic_direction"]),
        dict(name="业务布局与投资", query="{company} {segment} 业务 布局 投资 扩张",
             dims=["business", "strategy"], src=["annual_report", "research_report"],
             fields=["business_segments"]),
        dict(name="技术路线", query="{company} 技术 研发 路线 储备",
             dims=["technology"], src=["annual_report"], fields=["technologies"]),
        dict(name="市场机会", query="{company} {segment} 市场机会 增长 空间",
             dims=["market"], src=["research_report"], fields=[]),
        dict(name="综合研判报告", query="{company} 战略 综合 总结 前景",
             dims=["strategy"], src=["annual_report", "research_report"], fields=[]),
    ],
    ResearchIntent.MARKET_ANALYSIS: [
        dict(name="企业知识画像", query="{company} 企业概况 商业模式",
             dims=["business"], src=["annual_report"], fields=[]),
        dict(name="市场格局", query="{company} {segment} 市场 格局 份额 规模",
             dims=["market"], src=["annual_report", "research_report"], fields=[]),
        dict(name="出货与需求", query="{company} {segment} 出货量 销量 需求",
             dims=["market"], src=["annual_report"], fields=[]),
        dict(name="竞争态势", query="{company} {segment} 竞争 对手 格局",
             dims=["competition"], src=["research_report"],
             fields=["competitive_advantages"]),
        dict(name="增长前景", query="{company} {segment} 增长 前景 空间 渗透率",
             dims=["market", "strategy"], src=["research_report"], fields=[]),
        dict(name="综合研判报告", query="{company} {segment} 市场 综合 总结",
             dims=["market"], src=["annual_report", "research_report"], fields=[]),
    ],
    ResearchIntent.POLICY_ANALYSIS: [
        dict(name="企业知识画像", query="{company} 企业概况 商业模式",
             dims=["business"], src=["annual_report"], fields=[]),
        dict(name="相关政策梳理", query="{segment} 政策 补贴 法规 支持",
             dims=["policy"], src=["policy"], fields=[]),
        dict(name="监管与合规影响", query="{company} {segment} 监管 合规 影响",
             dims=["policy"], src=["policy", "news"], fields=[]),
        dict(name="行业趋势与机遇", query="{company} {segment} 行业 趋势 机遇",
             dims=["market", "policy"], src=["research_report"], fields=[]),
        dict(name="政策风险与应对", query="{company} 政策风险 应对 措施",
             dims=["policy", "risk"], src=["annual_report"], fields=["risks"]),
        dict(name="综合研判报告", query="{company} 政策 综合 总结 影响",
             dims=["policy"], src=["policy", "research_report"], fields=[]),
    ],
    ResearchIntent.TECHNOLOGY_ANALYSIS: [
        dict(name="企业知识画像", query="{company} 企业概况 商业模式",
             dims=["business"], src=["annual_report"], fields=[]),
        dict(name="核心技术", query="{company} 核心技术 研发投入 研发",
             dims=["technology"], src=["annual_report"], fields=["technologies"]),
        dict(name="技术路线与创新", query="{company} {segment} 技术路线 创新 前沿",
             dims=["technology"], src=["annual_report", "research_report"], fields=[]),
        dict(name="专利与壁垒", query="{company} 专利 技术壁垒 护城河 知识产权",
             dims=["technology", "competition"], src=["annual_report"], fields=[]),
        dict(name="技术趋势", query="{company} {segment} 技术趋势 未来 方向",
             dims=["technology", "strategy"], src=["research_report"], fields=[]),
        dict(name="综合研判报告", query="{company} 技术 综合 总结 领先性",
             dims=["technology"], src=["annual_report", "research_report"], fields=[]),
    ],
    ResearchIntent.GENERIC_RESEARCH: [
        dict(name="企业知识画像", query="{company} 企业概况 商业模式 主营",
             dims=["business"], src=["annual_report"], fields=[]),
        dict(name="业务与产品", query="{company} 业务 产品 板块",
             dims=["business", "product"], src=["annual_report"],
             fields=["business_segments", "products"]),
        dict(name="技术与研发", query="{company} 技术 研发 专利",
             dims=["technology"], src=["annual_report"], fields=["technologies"]),
        dict(name="市场与竞争", query="{company} {segment} 市场 竞争 客户",
             dims=["market", "competition"], src=["annual_report", "research_report"],
             fields=["geographic_markets", "customers", "competitive_advantages"]),
        dict(name="财务与风险", query="{company} 财务 风险 负债",
             dims=["financial", "risk"], src=["annual_report"], fields=["risks"]),
        dict(name="战略与前景", query="{company} 战略 未来 前景",
             dims=["strategy"], src=["annual_report", "research_report"],
             fields=["strategic_direction"]),
        dict(name="综合研判报告", query="{company} 综合 总结 展望",
             dims=["business", "strategy"], src=["annual_report", "research_report"], fields=[]),
    ],
}


def _fill(template: str, company: str, segment: str) -> str:
    """替换 {company}/{segment}；segment 为空移除 " {segment}" token。"""
    if not segment:
        template = template.replace(" {segment}", "").replace("{segment}", "")
    return template.replace("{company}", company or "").replace("{segment}", segment)


# 缺失维度 → 补充研究步骤模板（PR38 动态补步）。
# evaluate 发现某维度证据产出不足（low_yield_steps）→ Replan Node 据此生成补充步骤。
_MISSING_TEMPLATES: dict[str, dict] = {
    "risk": dict(
        name="{company}{segment}风险补充",
        query="{company} {segment} 风险 财务风险 经营风险 不确定性",
        dims=["risk"], src=["annual_report", "research_report"], fields=["risks"]),
    "competition": dict(
        name="{company}{segment}竞争补充",
        query="{company} {segment} 竞争对手 市场份额 壁垒 优势",
        dims=["competition"], src=["research_report"], fields=["competitive_advantages"]),
    "technology": dict(
        name="{company}{segment}技术补充",
        query="{company} {segment} 技术 研发投入 专利 技术壁垒",
        dims=["technology"], src=["annual_report"], fields=["technologies"]),
    "financial": dict(
        name="{company}{segment}财务补充",
        query="{company} 财务数据 毛利率 净利率 负债 现金流",
        dims=["financial"], src=["annual_report"], fields=[]),
    "market": dict(
        name="{company}{segment}市场补充",
        query="{company} {segment} 市场份额 客户 需求 全球化",
        dims=["market"], src=["annual_report"], fields=[]),
    "policy": dict(
        name="{company}{segment}政策补充",
        query="{segment} 法规 监管 补贴 政策 合规",
        dims=["policy"], src=["policy"], fields=[]),
}


def add_replan_step(plan: ResearchPlan, missing_dimension: str) -> ResearchStep | None:
    """为缺失维度生成一个补充研究步骤（PR38 Replan Node 用）。

    Args:
        plan:              已执行的 ResearchPlan（steps 可能部分完成）。
        missing_dimension:  ResearchDimension 值（"risk"/"competition"/...）。

    Returns:
        新 ResearchStep（order = len(plan.steps)+1），无匹配模板返回 None。
    """
    tmpl = _MISSING_TEMPLATES.get(missing_dimension)
    if not tmpl:
        return None
    order = len(plan.steps) + 1
    company, segment = plan.target.company, plan.target.segment
    name = _fill(tmpl["name"], company, segment)
    return ResearchStep(
        order=order,
        name=name,
        description=f"补充研究：{_fill(tmpl['query'], company, segment)}",
        retrieval_query=_fill(tmpl["query"], company, segment),
        dimensions=list(tmpl["dims"]),
        source_types=list(tmpl["src"]),
        profile_fields=list(tmpl["fields"]),
        depends_on=list(range(1, order)),
    )


class ResearchPlanner:
    """研究计划生成器：intent + target → 有序 ResearchStep 列表。"""

    def plan(
        self,
        intent: ResearchIntent,
        target: ResearchTarget,
        dimensions: list[ResearchDimension] | None = None,
    ) -> list[ResearchStep]:
        """生成步骤序列。dimensions 仅用于元信息，步骤由 intent 模板驱动。"""
        template = _TEMPLATES[intent]
        company, segment = target.company, target.segment
        steps: list[ResearchStep] = []
        for i, t in enumerate(template, start=1):
            name = _fill(t["name"], company, segment)
            steps.append(
                ResearchStep(
                    order=i,
                    name=name,
                    description=f"{name}：{_fill(t['query'], company, segment)}",
                    retrieval_query=_fill(t["query"], company, segment),
                    dimensions=list(t["dims"]),
                    source_types=list(t["src"]),
                    profile_fields=list(t["fields"]),
                    depends_on=list(range(1, i)),  # 顺序执行：依赖前序步骤
                )
            )
        return steps
