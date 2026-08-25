"""研究意图理解（PR #36）。

规则驱动：NL 请求 → 意图分类（关键词权重）+ 目标抽取（公司/业务板块）+ 维度推导。

设计（类比 QueryRewrite 的 RuleBasedQueryRewriter）：
    - 零依赖、可解释：每个关键词命中都可审计；
    - rule 模式够覆盖金融研究意图；LLM 意图理解留作后续增强（_parse seam）。

输出：ResearchIntent + ResearchTarget + dimensions，供 ResearchPlanner 生成步骤。
"""

from __future__ import annotations

from app.core.config import settings
from app.rag.research.schema import ResearchDimension, ResearchIntent, ResearchTarget

# 意图 → 触发关键词（按权重排序，key 命中得更多分）
_INTENT_KEYWORDS: dict[ResearchIntent, list[str]] = {
    ResearchIntent.COMPETITIVE_ANALYSIS: [
        "竞争力", "竞争优势", "竞争格局", "壁垒", "护城河", "领先优势",
    ],
    ResearchIntent.RISK_ANALYSIS: [
        "风险", "不确定性", "危机", "隐患", "威胁",
    ],
    ResearchIntent.STRATEGY_ANALYSIS: [
        "战略", "未来规划", "未来方向", "发展前景", "路线图", "愿景",
    ],
    ResearchIntent.FINANCIAL_ANALYSIS: [
        "财务", "收入", "营收", "盈利", "利润", "负债", "现金流", "毛利率", "净利率",
    ],
    ResearchIntent.MARKET_ANALYSIS: [
        "市场", "行业", "出货量", "销量", "市场份额", "渗透率", "需求",
    ],
    ResearchIntent.POLICY_ANALYSIS: [
        "政策", "补贴", "监管", "法规", "合规", "关税",
    ],
    ResearchIntent.TECHNOLOGY_ANALYSIS: [
        "技术", "研发", "专利", "创新", "技术路线", "软件", "芯片", "电池技术",
    ],
    ResearchIntent.BUSINESS_OVERVIEW: [
        "业务", "商业模式", "企业概况", "公司介绍", "主营", "产品矩阵",
    ],
}

# 业务板块关键词 → 标准化 segment 名
_SEGMENT_KEYWORDS: dict[str, list[str]] = {
    "汽车": ["汽车", "智能电动汽车", "新能源汽车", "电动车", "整车", "智能驾驶"],
    "手机": ["手机", "智能手机", "智能终端", "移动通信"],
    "电池": ["电池", "储能", "动力电池", "锂电", "电芯"],
    "IoT": ["物联网", "iot", "智能家居", "生态链"],
    "AI": ["人工智能", "ai", "大模型", "智能体"],
}

# 意图 → 默认研究维度（planner 用；segment 特定维度由 _infer_dimensions 追加）
_INTENT_DIMENSIONS: dict[ResearchIntent, list[ResearchDimension]] = {
    ResearchIntent.COMPETITIVE_ANALYSIS: [
        ResearchDimension.BUSINESS, ResearchDimension.TECHNOLOGY,
        ResearchDimension.MARKET, ResearchDimension.COMPETITION,
        ResearchDimension.RISK, ResearchDimension.STRATEGY,
    ],
    ResearchIntent.BUSINESS_OVERVIEW: [
        ResearchDimension.BUSINESS, ResearchDimension.PRODUCT,
        ResearchDimension.MARKET, ResearchDimension.STRATEGY,
    ],
    ResearchIntent.FINANCIAL_ANALYSIS: [
        ResearchDimension.FINANCIAL, ResearchDimension.BUSINESS, ResearchDimension.RISK,
    ],
    ResearchIntent.RISK_ANALYSIS: [
        ResearchDimension.RISK, ResearchDimension.FINANCIAL,
        ResearchDimension.MARKET, ResearchDimension.POLICY,
    ],
    ResearchIntent.STRATEGY_ANALYSIS: [
        ResearchDimension.STRATEGY, ResearchDimension.BUSINESS,
        ResearchDimension.TECHNOLOGY, ResearchDimension.MARKET,
    ],
    ResearchIntent.MARKET_ANALYSIS: [
        ResearchDimension.MARKET, ResearchDimension.COMPETITION,
        ResearchDimension.PRODUCT, ResearchDimension.STRATEGY,
    ],
    ResearchIntent.POLICY_ANALYSIS: [
        ResearchDimension.POLICY, ResearchDimension.MARKET, ResearchDimension.RISK,
    ],
    ResearchIntent.TECHNOLOGY_ANALYSIS: [
        ResearchDimension.TECHNOLOGY, ResearchDimension.PRODUCT, ResearchDimension.COMPETITION,
    ],
    ResearchIntent.GENERIC_RESEARCH: list(ResearchDimension),
}

# 已知公司（config 可覆盖）。含简称 → 标准名映射。
_COMPANY_ALIASES: dict[str, str] = {
    "小米": "小米", "小米集团": "小米", "xiaomi": "小米",
    "宁德时代": "宁德时代", "宁德": "宁德时代", "catl": "宁德时代",
    "小鹏": "小鹏汽车", "小鹏汽车": "小鹏汽车", "xpeng": "小鹏汽车",
}


def _known_companies() -> list[str]:
    """已知公司标准名列表（settings 优先，否则用别名表 key）。"""
    if settings.rag_known_companies:
        return list(settings.rag_known_companies)
    return list(dict.fromkeys(_COMPANY_ALIASES.values()))


class IntentParser:
    """研究意图解析器：NL 请求 → (intent, target, dimensions)。

    rule 模式为默认（确定性、零成本）；LLM 模式留作后续增强。
    """

    def __init__(self, *, _parse: callable | None = None) -> None:
        """Args:
            _parse: 测试 seam —— 注入 (request) -> (intent, target, dimensions)。
        """
        self._parse_fn = _parse

    # ── 主入口 ─────────────────────────────────────────────────
    def parse(self, request: str) -> tuple[ResearchIntent, ResearchTarget, list[ResearchDimension]]:
        """解析请求 → (intent, target, dimensions)。"""
        if self._parse_fn is not None:
            return self._parse_fn(request)
        intent = self._classify_intent(request)
        target = self._extract_target(request)
        dimensions = self._infer_dimensions(intent, target)
        return intent, target, dimensions

    # ── 意图分类 ───────────────────────────────────────────────
    def _classify_intent(self, request: str) -> ResearchIntent:
        """关键词加权分类：命中数最多者胜；无命中 → GENERIC_RESEARCH。"""
        text = request.lower()
        best: ResearchIntent = ResearchIntent.GENERIC_RESEARCH
        best_score = 0
        for intent, keywords in _INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in text)
            if score > best_score:
                best, best_score = intent, score
        return best

    # ── 目标抽取 ───────────────────────────────────────────────
    def _extract_target(self, request: str) -> ResearchTarget:
        """抽取 company + segment（规则）。"""
        return ResearchTarget(
            company=self._extract_company(request),
            segment=self._extract_segment(request),
        )

    def _extract_company(self, request: str) -> str:
        """公司匹配：优先最长别名命中（避免"小米"先命中遮住"小鹏"）。"""
        text = request.lower()
        matched = ""
        # 按别名长度降序，取首个命中
        for alias in sorted(_COMPANY_ALIASES, key=len, reverse=True):
            if alias.lower() in text:
                matched = _COMPANY_ALIASES[alias]
                break
        # 兜底：已知公司名直接匹配
        if not matched:
            for name in _known_companies():
                if name.lower() in text:
                    matched = name
                    break
        return matched

    def _extract_segment(self, request: str) -> str:
        """业务板块匹配：关键词命中（复用 query rewrite 同义词思路）。"""
        text = request.lower()
        for segment, keywords in _SEGMENT_KEYWORDS.items():
            if any(kw.lower() in text for kw in keywords):
                return segment
        return ""

    # ── 维度推导 ───────────────────────────────────────────────
    def _infer_dimensions(
        self,
        intent: ResearchIntent,
        target: ResearchTarget,
    ) -> list[ResearchDimension]:
        """意图默认维度 + segment 特定维度（汽车→政策、电池→技术）。"""
        dims = list(_INTENT_DIMENSIONS[intent])
        if target.segment == "汽车" and ResearchDimension.POLICY not in dims:
            dims.append(ResearchDimension.POLICY)
        if target.segment == "电池" and ResearchDimension.TECHNOLOGY not in dims:
            dims.append(ResearchDimension.TECHNOLOGY)
        # 去重保序
        return list(dict.fromkeys(dims))
