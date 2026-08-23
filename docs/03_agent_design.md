# 03 · Research Agent 设计

> Research Agent V1 · 2026-08-07 · 对应代码 `app/rag/research/`

---

## 1. 定位

Research Agent 是 V1 交付的**唯一生产 Agent**，负责企业深度研究：理解研究意图 → 生成结构化计划 → 逐步证据检索 → 合成带证据链的报告 → 质量评测。

**职责边界**：只做企业知识研究与证据检索。财务计算、实时舆情不在 V1 范围（数据分类原则见 `docs/02_rag_pipeline.md` §1）。

---

## 2. 整体流程（六步）

```
研究请求
  │
  ▼
① 意图理解 intent.py
  │    9 类意图关键词加权分类 + 公司别名归一
  │    （business_overview / competitive_analysis / risk_analysis / financial / market / ...）
  ▼
② 计划生成 planner.py
  │    按意图选模板（_TEMPLATES），补步骤、去重
  │    e.g. business_overview → 企业知识画像 + 业务发展(财务/产品/市场/客户)
  ▼
③ 画像构建 profile/
  │    9 字段 LLM 抽取（主营/客户/竞争/风险...）+ 证据归因
  │    缓存 data/profiles/{company}.json，每公司构建一次
  ▼
④ 逐步执行 executor.py
  │    遍历计划步骤 → 每步调 Hybrid RAG 检索 → 收集 EvidenceRef
  │    步骤名含"画像/企业概况/企业知识" → 走 profile_lookup，否则 evidence_search
  ▼
⑤ 报告合成 report.py
  │    LLM 只输出 JSON：claims[带证据索引] + uncertainties
  │    后端用 ref_map 补全真实 EvidenceRef；越界索引静默丢弃
  ▼
⑥ 质量评测 evaluate.py
  │    coverage / citation / alignment / completeness / yield
```

---

## 3. 抗幻觉设计（核心）

**原则：LLM 永不直接输出证据内容，只输出证据索引。**

```
LLM 输出（JSON）:
  { "claims": [
      {"text": "2025年小鹏交付量创新高",
       "evidence_refs": [2, 5]}     ← 只给索引
  ]}

后端补全:
  ref_map = {0: EvidenceRef(chunk...), 1: ..., 2: EvidenceRef(...), ...}
  claim.evidence = [ref_map[2], ref_map[5]]

越界保护:
  refs = [i for i in refs if 0 <= i < len(ref_map)]   ← 越界静默丢弃
```

**为什么有效**：
1. 模型编造"第 3 页说……"很容易，编造"引用第 3 条证据"后如果该索引在范围内，证据链由后端从真实 chunk 补全——**文本必然真实**。
2. 无证据的 claim 允许存在，但证据列表为空（宁缺毋滥），评测会扣 `coverage`。

---

## 4. 企业知识画像（Profile）

`app/rag/profile/` 负责从年报抽取 9 个维度结构化画像：

```
主营产品 / 目标客户 / 竞争对手 / 商业模式 / 财务亮点 /
发展战略 / 竞争优势 / 主要风险 / 行业环境
```

- 每字段由 LLM 抽取 + **Evidence 归因**（可回源到具体 chunk）。
- 构建成本高（9+1 次 LLM 调用，约 6 分钟）→ `storage.py` 缓存到 `data/profiles/{company}.json`。
- 生产模式：每公司构建一次，后续研究直接命中缓存。

---

## 5. 计划模板（Planner）

按意图选择模板，每个模板第一步固定为「企业知识画像」，后续步骤按意图维度展开。部分步骤可标注 `src=["policy"]` 等语料来源约束——若该语料未入库，该步产出低是**预期行为**而非 bug（评测 `yield` 会体现）。

**自适应（PR38）**：质量反馈路由自动决定继续查什么 / 补什么 / 何时结束，避免机械走完所有步骤。

---

## 6. 质量评测五指标（PR37.5）

| 指标 | 含义 |
|------|------|
| `coverage` 证据覆盖率 | 有证据支撑的 claim 占比 |
| `citation` 引用准确率 | 证据字段完整性（chunk_id/source/quote/page 四字段） |
| `alignment` 论点对齐度 | claim 文本与证据语义一致性 |
| `completeness` 步骤完整度 | 计划步骤完成比例 |
| `yield` 步骤产出率 | 各步实际产出 evidence 比例 |

评测结果随报告落库，可量化对比不同版本 Agent 的退化/提升。

---

## 7. 真实生产流程验证（小鹏 2025 年报）

| 维度 | 结果 |
|------|------|
| 入库 | 772 chunks（351 页年报） |
| 意图命中 | 3/3（业务/竞争力/风险） |
| 计划完成 | 20/20 步全部执行 |
| claims 带证据 | 31/31（100%） |
| 证据链四字段完整 | 35/35（100%） |
| 报告可溯源 | 每 claim 带章节/页码/原文 chunk |

可视化报告：`data/xpeng_research_report.html`（运行时生成）。复现：`python scripts/run_xpeng_research_demo.py`。
