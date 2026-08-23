# 06 · 评测基线与回归门禁

> Research Agent V1 · 2026-08-07 · 基于 `evaluation/RESULTS.md` 整合

---

## 1. 评测体系两套指标

| 层 | 评测对象 | 指标 | 出处 |
|----|---------|------|------|
| **RAG 检索质量** | 检索链路（Dense/Sparse/RRF/Rerank） | Recall@5 / MRR / NDCG@5 / Top1 | PR32 |
| **研究报告质量** | Research Agent 产出 | 证据覆盖 / 引用准确 / 论点对齐 / 完整度 / 步骤产出率 | PR37.5 |

---

## 2. RAG 检索基线（`evaluation/RESULTS.md`）

### 数据集

| 数据集 | 文档 | 说明 |
|--------|------|------|
| `catl_2025.json` | 宁德时代 2025 年报（232 页） | 文本层正常，10 题 |
| `xiaomi_2025.json` | 小米 2025 年报 | 文本层损坏需 OCR，10 题 |

模型：BGE-M3（embedding）+ bge-reranker-v2-m3（CrossEncoder）

### 指标口径

- **expected_sections 命中任意**：企业知识查询不是单一章节定位问题（如"营业收入"横跨多章），命中任一即 HIT。
- Recall@5 = 期望章节出现在 top-5 的比例；MRR = 首个命中的平均倒数；Top1 = 期望章节为 top-1。

### 当前基线（V1 交付，确定性 rule 模式）

| 数据集 | R@1 | R@5 | R@10 | MRR | NDCG@5 | Top1 |
|--------|-----|-----|------|-----|--------|------|
| **CATL**（干净文本） | 90% | **100%** | 100% | **0.950** | 0.957 | 90% |
| **Xiaomi**（OCR 受损） | 30% | **80%** | — | **0.423** | 0.474 | 30% |

> Xiaomi 为 metadata-aware rerank 基线（PR33）；CATL 为纯 CrossEncoder 直通（未配置公司权重即直通，零回归）。

### 关键演进记录（节选）

| 阶段 | Xiaomi R@5 | Xiaomi MRR | 说明 |
|------|-----------|-----------|------|
| OCR 结构适配前（PR29） | 10% | 0.100 | OCR 恢复了文本层，但结构（章节/TOC）未恢复 |
| + OCR 结构适配（PR30） | 70% | 0.265 | PDF outline 优先 TOC，数字误判过滤，繁体归一 |
| + LLM Query Rewrite（PR31） | 80% | 0.310 | 理解意图后 MISS 变 rank 2 |
| + Metadata Rerank（PR33） | **80%** | **0.423** | `final = α*CE + β*section + γ*keyword`，Top1 10%→30% |

**关键教训**：

1. **expected_sections（多期望章节）**：此前单期望章节把模型正确召回判为 MISS（评测数据问题，非模型问题）。
2. **per-company 权重**：单一全局 metadata 权重伤 CATL（MRR 0.95→0.77）→ 配置公司走融合，未配置直通纯 CE。
3. **LLM 改写是随机的**：每次生成不同变体，指标在 70~80% 波动 → 回归门禁用**确定性 rule 模式**，LLM 模式仅作信息性基准。
4. **base-first multi-query**：均权 meta-RRF 会把正确结果挤出 top-5 → 原始 query 保持权威序，扩展 query 只补新候选。

### 在线延迟（CATL 缓存库，稳态）

| 阶段 | 耗时 |
|------|------|
| Rewrite（rule） | 0ms |
| Retrieval（dense+bm25+rrf） | 48ms |
| Rerank（CrossEncoder） | 303ms（瓶颈） |
| **Total** | **~351ms** |

---

## 3. Research 报告质量指标（PR37.5）

| 指标 | 定义 | 用途 |
|------|------|------|
| **Evidence Coverage** | 有证据 claim 数 / 总 claim 数 | 报告幻觉度：越低越像编造 |
| **Citation Accuracy** | 四字段完整证据数 / 总证据数（chunk_id/source/quote/page 缺一即不完整） | 引用可定位性 |
| **Claim Alignment** | claim↔quote jieba Jaccard ≥0.2 的 claim 占比 | "chunk 匹配"：证据是否真支撑论点 |
| **Completeness** | 实际完成步骤 / 计划步骤 | 研究是否跑完 |
| **Step Yield** | 有证据产出的检索步骤占比 + low_yield_steps | 信息缺口（补步候选） |

> 全规则驱动、零 LLM 成本、可审计。

---

## 4. 真实生产流程验证（小鹏 2025 年报）

| 维度 | 结果 |
|------|------|
| 入库 | 772 chunks（351 页年报） |
| 意图命中 | 3/3（业务 / 竞争力 / 风险） |
| 计划完成 | 20/20 步 |
| claims 带证据 | 31/31（100%） |
| 证据链四字段完整 | 35/35（100%） |
| 单问题指标示例 | Q1: coverage 1.0 / citation 1.0 / alignment 0.636 / yield 1.0 |

复现：`python scripts/run_xpeng_research_demo.py` → `python scripts/render_xpeng_report.py` → `data/xpeng_research_report.html`。

---

## 5. 回归门禁（CI 守门员）

```bash
PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe -m pytest \
  --run-real tests/test_regression.py -s
```

| 门槛 | 指标 |
|------|------|
| Xiaomi | R@5 ≥ 75% · MRR ≥ 0.30 · NDCG@5 ≥ 0.35 |
| CATL | R@5 = 100% · MRR ≥ 0.90 |

**规则**：任何 RAG / Agent 链路修改必须跑回归门禁，双数据集任一退化即判回归。
LLM 模式（改写）仅作信息性报告，不做硬门禁（随机性）。

---

## 6. 评测存档与复现

- 评测存档 `temp/rag_eval_<company>/`（版本标记），二次运行仅查询 embed（~50s）。
- 管线版本变化时自动重入库（`_PIPELINE_VERSION` 递增）。
- 离线评测**钉死 FAISS**（PR44.4）：冻结基线严格可复现，不随生产配置（milvus）漂移。
