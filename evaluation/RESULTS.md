# RAG 评测基线（evaluation/）

> 数据集：`catl_2025.json`（10 题）· `xiaomi_2025.json`（10 题，待干净 PDF）
> 文档：宁德时代 2025 年报（232 页，文本正常）；小米 2025 年报（文本层损坏，需 OCR）
> 模型：BGE-M3（embedding）+ bge-reranker-v2-m3（CrossEncoder）
> 运行：`pytest --run-real tests/test_real_catl_eval.py -s`

## 指标口径

- **expected_sections 命中任意**：企业知识查询不是单一章节定位问题
  （如"营业收入"横跨 公司简介/管理层讨论/财务报告），命中任一即 HIT。
- Recall@5：期望章节出现在 top-5。
- MRR：期望章节首个命中位置的平均倒数。
- Top1：期望章节是 top-1。

## 检索链路演进（宁德时代，10 题）

| 阶段 | Recall@5 | MRR | Top1 | 说明 |
| ---- | ---- | ---- | ---- | ---- |
| Dense（BGE-M3 + FAISS） | 70% | 0.583 | 50% | 基础语义检索 |
| Hybrid（+BM25 + RRF） | 70% | 0.700 | 70% | 关键词互补 |
| **CrossEncoder + expected_sections** | **100%** | **0.950** | **90%** | 真实 reranker + 多期望章节 |

**关键修正**：
1. **expected_sections（多期望章节）**：企业查询非单章节定位，"营业收入"横跨多章，命中任意即 HIT。
   此前单期望章节把模型正确召回判为 MISS（评测数据问题，非模型问题）。
2. **fp16 修复 CrossEncoder OOM**：BGE(2.2G) + Reranker(2.2G) 同载 8GB 显卡 97% 显存 → 内存抖动卡死。
   reranker `.half()`（fp16 ~1.1G）后共存不 OOM，评测 63s 完成。

## 存档复用

- 评测存档 `temp/rag_eval_<company>/`（版本标记），二次运行仅查询 embed（~50s）。
- 管线版本变化时自动重入库（`_PIPELINE_VERSION` 递增）。

## 双数据集回归（PR #29，OCR 加入后稳定性）

| 数据集 | PDF 情况 | Recall@5 | MRR | Top1 | load | query |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 宁德时代 | 正常文本层 | **100%** | 0.950 | 90% | 34s | 15s |
| 小米 | 损坏文本层 → OCR | **10%** | 0.100 | 10% | 464s | 7s |

### 关键结论：OCR 恢复文本层，但结构未恢复

- **OCR 单页质量 100%**（文本层恢复成功，464s/413页）
- 但 **chunk 结构失效**：1107 chunks 中 790 有 section 但**全是数字误判**
  （`4.063,148,182` 被标题正则当成 level-2 标题）；**0 个 chapter**（TOC 解析在 OCR 输出上失败）
- → 检索匹配不到期望章节，Recall@5=10%

**含义**：OCR 解决了"文本可读"，但**结构管线（标题识别 + TOC）需要 OCR 适配**
（数字误判过滤、OCR 输出上的章节识别），否则检索质量无法恢复。
这是下一步（Query Rewrite 之前）的结构优化方向。

## OCR Structure Adapter（PR #30，小米结构恢复）

| 阶段 | Recall@5 | MRR | Top1 | 说明 |
| ---- | ---- | ---- | ---- | ---- |
| 小米 OCR（PR #29 基线） | 10% | 0.100 | 10% | 790 数字误判 section，0 chapter |
| **+ OCR 结构适配（PR #30）** | **70%** | **0.265** | **10%** | 数字误判→0 + 章节恢复 |

### 修复内容

1. **数字误判过滤**：`X.Y` 负向断言扩展到"空格后接数字/财务单位"（`(?!\s*[\d.,亿元万%％])`），
   OCR 在数字内插空格（`1.2 37.567.000元`）也拒判。790 误判 section → 0。
   原则：宁可漏掉标题，也不把财务数据当章节标题。
2. **OCR 容错 TOC**：选页从"标题行最多"改为"前部首个达标目录页"，正确选中页3（9 章节），
   排除页7 公司资料页误判（香港/湾仔/皇后大道东183号…43 个噪音行）。
3. **PDF outline 优先 TOC**（关键）：损坏文本层 PDF 的图形分隔页 OCR 读不出章节标题
   （管理层讨论及分析/董事会报告/企业管治报告，Tesseract 4x DPI 仍失败），但 PDF 内置书签
   （`fitz.get_toc()`）含**干净章节名 + 真实页码**，是 PDF 元数据，独立于文本层。
   `parse_toc` 路径优先级：**outline → 点线目录 → OCR 标题**。CATL 无 outline，走原路径，无回归。
4. **繁简归一补全**：`_TRADITIONAL_CHARS` 补全缺失繁体字符（層/討/論/會 等），
   确保 outline 章节名（传统）可靠转简体，与评测期望对齐。

### 评测助手修复（非管线）

- `run_company_eval` 重建时未清 `get_vector_store` 内存缓存 → 新数据**追加**到旧索引
  （新旧混合：4415 chunks = 旧2209 + 新2206）。修复：重建前 `_default_stores.pop(company)`。
- 命中检测改为 section 与 chapter 任一命中即 HIT（子标题可能覆盖 outline 章节名）。

### 命中分布（7/10）

- HIT：研发投入(5)、IoT连接数(3)、董事会构成(5)、主要经营风险(1)、公司治理结构(4)、现金(3)、股东权益变动(3)
- MISS：汽车业务未来战略、营业收入、智能手机出货量（期望"管理层讨论及分析"，top-5 内未命中）

**含义**：OCR 结构适配 + PDF outline 把小米检索质量从 10% 恢复到 70%（达标），
且 MRR 低（命中多在 rank 3-5）指向**排序优化**（Query Rewrite）为下一阶段方向。

## 性能优化（PR #27）

### Top-K sweep（Retrieve 20/30/50/100 → Rerank 5）

| Retrieve | Recall@5 | MRR | Top1 | 耗时 |
| ---- | ---- | ---- | ---- | ---- |
| 20 | 100% | 0.950 | 90% | 44s |
| 30 | 100% | 0.950 | 90% | 45s |
| 50 | 100% | 0.950 | 90% | 48s |
| 100 | 100% | 0.950 | 90% | 55s |

**结论**：候选 20~100 准确率完全一致，**Top-20 最优**（延迟最低）→ 默认 `rag_retrieve_top_k=20`。

### 在线稳态 Latency（Retrieve 20 → Rerank 20，warm）

| 阶段 | 冷启动（首 query） | **稳态（warm）** |
| ---- | ---- | ---- |
| Embedding | 4.6s（模型加载） | **22ms** |
| CrossEncoder | 3.9s（首次预热） | **157ms** |
| FAISS/BM25/RRF | ~30ms | ~30ms |
| **总 Retrieval** | ~8.5s | **~210ms** |

**结论**：真实在线场景（模型已加载）单 query **~210ms**。之前 9s 是冷启动（一次性）。优化：Top-20 + 输入截断(max_length=1024) + fp16。

### 基准脚本

- `evaluation/latency_benchmark.py`：分阶段耗时（warmup 后测稳态）
- `evaluation/topk_sweep.py`：Top-K 参数对比（`retriever._FETCH_K` 扫描）

## Query Rewrite（PR #31，2026-08-06）

### 背景

小米 MRR 0.265 低（命中多在 rank 3-5）、3 个 MISS 完全在线外，指向**查询理解**问题。
PR #31 新增 `app/rag/query/` 查询改写层：`QueryRewriter` ABC + `RuleBasedQueryRewriter`
（同义词扩展）+ `LLMQueryRewriter`（DeepSeek，失败回退规则版）+ base-first multi-query 召回。

### 规则版实测：触及天花板（MRR 持平）

真实小米库（BGE-M3 + CrossEncoder）对比 rewrite 开/关：

| 指标 | PR #30 基线 | 规则版改写 |
| ---- | ---- | ---- |
| Recall@5 | 70% | 70%（持平） |
| MRR | 0.265 | ~0.260（持平） |
| Top1 | 10% | 10%（持平） |

**根因（非规则质量问题，是结构性）**：

1. **评测数据 bug**：`小米汽车业务未来战略` 期望 `["管理层讨论及分析", "未来展望"]`，
   但 `未来展望` 在小米库 **0 个 chunk**（PDF 无此章节）→ 该期望永远无法命中。
2. **表格切分问题**：`小米2025年营业收入` 期望 `五年财务概要`，但该章 chunk 是
   **纯数字财务表**，语义检索天然难命中（rank 10）。
3. **CrossEncoder 边界**：`小米智能手机全球出货量` 召回层已到 MD&A rank 6-7，
   但 CrossEncoder 用原始 query 裁决压不进 top-5 → 精排问题，非召回/改写。
4. 规则改写唯一稳定收益：`公司治理结构`（同义词"企业管治"直接命中章节名）4→3。

**设计教训**：第一版 multi-query 做**均权 meta-RRF**，扩展查询的噪声 chunk 会把原始
query 的正确结果挤出 top-5（研发投入 5→7 回归）。改为 **base-first**：原始 query 结果保持
权威序，扩展 query 只补充 base 之外的新候选（recall booster）→ 消除回归，Recall@5 保持 70%。

### LLM 改写实测（deepseek-v4-flash，2026-08-06）

| 指标 | PR #30 基线 | 规则版 | **LLM 改写** |
| ---- | ---- | ---- | ---- |
| Recall@5 | 70% | 70% | **80%** |
| MRR | 0.265 | 0.260 | **0.310** |
| Top1 | 10% | 10% | 10% |

- **改善来源**：LLM 理解了"小米2025年营业收入"意图（改写出"经营情况讨论与分析之营业收入"）→
  MISS 变 rank 2；其余查询 rank 稳定无回归。
- **剩余 2 个 MISS（结构性，非改写可解）**：
  1. `小米汽车业务未来战略`：期望章节 `未来展望` 在库 **0 chunk**（评测数据 bug）。
  2. `小米智能手机全球出货量`：top-20 已召回 4 个 MD&A chunk（rank 6/9/17/18），
     但文本为 **OCR 乱码**（`加 ii (iv ...`），CrossEncoder 正确排后；干净文本在"主席报告"
     （非期望章节）。→ 需修复 MD&A 章节 OCR 质量（如更高 DPI / 表格识别）。

### 配置

```
RAG_QUERY_REWRITER=rule|llm|off   # .env 切换；llm 需 DEEPSEEK_API_KEY
LLM_REWRITE_MODEL=deepseek-v4-flash
```

- `app/rag/query/`：`QueryRewriter` ABC + `RuleBasedQueryRewriter`（离线可测）+ `LLMQueryRewriter`
  （DeepSeek，失败回退规则版，零崩溃）
- `retriever` base-first multi-query：原始 query 结果保持权威序，扩展 query 只补充新候选
- 测试：`tests/test_query_rewriter.py` 16 个用例（含 LLM mock / 回退 / 工厂模式）

## Evaluation 评测体系（PR #32，2026-08-06）

不是让 RAG 变强，而是建立"知道 RAG 是否变强"的测量系统。新增 `app/rag/evaluation/`
（dataset / metrics / evaluator / benchmark）+ 标准化数据集 `evaluation/datasets/` + 回归门禁。

### 指标基线（--run-real，deepseek-v4-flash）

| 数据集 | 模式 | R@1 | R@5 | R@10 | MRR | NDCG@5 | Top1 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CATL | rule（确定性） | 60% | **100%** | 100% | 0.950 | 0.957 | 90% |
| Xiaomi | rule（确定性） | 10% | **70%** | 90% | 0.260 | 0.363 | 10% |
| Xiaomi | llm（随机） | 10% | 70~80% | — | 0.252~0.310 | — | 10% |

### 关键发现：LLM 改写是随机的

PR #31 实测 Xiaomi LLM R@5=80%，但回归测试首次运行掉到 70% —— **LLM 改写每次生成不同变体，
指标在 70~80% 波动**。因此：

- **回归门禁用确定性 rule 模式**（`tests/test_regression.py`）：R@5≥65% / MRR≥0.2 / NDCG@5≥0.25
  （Xiaomi）、R@5=100% / MRR≥0.9（CATL）。专门抓代码回归，不受 LLM 噪声干扰；
- **LLM 模式作为信息性基准**单独测量/报告，不做硬门禁。

### 分阶段 Latency（CATL 缓存库，稳态）

| 阶段 | 耗时 |
| ---- | ---- |
| Rewrite（rule） | 0ms |
| Retrieval（dense+bm25+rrf） | 48ms |
| Rerank（CrossEncoder） | 303ms ← 瓶颈 |
| **Total** | **351ms** |

### 回归门禁（RAG CI）

```
PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe -m pytest \
  --run-real tests/test_regression.py -s   # 8 passed（6 单测 + 2 回归）
```

以后任何 PR（如 Metadata Rerank）跑该命令，CATL R@5<100% 或 Xiaomi R@5<65% 即判回归。

## Metadata-aware Rerank（PR #33，2026-08-06）

从 Text Rerank 升级为 Document Intelligence Rerank：CrossEncoder 输入从 `chunk.text` 变为
`[Company][Chapter][Section][Page][Content]` 结构化上下文，并叠加元数据信号做 Hybrid Score Fusion：

    final = α*ce_norm + β*section_signal + γ*keyword_signal

### 实测（rule 改写 + metadata 精排，--run-real）

| 数据集 | 配置 | R@1 | R@5 | MRR | NDCG@5 | Top1 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Xiaomi | CrossEncoder（基线） | 10% | 70% | 0.260 | 0.363 | 10% |
| **Xiaomi** | **metadata (0.90,0.08,0.02)** | **30%** | **80%** | **0.423** | **0.474** | **30%** |
| CATL | CrossEncoder（基线） | 90% | 100% | 0.950 | 0.957 | 90% |
| CATL | metadata（未配置→纯CE直通） | 90% | 100% | 0.950 | 0.957 | 90% |

MRR +63%（0.26→0.42）、Top1 10%→30%、Recall@5 70%→80%。

### 两个关键发现

1. **单一全局 metadata 权重无法服务两种语料**：metadata 增强帮 Xiaomi（分析师查询 + OCR 受损，
   CE 结构盲），但伤 CATL（精确财务查询 + 干净文本，CE 已最优）——MD&A 加成把正确"财务报告"
   chunk 挤下去（MRR 0.95→0.77）。→ **per-company 权重**（`RAG_METADATA_COMPANY_WEIGHTS`）。
2. **格式化 context 本身改变 CE 打分**：即使 α=1/β=0/γ=0，[Company][Chapter] 标签也非 CE
   训练见过的原始文本。→ 未配置公司必须**完全直通 CrossEncoderReranker**（原始 text），
   不能只设 β=0。

### 配置

```
RAG_RERANKER_MODEL=metadata                     # 生产精排模式
RAG_METADATA_COMPANY_WEIGHTS={"小米":"0.90,0.08,0.02"}  # 配置公司走融合；未配置直通纯CE
```

### 回归门禁（PR #32 升级）

```
pytest --run-real tests/test_regression.py -s   # 8 passed
# Xiaomi: R@5≥75% MRR≥0.30 NDCG@5≥0.35（metadata 基线）
# CATL:   R@5=100% MRR≥0.90（纯CE直通）
```

## Enterprise Knowledge Profile + Evidence Chain（PR #34，2026-08-06）

从 Document Knowledge 到 Enterprise Model 的第一次结构化升级：
不是 PDF→LLM→公司简介 的简单抽取，而是每个字段附带证据链（source/chapter/section/page/quote/chunk_id）。

### 模块

```
app/rag/profile/
├── schema.py      # CompanyProfile + ProfileItem + EvidenceRef
├── extractor.py   # ProfileExtractor：RAG 召回 + LLM 逐字段抽取 + 证据归因
└── storage.py     # save/load profile JSON（data/profiles/）
```

### 验收输出（输入小米年报 → data/profiles/小米.json）

| 维度 | 实体数 | 说明 |
| ---- | ---- | ---- |
| industry | 1 | 公司定位 |
| business_segments | 2 | 业务矩阵 |
| products | 8 | 产品矩阵 |
| technologies | 8 | 核心技术 |
| customers | 3 | 客户 |
| geographic_markets | 4 | 地理市场 |
| competitive_advantages | 0 | 检索无明确证据 → 宁缺毋滥（抗幻觉） |
| risks | 4 | 风险因素 |
| strategic_direction | 2 | 战略方向 |

每个字段结构（可审计、可追溯）：
```json
{"name": "智能手机", "description": "核心产品", "evidence": [
  {"source": "小米集团2025年报.pdf", "chapter": "主席报告", "page": 16,
   "quote": "2025年，小米硬件业务主要包含智能手机...", "chunk_id": "..."}]}
```

### 抗幻觉设计

1. **LLM 只返回 chunk 索引号**（evidence_refs），source/page/chapter 由后处理从已知
   metadata 补全 —— LLM 无法编造章节页码；
2. **越界索引静默丢弃**；无有效证据的实体丢弃（宁缺毋滥）；
3. quote 缺失时用真实 chunk 文本兜底。

### 关键发现：deepseek-v4-flash 是推理模型

长抽取 prompt 时 **reasoning_content 先消耗 token**（实测单次抽取推理占 2245+ token），
max_tokens 不足会返回空 content。修复：
- max_tokens 1200 → 6000（给 content 留足空间）
- chunk 文本 600 → 300 字符（缩小 prompt，减推理负担）
- **空 content 自动重试**（最多 3 次，退避 1s/2s）——推理模型间歇性返回空，重试最有效

### 生成命令

```
PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe -c "
from app.rag.profile import build_profile, save_profile
save_profile(build_profile('小米'))"   # 需 DEEPSEEK_API_KEY + 已入库向量库
```

### 边界

- ❌ 不做知识图谱（单企业年报规模不足，先 Profile JSON + Evidence）
- ❌ 不修改检索/排序/评测管线（纯消费侧）

## Multi-source Knowledge Fusion（PR #35，2026-08-06）

### 目标

从 Single-source Enterprise Profile（仅年报）升级为 Multi-source Enterprise Knowledge Base。
核心不是"多塞文件"，而是建立**不同来源之间的证据融合与冲突理解能力**。

### 架构

```
Vector Store (company=小米, 含所有 source_type)
    |
    +── source_type: annual_report → ProfileExtractor(source_type=...) → CompanyProfile
    +── source_type: research_report → ProfileExtractor(source_type=...) → CompanyProfile
    +── source_type: policy → ProfileExtractor(source_type=...) → CompanyProfile
    |
    v
SourceFusion
    ├── 同名实体跨源 → 合并证据（跨源确认）
    ├── 冲突实体 → 保留高优先级源（默认年报）+ 写入 conflicts
    └── 单源实体 → 原样保留
    |
    v
EnterpriseKnowledgePackage (profiles + fused + conflicts + evidence_summary)
```

### 关键设计

1. **doc_type vs source_type 分离**：doc_type 是格式级（"pdf"，splitter 依赖做分层切分）；
   source_type 是语义级（annual_report/research_report/policy/news），入库时由 ingest()
   标注，独立于 doc_type，零破坏性。
2. **证据归因扩展**：EvidenceRef 新增 source_type，每段证据可追溯"来自哪个源"——
   Research Agent 可据此区分公司自述 vs 市场观点 vs 外部环境。
3. **source_type 过滤在 extractor 层**：不进检索管线（retriever/vector_store 零改动），
   ProfileExtractor 做 post-retrieval filter；空 source_type = 混源检索（向后兼容）。
4. **冲突检测规则驱动**（零 LLM 成本）：实体名匹配（子串/bigram/共享核心词）+
   描述 jieba Jaccard < 0.3 → 冲突。识别"公司说长期增长" vs "研报说短期亏损"。
5. **冲突不自动裁决**：conflicts 列表保留双方 claim + 各自证据链，留 Research Agent 判断。

### 验收（单元 + 回归）

- 单测：`tests/test_source_fusion.py` 12 项（schema/冲突检测/融合/数据通道）✓
- 全量单测：196 passed ✓
- 回归门禁（--run-real）：8 passed ✓（Xiaomi R@5≥75% / CATL R@5=100%，零影响）

### 用法

```python
from app.rag import ingest
ingest("小米集团2025年报.pdf", company="小米", source_type="annual_report")   # 入库标注源类型
ingest("小米_行业研报.pdf", company="小米", source_type="research_report")   # 未来研报

from app.rag.source import SourceFusion
package = SourceFusion("小米", sources=["annual_report", "research_report", "policy"]).build()
# package.profiles / package.fused / package.conflicts / package.evidence_summary
```

### 边界（不做）

- ❌ Neo4j / 知识图谱；❌ 自动生成投资结论；❌ Agent workflow
- ❌ 检索基础设施改动（retriever/vector_store）
- ❌ 真实研报/政策 PDF（数据文件待用户提供，框架已就绪）

## Research Intent Understanding + Planning（PR #36，2026-08-06）

### 目标

补齐分析任务理解层：NL 研究请求 → 意图分类 + 目标抽取 + 维度识别 → 有序研究步骤序列。
纯规划层（不执行检索/抽取），执行在 PR #37 Agent Workflow。

### 架构

```
用户请求 "分析小米汽车未来竞争力"
    |
    v
IntentParser.parse()
    +── 意图分类: 关键词加权 → competitive_analysis
    +── 目标抽取: company=小米, segment=汽车（别名表 + 板块关键词）
    +── 维度推导: 意图默认 + segment 特定（汽车→policy）
    |
    v
ResearchPlanner.plan()
    +── 匹配 intent 模板（8 意图各有步骤序列）
    +── {company}/{segment} 替换（segment 空自动省略）
    |
    v
ResearchPlan {intent, target, dimensions, steps[], confidence}
```

### 关键设计

1. **9 意图分类**：competitive_analysis / business_overview / financial_analysis /
   risk_analysis / strategy_analysis / market_analysis / policy_analysis /
   technology_analysis / generic_research（兜底）。
2. **规则驱动**（零 LLM 成本，可审计）：关键词加权分类 + 公司别名表 + 板块关键词；
   LLM 意图理解留作后续增强（pattern 同 Query Rewrite rule→llm）。
3. **ResearchStep 是执行最小单元**：order / retrieval_query / dimensions / source_types /
   profile_fields / depends_on —— PR #37 Agent 直接消费。
4. **segment 感知**：汽车 → 自动补政策维度与政策源步骤；空 segment 自动省略关键词。

### 验证

- 单测 `tests/test_research.py` 18 项（schema/分类/目标抽取/维度/模板/端到端）✓
- 全量单测 214 passed ✓
- 回归门禁（--run-real）验证检索零影响 ✓
- E2E："分析小米汽车未来竞争力" → competitive_analysis + 小米 + 汽车 + 8 步 ✓

### 用法

```python
from app.rag.research import build_research_plan
plan = build_research_plan("分析小米汽车未来竞争力")
# plan.intent / plan.target / plan.dimensions / plan.steps / plan.confidence
```

### 边界（不做）

- ❌ 不执行 ResearchPlan（PR #37 Agent 做）
- ❌ 不连接 LangGraph workflow
- ❌ 不调用 profile/source 实际抽取

## Research Execution Engine（PR #37，2026-08-06）

### 目标

把 ResearchPlan 变成带证据链的研究结果。不是 Agent Loop（while True: llm.invoke()），
而是 Planner → Executor → Tools → State → Report，每步可溯源、每条 claim 有证据。

### 架构

```
build_research_plan("分析小米汽车未来竞争力")
    |
    v
ResearchPlan (8 steps)
    |
    v
ResearchExecutor.execute()
    for step in plan.steps:
        ├── 画像步骤 → tools.profile_lookup(company) → state.profile
        └── 搜索步骤 → tools.evidence_search(query, company, source_types)
    |
    v
ResearchState {profile, findings, evidence_pool}
    |
    v
ReportBuilder.build(state)
    +── 证据格式化（LLM 只见索引，source 由 ref_map 补全 —— 抗幻觉）
    +── LLM 合成 → {title, summary, advantages, risks, uncertainties}
    |
    v
ResearchReport（结构化报告 + 完整证据链）
```

### 关键设计

1. **工具抽象层**（ResearchTools）：profile_lookup / knowledge_search / evidence_search /
   conflict_analysis —— 每个都是已有能力的薄封装，可 mock、未来 LangGraph Tool 化零改动。
2. **证据链贯穿**：evidence_search 把 DocumentChunk 转 EvidenceRef（source/page/quote）；
   报告每条 claim 的 evidence_refs 由后处理解析成真实证据（越界丢弃，同 PR #34）。
3. **单次 LLM 合成**（非每步 LLM）：8 步骤只做检索收集证据，最终 1 次 LLM 合成报告 ——
   成本可控，且避免 Agent 黑盒。
4. **ResearchState 可序列化**：profile + findings + evidence_pool + completed_steps，
   天然可迁移到 LangGraph StatefulNode。

### 验证

- 单测 `tests/test_executor.py` 12 项（state/executor/tools/report/端到端 mock）✓
- 全量单测 226 passed ✓
- 回归门禁（--run-real）验证检索零影响 ✓
- E2E 验收：run_research("分析小米汽车未来竞争力") → 结构化报告 + 证据链 ✓

### 用法

```python
from app.rag.research import run_research
report = run_research("分析小米汽车未来竞争力")
# report.title / .summary / .advantages[i].claim+evidence / .risks / .uncertainties / .evidence
```

### 边界（不做）

- ❌ 不做 LangGraph Agent workflow（架构可迁移，预留下一步 PR #38）
- ❌ 不做 while True LLM 自主循环（黑盒 Agent）
- ❌ 不做 per-step LLM 分析（成本控制，只做最终报告合成）

## Research Report Evaluation（PR37.5，2026-08-06）

### 目标

在 RAG 检索评测（PR #32，Recall@K/MRR/NDCG）之上，增加**研究报告产出质量**评测——
回答"报告到底可不可信、研究是否充分"。PR38 动态 Agent 的"何时补步"决策需要这套指标。

### 指标（规则驱动，零 LLM 成本）

| 指标 | 定义 | 用途 |
| ---- | ---- | ---- |
| **Evidence Coverage** | 有证据 claim 数 / 总 claim 数（advantages+risks） | 报告幻觉度：越低越像编造 |
| **Citation Accuracy** | 四字段完整证据数 / 总证据数（chunk_id/source/quote/page 缺一即不完整） | 引用可定位性：能否回溯到原文 |
| **Claim Alignment** | claim↔quote 的 jieba Jaccard ≥0.2 的 claim 占比 | "chunk 匹配"：引用的证据是否真支撑论点 |
| **Completeness** | 实际完成步骤 / 计划步骤 | 研究是否跑完（PR38 动态补步才有 <1.0） |
| **Step Yield** | 有证据产出的检索步骤占比 + low_yield_steps | **PR38 补步候选**：0 证据步骤 = 信息缺口 |

对齐度复用 `app/rag/source/conflict.py` 的 jieba Jaccard（claim 是 LLM 摘要、quote 是原文，
阈值 0.2 比冲突检测的 0.3 更宽松，只拦截明显"引错 chunk"）。

### 模块

```
app/rag/research/evaluate.py   # ResearchMetrics + ClaimEval + evaluate_report()
```

### 验证

- 单测 `tests/test_research_eval.py` 10 项（coverage/citation/alignment/completeness/yield/退化路径）✓
- 全量单测 236 passed ✓
- 回归门禁（--run-real）8 passed ✓（Xiaomi R@5=80% / CATL R@5=100%，检索零影响）

### 用法

```python
from app.rag.research import build_research_plan, ResearchExecutor, ReportBuilder, evaluate_report

plan = build_research_plan("分析小米汽车未来竞争力")
state = ResearchExecutor().execute(plan)
report = ReportBuilder().build(state)
m = evaluate_report(report, state=state)

print(f"Evidence Coverage: {m.evidence_coverage:.0%}")   # 幻觉度
print(f"Citation Accuracy: {m.citation_accuracy:.0%}")    # 引用可定位性
print(f"Step Yield: {m.step_yield:.0%} low-yield: {m.low_yield_steps}")  # PR38 补步信号
```

### 边界（不做）

- ❌ 不做 LLM 参与评测（规则驱动，可审计）
- ❌ 不做 gold-standard 标注数据集（`evaluation/datasets/research.json` 预留，需人工标注）
- ❌ 不改 `run_research` 返回值（state 可选，缺省退化，保持兼容）

## Adaptive Research Agent（PR38，2026-08-06）

### 目标

PR37.5 补上质量反馈，但研究流程仍是单次 `Plan→Execute→Report`。PR38 的核心不是"接
LangGraph"，而是 **让 Agent 根据研究质量自动决定：继续查、补什么、什么时候结束**。
LangGraph StateGraph 是实现工具，自适应决策由 PR37.5 指标驱动。

### 架构

```
app/rag/agent/
├── __init__.py   # run_adaptive_research() → AgentState
├── state.py      # AgentState(ResearchState) —— 复用 PR37，不重新设计
├── graph.py      # StateGraph(AgentState) 编译
├── nodes.py      # 6 节点（intent/planning/execute/report/evaluate/replan）
├── edges.py      # router 条件边
└── router.py     # quality_ok + decide_next_action
```

### Graph 流（最多 3 轮，防无限循环）

```
START → intent → planning → execute → report → evaluate
                    ↓                                  ↓
                    └── [router] ── replan ──→ execute ←┘
                                   （缺维度补步）
                    ↓ end
                    END
```

### 自适应决策（router.py，规则驱动零 LLM）

| 条件 | 动作 |
| ---- | ---- |
| 迭代 ≥ 3 | **end**（第 3 轮强制输出，成本可控） |
| 有证据缺失维度（某维度下所有步骤 0 证据） | **replan**（动态补步，即使整体质量尚可） |
| 质量达标（coverage≥0.5 + citation≥0.5 + completeness≥0.8） | **end** |
| 无缺失但质量不足 | **end**（不盲目循环） |

缺失维度 → `planner.add_replan_step()` 从 `_MISSING_TEMPLATES`（6 维度模板）生成补充
步骤；`replanned_dimensions` 去重防步骤膨胀。每轮只执行未完成步骤（`executor.resume()`
增量执行，已收集证据保留）。

### 验收案例（tests/test_adaptive_agent.py，全 mock 零 LLM）

| Case | 构造 | 预期 |
| ---- | ---- | ---- |
| 1 正常 | 所有步骤产出证据 | iteration=1, next_action=end, quality_ok |
| 2 缺失触发 | 风险步骤首轮 0 证据 | iteration=2, 补步后风险覆盖, end |
| 3 最大轮次 | 证据持续缺失 | iteration=3, 强制结束 |

### 验证

- 单测 `tests/test_adaptive_agent.py` 11 项（graph/状态/3 案例/router/补步/resume 增量/入口）✓
- 全量单测 247 passed ✓
- 回归门禁（--run-real）8 passed ✓（检索零影响）
- **真实 E2E**（小米向量库 + DEEPSEEK，2026-08-06）：
  `run_adaptive_research("分析小米汽车未来竞争力")` → iteration=3（policy 缺口
  不可填——库中无 policy 源文档 → 补步无产出 → 第 3 轮强制结束），
  报告 coverage=100% / citation=100% / completeness=100%，6 优势 + 4 风险。
  验证自适应三机制：缺口检测 + 动态补步 + 迭代上限强制收尾 ✓

### 用法

```python
from app.rag.agent import run_adaptive_research

state = run_adaptive_research("分析小米汽车未来竞争力")
print(f"iterations={state.iteration} action={state.next_action}")
print(f"coverage={state.evaluation.evidence_coverage:.0%}")   # 若 evaluation 非空
print(state.current_report.title)                              # 最终报告
```

### 边界（不做）

- ❌ 不做 checkpoint 持久化（延后 PR39 Research Memory + Checkpoint）
- ❌ 不做 per-step LLM（与 PR37 一致：收集零 LLM，报告合成 1 次 LLM / 轮）
- ❌ 不改 `run_research` 返回值（`run_adaptive_research` 是新入口，向后兼容）

## Research Memory + Durable Checkpoint（PR39，2026-08-06）

### 目标

PR38 的 Adaptive Agent 可动态补步，但无持久化——任务执行 10 分钟，API 断开 / 服务重启 /
人工暂停后状态全丢。PR39 让 Agent 拥有"生命力"：**任务可暂停、可恢复、可跨进程重启**。

### 架构

```
app/rag/memory/
├── __init__.py      # ResearchCheckpointer 公共入口
├── schema.py        # RecordStatus + ResumeAction + ResearchRecord
├── serializer.py    # Memory Boundary（to_record 派生）
├── store.py         # get_checkpointer 工厂（memory | sqlite）
└── checkpoint.py    # run / resume / record per-thread
```

Agent 图零侵入：仅 +current_step 字段（进度）+ build_graph/run_adaptive_research 透传
checkpointer + thread_id（共 ~14 行）。

### Memory Boundary（什么持久化 / 什么重新生成）

| 持久化（ResearchRecord 字段） | 重新生成（不存） |
| ---- | ---- |
| 任务信息：research_id/query/company/intent/created_at | Embedding vector |
| 执行进度：current_step/completed_steps | LLM raw response |
| Agent 决策：iteration/next_action/missing_dimensions | 模型 cache / debug info |
| 产物摘要：evidence_count/finding_count/coverage | （AgentState 本就是序列化 IR，天然满足） |

### 存储后端（store.py 抽象，生产切 Postgres 零改动）

```
RAG_CHECKPOINT_STORE=memory|sqlite    # .env 配置
RAG_CHECKPOINT_DB_PATH=data/checkpoints.db
```

- `memory` → LangGraph MemorySaver（进程内）
- `sqlite` → SqliteSaver（`langgraph-checkpoint-sqlite` 3.1.1，本地磁盘，**跨进程重启可恢复**）
- `postgres` → PR41（AsyncPostgresSaver）——仅新增一个分支

msgpack 显式注册全部 AgentState 涉及类型（`JsonPlusSerializer(allowed_msgpack_modules=...)`），
消除 LangGraph 4.1.1 的 unregistered type 未来兼容警告。

### 验收案例（tests/test_memory.py，全 mock 零 LLM）

| Case | 构造 | 预期 |
| ---- | ---- | ---- |
| 1 服务重启恢复 | run → 新实例（同 SqliteSaver db）→ resume | 幂等返回相同终态（iteration/evidence 一致） |
| 2 人工暂停 → 审核 → 继续 | interrupt 节点暂停 → record=PAUSED → resume(approve) | 继续到 END → record=COMPLETED |
| 3 多任务线程隔离 | thread_A(小米) + thread_B(宁德时代) | 各线程 company/evidence 独立，resume 不交叉污染 |

### 验证

- 单测 `tests/test_memory.py` 9 项（store 工厂 / Memory Boundary / 3 案例 / 未知线程 / action 构造）✓
- 全量单测 256 passed ✓
- 回归门禁（--run-real）8 passed ✓（检索零影响）

### 用法

```python
from app.rag.memory import ResearchCheckpointer, ResumeAction

cp = ResearchCheckpointer(backend="sqlite")   # 生产切 postgres 只改 backend
state = cp.run("分析小米汽车未来竞争力", thread_id="xiaomi_auto_001")
state = cp.resume("xiaomi_auto_001")                        # 幂等恢复
state = cp.resume("xiaomi_auto_001",                        # 人工审核续传（PR41 接入）
                  action=ResumeAction(decision="approve"))
record = cp.record("xiaomi_auto_001")                       # 任务压缩记录
# record.status / .company / .completed_steps / .iteration / .coverage
```

### 边界（不做）

- ❌ 不做 Chat Memory（历史压缩成 summary——聊天记忆，非研究记忆）
- ❌ 不做 User Memory（用户偏好/长期知识）
- ❌ 不做 PostgresSaver（延后 PR41；store.py 已预埋 seam）
- ❌ 不做 `list_records()` 任务列表（延后 PR41 FastAPI）
- ❌ 不改 Agent 研究图结构（PR39 当时延后的 review_node 已由 PR40 落地）

## Research Agent Service Layer（PR40，2026-08-06）

### 目标

把"能自主研究的 Agent"变成**用户可以调用、观察、干预、恢复的 Agent 服务**——
从实验型 RAG Agent 走向企业应用的关键一层。PR40 四层架构：

```
User → FastAPI API Layer → ResearchService → LangGraph Agent → Checkpoint
                                    ↓
                              Human Review（人工审核闸口）
```

### 1. Human-in-the-loop（Agent Graph 接入）

PR39 有 interrupt 机制但未业务接入；PR40 加入 review_node 闸口：

```
execute → report → evaluate → [router] → review（证据不足 + human_review=True 时）
                                              ↓ interrupt 暂停
                                        人工审核 → resume
                                              ↓
                                   approve/modify → replan → execute
                                   reject → END
```

- `app/rag/agent/human.py`：HumanDecision（approve/reject/modify + feedback）+ validate_decision
- `app/rag/agent/interrupt.py`：LangGraph interrupt/Command 薄封装（PR41 换 SSE 零改动）
- `app/rag/agent/review.py`：HumanReviewRequest + build_review_payload（缺失维度→决策问题）
- 默认 `human_review=False` → PR38 路径完全不变（回归门禁验证）

### 2. Research Service 层（app/services/research_service.py）

API → Service → Agent → Memory，避免 API 直接耦合 LangGraph：
- create_task（thread_id 生成 `r{ts}_{company}_{uuid6}`）/ get_task / get_report / resume
- RecordStatus → API status 映射（PAUSED→waiting_human / COMPLETED→completed）

### 3. FastAPI API Layer（app/api/）

| Method | Path | 说明 |
|--------|------|------|
| POST | /api/v1/research/start | 创建任务 {query, company?, human_review?} |
| GET | /api/v1/research/{id} | 任务状态（status/current_step/iteration/missing_dimensions） |
| GET | /api/v1/research/{id}/report | 研究报告 |
| POST | /api/v1/research/{id}/resume | 人工审核恢复 {action, feedback} |
| GET | /api/v1/knowledge/search | RAG 检索（薄封装 retrieve） |
| GET | /api/v1/profile/{company} | 企业画像（薄封装 load_profile） |
| GET | /api/v1/health | 健康检查 |

### 4. 统一异常/响应（app/core/）

```
{code, message, data} 信封：code=0 成功；非 0 业务错误
ResearchNotFound(40001/404) / CheckpointExpired(40002/410)
InvalidDecision(40003/400) / ResearchReportNotReady(40004/409)
```

### 验收案例（tests/test_api.py，全 mock）

| Case | 预期 |
| ---- | ---- |
| 1 创建→状态→报告 | /start → completed；/report → 报告 |
| 2 人工暂停→审核→恢复 | waiting_human → resume(approve) → completed；resume(reject) → completed |
| 3 多任务隔离 | 小米/宁德时代报告证据互不污染 |
| 4 异常信封 | 404/400/409 统一 {code,message,data} |

### 验证

- 单测 `tests/test_api.py` 10 项（health/创建/状态/报告/HITL 暂停恢复/隔离/异常/knowledge/profile）✓
- 全量单测 266 passed ✓
- 回归门禁（--run-real）8 passed ✓（检索零影响，agent graph 默认路径不变）

### 用法

```bash
PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe -c "
from app.api.app import create_app
import uvicorn
uvicorn.run(create_app(), host='0.0.0.0', port=8000)"   # 需真实向量库 + DEEPSEEK_API_KEY

curl -X POST http://localhost:8000/api/v1/research/start \
  -H 'Content-Type: application/json' \
  -d '{\"query\":\"分析小米汽车未来竞争力\",\"company\":\"小米\",\"human_review\":true}'
```

### 边界（不做）

- ❌ 不做 async（同步 invoke；/start 阻塞到完成或暂停 → PR41 ainvoke）
- ❌ 不做 SSE（Agent 事件流 → PR41）
- ❌ 不做 PostgresSaver（SQLite 够用 → PR41）
- ❌ 不做 Milvus / 多租户（→ PR42）

---

## PR41 Production Runtime（2026-08-06）

### 目标

把 PR40 的"同步阻塞 Agent 服务"升级为"异步生产 Runtime"：
- POST /start 立即返回 queued，后台 worker 执行（不再阻塞 10 分钟）
- SQLite → PostgreSQL（AsyncPostgresSaver，多 worker/多任务并行）
- SSE 实时事件流（节点级进度反馈）
- 状态模型扩展：queued / running / paused / completed / failed / cancelled

### 架构

```
User → FastAPI(async def) → ResearchService → TaskManager → ResearchCheckpointer → Agent
                                                              │
                                               AsyncPostgresSaver（psycopg pool）
                                                              │
                                          PostgreSQL（public 业务表 + langgraph checkpoint）
                                              │
                                  SSE: GET /research/{id}/stream ← EventBus
```

### 实现要点

| 层 | 变更 |
|----|------|
| 基础设施 | scripts/setup_postgres.sh + scripts/init_db.py（public 业务表 + langgraph schema） |
| 存储 | store.py +postgres 分支（AsyncPostgresSaver + psycopg pool）；AsyncSqliteSaver 本地 async 测试 |
| Agent | agent/__init__.py +arun_adaptive_research（ainvoke + event_sink） |
| Checkpoint | ResearchCheckpointer +arun/aresume/aget_state/arecord（async）；sync 保留兼容 |
| Runtime | app/runtime/（TaskManager 后台 worker / EventBus 事件总线 / NodeEvent） |
| Service | ResearchService 双接口（sync PR40 兼容 + async PR41 生产） |
| API | research.py 全 async def + /cancel 端点；stream.py SSE 端点 |
| 状态 | RecordStatus +queued/+cancelled；API status 兼容 waiting_human |

### 验收（真实 PostgreSQL 集成）

| # | 场景 | 结果 |
|---|------|------|
| 1 | POST /start → queued（立即返回） | ✓ submit 立即返回 queued |
| 2 | 后台 worker → completed | ✓ arun 完成，iteration≥1 |
| 3 | GET /report | ✓ 报告可取 |
| 4 | cancel 幂等 | ✓ 已结束任务 cancel 返回原状态 |
| 5 | SSE 事件流 | ✓ event: node_start / node_end / done 格式 |

### 验证

- 单测 `tests/test_runtime.py` 9 项（async agent / EventBus / SSE 端点 / 状态扩展 / postgres sync 防护）✓
- 全量单测 275 passed ✓（PR40 基线 266 + 新增 9，零回归）
- 回归门禁（--run-real）8 passed ✓（Xiaomi R@5=80% / CATL R@5=100%）
- PostgreSQL 集成：queued→completed→report→cancel 全链路 ✓

### 用法

```bash
# 1. 启动 PostgreSQL（幂等）
bash scripts/setup_postgres.sh
# 2. 初始化表结构（business + langgraph schema，幂等）
PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe scripts/init_db.py
# 3. .env 配置
#    RAG_CHECKPOINT_STORE=postgres
#    DATABASE_URL=postgresql://eduagent_user:123456@localhost:5433/finance_agent
#    CHECKPOINT_DB_URL=postgresql://eduagent_user:123456@localhost:5433/finance_agent
# 4. 启动服务
PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe -c "
from app.api.app import create_app
import uvicorn
uvicorn.run(create_app(), host='0.0.0.0', port=8000)"

# SSE 流式监听
curl -N http://localhost:8000/api/v1/research/{id}/stream
```

### 边界（不做，延后）

- ❌ 不迁移 Milvus（向量基础设施 → PR42）
- ❌ 不改 RAG Pipeline（Hybrid Retrieval / Rerank 稳定）
- ❌ 不增强 Agent 能力（不加新 tool/planner/node）
- ❌ 不用 Redis Queue / Celery（当前 asyncio 简单 worker 过渡 → PR42 如需）
- ❌ 不引入 sse-starlette（网络 SSL 阻断；StreamingResponse 已覆盖心跳/事件类型）
- ❌ 不做 worker 故障恢复 / 多 worker 水平扩展（→ PR42）
