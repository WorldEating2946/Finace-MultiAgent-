# 02 · RAG 检索链路设计

> Research Agent V1 · 2026-08-07 · 对应代码 `app/rag/`

---

## 1. 数据分类原则（三条腿）

| 数据类型 | 示例 | 处理方式 | 原理 |
|---------|------|---------|------|
| 非结构化知识 | 企业介绍 / 行业知识 / 政策文件 / 年报正文 | **RAG**（切片→向量→检索） | 语义相似度检索，无精确计算要求 |
| 结构化数据 | 财务指标 / 股票数据 / 实时行情 | **Tool API + Python 计算** | LLM 不负责精确计算 |
| 实时信息 | 新闻 / 舆情 | 外部 Search API | 时效性要求高 |

> 边界原则：**凡是能通过程序精确计算或实时获取的数据，禁止进入 RAG**。RAG 只承载需要语义理解的静态/长文本知识。

**禁止**：让 LLM 直接计算财务指标。正确流程：数据获取 → Python 计算 → 指标验证 → LLM 解释。

---

## 2. 离线入库链路（知识库构建）

```
PDF 年报/研报
   │
   ▼
① 文档解析 (app/rag/parsers/ + loaders/)
   │    PDF outline (fitz.get_toc) 优先恢复章节结构 —— 关键！
   │    OCR 兜底（扫描版）
   ▼
② 智能切片 (splitters/)
   │    章节感知切片 + 重叠窗口 + 元数据(章节/页码/公司/来源类型)
   ▼
③ 向量化 (embeddings/ BGE-M3, CUDA)
   │    Dense 向量 + Sparse 词权重
   ▼
④ 写入向量库 (vectorstore/)
   │    VectorStore.add() —— FAISS 或 Milvus
   ▼
⑤ 元数据关联
      company / source_type / section / page 全部随 chunk 落库
```

**入库原则**：

- **幂等**：chunk_id = `MD5(document_id + chunk_index + content 前缀)`，同内容重复入库不重复插入。
- **先删后插**：文档更新按 `document_id` 删除旧 chunk 再 upsert，保证无残留旧数据。
- **company 第一级过滤**：切片不可跨企业混排，`company` 作为检索前置过滤。
- **可读溯源**：每个 chunk 携带 `source_name`（如"年报 > 第3章 > 3.1 商业模式"）。
- **入库与查询共用同一 Embedding 模型**，禁止版本漂移。

> PR44 向量后端迁移工具 `scripts/migrate_faiss_to_milvus.py`：幂等 + 分批 + 对账校验 + 回滚。

---

## 3. 在线检索链路（Hybrid RAG）

```
Query
  │
  ▼
① Query Rewrite (query/rewrite.py)
  │    规则同义词 + LLM 改写（消歧、扩展专业术语）
  ▼
② Dense Retrieval (dense_retriever.py)
  │    BGE-M3 编码 → 向量 Top-K（公司 scalar filter 隔离）
  │
  ├── 并行 ── ③ Sparse Retrieval (sparse_retriever.py)
  │              BM25 词权重 Top-K
  ▼
④ 融合 (fusion.py)  RRF 融合
  │    Reciprocal Rank Fusion：dense/sparse 排名合并
  ▼
⑤ 精排 (reranker/)
  │    CrossEncoder (bge-reranker-v2-m3) 对 Top-N 重排
  │    Metadata-aware Rerank（章节/来源类型加权）
  ▼
⑥ 结果返回
      EvidenceChunk[]（chunk_id / source / section / page / text）
```

### 置信度兜底（反幻觉核心机制）

精排输出带 Top-1 置信度（CrossEncoder [0,1]）。**`confidence < 0.75` 视为检索不足，必须显式降级**（Web Search 兜底或返回"未检索到相关信息"），禁止 LLM 凭记忆硬答。

### 唯一入口契约

`app/rag/pipeline.py` 是 RAG 对外的**唯一入口**，内部模块（loader/splitter/embedding/vector_store/retriever/reranker）不对外暴露：

```python
def retrieve(query: str, company: str, top_k: int = 5,
             doc_type: str | None = None) -> RetrievalResult:
    """Agent 只提供问题 + 公司，不感知底层向量库实现。"""
```

> 分层规则：Research Agent 只依赖 `pipeline.retrieve`；调用方统一 `from app.rag import retrieve`，不直接 import 内部模块。

---

## 4. 向量存储层（PR44.1-44.4）

### 抽象契约（`vectorstore/base.py`）

```
VectorStore ABC:
    add(ids, embeddings, metadatas, texts) → None
    search(query, top_k, company=None, ...) → list[ScoredChunk]
    delete / update / count
```

### 双实现

| 后端 | 适用场景 | 说明 |
|------|---------|------|
| **FAISS**（默认） | 本地开发 / 离线评测 / 紧急回滚 | 零依赖、可落盘存档 `data/vector_store/` |
| **Milvus**（生产） | 生产环境 | `finance_agent` 库、FLAT/COSINE、多公司 scalar filter |

### 配置切换

```env
RAG_VECTOR_BACKEND=faiss     # 或 milvus
```

启动时 `vectorstore/health.py` 对所选后端做健康检查，**fail-fast**：后端不可用则拒绝启动，避免"启动成功但检索全空"的假健康。

### 一致性基准（PR44）

FAISS 与 Milvus 检索结果**逐项精度一致**，Milvus 延迟仅高 6~9%（网络往返），可作为生产与开发环境互换的信任依据。

---

## 5. 关键工程决策

| 决策 | 原因 |
|------|------|
| PDF outline 优先 TOC | OCR 直接切分丢失章节结构，小米 Recall@5 10%→70% |
| Query Rewrite | 规则同义词 + LLM 改写，小米 Recall 70%→80% |
| RRF 而非加权和 | 对 dense/sparse 分数尺度差异鲁棒 |
| 公司 scalar filter | 多公司知识库隔离，检索不跨公司串扰 |
| BGE-M3 同时出 Dense+Sparse | 单模型覆盖双路，避免两套嵌入模型维护成本 |

---

## 6. 评测回归门禁

`app/rag/evaluation/` 提供 Recall@K / MRR / NDCG 评测，`evaluation/RESULTS.md` 记录双数据集基线。

任何 RAG 链路修改必须跑回归门禁，Xiaomi + CATL 双数据集不允许退化。详见 `docs/06_evaluation_baseline.md`。
