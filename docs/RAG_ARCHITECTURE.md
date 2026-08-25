# FinaceAgent RAG 架构设计

> **文档版本**: v1.6
> **适用阶段**: Phase 1 — RAG 基础闭环（Demo）
> **最后更新**: 2026-08-03
> **文档状态**: 待团队评审确认
> **修订说明**: v1.1 借鉴 edu-agent RAG 实践，引入 Hybrid 召回 + Reranker 精排 + 置信度兜底、Contextual RAG 可选步骤；v1.2 定稿对外检索接口契约；v1.3 对外入口由 service.py 更名为 pipeline.py，内部实现不对外暴露；v1.4 统一模型命名为 DocumentChunk，新增 load_documents 加载接口（Phase 1：Markdown/TXT）；v1.5 新增 split_documents 切片接口（MD 标题感知 + 中文边界）；v1.6 新增 Embedding 抽象层（接口 + 模型入口，不绑定具体模型）

---

## 1. 文档目的

本文档定义 FinaceAgent 项目 **RAG（检索增强生成）** 模块的职责边界、数据流、模块划分与技术选型，作为 Phase 1 RAG Demo 开发的统一依据。

**设计约束（来自 CLAUDE.md，不可违反）：**

- 非结构化知识 → RAG；结构化数据 → Tool；实时舆情 → Search（三层数据分流原则）
- RAG 检索能力仅服务于 **Research Agent**（Agent 职责隔离）
- `app/rag/` 目录职责为：**文档加载、切片、检索**
- RAG 的检索结果作为上下文供 LLM 解释，**LLM 不负责计算财务指标**

---

## 2. RAG 职责

### 2.1 职责边界（什么进 RAG）

| 数据类型 | 处理机制 | 进入 RAG？ |
| -------- | -------- | ---------- |
| 企业介绍、商业模式 | RAG | ✅ 是 |
| 招股书、财报文本、公告 | RAG | ✅ 是 |
| 行业知识、产业链信息 | RAG | ✅ 是 |
| 政策文件、历史事件 | RAG | ✅ 是 |
| 财务指标、股价、实时行情 | Tool API + Python 计算 | ❌ 否 |
| 实时新闻、市场情绪 | Search API + FinBERT | ❌ 否 |

> 边界原则：**凡是能通过程序精确计算或实时获取的数据，禁止进入 RAG**。RAG 只承载需要语义理解的静态/长文本知识。

### 2.2 核心职责

1. **知识入库**：加载金融文档，清洗、结构化切片（可选 Contextual 增强）、向量化，写入向量库，并维护可追溯的元数据。
2. **语义检索**：针对 Research Agent 的查询，按 `company` 过滤后做 Hybrid 召回（dense + sparse），再经 Reranker 精排。
3. **上下文组装**：将精排片段拼接为结构化上下文，携带来源引用（source_name / page），供 LLM 生成可溯源的回答。
4. **置信度兜底**：精排输出 Top-1 置信度，低于阈值（0.75）时显式降级，禁止 LLM 凭记忆硬答——反幻觉核心机制。
5. **来源可溯源**：检索结果必须带来源元数据（source / page / chunk_id），LLM 输出引用必须可回溯到原文。

### 2.3 非职责（禁止越界）

- ❌ 不做财务指标计算（交给 Financial Agent + Tool）
- ❌ 不做实时舆情抓取（交给 Sentiment Agent + Search）
- ❌ 不直接管理数据库连接（属于 `app/database/`）
- ❌ 不包含复杂业务编排逻辑（属于 `app/services/`）

---

## 3. 数据流

RAG 由两条独立链路组成：**离线索引链路（入库）** 与 **在线查询链路（检索）**。

### 3.1 离线索引链路（Ingestion）

```
源文档（PDF / Markdown / TXT）
        │
        ▼
  文档加载 Loader ──────── LangChain Loader / PyMuPDF
        │
        ▼
  文本清洗 Normalizer ──── 去空白、页码/页眉噪声、统一编码
        │
        ▼
  语义切片 Splitter ────── Markdown 按标题层级 / PDF 过滤空页后 RecursiveCharacterTextSplitter
        │
        ▼
  Contextual 增强（可选）─ LLM 生成 chunk 位置描述前缀，提升检索精度
        │
        ▼
  向量化 Embedding ─────── BGE-large-zh（Phase 1）→ BGE-M3 dense+sparse（评估中）
        │
        ▼
  向量库 Vector Store ──── FAISS（Phase 1）→ Milvus / pgvector（Phase 2，支持 Hybrid）
        │
        ▼
  元数据登记 ──────────── company / source / source_name / page / chunk_id
```

**入库原则：**

- 切片不可跨企业/文档混排，`company` 作为第一级元数据过滤条件；
- 单次入库幂等：chunk ID 用 `MD5(document_id + chunk_index + content 前缀)` 生成，同内容重复入库不重复插入；
- 文档更新采用**先删后插**（按 `document_id` 删除旧 chunk 再 upsert），保证无残留旧数据；
- 每个 chunk 必须携带可读的 `source_name`（如"招股书 > 第3章 > 3.1 商业模式"），供检索结果溯源；
- 入库与查询共用同一套 Embedding 模型，禁止版本漂移。

### 3.2 在线查询链路（Query）

```
Research Agent 查询（如："宁德时代的商业模式与竞争壁垒"）
        │
        ▼
  查询分类 / 意图归一 ───── 提取 company，判定检索策略（精准/模糊/多主题）
        │
        ▼
  Hybrid 召回 ──────────── Dense 语义 + Sparse 关键词 双路融合 → top_k 候选（如 10）
        │
        ▼
  Reranker 精排 ────────── CrossEncoder 逐对打分 → 取 top_n（如 3）+ Top-1 置信度
        │
        ▼
  置信度判断 ────────────── confidence >= 0.75 → 走 LLM；< 0.75 → 降级兜底（Web Search）
        │
        ▼
  上下文组装 ───────────── 拼接片段 + 来源引用（source_name / page）
        │
        ▼
  LLM 生成 ─────────────── 结合 Prompt 输出结构化分析
```

**查询原则：**

- 先按 `company` 元数据过滤，再做 Hybrid 召回，避免跨企业噪声；
- **置信度兜底**：Top-1 文档 Reranker 置信度 < 0.75 时视为检索不足，必须显式降级（Web Search 兜底或返回"未检索到相关信息"），禁止 LLM 凭记忆编造；
- 每次检索返回的引用列表（source_name / page）随上下文一同传给 LLM，确保输出可溯源。

---

## 4. 模块划分

### 4.1 目录结构（对应 `app/rag/`）

```
app/rag/
├── __init__.py
├── loaders/
│   ├── __init__.py        # 类型分发（get_loader / load_documents）
│   ├── base.py            # DocumentLoader 统一接口 + file_created_time
│   ├── txt_loader.py      # TXT 加载（doc_type="text"）
│   ├── markdown_loader.py # Markdown 加载（doc_type="markdown"）
│   └── pdf_loader.py      # PDF 逐页加载（PyMuPDF，page 从 1 起）
├── splitter.py          # 文本切片策略（按 doc_type 分发）
├── embedding.py         # Embedding 抽象接口 + 工厂（默认 BGE-M3）
├── embeddings/
│   └── bge_m3.py        # BGE-M3 具体实现（1024 维 dense，惰性加载）
├── vector_store.py      # 向量库抽象层（写/查 + save/load 持久化 + 多公司隔离）
├── retriever.py         # 检索器（Hybrid 召回：dense + sparse）
├── reranker.py          # 精排器（CrossEncoder 打分 + 置信度）
├── document.py          # 数据模型（Document / DocumentMetadata / DocumentChunk）
└── pipeline.py          # 对外唯一入口（retrieve），内部模块不对外暴露
```

### 4.2 各模块职责

| 模块 | 职责 | 关键输入 → 输出 |
| ---- | ---- | --------------- |
| `loaders/base.py` | `DocumentLoader` 统一接口 + `file_created_time` 助手 | 文件路径 → list[Document] |
| `loaders/txt_loader.py` | TXT 加载（doc_type="text"） | 文件路径 → list[Document] |
| `loaders/markdown_loader.py` | Markdown 加载（doc_type="markdown"） | 文件路径 → list[Document] |
| `loaders/pdf_loader.py` | PDF 逐页加载（PyMuPDF，page 从 1 起，跳过无文本层页） | 文件路径 → list[Document] |
| `splitter.py` | 按 doc_type 分发切分：markdown 标题感知 / 纯文本递归；提取 metadata 到 chunk | list[Document] → list[DocumentChunk] |
| `embedding.py` | Embedding 抽象接口 + 工厂（get_embedding_model，按配置选 bge-m3 / dummy） | 文本 → dense 向量 |
| `embeddings/bge_m3.py` | BGE-M3 真实模型实现（1024 维 dense，惰性加载） | 文本 → 1024 维向量 |
| `vector_store.py` | 向量库写/查抽象 + save/load 持久化 + 多公司隔离（`get_vector_store(company)`） | 向量 + 元数据 → 入库 / 召回候选 |
| `retriever.py` | Hybrid 召回（dense + sparse 双路融合） | 查询向量 → 候选片段 |
| `reranker.py` | CrossEncoder 精排 + 置信度输出 | 候选 → 有序 top_n + confidence |
| `document.py` | `Document` / `DocumentMetadata` / `DocumentChunk` / `RetrievalResult` | — |
| `pipeline.py` | 对外唯一入口：暴露 retrieve()，编排入库/查询完整流程 | 外部调用 → RetrievalResult |

### 4.3 数据模型（`document.py` 核心定义）

```python
from pydantic import BaseModel


class DocumentChunk(BaseModel):
    chunk_id: str          # 唯一标识：{source_hash}-{seq}
    company: str           # 所属企业（一级过滤字段；加载阶段未知则留空）
    doc_type: str          # 文档类型：招股书/财报/行业报告/政策；加载阶段默认 "text"
    source: str            # 来源文件路径
    source_name: str       # 可读溯源标注，如"招股书 > 第3章 > 3.1 商业模式"
    page: int              # 页码（Markdown/TXT 无分页，取 0）
    text: str              # 文本内容
    metadata: dict = {}    # 扩展元数据

class RetrievalResult(BaseModel):
    query: str
    chunks: list[DocumentChunk]    # 精排后按相关度排序
    scores: list[float]            # Reranker 置信度 [0, 1]
    confidence: float              # Top-1 置信度，< 0.75 触发降级兜底
```

### 4.4 与外部模块的关系

```
Research Agent
    │  调用
    ▼
app/rag/pipeline.py ── 唯一入口（retrieve）
    │
    ├── app/rag/*.py      检索/入库
    │
    └── app/database/     向量库与 PG 连接（由 database 层管理）

app/services/ ── 禁止绕过 rag 层直接拼装检索逻辑
```

> 分层规则：Research Agent 只依赖 `rag.pipeline`（retrieve），不直接触碰 loader / splitter / embedding / vector_store / retriever / reranker 内部细节；向量库连接交由 `app/database/` 统一管理。

#### 对外接口契约（Research Agent 对接定稿，ADR-001）

```python
# app/rag/pipeline.py —— 对 Agent 的唯一入口（对外仅暴露 retrieve）
from app.rag.document import RetrievalResult

def retrieve(
    query: str,
    company: str,                  # 一级过滤维度，必填
    top_k: int = 3,                # 精排后返回条数
    doc_type: str | None = None,   # 可选：招股书/财报/行业报告
) -> RetrievalResult:
    """RAG 检索唯一入口。Agent 只提供问题+公司，不感知底层向量库实现。"""
```

```python
class RetrievalResult(BaseModel):
    query: str
    chunks: list[DocumentChunk]    # 精排后，最多 top_k 个；含 source_name/page
    scores: list[float]            # 各 chunk 的 Reranker 置信度 [0,1]
    confidence: float              # Top-1 置信度；< 0.75 时由 Agent 决定降级
```

#### 对外接口契约（文档加载，Phase 1）

```python
# app/rag/loader.py —— 加载接口（经 app.rag 包暴露）
from app.rag.document import DocumentChunk

def load_documents(file_path: str) -> list[DocumentChunk]:
    """加载文档，转换为统一的 DocumentChunk 列表。"""
```

- 第一阶段仅支持 Markdown / TXT；PDF / Word（OCR、表格、复杂布局）Phase 2 支持；
- 对外访问：`from app.rag import load_documents`，不暴露加载内部实现；
- Phase 1 一个文件对应一个 chunk，细粒度切分由 splitter 模块负责；
- `company` 加载阶段未知留空，由入库编排填充。

#### 对外接口契约（文档切片，Phase 1）

```python
# app/rag/splitter.py —— 切片接口（经 app.rag 包暴露）
def split_documents(
    chunks: list[DocumentChunk],
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> list[DocumentChunk]:
    """将加载后的 DocumentChunk 切分为更细粒度的 DocumentChunk。"""
```

- Markdown 按标题层级切分（source_name 附加标题路径，如"招股书 > 第一章 > 1.1 商业模式"），超大节再按 chunk_size 细分；纯文本直接递归切分；
- 分隔符含中文标点（"。" "！" "？" "；" "，"），保证中文语义边界；
- 继承 company / source / source_name / doc_type / page；
- chunk_id 为 MD5(source + 序号 + 内容前缀)，内容不变则 ID 稳定。

#### Embedding 抽象层（内部 SPI，不对外暴露）

```python
# app/rag/embedding.py —— Embedding 抽象（内部模块）
class EmbeddingModel(ABC):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

def get_embedding_model() -> EmbeddingModel:   # 模型切换唯一入口
```

- 默认接入本地 **BGE-M3**（1024 维 dense）；`DummyEmbeddingModel`（MD5 确定性伪向量）保留，供单元测试显式注入；
- 切换 OpenAI / 其他模型**只改 `get_embedding_model()`**，RAG 主流程不变；
- BGE-M3 惰性加载：构造不加载模型（约 2.2GB），首次 `embed()` 才加载，避免拖慢单测与无关模块；
- vector_store / retriever 依赖该抽象，不依赖具体模型。

**接口约束：**

- `company` 为必填参数，内部先做元数据过滤再 Hybrid 召回，避免跨公司噪声；
- `confidence < 0.75` 视为检索不足，由 Agent 决定降级策略（Web Search 兜底 / 明示未检索到）；
- 输出必带来源（source_name / page），研报引用可溯源；
- Agent 不关心底层向量库实现（Phase 1 FAISS / Phase 2 Milvus 内部切换，Agent 零感知）；
- **内部实现不对外暴露**：调用方统一 `from app.rag import retrieve`，不直接 import 内部模块（loader / splitter / embedding / vector_store / retriever / reranker）。

---

## 5. 技术选型

### 5.1 组件选型

| 环节 | 选型 | 版本/说明 | 依据 |
| ---- | ---- | --------- | ---- |
| 文档解析 | LangChain Loaders + PyMuPDF | — | 生态统一、PDF 定位页码 |
| 文本切片 | MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter | chunk_size≈512，overlap≈100，separators 含中文标点 | 结构化标题路径 + 空页过滤，中文语义边界友好 |
| Embedding | **BGE-M3（本地）**：dense 1024 维 | 多语言 + 中文金融场景；Phase 1 仅启用 dense，sparse 随 Hybrid 检索 Phase 2 启用 | M3 原生支持 dense + sparse + colbert 三路，为 Hybrid 预留 |
| 向量库 | **FAISS（Phase 1）** | 本地文件 | 零服务依赖，轻量验证流程 |
| Hybrid 召回 | Dense + Sparse 双路 → WeightedRanker 融合 | Phase 1 可先仅 Dense；完整 Hybrid 随 Phase 2 启用 | 语义 + 关键词互补，解决"语义远但关键词命中" |
| 精排 | **BAAI/bge-reranker-v2-m3**（CrossEncoder） | 候选 10 → 精排 3 | 二次相关度打分，输出 [0,1] 置信度 |
| 置信度兜底 | Top-1 confidence < 0.75 → 降级（Web Search） | 阈值按场景校准 | 反幻觉核心机制，检索不足不硬答 |
| 元数据过滤 | 向量库 Filter | `company` 前置过滤 | 隔离跨企业噪声 |

### 5.2 向量库选型（分阶段决策）

| 方案 | 优点 | 缺点 | 决策 |
| ---- | ---- | ---- | ---- |
| **FAISS（本地文件）** | 零服务依赖、上手快、验证成本低 | 无分布式、数据量大后性能受限 | ✅ **Phase 1 采用** |
| **pgvector（PostgreSQL 16）** | 复用现有 PG、SQL 统一、Alembic 可管理 | 大规模向量召回性能弱于专用库 | ⏳ Phase 2 评估 |
| **Milvus** | 大规模向量检索性能最优 | 引入独立服务，运维成本高 | ⏳ 大数据量再评估 |

> **决策理由**：Phase 1 目标是快速验证 RAG 链路，FAISS 无需新增 Docker 服务，符合"基础设施最小化"原则。Phase 2 数据规模增长后，优先评估 **Milvus**（原生支持 Dense+Sparse Hybrid 检索与 WeightedRanker 融合）与 **pgvector**（复用 PostgreSQL 16，pgvector 0.5+ 支持 sparsevec）；若需完整 Hybrid 检索，Milvus 更顺滑。
>
> **架构保障**：通过 `vector_store.py` 抽象层隔离底层实现，**预留 `hybrid_search` 接口**（dense + sparse 双路入参、加权融合），切换向量库无需改动上层检索与 Agent 代码。

### 5.3 依赖管理

新增依赖按项目规范进入 `requirements.txt`，属于 **模块扩展依赖**：

```text
# RAG 模块（Phase 1，已启用）
langchain-text-splitters # 文档切片（splitter 直接引用）
faiss-cpu               # Phase 1 向量库
sentence-transformers   # 加载本地 BGE-M3 向量模型（Embedding 实现）
torch                   # 深度学习运行环境（BGE-M3 基于 Transformer）
pymupdf                 # PDF 解析（pdf_loader，fitz）

# RAG 模块（Phase 2，评估后启用）
# pymilvus              # Milvus 客户端（完整 Hybrid 检索）
# FlagEmbedding         # BGE-M3 sparse 通路（Hybrid 检索评估后启用）
```

> 注意：新增依赖需按 ENVIRONMENT.md §7 规范流程——确认必要性、更新 `requirements.txt`、提交说明、团队同步。`torch` / `sentence-transformers` 体积较大，与项目"模块按需扩展"原则一致，仅 RAG 开发阶段加入；依赖变更已随「!10 真实 Embedding 接入」同步。

### 5.4 评估指标

| 指标 | 说明 | 目标（Phase 1） |
| ---- | ---- | --------------- |
| Recall@k | 相关片段是否被召回 | 检索覆盖关键段落 |
| 置信度分布 | Top-1 Reranker 置信度质量 | 有效回答 ≥ 0.75 |
| 降级正确率 | 置信度不足时是否正确降级而非硬答 | 低置信度全部降级 |
| 引用可溯源 | LLM 引用必须能在原文定位 | 100% |
| 切片纯净度 | 无跨企业/跨主题混排 | 人工抽检通过 |

---

## 6. 待确认事项（需团队评审）

1. 向量库 Phase 2 方案：Milvus（完整 Hybrid）vs pgvector（复用 PG），需结合数据规模与部署复杂度定案。
2. 切片参数（chunk_size / overlap）：参考 512/100，需按金融文档结构（招股书章节）实测校准。
3. 重排模型确认：拟采用 BGE-reranker-v2-m3，置信度阈值 0.75 需按金融问答场景校准。
4. Embedding 升级评估：BGE-M3（dense + sparse 双向量）是否替代 bge-large-zh，依赖 Hybrid 检索需求。
5. Contextual RAG（建库时 LLM 增强 chunk 上下文）是否启用，涉及建库成本与检索精度权衡。
6. `requirements.txt` 新增 RAG 依赖的最终确认。

---

## 7. 总结

FinaceAgent RAG 模块一句话：

> **只承载非结构化知识，通过"离线入库 + 在线检索 + 引用溯源"为 Research Agent 提供可验证的上下文，其余数据一律走 Tool 与 Search。**

```
源文档 ──→ Loader ──→ Splitter ──→ Contextual增强(可选) ──→ Embedding ──→ FAISS（Phase 1）
                                                                              │
Research Agent ──→ Hybrid召回(dense+sparse) ──→ Reranker精排 ──→ 置信度判断 ←────┘
     │                                                   │ confidence < 0.75
     │                                                   ▼
     │                                            降级兜底（Web Search / 明示不足）
     ▼
  LLM 生成（带来源引用）──→ Research Agent 输出
```

---

## 8. 附录：BGE-M3 部署说明（模型放置与已知问题）

> 本附录说明真实 Embedding 接入后的**本地模型放置、依赖安装、加载已知问题与验证**，团队其他成员按此同步即可（对应 ENVIRONMENT.md §7 流程）。

### 8.1 模型放置（不进 Git）

- 模型权重约 **2.2GB**，已加入 `.gitignore`（`app/models/embedding/`），**不会随代码分发**，需成员自行放置：
  ```
  app/models/embedding/bge-m3/
  ```
- 来源：从 HuggingFace 下载 `BAAI/bge-m3`（或向团队索取拷贝）。模型为 sentence-transformers 2.2.2 打包格式，含 `pytorch_model.bin`、`modules.json`、`config_sentence_transformers.json` 等。

### 8.2 依赖安装

新增依赖已提交 `requirements.txt`：

| 包 | 用途 |
| ---- | ---- |
| `sentence-transformers` | 加载本地 BGE-M3 向量模型 |
| `torch` | 深度学习运行环境（BGE-M3 基于 Transformer） |

```
uv pip install -r requirements.txt   # 或 pip install -r requirements.txt
```

> torch 安装包较大（CPU wheel 约 100MB+），首次安装稍慢属正常。

### 8.3 加载方式（惰性）

`get_embedding_model()` 返回 `BGE_M3EmbeddingModel` 实例但**不加载模型**；首次调用 `embed()` 才加载（约 10~30s），进程内单例复用。`DummyEmbeddingModel` 保留供单元测试显式注入。

### 8.4 已知问题：Pooling 加载报错

**症状**：加载模型时报

```
TypeError: Pooling.__init__() missing 1 required positional argument: 'embedding_dimension'
```

**原因**：BGE-M3 为 sentence-transformers 2.2.2 打包，`modules.json` 引用的 `1_Pooling/` 子目录缺失；新版 sentence-transformers（5.x）的 `Pooling` 模块要求显式 `embedding_dimension`。

**修复**：在模型目录下补建 `1_Pooling/config.json`：

```json
{
  "embedding_dimension": 1024,
  "pooling_mode": "mean",
  "include_prompt": true
}
```

### 8.5 验证

```bash
# 1. 维度应为 1024
python -c "from app.rag.embedding import get_embedding_model; print(len(get_embedding_model().embed(['测试'])[0]))"

# 2. 全套测试
python -m pytest -q   # 预期 63 passed
```
