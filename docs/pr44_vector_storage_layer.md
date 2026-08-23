# PR44 — Enterprise Vector Storage Layer

> 状态：**正式定义（2026-08-07）**。基线 Tag `v0.43-runtime-stable`，PR47 Runtime 文档已冻结。
> 下一步：PR44.1 抽象层 → PR44.2 Milvus Adapter → PR44.3 Migration + Benchmark → PR45 Worker Isolation。

## 1. 定位：不是"FAISS → Milvus"

本 PR 不是技术替换，而是把当前**单机向量检索**升级为**企业级、多租户、可扩展知识存储层**。

```
     Retriever
        │
        │
  VectorStore Interface
        │
  ┌──────┴──────┐
  ▼             ▼
 FAISS        Milvus
```

## 2. 第一原则：不破坏现有 RAG

**第一步不是删 FAISS。** 先形成抽象层，FAISS 与 Milvus 平级共存、可切换，现有 RAG 零感知。

## 3. 现状对照（PR44.1 的起点）

`app/rag/vector_store.py` **已存在**抽象层（不是从零建）：

| 现状 | 位置 |
|------|------|
| `VectorStore` ABC：`add(chunks, vectors)` / `search(query_vector, company, top_k)` / `hybrid_search(...)` / `all_chunks()` / `vector_of(chunk_id)` / `save()` / `load()` | `vector_store.py:28` |
| `FAISSVectorStore`：IndexFlatIP + 归一化内积 + "多召回(×5)+后置 company 过滤" | `vector_store.py:126` |
| 持久化：`<rag_vector_store_path>/<company>/index.faiss + metadata.json` | `vector_store.py:247` |
| 多公司单例：`get_vector_store(company, dim)`，每公司独立子目录 | `vector_store.py:319` |

**PR44.1 的真实增量**（接口已存在，需扩展/重构）：

1. **包结构**：单文件 `vector_store.py` → 包 `app/rag/vectorstore/`（`base.py` / `faiss_store.py` / `milvus_store.py`）。
2. **结构化 filters**：现 `search(query_vector, company, top_k)`（company 位置参数）→ `search(query_embedding, top_k, filters=None)`（filters 字典，支持 company / year / document_type 组合）。
3. **delete(ids) + 增量 insert + update chunk**：现 ABC 无删除/更新，仅整批 append。
4. **单集合设计**：现"每公司一子目录" → Milvus 单 collection `finance_knowledge` + company 过滤字段（**架构性差异，需迁移验证**）。

## 4. PR44.1 VectorStore 抽象

新增包结构：

```
app/rag/vectorstore/
├── base.py          # VectorStore 接口
├── faiss_store.py   # FAISS 实现（迁移自 vector_store.py）
└── milvus_store.py  # Milvus 实现（PR44.3）
```

接口：

```python
class VectorStore:
    async def search(self, query_embedding, top_k, filters=None): ...
    async def add(self, vectors, metadata): ...
    async def delete(self, ids): ...
```

> 注：现 `vector_store.py` 为**同步**接口（`save`/`load` 同步 IO、调用方同步消费）。
> PR44.1 需决策：接口是否异步化（`async def`），及与现有 `hybrid_search` / `all_chunks` /
> `vector_of` 如何并入新包——**hybrid 与 BM25 依赖 `all_chunks`/`vector_of`，必须保留**。

## 5. PR44.2 FAISS Adapter

把现有逻辑迁移：

```
faiss_index.search()  →  vector_store.search()
底层仍 FAISS
```

**基准门禁（必须保持）**：

| 公司 | 指标 | 门禁 |
|------|------|------|
| Xiaomi | Recall@5 ≥ 80% / MRR ≥ 0.423 | 不可下降 |
| CATL | Recall@5 = 100% / MRR = 0.950 | 不可下降 |

回归用 `tests/test_regression.py --run-real`（现有评测基线）。

## 6. PR44.3 Milvus Adapter

Docker 部署：

```yaml
milvus:
  image: milvusdb/milvus
# 依赖：etcd + minio
```

生产架构：

```
Milvus Cluster
        │
        │
     FinanceAgent
        │
     VectorStore
```

**Collection 设计（重点）**——不要简单叫 `documents`，建议 `finance_knowledge`：

```json
{
  "id": "uuid",
  "embedding": [float],
  "company": "xiaomi",
  "document_type": "annual_report",
  "year": 2025,
  "section": "financial_statement",
  "source": "xiaomi_2025_report.pdf",
  "chunk_id": "xxx"
}
```

支持：

- **公司隔离**：`filter: {"company": "xiaomi"}`（如"小米 2025 年现金流"）
- **多源知识**（未来）：`annual_report` / `research_report` / `announcement` / `regulation` / `internal_doc`

## 7. 最大风险：Metadata 迁移

不是 Milvus API，而是 **chunk id / metadata / embedding 三者一致性**。

```
现在 FAISS:        index.faiss  +  metadata.json（两个文件，按行号对齐）
Milvus:            vector + metadata（统一存储，field 绑定）
```

必须保证迁移后：

- 同一 chunk 的 `chunk_id`、`metadata`、`embedding` 三者指向一致
- 迁移脚本做**逐条对账**（如：按 chunk_id 抽样比对原库/新库结果一致）

## 8. 验收标准

**Retrieval 不下降**：

| 公司 | 门槛 |
|------|------|
| Xiaomi | Recall@5 ≥ 80%，MRR ≥ 0.423 |
| CATL | Recall@5 = 100%，MRR = 0.950 |

**新增功能**：

- [ ] company filter
- [ ] year filter
- [ ] document_type filter
- [ ] incremental insert
- [ ] delete / update chunk

## 9. Runtime 不受影响

```
Research Agent
      │
      │
  VectorStore
      │
  FAISS / Milvus    ← 切换透明
```

Runtime（TaskManager / Worker / Lease）不感知检索层改动，回归保持 316 passed。

## 10. 后续 PR45（本 PR 不做）：Runtime Execution Isolation

线程级 Worker 隔离**放后面**。理由：PR43 asyncio Worker + lease×3 已稳定，线程隔离解决的是
"Agent 同步阻塞"问题，属 Runtime v2。演进路线：

```
async worker → thread executor → process worker → （未来）K8s Job
```

## 路线图

```
PR47 Runtime 文档冻结（已合并）
        ↓
⭐ PR44 Vector Storage Layer（本 PR）
        ↓
  PR44.1 抽象层
        ↓
  PR44.2 Milvus Adapter
        ↓
  PR44.3 Migration + Benchmark
        ↓
PR45 Worker Isolation
```

## 关联

- [Runtime v1 架构](runtime_v1_architecture.md)
- [RAG 架构](RAG_ARCHITECTURE.md)
- [架构决策记录](architecture_decisions.md)
