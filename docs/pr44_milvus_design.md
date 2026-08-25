# PR44.3 Milvus Adapter 设计（Design Review）

> **文档版本**: v1.0（Design Review，2026-08-07 待冻结）
> **前置依赖**: PR44.1（VectorStore 抽象层，已合并）+ PR44.2（FAISS Adapter，PR #50 待合并）
> **冻结目标**: Milvus Database / Collection Schema / Primary Key / Index 策略 / Hybrid 保持 / Migration 方案 / 多公司隔离
> **范围锁定**: **只做设计文档**，不写 Adapter 代码。设计冻结后再进入 PR44.3.1 实施。

---

## 0. 目标与定位

PR44.3 的目标不是"把 FAISS 换成 Milvus"这个技术动作本身，
而是把当前**单机知识库**升级为**企业级向量基础设施**：

```
应用级 RAG（已完成）
   ↓
模块化 RAG（已完成：Loader / Splitter / Embedder / Store / Retriever / Reranker 分层）
   ↓
可替换存储 RAG（PR44.1+44.2 已完成：VectorStore ABC + filters + delete/update/count + compact）
   ↓
企业级向量基础设施（PR44.3：独立向量服务 Milvus，多公司隔离 + 可扩展 + 高并发）
```

前置条件（已达成）保证本 PR 不必改上层：

- `app.rag.vectorstore.VectorStore` ABC 已冻结 `add / search / delete / update / count`
- 调用方（retriever / ingestion / evaluation / benchmark）已全部走新接口
- 检索管线保持 `Dense + Sparse → RRF → Reranker`，**本 PR 不改变召回逻辑**

---

## 1. 架构决策（冻结项）

### AD-1 Milvus Database：`finance_agent`

不使用 Milvus 默认的 `default` database，显式创建 `finance_agent`，隔离本项目向量数据。

```
Milvus Cluster
   └── finance_agent            ← Database（AD-1）
          └── finance_knowledge ← Collection（AD-2）
```

**理由**：`default` 库承载 Milvus 内部默认对象，业务数据单独建库可避免
与未来其他业务 collection 混杂，运维时按库整体备份/清理更清晰。

### AD-2 Collection Schema：`finance_knowledge`

| 字段 | 类型 | 说明 |
| ---- | ---- | ---- |
| `chunk_id` | VARCHAR(128) | 主键（见 AD-3） |
| `company_id` | VARCHAR(64) | 公司隔离（见 AD-7） |
| `document_id` | VARCHAR(128) | 所属文档（`{source_hash}`，粒度 < company，便于按文档级联管理） |
| `year` | INT | 年份（数值过滤，>= 语义） |
| `section` | VARCHAR(256) | 章节路径（如 "财务分析"） |
| `text` | VARCHAR(8192) | 原文（chunk 文本） |
| `embedding` | FLOAT_VECTOR(1024) | BGE-M3 稠密向量 |
| `metadata` | JSON | 扩展元数据（source_type / page / source / source_name / deleted 等） |

**要点**：
- 维度固定 **1024**（BGE-M3）。`dim` 在 `create_collection` 时一次性决定，
  与 FAISS"首批向量自动适配维度"不同——这是迁移到 Milvus 的**语义差异**，必须在 Adapter 层处理
  （dim 来自配置/embedding 模型，不再运行时推断）。
- `document_id` 与 `chunk_id` 的关系沿用现有约定：`chunk_id = {source_hash}-{seq}`，
  故 `document_id` 取 `{source_hash}` 前缀即可，无需在 chunk 里额外维护。
- `text` 冗余存一份（Milvus 可按 primary key 回查原文，省去 FAISS 式的
  metadata.json + reconstruct 组合，也让迁移验证可直接对账）。

### AD-3 Primary Key：`chunk_id`（禁止 `auto_id`）

**必须**用业务主键 `chunk_id`，**禁用** Milvus `auto_id` 自增主键。

**理由（迁移验证的基石）**：FAISS 与 Milvus 必须能按同一 `chunk_id` 对账：

```
FAISS  chunk_id ──对应── Milvus chunk_id
        │                    │
   embedding 一致 ──验证── embedding hash
        │                    │
    metadata 一致 ──验证── metadata JSON
```

若用 `auto_id`，两库之间失去稳定关联键，**embedding / metadata 一致性无法逐条验证**，
迁移即"黑盒搬运"，回归门禁无从谈起。

**连带收益**：
- `delete(ids)` 直接用 `expr: chunk_id in [...]`，物理删除，无需 FAISS 的逻辑删除标记
- `update(record)` = `delete(chunk_id)` + `insert`，与 FAISS 语义对齐
- 幂等 `add`：`upsert` 能力（Milvus 支持按主键 upsert）可原生处理重复入库

### AD-4 Index 策略：metric=COSINE；开发 FLAT → 生产 Benchmark 后 HNSW/IVF_FLAT

**不提前优化**。当前 BGE-M3（1024 维）规模为小米 ~5000 chunks / CATL 量级：

| 阶段 | Index | metric_type | 理由 |
| ---- | ----- | ----------- | ---- |
| **开发/验证** | `FLAT` | `COSINE` | 暴力扫描 = 精确结果，作为正确性基准 |
| **生产（Benchmark 后）** | `HNSW` 或 `IVF_FLAT` | `COSINE` | Benchmark（PR44.3.3）出 P95/QPS 数据后再定 |

**要点**：
- `metric_type=COSINE` 与 FAISS `IndexFlatIP` + L2 归一化**数学等价**（归一化后内积 = 余弦）。
  保证迁移后召回结果与回归门禁可比对。
- 不要提前用 HNSW：近似检索会引入召回噪声，回归门禁（Recall@5≥80%）在数据量级下
  FLAT 已够；等真实数据规模/并发撑不住再切换，切换只改 `create_index` 参数，不动代码。

### AD-5 Hybrid Retrieval **保持不变**

**不改召回架构**。当前与未来：

```
现在：
  Dense(FAISS) + BM25(SparseRetriever) ──RRF──▶ Reranker

迁移后：
  Dense(Milvus) + BM25(SparseRetriever) ──RRF──▶ Reranker
```

**禁止**把 Milvus Hybrid Search（dense+sparse 双路 + WeightedRanker 融合）作为核心召回。

**理由**：
1. **行为一致优先**：RRF 融合 + 现有 reranker 已通过回归门禁；Milvus Hybrid Search 的
   融合权重/打分语义不同，会改变召回序，需要重新调参 + 重新验收。
2. **Sparse 通路不迁移**：BM25（jieba 中文分词）是独立的 SparseRetriever，Milvus 的
   sparse 是 SPLADE 风格，语义不等价，迁移属于另一条工作线。
3. **渐进可逆**：Dense 层换后端（FAISS→Milvus）失败可一键回退；若把 Hybrid 也迁了，
   回归定位无从下手。

Milvus Hybrid Search 记为**后续可选项**（PR44 之后单独立项），不作为本 PR 目标。

### AD-6 Migration 流程：Export VectorRecord → Milvus Insert Batch → Validation

```
FAISS                          Milvus
index.faiss + metadata.json
        │
        ▼
  Export VectorRecord          (chunk_id + text + embedding + metadata)
  （逐条/分批读取，含 deleted 标记）
        │
        ▼
  Milvus insert batch          (映射到 finance_knowledge 字段)
        │
        ▼
  Validation ── 数量 + 抽样对账
```

**验证（迁移必须通过才算成功）**：
1. **数量**：`FAISS count()` == `Milvus count()`（均取**活跃**数，deleted 不迁移）
2. **抽样**（≥ N 条，如每公司 100 条）：
   - `chunk_id` 完全一致
   - `embedding hash`：`md5(embedding.tobytes())` 两库一致
   - `metadata`：关键字段（company_id / source_type / year / section）两库一致
3. **行为**：抽样 query 双库检索，Top-K 交集 ≥ 阈值（或用完整回归门禁兜底）

### AD-7 多公司隔离：`company_id` 作 **scalar filter**（不默认用 Partition）

Milvus 提供两种公司隔离手段：

| 方案 | 机制 | 取舍 |
| ---- | ---- | ---- |
| **Scalar filter**（**推荐**） | `expr: company_id == "xiaomi"` 每次检索带过滤 | 无管理开销；过滤成本随 partition 数无关；与 VectorStore ABC `filters["company_id"]` 语义一致 |
| Partition | 每公司一个 partition，检索限定 partition | 分区上限 4096；与 delete/upsert 语义交互复杂；需要维护分区生命周期 |

**决策**：默认用 **scalar filter** 实现 `filters["company_id"]`，与 FAISS 后置过滤行为对齐。
数据量到达单 partition 无法满足（如单公司 > 千万级）再评估 PartitionKey 方案。

---

## 2. VectorStore Contract → Milvus 映射

`MilvusStore` 实现 `app.rag.vectorstore.VectorStore`（+ `LocalVectorStoreMixin` 弃用、
`HybridSupportMixin` 保留）。与 PR44.2 `FAISSStore` 的逐方法对照：

| ABC 方法 | FAISSStore（现状） | MilvusStore（设计） |
| -------- | ------------------ | ------------------- |
| `add(records)` | wrapper → 旧 add + metadata.json | `collection.upsert(rows)`（主键 chunk_id 幂等）；dim 不符则 `create_collection` 重建 |
| `search(query_embedding, top_k, filters)` | 多召回 + Python 后置过滤 | `collection.search(data, limit, expr=翻译(filters))` → 原生过滤；返回 SearchResult |
| `delete(ids)` | 逻辑删除 metadata["deleted"]=True | `collection.delete(expr="chunk_id in [...]")` **物理删除** |
| `update(record)` | delete 旧 + add 新 | `upsert`（同主键覆盖） |
| `count()` | 遍历活跃 chunk | `collection.num_entities` 或 query 计数（活跃 = 全量，因为 delete 物理） |
| `compact()` | 本地重建 index.faiss | `collection.compact()`（Milvus 原生 segment 合并，回收已删行磁盘） |
| `all_chunks()` | 遍历 _chunks | `collection.query(expr, output_fields=[...])` 分页拉全量（喂 BM25） |
| `vector_of(chunk_id)` | reconstruct | `query(expr="chunk_id==...", output_fields=["embedding"])` |

**filters 翻译**（结构化 filter → Milvus expr，比 FAISS 后置过滤更强）：

| filter 键 | Milvus expr | 说明 |
| --------- | ----------- | ---- |
| `company_id` | `company_id == "xiaomi"` | 标量字段原生过滤（AD-7） |
| `year` | `year >= 2025` | 数值 >= 语义，沿用现有约定 |
| `document_type` | `metadata["source_type"] == "annual_report"` | JSON 字段过滤（保留 document_type→source_type 别名映射） |
| `section` | `section == "..."` 或 `metadata["section"] == "..."` | 标量/JSON 均可 |

**LocalVectorStoreMixin（save/load/validate_integrity）**：
- Milvus 持久化由服务端负责，`save()` / `load()` 为 no-op（符合 PR44.1 分层设计——Mixin 不强制）。
- `validate_integrity()` 实现为：`num_entities` vs 全量 query 计数一致 + 抽样 embedding 非空。

---

## 3. Migration Tool 设计（PR44.3.2）

`scripts/faiss_to_milvus.py`（一次性迁移脚本，不进生产包）：

```
输入：FAISS store（company_id 指定）＋ Milvus 目标（finance_agent.finance_knowledge）
流程：
  1. FAISS 导出活跃 VectorRecord（skip deleted）
  2. 按 batch（如 512）upsert 到 Milvus
  3. 逐公司跑 Validation（数量 + 抽样对账）
输出：迁移报告（迁移数 / 失败数 / 抽样对账结果 / 差异明细）
```

**约束**：
- 幂等：可重复执行（主键 upsert），中断重跑不产生重复
- 分批：单批 ≤ 千条，避免 Milvus 大请求超时
- 失败可续：记录已完成 chunk_id 断点，重跑跳过
- 只迁活跃数据（逻辑删除不迁移；Milvus 侧 delete 即物理删除，无需保留标记）

---

## 4. 验收门禁

### 正确性（必须全过）

| 门禁 | 阈值 |
| ---- | ---- |
| 全量 pytest（含 MilvusStore 适配器测试） | 不回归现有 354 passed |
| 回归评测（FAISS 基线，--run-real） | Xiaomi R@5≥80% / MRR≥0.423 / NDCG@5≥0.474；CATL R@5=100% / MRR≥0.950 |
| **迁移后回归（Milvus，--run-real）** | 与 FAISS 基线**一致**（diff ≤ 一个评测点） |
| Migration Validation | count 相等 + 抽样 100% 对账 |

### 性能（PR44.3.3 Benchmark，仅作记录不设硬门禁）

| 指标 | 对比 |
| ---- | ---- |
| Recall@5 / MRR / NDCG | FAISS vs Milvus 无显著差异 |
| P95 latency | 记录（不提前设阈值，避免"过早优化"） |
| QPS | 记录并发上限（Milvus 的收益在并发/规模，非单机延迟） |

---

## 5. PR44.3 实施顺序（分步，不直接接入生产）

| 步骤 | 内容 | 交付物 | 验证 |
| ---- | ---- | ------ | ---- |
| **PR44.3.1** | Milvus Adapter：`MilvusStore` 实现 `add/search/delete/update/count` + 工厂 `get_store(backend="milvus")` | `app/rag/vectorstore/milvus_store.py` + 适配器测试 | 单元测试 + 回归门禁保持（FAISS 仍是默认后端） |
| **PR44.3.2** | Migration Tool | `scripts/faiss_to_milvus.py` | 真实 FAISS 存档迁移到 Milvus → Validation 全过 |
| **PR44.3.3** | Benchmark | `evaluation/` 基准脚本 | FAISS vs Milvus 指标对比表 |

**上线切换**（本 PR 范围之外，需用户确认）：生产默认后端由配置切换
`RAG_VECTOR_BACKEND=faiss → milvus`，`get_store(backend=...)` 决定。

> **PR44.4 已实现（2026-08-07）**：Production Vector Backend Switch。
> - `config.py` 新增 `rag_vector_backend`（Literal faiss/milvus，默认 faiss）+ `milvus_db_name`；
>   工厂 `get_store(backend=None)` 默认读配置，显式传参优先。
> - `app/rag/vectorstore/health.py` 启动健康检查（milvus：可达/库存在/collection 存在/维度匹配，
>   全只读 fail-fast），接入 `app/api/app.py` lifespan。
> - 策略：v1 无自动 fallback（数据一致性未实时保证）；回滚 = 改回 `RAG_VECTOR_BACKEND=faiss` 重启。
> - 离线评测钉死 FAISS（evaluator/eval_helpers）；`pymilvus` 已加入 requirements。
> - 不删 FAISS、不改 RAG pipeline/Hybrid/chunk schema。生产切换由部署期用户设置 `.env` 完成。

**部署形态**：Milvus 需新增 Docker 服务（开发期可用 Milvus Standalone 单容器
或 Milvus Lite 内嵌；生产独立部署），纳入现有 `deploy/` 编排。数据库建 `finance_agent`。

---

## 6. Review 待决问题（Review 时确认后冻结）

1. **dim 固定 1024**：测试用 Dummy(128) 与生产 BGE-M3(1024) 共用一个 collection 会维度冲突——
   是否接受"测试用独立 test collection / 测试 dim 走配置"的约定？
2. **`text` 冗余字段**：VARCHAR(8192) 上限是否满足最长 chunk？超长截断策略？
3. **`document_type` 过滤**：走 `metadata["source_type"]` JSON expr（推荐），还是 schema 加显式字段？
4. **迁移执行者**：faiss_to_milvus.py 由人工在 CI/本地跑，还是纳入部署脚本？
5. **Milvus 部署**：开发用 Milvus Lite（零运维）还是 Docker Standalone（贴近生产）？

---

## 7. 本次冻结清单（Checklist）

- [x] AD-1 Database = `finance_agent`（非 default）
- [x] AD-2 Collection = `finance_knowledge`，schema 含 chunk_id/company_id/document_id/year/section/text/embedding/metadata
- [x] AD-3 主键 = `chunk_id`（禁用 auto_id，保迁移可对账）
- [x] AD-4 Index = COSINE；开发 FLAT，生产 Benchmark 后再定 HNSW/IVF_FLAT
- [x] AD-5 Hybrid = **保持不变**（Dense Milvus + BM25 → RRF → Reranker；不用 Milvus Hybrid Search 作核心）
- [x] AD-6 Migration = Export VectorRecord → batch insert → Validation（count + 抽样 embedding hash/metadata）
- [x] AD-7 多公司隔离 = company_id scalar filter（不默认 Partition）

> 冻结后，PR44.3.1 的 MilvusStore 实现必须严格遵循本清单，任何偏差需回到本 PR 重新 Review。
