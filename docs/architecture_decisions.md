# 架构决策记录（Architecture Decision Records）

> 项目：FinaceAgent
> 说明：记录已确认的架构决策。**禁止重复讨论已确定的决策**（CLAUDE.md §11）。

---

## ADR-001: RAG 模块对外检索接口

- **状态**: 已接受
- **决策时间**: 2026-08-03
- **决策者**: 项目组（与 Agent 负责人同步确认）
- **相关文档**: [docs/RAG_ARCHITECTURE.md](RAG_ARCHITECTURE.md)
- **修订**: 2026-08-03，入口文件由 `service.py` 更名为 `pipeline.py`（语义贴合对外流水线），接口契约不变；同日统一模型命名为 `DocumentChunk`，新增加载接口 `load_documents()`

### 背景

Research Agent 需要从企业知识库检索非结构化知识（招股书、财报、行业报告等），
但不应感知底层向量库的实现细节（Phase 1 FAISS / Phase 2 Milvus 评估中）。

### 决策

RAG 模块通过 `app/rag/pipeline.py` 暴露唯一检索入口：

```python
def retrieve(
    query: str,
    company: str,                  # 一级过滤维度，必填
    top_k: int = 3,
    doc_type: str | None = None,
) -> RetrievalResult: ...
```

```python
class RetrievalResult(BaseModel):
    query: str
    chunks: list[DocumentChunk]    # 精排后，最多 top_k 个；含 source_name/page
    scores: list[float]            # 各 chunk 的 Reranker 置信度 [0,1]
    confidence: float              # Top-1 置信度；< 0.75 时由 Agent 决定降级
```

决策要点：

1. **`company` 为必填参数**，作为一级元数据过滤维度，避免跨公司检索噪声；
2. **返回 `confidence`（Top-1 置信度）**，`< 0.75` 时由 Agent 决定降级策略；
3. **输出必带来源**（source_name / page），保证研报引用可溯源；
4. **Agent 不感知向量库实现**，底层切换（FAISS → Milvus）对 Agent 零影响。

### 备选方案

- **最简 `retrieve(query) -> documents`**：签名更简洁，但存在两个致命缺陷：
  - 缺 `company` 过滤维度，多公司知识库跨公司噪声无法避免；
  - 缺置信度信号，反幻觉兜底机制（置信度阈值降级）失效。
  已否决。

### 理由

- 对齐 CLAUDE.md §4（Agent 禁止直接写 SQL / 调接口 / 管连接）与 §2.2（非结构化知识 → RAG）；
- `company` 是 FinaceAgent 知识库的领域固有维度，必须由接口显式承载，不能依赖 LLM 从问题中猜测；
- 置信度兜底是项目反幻觉的核心机制，接口必须透出该信号供 Agent 决策。

### 影响

- Research Agent 按此接口对接；`rag.pipeline` 内部封装 company 过滤 → Hybrid 召回 → Reranker 精排 → 置信度计算；
- 向量库抽象层（`vector_store.py`）负责底层切换，接口契约不随向量库变更。

### 回滚 / 演进

- 若后续需扩展（如新增过滤维度、检索策略参数），按"保持公共 API 稳定"原则走增量演进（新增可选参数），不破坏现有签名。
