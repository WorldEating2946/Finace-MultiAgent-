"""RAG 数据模型（对外接口契约的返回类型）。

定义见 docs/RAG_ARCHITECTURE.md §4.3 与 ADR-001。
"""

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """文档元数据统一规范（所有 loader 输出同一组字段，防多类型文档字段漂移）。

    固定字段（PDF 用 page / Markdown 用 section，统一到同一键，避免
    page_num / title 等各自命名导致检索过滤混乱）：

        source         来源（文件路径 / URL），稳定标识
        source_name    可读溯源名（loader 填文件名；splitter 附加上 section 路径）
        company        所属企业（入库编排填充）
        doc_type       文档格式：markdown / text / pdf（splitter 依赖格式做切分）
        source_type    文档语义类型：annual_report / research_report / policy / news
                      （PR #35 多源融合：与 doc_type 分离，入库编排填充，可空）
        page           页码（PDF 必填；markdown/txt 无分页为 None）
        section        章节路径（如 "第四章 > 4.1"），splitter 填充
        chapter        章标题（如 "第四章 业务回顾"），结构识别填充
        title          文档标题（默认取文件 stem）
        table          是否含表格（"true"/""，布局识别预留）
        header         页眉文本（布局识别预留）
        footer         页脚文本（布局识别预留）
        original_text  原始抽取文本（繁简归一前，供最终回答引用原文）
        created_time   创建/发布时间（ISO 字符串，默认取文件 mtime），可选
    """

    source: str = ""
    source_name: str = ""
    company: str = ""
    doc_type: str = ""
    source_type: str = ""
    page: int | None = None
    section: str = ""
    chapter: str = ""
    title: str = ""
    table: str = ""
    header: str = ""
    footer: str = ""
    content_type: str = ""       # "text" / "table"（表格 Document）
    table_title: str = ""        # 表格标题（如 "五年财务概要"）
    table_headers: list[str] = Field(default_factory=list)  # 表头（年份等）
    original_text: str = ""
    created_time: str = ""


class Document(BaseModel):
    """原始文档（loader 输出，未切分）。

    与 DocumentChunk 的分工：
        Document       → 原始文档：text + 统一 DocumentMetadata
        DocumentChunk  → 切分后的检索单元（splitter 输出，含 chunk_id）
    """

    text: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)


class DocumentChunk(BaseModel):
    """统一文档块（加载 / 切片 / 检索的公共单元）。"""

    chunk_id: str        # 唯一标识：{source_hash}-{seq}
    company: str         # 所属企业（一级过滤字段；加载阶段未知则留空，由入库编排填充）
    doc_type: str        # 文档类型：招股书/财报/行业报告/政策；加载阶段默认 "text"
    source: str          # 来源文件路径
    source_name: str     # 可读溯源标注，如"招股书 > 第3章 > 3.1 商业模式"
    page: int | None = None  # 页码（PDF 有；Markdown/TXT 无分页为 None）
    text: str            # 文本内容
    dense_vector: list[float] = Field(default_factory=list)  # 稠密向量（Hybrid 检索时填充，不持久化）
    sparse_tokens: list[str] = Field(default_factory=list)   # BM25 稀疏词（Hybrid 检索时填充，不持久化）
    metadata: dict = Field(default_factory=dict)  # 扩展元数据（含 DocumentMetadata 字段 + chunk_index）


class RetrievalResult(BaseModel):
    """retrieve() 的返回结构。"""

    query: str                    # 原始查询
    chunks: list[DocumentChunk]   # 精排后，最多 top_k 个；含来源引用
    scores: list[float]           # 各 chunk 的 Reranker 置信度 [0,1]
    confidence: float             # Top-1 置信度；< 0.75 时由调用方决定降级
