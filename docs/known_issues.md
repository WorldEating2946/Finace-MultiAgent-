# Known Issues（已知问题与约定）

> 记录 RAG 开发过程中踩过的问题与团队约定，避免重复踩坑。
> 开发经验类沉淀见 `docs/lessons_learned.md`，架构决策见 `docs/architecture_decisions.md`。

## 1. PDF 加载：依赖、Git 约定与扫描件限制

**依赖**：PDF Loader（`app/rag/loaders/pdf_loader.py`）依赖 `pymupdf`（fitz）。
安装：`uv pip install -r requirements.txt`。

**约定：企业 PDF 不进 Git**

- `*.pdf` 已加入 `.gitignore`（年报/研报体积大，按需本地放置）；
- 测试套件**不依赖真实 PDF**：`tests/conftest.py` 程序化生成 205 页模拟年报
  （`tests/data/`，同样 gitignore），保证 CI 确定性可复现；
- 真实年报验证：本地放置 PDF 后跑 `ingest("xxx.pdf", company="公司名")` 即可。

**已知限制**

- **扫描件 PDF（无文本层，纯图片）整页跳过**，当前不解析 → 需 OCR 的文档暂不支持；
- **文本层损坏（ToUnicode 乱码 / (cid:NNN) 占位）**：部分 PDF（如某些年报导出工具）内嵌
  字体 ToUnicode 映射损坏，`page.get_text()` 返回乱码（如 Xiaomi 2025 AR 实测 43% 质量分）；
  pdfplumber 则输出 `(cid:NNN)` 占位符（映射失败记号）。**两者均无法恢复正确文本**。
  **防御**：`pdf_loader` 多解析器 fallback + 质量门禁（`quality_checker`，阈值 0.8）——
  PyMuPDF → pdfplumber，质量均不达标时抛 `PDFQualityError` **阻断**，防止垃圾进 embedding；
  真实评测（`tests/test_real_xiaomi_eval.py`）捕获阻断后自动跳过；
- v1 为**每页扁平切块**：不做表格/图片/章节（section）解析，长文本结构化切分规划中；
- 标题识别为**正则启发式**：财务百分比（如 "22.3%"）等纯数字行已被排除，但运行页眉
  （每页重复章节名）仍可能使章节路径退化为章级。

## 2. Windows 非 ASCII 路径（faiss 序列化绕过）

- **症状**：`faiss.write_index` 在 Windows 上对含中文的路径（如 `data/vector_store/小米/`）
  经 C++ `fopen` 窄字符转码后写错路径，抛 `could not open ... No such file or directory`；
- **修复**：`vector_store.save/load` 改用 `faiss.serialize_index` + Python 文件读写
  （Python 原生处理 Unicode 路径），中英文 company 目录名均可正常存取；
- 有回归测试：`test_save_load_with_chinese_company_dir`。

## 3. pydantic-settings 的 `extra` 默认行为

- pydantic-settings **v2 的 `BaseSettings` 默认 `extra="forbid"`**，`.env` 中的
  非配置键（如 `POSTGRES_*`、`GITEE_TOKEN` 等）会导致 `Settings()` 校验失败；
- **约定**：`app/core/config.py` 的 `SettingsConfigDict` 必须显式 `extra="ignore"`，
  否则新增任意 `.env` 键都会崩。

## 4. 开发工具：Gitee API 中文乱码（非运行时）

- Windows Git Bash + `curl` 传中文参数给 Gitee API v5 会变 `�`（U+FFFD，不可逆损坏）；
- **约定**：传中文的 Gitee API 调用一律用 Python `urllib`（原生 UTF-8），
  仅纯 ASCII（分支名、token）可用 `curl`。

## 5. 小米 MD&A 章节检索瓶颈（OCR / 排序，2026-08-06）

PR #31（Query Rewrite）实测后确认：`小米智能手机全球出货量` 期望命中"管理层讨论及分析"，
top-20 已召回 4 个 MD&A chunk（rank 6/9/17/18），但 **CrossEncoder 用原始 query 精排时
压不进 top-5** —— 是排序问题，不是召回/改写问题。

**OCR 排查结论（单页实验，2026-08-06）**：

- `app/rag/loaders/pdf/ocr_parser.py`：`_RENDER_SCALE=2`、无 PSM（默认 3）、无图像预处理
  （无二值化/纠偏/去噪）、无置信度过滤；
- 但 **2x/PSM3 与 4x/PSM6/PSM3 实测质量相近**（p19 收入/利润/毛利均识别；4x 反而丢
  "税前"）。**没有低成本 OCR 配置能带来明显提升**；
- 质量门禁（`quality_checker`，阈值 0.8）是**全文级**评分，检测不出"语义乱码但字符合法"
  的页（如表格 OCR 噪声 `人民囊百万元` 仍可通过）；
- 根因更可能是：MD&A 智能手机出货量正文稀疏（p19 为收入财务表，非出货量讨论），
  CrossEncoder 偏好干净的"主席报告"文本。

**待办（非紧急）**：若需突破手机/汽车 Miss，方向是 MD&A 出货量正文的 OCR 结构化恢复
（per-page 质量门禁 + 更高针对性预处理），或补入招股书等补充数据源。当前不阻塞检索可用性。
