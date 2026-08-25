

# FinaceAgent 环境规范

## 1. 文档目的

本文档用于统一 FinaceAgent 项目开发环境，保证团队成员在不同机器上具有一致的基础运行环境。

**环境管理原则：**

- 所有成员使用统一基础环境
- 公共依赖统一维护
- 模块专属依赖按需扩展
- 新增依赖必须经过团队同步

---

## 2. 基础环境说明

### Python 版本

项目统一使用：**Python 3.11**

**选型原因：**
- 性能较好且生态成熟
- 兼容 FastAPI、LangChain、LangGraph 等主流框架
- 避免不同 Python 版本导致依赖冲突

---

## 3. 环境管理方案

项目采用：**Conda + pip** 结合的方式管理环境。

### 职责划分

| 文件               | 作用                 |
| :----------------- | :------------------- |
| `environment.yml`  | 创建基础 Python 环境 |
| `requirements.txt` | 管理 Python 项目依赖 |
| `.env.example`     | 管理环境变量模板     |

### 目录结构

```text
FinaceAgent/
├── environment.yml
├── requirements.txt
└── .env.example
```

## 4. 创建开发环境

首次拉取项目后，请按照以下步骤配置开发环境。

### 4.1 进入项目根目录

确保当前目录位于项目根目录：

Bash

```
cd FinaceAgent
```

目录中应包含：

Plaintext

```
FinaceAgent/
├── environment.yml
├── requirements.txt
├── README.md
└── app/
```

> **注意：**
>
> `environment.yml` 位于项目根目录，与 `README.md`、`requirements.txt` 同级。
>
> 请勿在 `docs/`、`app/` 等子目录中执行环境创建命令。

### 4.2 创建 Conda 环境

执行：

Bash

```
conda env create -f environment.yml
```

该命令会：

- 创建 Python 3.11 环境
- 根据 `requirements.txt` 自动安装项目依赖

### 4.3 激活环境

Bash

```
conda activate finance-agent
```

### 4.4 环境验证

检查 Python 版本：

Bash

```
python --version
```

预期输出：

Plaintext

```
Python 3.11.x
```

验证核心依赖：

Bash

```
python -c "import fastapi,langchain,langgraph,sqlalchemy; print('environment ok')"
```

预期输出：

Plaintext

```
environment ok
```

输出 `environment ok` 表示基础开发环境创建成功。

### 环境更新

当项目依赖发生变化时：

Bash

```
pip install -r requirements.txt
```

或者：

Bash

```
conda env update -f environment.yml
```

## 5. 基础依赖管理原则

公共依赖进入基础环境，主要包含以下类型：  

- **Web 服务：** `fastapi`, `uvicorn`

    

- **配置管理：** `pydantic-settings`, `python-dotenv`

    

- **数据库：** `sqlalchemy`, `asyncpg`, `redis`

    

- **AI 框架：** `langchain`, `langgraph`, `langchain-text-splitters`（RAG 文档切片）

    

- **开发工具：** `pytest`, `ruff`, `black`

    

## 6. 模块依赖管理

项目采用：**基础环境 + 模块扩展** 的依赖管理模式。

### 为什么不一次性安装所有依赖？

项目初期不将所有可能使用的库全部加入基础环境，原因：

- 避免环境体积过大，增加安装时间；
- 避免不同模块之间产生版本冲突；
- 避免未使用依赖污染开发环境；
- 降低后续技术方案调整成本。

例如，以下依赖暂不加入基础环境：

| **依赖**                 | **所属模块**       | **添加阶段**     |
| ------------------------ | ------------------ | ---------------- |
| `torch`                  | 模型训练、推理     | 模型开发阶段     |
| `transformers`           | 大模型/预训练模型  | 模型开发阶段     |
| `sentence-transformers`  | Embedding 向量模型 | RAG 开发阶段     |
| `faiss-cpu`             | 向量检索           | RAG 开发阶段     |
| `pandas`、`scikit-learn` | 数据分析、机器学习 | 算法模块开发阶段 |

### 管理原则

- 基础环境保持稳定；
- 模块依赖根据实际开发需求增加；
- 新增依赖必须经过团队同步；
- 不允许个人直接修改本地环境后继续开发。

### 新增模块依赖示例

例如 RAG 模块需要：

Plaintext

```
sentence-transformers
chromadb
```

开发成员需要：

1. 修改 `requirements.txt`
2. 提交 Git
3. 说明依赖用途
4. 通知团队成员同步

最终形成层次结构：

Plaintext

```
基础环境
    │
    ├── Web服务
    ├── 数据库
    ├── Agent框架
    │
    ▼
模块扩展依赖
    │
    ├── RAG
    ├── 模型训练
    └── 数据处理
```

## 7. 新增依赖流程

任何新增依赖必须遵循以下规范流程[cite: 1]：

1. **第一步：确认必要性**

   开发成员确认是否已有替代方案、是否所有模块都需要该依赖[cite: 1]。

2. **第二步：更新依赖文件**

   将新增依赖（如 `sentence-transformers`）追加写入 `requirements.txt`[cite: 1]。

3. **第三步：提交说明**

   在 Git 提交与 PR 中明确标注用途：

   - **Commit 示例：** `git commit -m "配置: 添加RAG模块依赖"`

   - **PR 说明：**

     > **新增依赖：** `sentence-transformers`
     >
     > [cite: 1]
     >
     > **用途：** 用于 RAG 向量模型加载[cite: 1]

4. **第四步：团队同步**

   其他团队成员拉取代码后运行同步：

   

   ```Bash
   pip install -r requirements.txt
   ```

## 8. 环境变量管理

敏感信息禁止直接提交至仓库，包括 API Key、数据库密码、Token 等。

- **实际运行环境：** 使用 `.env` 文件（**禁止提交**至 Git）
- **模板配置文件：** 维护 `.env.example` 文件（**必须提交**至 Git）

### `.env.example` 示例

代码段

```
OPENAI_API_KEY=
DATABASE_URL=
REDIS_URL=
```

## 9. Docker 环境说明

当前阶段 Docker 主要用于统一项目基础服务环境。

暂不使用 Docker 部署 Python 开发环境。

**原因：**

- Python 开发环境由 Conda 统一管理；
- 便于本地调试和代码开发；
- 降低团队初期环境复杂度。

**后续计划使用 Docker Compose 管理：**

- PostgreSQL
- Redis
- 向量数据库（根据 RAG 方案确定）

**Docker 使用原则：**

- 数据库、中间件等基础服务优先容器化；
- 应用代码保持本地开发；
- 生产环境再统一容器化部署。

**架构目标：**

Plaintext

```
开发环境：

Conda
 │
Python应用
 │
Docker
 │
PostgreSQL / Redis / Vector DB
```

**实现目标：** 统一服务环境 + 灵活代码开发。

## 10. 环境问题排查与维护

Bash

```
# 查看当前 Python 版本
python --version

# 查看已安装依赖
pip list

# 查看所有 Conda 环境
conda env list

# 更新 Conda 基础环境
conda env update -f environment.yml
```

## 11. 环境规范总结

FinaceAgent 整体采用：

$$\text{统一基础环境} + \text{模块按需扩展} + \text{Git 管理依赖变化} + \text{Docker 统一基础服务}$$

**核心目标：**

保证团队成员开发环境高度一致，同时保持项目依赖可维护、可扩展。