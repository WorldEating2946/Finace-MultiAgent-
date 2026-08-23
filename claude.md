# Claude.md

> AI Coding Assistant Collaboration Protocol
> 项目：FinaceAgent 多 Agent 金融智能分析平台

------

# 0. AI Assistant Operating Rules（最高优先级）

## 0.1 核心原则

Claude 作为本项目 AI 开发助手，必须：

1. 理解现有架构后再修改代码
2. 优先小范围修改，而不是大规模重构
3. 保持已有接口和模块边界稳定
4. 修改必须可解释、可验证、可回滚

## 0.2 禁止行为

未经用户明确确认，禁止：

- 删除已有模块
- 修改整体架构
- 大规模目录调整
- 修改公共 API
- 修改数据库 Schema
- 引入大型第三方依赖
- 重写已有核心逻辑

## 0.3 修改原则

优先级：

```
Bug修复
  >
功能增加
  >
局部优化
  >
代码重构
```

禁止：

为了“代码更优雅”进行无必要重构。

------

# 1. Project Overview（项目背景）

## 1.1 项目定位

FinaceAgent 是面向投资研究场景的多 Agent 智能分析平台。

目标：

通过：

- Large Language Model
- LangGraph Workflow
- RAG
- 数据工具调用

模拟真实金融研究团队流程。

主要能力：

- 企业基本面分析
- 财务数据分析
- 市场舆情分析
- 风险评估
- 自动研报生成

## 1.2 核心理念

本系统不是普通 Chatbot。

它是：

```
用户请求

↓

Manager Agent

↓

专业 Agent 协作

↓

数据计算

↓

风险推理

↓

报告生成
```

------

# 2. Architecture Invariants（架构不可变规则）

以下规则未经确认不得修改。

# 2.1 Agent 职责隔离

Agent 必须保持单一职责。

```
Research Agent

负责：
企业知识、行业信息、RAG检索


Financial Agent

负责：
财务数据处理、指标计算


Sentiment Agent

负责：
新闻、舆情分析


Risk Agent

负责：
风险推理


Report Agent

负责：
报告生成
```

禁止：

- 一个 Agent 承担多个领域职责
- Agent 之间复制相同能力

------

# 2.2 数据处理原则

## 非结构化数据

使用：

```
RAG
```

例如：

- 企业介绍
- 行业知识
- 政策文件
- 历史事件

## 结构化数据

使用：

```
Tool API
```

例如：

- 财务指标
- 股票数据
- 实时行情

原则：

```
结构化数据 → Tool

非结构化知识 → RAG
```

禁止：

让 LLM 直接计算财务指标。

正确流程：

```
数据获取

↓

Python计算

↓

指标验证

↓

LLM解释
```

------

# 2.3 LangGraph State 规则

统一使用：

```
TypedDict
```

管理 Workflow State。

禁止：

- Agent 修改全局变量
- 隐式传递状态
- 非序列化对象进入 State

State 必须：

- 可追踪
- 可序列化
- 可恢复

------

# 3. Tech Stack

| 模块            | 技术                           |
| --------------- | ------------------------------ |
| 语言            | Python 3.11+                   |
| Web             | FastAPI                        |
| 数据库          | PostgreSQL                     |
| ORM             | SQLAlchemy Async               |
| 缓存            | Redis                          |
| Agent Framework | LangGraph                      |
| LLM             | DeepSeek/OpenAI/Claude         |
| RAG             | LangChain + BGE + FAISS/Milvus |
| 数据计算        | Pandas/Numpy                   |
| 金融数据        | AkShare/Tushare                |
| 前端            | Streamlit → React              |

------

# 4. Directory Rules（目录职责）

```
app/

├── agents/
│
│   Agent角色实现

├── workflow/
│
│   LangGraph State 和 Graph

├── rag/
│
│   文档加载、切片、检索

├── tools/
│
│   外部能力封装

├── database/
│
│   数据库连接和Session

├── models/
│
│   Pydantic / ORM模型

├── services/
│
│   业务编排逻辑

├── api/
│
│   FastAPI接口

└── utils/
    公共工具
```

禁止：

Agent 中直接：

- 写 SQL
- 调接口
- 管理数据库连接

------

# 5. Development Rules（开发规范）

# 5.1 Agent规范

每个 Agent 必须：

拥有：

```
Input Schema

↓

独立 Prompt

↓

独立 Tools

↓

Output Schema
```

输出必须结构化。

示例：

```python
class FinancialReport(BaseModel):

    summary: str

    metrics: dict

    insight: str
```

------

# 5.2 Tool规范

所有外部能力必须封装为：

```
LangChain @tool
```

要求：

- 输入明确
- 输出结构化
- 单一职责

禁止：

Tool 内包含复杂业务流程。

------

# 5.3 Prompt规范

Prompt 只负责：

- 角色定义
- 输出格式
- 分析角度

禁止：

在 Prompt 中：

- 编写复杂业务逻辑
- 硬编码判断规则

------

# 6. Modification Policy（修改预算）

为了保持代码可理解性：

默认单次任务：

允许：

```
修改文件 <= 5个

新增代码 <= 300行
```

如果超过：

必须先说明：

```
1. 修改原因

2. 影响范围

3. 风险

4. 是否需要拆分任务
```

禁止：

一个小需求引发全项目重构。

------

# 7. AI Loop Governance（循环治理）

每轮 AI 修改必须遵循：

```
Understand

↓

Plan

↓

Modify

↓

Test

↓

Summarize

↓

Checkpoint
```

每轮结束必须输出：

```
修改文件：

原因：

影响：

测试：

风险：
```

连续修改超过：

```
5轮
```

必须：

- 总结当前状态
- 创建 Git checkpoint
- 建议开启新的 Session

避免长期上下文导致架构漂移。

------

# 8. Testing Rules

完成代码修改必须验证：

最低要求：

```
代码格式检查

↓

单元测试

↓

接口测试
```

Agent：

必须支持独立测试。

例如：

Financial Agent

不依赖 LangGraph 也可以运行。

------

# 9. Git Rules

提交格式：

```
动作 + 模块 + 目的
```

示例：

```
新增: Financial Agent财务分析能力

修改: 优化RAG检索策略

修复: 修复State字段丢失问题
```

必须 Commit：

- 完成一个 Agent
- 完成 Workflow 节点
- 数据库结构变化
- 大规模修改前

------

# 10. Change Summary（修改报告）

每次代码修改完成后必须输出：

## Change Summary

### Files Changed

```
xxx.py
xxx.py
```

### Reason

说明为什么修改。

### Impact

说明影响范围。

### Validation

说明如何验证。

### Risk

说明潜在风险。

### Rollback

说明如何恢复。

------

# 11. Memory & Decision Records

长期经验必须沉淀。

推荐维护：

```
docs/

├── architecture_decisions.md

架构决策记录


├── known_issues.md

已知问题


└── lessons_learned.md

开发经验
```

禁止：

重复讨论已经确定的架构决策。

------

# 12. Security Rules

禁止：

代码中出现：

```
API_KEY

PASSWORD

SECRET
```

统一使用：

```
.env
```

管理：

- API Key
- Database URL
- Redis配置

------

# 13. Current Roadmap

## Phase 1

基础闭环：

- LangGraph Workflow
- Agent接口
- RAG Demo
- 财务工具
- Markdown研报

## Phase 2

能力增强：

- 多数据源验证
- Hybrid Retrieval
- 风险推理

## Phase 3

平台化：

- 用户认证
- 多租户
- Human-in-the-loop

------

# Final Rule

任何代码修改必须满足：

```
代码正确

+

架构稳定

+

开发者理解

+

未来可维护
```

AI 的目标不是替代开发者设计系统。

AI 的目标是：

在保持开发者控制力的前提下，提高工程效率。