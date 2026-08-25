# FinaceAgent Git Workflow

## 1. 分支管理规范

项目采用 **Git Flow 简化模型**，分支层次结构如下：

$$\text{master} \longrightarrow \text{develop} \longrightarrow \text{feature/*}$$

### 分支说明

#### `master` 分支

- **用途**：生产环境稳定版本。
- **规则**：
  - 禁止直接开发
  - 禁止直接 push
  - 只能通过 Pull Request 合并

#### `develop` 分支

- **用途**：日常开发集成与团队测试。
- **规则**：
  - 所有功能开发完成后统一合并至 `develop`

#### `feature/*` 分支

- **用途**：个人功能开发。
- **命名规范**：`feature/<功能名称>`
- **示例**：
  - `feature/langgraph-agent`
  - `feature/rag-module`
  - `feature/database-layer`

## 2. 开发流程

### 第一步：同步 develop

开始新功能开发前，请先拉取并同步最新的 `develop` 分支代码：



```
git checkout develop
git pull origin develop
```

### 第二步：创建功能分支

从 `develop` 分支创建个人开发分支：

Bash

```
git checkout -b feature/<功能名称>
```

- **示例**：

  Bash

  ```
  git checkout -b feature/rag-module
  ```

- **命名要求**：

  - 使用英文小写
  - 使用短横线分隔
  - 描述具体功能
  - **推荐**：`feature/langgraph-agent`、`feature/postgresql-schema`、`feature/redis-cache`
  - **不推荐**：`feature/test`、`feature/mybranch`、`feature/update`

### 第三步：开发与本地验证

开发过程中可随时检查代码状态与差异：

- **查看修改状态**：`git status`
- **查看具体变化**：`git diff`

开发完成后，**必须进行**：

1. 本地运行验证
2. 基础功能测试
3. 确认无明显错误

### 第四步：提交代码

- **暂存修改**：

  Bash

  ```
  git add .
  ```

- **提交 Commit**：

  Bash

  ```
  git commit -m "类型: 描述"
  ```

- **示例**：

  Bash

  ```
  git commit -m "新增: 添加订单查询Agent"
  ```

### 第五步：推送个人分支

- **首次推送**（建立远程关联）：

  Bash

  ```
  git push -u origin feature/<功能名称>
  ```

- **后续推送**：

  Bash

  ```
  git push
  ```

- **示例**：

  Bash

  ```
  git push -u origin feature/rag-module
  ```

### 第六步：创建 Pull Request

提交完成后，将代码合并入开发主干：

$$\text{feature/*} \longrightarrow \text{develop}$$

**PR 要求**：

- 标明修改内容
- 标明影响模块
- 标明测试情况
- 关联相关 Issue（如果存在）

## 3. Commit 规范

### Commit 格式

Plaintext

```
类型: 简短描述
```

### 类型说明

| **类型** | **使用场景**   |
| -------- | -------------- |
| **新增** | 新增功能       |
| **修改** | 修改已有逻辑   |
| **修复** | Bug 修复       |
| **优化** | 性能或体验优化 |
| **重构** | 代码结构调整   |
| **配置** | 环境或工程配置 |
| **文档** | 文档修改       |

### 示例

- `新增: 添加RAG检索流程`
- `修改: 优化Agent状态管理`
- `修复: 修复Redis连接异常`
- `配置: 更新Docker开发配置`
- `文档: 更新Git协作规范`

### 要求

- 一次 commit 只完成一个明确目标
- 避免巨大 commit
- commit 信息必须能够清晰描述修改目的

## 4. Pull Request 规范

所有代码进入 `develop` 前必须经过 Pull Request。

### PR 标题格式

Plaintext

```
类型: 功能描述
```

- **示例**：`新增: 实现LangGraph工作流`

### PR 描述必须包含

#### 修改内容

说明完成的功能。

#### 影响范围

例如：

- Agent 模块
- RAG 模块
- Database 模块

#### 测试情况

说明：

- 是否通过本地测试
- 是否验证核心流程

### Review 要求

- 至少一名团队成员 Review 通过后方可合并。

## 5. Code Review 规范

### Review 重点

#### 架构

- 是否符合项目目录设计
- 是否破坏模块边界

#### 代码质量

- 是否存在重复代码
- 是否存在明显风险

#### 依赖

- 是否新增必要依赖
- 是否影响环境一致性

#### AI 生成代码

- 是否理解代码逻辑
- 是否经过人工验证

### Review 意见分类

- **Approve**：通过
- **Request Changes**：需要修改
- **Comment**：建议优化

## 6. 分支同步与冲突处理

开发过程中请保持 `feature` 分支最新：

Bash

```
git checkout feature/xxx
git merge develop
```

### 冲突处理原则

如果发生冲突：

1. 不直接覆盖他人代码
2. 理解双方修改目的
3. 协商解决方案
4. 完成测试验证

解决冲突后提交：

Bash

```
git add .
git commit -m "修复: 解决分支冲突"
```

## 7. Git 常用操作

| **操作**           | **命令**            |
| ------------------ | ------------------- |
| **查看状态**       | `git status`        |
| **查看提交记录**   | `git log --oneline` |
| **查看分支状态**   | `git branch -vv`    |
| **同步远程**       | `git pull`          |
| **查看远程仓库**   | `git remote -v`     |
| **撤销未提交修改** | `git restore .`     |

## 8. AI 辅助开发规范

本项目允许使用 AI Coding 工具（如 **Claude Code**, **Cursor**, **ChatGPT** 等）辅助开发，但**AI 生成的代码必须经过人工理解和验证**。

### 开发要求

1. 保持小范围修改
2. 修改前明确目标
3. 修改后检查 diff
4. 保留 Git checkpoint

### 明确禁止

- 禁止 AI 一次性重构多个核心模块
- 禁止未经 Review 直接合并 AI 代码
- 禁止删除已有架构设计
- 禁止引入未经确认的新依赖

> 详细规范参考项目中的 `claude.md` 文件。

## 9. 禁止事项

- ❌ **禁止**直接 push 到 `master`
- ❌ **禁止**提交 `.env` 文件
- ❌ **禁止**提交密钥 / Token 等敏感信息
- ❌ **禁止**提交大型无关文件
- ❌ **禁止**未提前沟通修改公共配置
- ❌ **禁止**在大范围重构前不进行讨论

以下文件发生修改时，**必须提前通知团队**：

- `requirements.txt`
- `environment.yml`
- `docker-compose.yml`
- 数据库 schema 变动
- 核心配置文件

---

## 12. 多人协作开发流程

多人协作时，所有成员遵循统一流程，确保代码、文档、依赖变更可追踪、可审查。

### 12.1 整体协作流程

```
领取任务
   │
   ▼
同步最新 develop（git checkout develop && git pull）
   │
   ▼
创建个人 feature 分支（feature/<功能名称>）
   │
   ▼
分支内开发 + 本地验证
   │
   ▼
commit（格式：类型: 描述）+ push 远程 feature 分支
   │
   ▼
创建 Pull Request（feature → develop）
   │
   ▼
Code Review（至少一名成员通过）
   │
   ▼
合并到 develop
```

### 12.2 任务隔离与分支管理

- 每个任务独立一个 `feature/<功能名称>` 分支，禁止多个任务共用分支；
- 分支生命周期 = 一个任务：创建 → 开发 → Review → 合并 → 删除；
- 分支命名必须反映功能，禁止 `test` / `update` 等无意义命名（见 §2）；
- 合并前保持分支与 develop 同步，优先解决冲突（见 §6）。

### 12.3 同步节奏

- 开始任务前：`git checkout develop && git pull`，确保基于最新 develop；
- 任务进行中：每次提交前同步一次 develop，避免冲突积累；
- 合并前：再次同步 develop，本地跑通验证后再创建 / 更新 PR。

### 12.4 Pull Request 协作规范

- PR 描述必须包含：修改内容、影响模块、测试情况（见 §4）；
- 涉及**依赖 / 接口 / 架构**变化，必须在 PR 中同步列出对应文档更新；
- 至少一名成员 Review 通过后方可合并（见 §5），不允许作者自行合并；
- 一次 PR 只完成一个明确目标，禁止混入无关改动。

### 12.5 文档同步责任

以下变化必须同步更新对应文档：

| 变化类型 | 必须更新的文档 |
| -------- | -------------- |
| 新增 / 升级依赖 | `requirements.txt` + `docs/ENVIRONMENT.md` |
| 公共接口变化 | 对应架构文档（如 `RAG_ARCHITECTURE.md`）+ `docs/architecture_decisions.md` |
| 架构调整 | `docs/architecture_decisions.md` |
| 数据库 Schema 变化 | 迁移脚本 + 相关文档 |
| 协作规范调整 | 本文档 `docs/GIT_WORKFLOW.md` |

### 12.6 冲突与交接

- 发生冲突时：不直接覆盖他人代码，理解双方修改目的后协商解决（见 §6）；
- 任务中断 / 交接：确保 feature 分支已 push、PR 状态说明清楚，不留未提交的本地改动；
- 合并完成后及时删除已合并的 feature 分支，保持远程整洁。

### 12.7 成员禁止事项

- ❌ 禁止直接 push 到 `master` / `develop`；
- ❌ 禁止未经 Review 合并代码；
- ❌ 禁止长期持有未同步 develop 的 feature 分支；
- ❌ 禁止在 PR 中混入无关改动。

