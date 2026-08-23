# Lessons Learned（开发经验）

> 记录项目开发过程中的经验教训，避免重复踩坑。
> 架构决策见 `docs/architecture_decisions.md`，已知问题见 `docs/known_issues.md`。

## 1. 流程例外：#27 直接推送 develop（2026-08-05）

**事件**：PR #27（Reranker 在线性能优化）因故直接 commit + push 到 develop，
未走 feature → PR → merge 规范流程。

**结论**：#27 为**一次性流程例外**，功能已验证充分（Recall@5=100%、Top1=90%、稳态 ~210ms），保留不重做。
**后续恢复规范**：从 #28 起严格执行 **feature 分支 → PR → 用户确认合并**，
禁止直接推送 master/develop（见 `docs/GIT_WORKFLOW.md`）。
