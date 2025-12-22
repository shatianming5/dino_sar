# dino_sar

**GitHub 仓库（所有阶段都 push 到这里）：** https://github.com/shatianming5/dino_sar

**工作流（最重要）：** `EXECUTION_PLAN.md` 里每完成一个步骤（阶段）→ 立刻 `git commit` → `git push` 到 `origin/main`。

## 这是什么

用 **DINOv3 + LoRA** 做 **SAR 船只检测** 的工程与实验记录；主线数据集 **RSAR（DOTA/OBB）**，兜底验证 **HRSID（HBB）**。

## 文档入口

- `plan.md`：原始 Proposal（分阶段目标/验收/注意事项）
- `EXECUTION_PLAN.md`：把 `plan.md` 落到“可执行步骤 + 每步结束必须 push”的执行清单
- `docs/REPRODUCE.md`：从零复现（环境→训练→评估）
- `docs/RESULTS.md`：结果表（统一口径）
- `docs/ROBUSTNESS.md`：鲁棒性曲线（干扰注入）

## 数据说明（不进 Git）

本仓库不提交数据集文件（`train/`、`val/`、`test/` 已在 `.gitignore` 中忽略）。在本机准备好数据后再进行训练/评估。
