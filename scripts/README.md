# scripts

统一放一键脚本（训练 / 评估 / 可视化）。  
每个阶段（Step）结束后按 `EXECUTION_PLAN.md` 要求 `commit + push`。

## 自动记录 + push（MMRotate1.x）

- 只记录并 push（已跑完 test）：`./scripts/record_mmrotate1x_test_and_push.sh <RSAR_test_log> [note...]`
- 一键评估 + 记录 + push：`./scripts/eval_rsar_mmrotate1x_test_and_push.sh <RSAR_config.py> <checkpoint.pth> [note...]`
