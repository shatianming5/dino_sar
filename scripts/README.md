# scripts

统一放一键脚本（训练 / 评估 / 可视化）。  
每个阶段（Step）结束后按 `EXECUTION_PLAN.md` 要求 `commit + push`。

## 自动记录 + push（MMRotate1.x）

- 只记录并 push（已跑完 test）：`./scripts/record_mmrotate1x_test_and_push.sh <RSAR_test_log> [note...]`
- 一键评估 + 记录 + push：`./scripts/eval_rsar_mmrotate1x_test_and_push.sh <RSAR_config.py> <checkpoint.pth> [note...]`
- 训练 + 评估 + 记录 + push：`./scripts/train_eval_rsar_mmrotate1x_and_push.sh <RSAR_config.py> <work_dir_name> [note...]`
- RK-LoRA(stage2+3) 消融扫参（顺序执行）：`WAIT_FOR_TMUX_SESSION=<tmux_session> ./scripts/sweep_rsar_mmrotate1x_rk_lora_stage23.sh`
- RK-LoRA(stage2+3) 额外消融（neck+head）：`WAIT_FOR_TMUX_SESSION=<tmux_session> ./scripts/sweep_rsar_mmrotate1x_rk_lora_stage23_extras.sh`
- A-LoRA 消融扫参（顺序执行）：`WAIT_FOR_TMUX_SESSION=<tmux_session> ./scripts/sweep_rsar_mmrotate1x_a_lora.sh`

Notes:
- README 记录默认会额外解析 `iou_thr: 0.75` 表格，写入 `tank/bridge/harbor` 的 AP75。
