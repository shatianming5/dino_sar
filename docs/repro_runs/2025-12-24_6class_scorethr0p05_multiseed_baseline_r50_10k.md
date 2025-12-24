# 复现记录：RSAR 6-class / Baseline R50-FPN（10k iters）多 seed（test）

- 日期：2025-12-24
- 代码版本：95e3ac4
- RUN_TAG：6class_scorethr0p05
- 命令：`DINO_SAR_RUN_TAG=6class_scorethr0p05 bash scripts/run_multiseed_baseline_r50_10k_6class.sh 1`
- 配置：`configs/baselines/rotated_retinanet_obb_r50_fpn_rsar_le90_10kiter.py`
- seeds：`0/1/2`

## 指标（mAP@0.5，test）

| seed | mAP |
|---:|---:|
| 0 | 0.07764682173728943 |
| 1 | 0.10954006761312485 |
| 2 | 0.11523851007223129 |
| mean | 0.1008084664742152 |
| std (sample) | 0.020259921071035972 |

## 目录（本地，不入 Git）

- 日志：`outputs/_rerun_6class/logs/6class_scorethr0p05/20251224_134958/`
- 训练输出：`outputs/multiseed_6class_scorethr0p05/baseline_r50_10k/seed*/`
- 评估输出：`outputs/eval_6class_scorethr0p05/multiseed_baseline_r50_10k/seed*/`

