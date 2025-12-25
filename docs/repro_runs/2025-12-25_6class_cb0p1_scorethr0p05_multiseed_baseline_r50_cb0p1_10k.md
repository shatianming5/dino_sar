# 复现记录：RSAR 6-class / Baseline R50-FPN（ClassBalanced t=0.1, 10k iters）多 seed（test）

- 日期：2025-12-25
- 代码版本：5948b75
- RUN_TAG：6class_cb0p1_scorethr0p05
- 命令：`DINO_SAR_RUN_TAG=6class_cb0p1_scorethr0p05 bash scripts/run_multiseed_baseline_r50_10k_cb0p1_6class.sh 1`
- 配置：`configs/baselines/rotated_retinanet_obb_r50_fpn_rsar_le90_cb0p1_10kiter.py`
- seeds：`0/1/2`

## 指标（mAP@0.5，test）

| seed | mAP |
|---:|---:|
| 0 | 0.08507496863603592 |
| 1 | 0.1057405173778534 |
| 2 | 0.11602496355772018 |
| mean | 0.10228014985720317 |
| std (sample) | 0.015762491996716404 |

## 目录（本地，不入 Git）

- 日志：`outputs/_rerun_6class/logs/6class_cb0p1_scorethr0p05/20251224_235955_multiseed_baseline_r50_cb0p1_10k.log`
- 训练输出：`outputs/multiseed_6class_cb0p1_scorethr0p05/baseline_r50_cb0p1_10k/seed*/`
- 评估输出：`outputs/eval_6class_cb0p1_scorethr0p05/multiseed_baseline_r50_cb0p1_10k/seed*/`

