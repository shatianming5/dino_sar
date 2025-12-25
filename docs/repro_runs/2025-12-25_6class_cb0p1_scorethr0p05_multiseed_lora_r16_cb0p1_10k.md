# 复现记录：RSAR 6-class / LoRA r=16（ClassBalanced t=0.1, 10k iters, full targets）多 seed（test）

- 日期：2025-12-25
- 代码版本：f99e491
- RUN_TAG：6class_cb0p1_scorethr0p05
- 命令：`DINO_SAR_RUN_TAG=6class_cb0p1_scorethr0p05 bash scripts/run_multiseed_lora_r16_10k_cb0p1_6class.sh 1`
- 配置：`configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_10kiter_cb0p1.py`
- seeds：`0/1/2`

## 指标（mAP@0.5，test）

| seed | mAP |
|---:|---:|
| 0 | 0.08026016503572464 |
| 1 | 0.08211181312799454 |
| 2 | 0.08375352621078491 |
| mean | 0.08204183479150136 |
| std (sample) | 0.001747731615706023 |

## 目录（本地，不入 Git）

- 日志：`outputs/_rerun_6class/logs/6class_cb0p1_scorethr0p05/20251225_013418_multiseed_lora_r16_cb0p1_10k.log`
- 训练输出：`outputs/multiseed_6class_cb0p1_scorethr0p05/lora_r16_cb0p1_10k/seed*/`
- 评估输出：`outputs/eval_6class_cb0p1_scorethr0p05/multiseed_lora_r16_cb0p1_10k/seed*/`

