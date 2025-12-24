# 复现记录：RSAR 6-class / LoRA r16（2k iters）多 seed（test）

- 日期：2025-12-24
- 代码版本：e90a8b8
- RUN_TAG：6class_scorethr0p05
- 命令：`DINO_SAR_RUN_TAG=6class_scorethr0p05 bash scripts/run_multiseed_lora_r16_2k_6class.sh 1`
- 配置：`configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_2kiter.py`
- seeds：`0/1/2`

## 指标（mAP@0.5，test）

| seed | mAP |
|---:|---:|
| 0 | 0.025282567366957664 |
| 1 | 0.03539997711777687 |
| 2 | 0.03523268923163414 |
| mean | 0.031971744572122894 |
| std (sample) | 0.005793601217790976 |

## 目录（本地，不入 Git）

- 日志：`outputs/_rerun_6class/logs/6class_scorethr0p05/20251224_100839`
- 训练输出：`outputs/multiseed_6class_scorethr0p05/lora_r16_2k/seed*/`
- 评估输出：`outputs/eval_6class_scorethr0p05/multiseed_lora_r16_2k/seed*/`
