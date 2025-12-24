# 复现记录：RSAR 6-class / LoRA r16 target=stages2+3（10k iters）多 seed（test）

- 日期：2025-12-24
- 代码版本：e90a8b8
- RUN_TAG：6class_scorethr0p05
- 命令：`DINO_SAR_RUN_TAG=6class_scorethr0p05 bash scripts/run_multiseed_lora_r16_stage23_10k_6class.sh 1`
- 配置：`configs/lora_target_ablation/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_stage23_10kiter.py`
- seeds：`0/1/2`

## 指标（mAP@0.5，test）

| seed | mAP |
|---:|---:|
| 0 | 0.08279760926961899 |
| 1 | 0.08392838388681412 |
| 2 | 0.08713585883378983 |
| mean | 0.08462061733007431 |
| std (sample) | 0.0022504427731522107 |

## 目录（本地，不入 Git）

- 日志：`outputs/_rerun_6class/logs/6class_scorethr0p05/20251224_100839`
- 训练输出：`outputs/multiseed_6class_scorethr0p05/lora_r16_stage23_10k/seed*/`
- 评估输出：`outputs/eval_6class_scorethr0p05/multiseed_lora_r16_stage23_10k/seed*/`
