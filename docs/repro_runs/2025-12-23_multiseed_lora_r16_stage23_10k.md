# 复现记录：LoRA r16 target=stages2+3（10k iters）多 seed（RSAR test）

- 日期：2025-12-23
- 命令：`bash scripts/run_multiseed_lora_r16_stage23_10k.sh 1`
- 配置：`configs/lora_target_ablation/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_stage23_10kiter.py`
- seeds：`0/1/2`

## 指标（mAP@0.5，test）

| seed | mAP |
|---:|---:|
| 0 | 0.5245473 |
| 1 | 0.5256730 |
| 2 | 0.5075258 |
| mean | 0.5192487 |
| std (sample) | 0.0101679 |

## 运行日志（本地，不入 Git）

- 运行总日志：`outputs/_codex_runs/logs/run_multiseed_lora_r16_stage23_10k_20251223_184458.log`
- 训练输出：`outputs/multiseed/lora_r16_stage23_10k/seed*/`
- 评估输出：`outputs/eval/multiseed_lora_r16_stage23_10k/seed*/`

