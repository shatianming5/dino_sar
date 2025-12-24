# 复现记录：LoRA r16（2k iters）多 seed（RSAR test）

> 注意：该记录为 **ship-only（仅 ship 类）** 口径下的历史结果；当前仓库默认使用 RSAR **6 类**，mAP 不可直接对比。

- 日期：2025-12-23
- 命令：`bash scripts/run_multiseed_lora_r16_2k.sh 1`
- 配置：`configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_2kiter.py`
- seeds：`0/1/2`

## 指标（mAP@0.5，test）

| seed | mAP |
|---:|---:|
| 0 | 0.2004206 |
| 1 | 0.1866457 |
| 2 | 0.2739225 |
| mean | 0.2203296 |
| std (sample) | 0.0469211 |

## 运行日志（本地，不入 Git）

- 运行总日志：`outputs/_codex_runs/logs/run_multiseed_lora_r16_2k_20251223_165012.log`
- 训练输出：`outputs/multiseed/lora_r16_2k/seed*/`
- 评估输出：`outputs/eval/multiseed_lora_r16_2k/seed*/`
