# 复现记录：RSAR 6-class / LoRA r=16（ConvNeXt-Base, 10k iters）多 seed（test）

- 日期：2025-12-24
- 代码版本：f2f22f3
- RUN_TAG：6class_scorethr0p05
- 命令：`DINO_SAR_RUN_TAG=6class_scorethr0p05 bash scripts/run_multiseed_lora_r16_convnext_base_10k_6class.sh 1`
- 配置：`configs/dinov3_lora/retinanet_dinov3_timm_convnext_base_fpn_rsar_le90_lora_r16_10kiter.py`
- seeds：`0/1/2`

## 指标（mAP@0.5，test）

| seed | mAP |
|---:|---:|
| 0 | 0.08289607614278793 |
| 1 | 0.09245046228170395 |
| 2 | 0.09233380109071732 |
| mean | 0.08922677983840306 |
| std (sample) | 0.005482860514332544 |

## 目录（本地，不入 Git）

- 日志：`outputs/_rerun_6class/logs/6class_scorethr0p05/20251224_181914_multiseed_lora_r16_convnext_base_10k.log`
- 训练输出：`outputs/multiseed_6class_scorethr0p05/lora_r16_convnext_base_10k/seed*/`
- 评估输出：`outputs/eval_6class_scorethr0p05/multiseed_lora_r16_convnext_base_10k/seed*/`

