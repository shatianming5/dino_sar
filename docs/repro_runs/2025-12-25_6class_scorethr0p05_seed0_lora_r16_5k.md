# 复现记录：RSAR 6-class / LoRA r=16（5k iters）seed0（test）

- 日期：2025-12-25
- 代码版本：ad2734b
- RUN_TAG：6class_scorethr0p05
- 命令：`DINO_SAR_RUN_TAG=6class_scorethr0p05 bash scripts/reproduce_lora_r16_5k.sh 1 0`
- 配置：`configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_5kiter.py`
- seed：`0`

## 指标（mAP@0.5，test）

- mAP：`0.06966463476419449`

## 目录（本地，不入 Git）

- 训练输出：`outputs/reproduce_6class_scorethr0p05/lora_r16_5k/seed0/`
- 评估输出：`outputs/eval_6class_scorethr0p05/reproduce_lora_r16_5k/seed0/`

