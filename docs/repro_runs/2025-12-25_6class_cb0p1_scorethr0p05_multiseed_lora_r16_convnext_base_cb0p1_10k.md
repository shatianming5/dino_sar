# 复现记录：RSAR 6-class / LoRA r=16（ConvNeXt-Base, ClassBalanced t=0.1, 10k iters）多 seed（test）

- 日期：2025-12-25
- 代码版本：f1bdba9
- RUN_TAG：6class_cb0p1_scorethr0p05
- 命令：`DINO_SAR_RUN_TAG=6class_cb0p1_scorethr0p05 bash scripts/run_multiseed_lora_r16_convnext_base_10k_cb0p1_6class.sh 1`
- 配置：`configs/dinov3_lora/retinanet_dinov3_timm_convnext_base_fpn_rsar_le90_lora_r16_10kiter_cb0p1.py`
- seeds：`0/1/2`

## 指标（mAP@0.5，test）

| seed | mAP |
|---:|---:|
| 0 | 0.08224842697381973 |
| 1 | 0.07444778829813004 |
| 2 | 0.10463845729827881 |
| mean | 0.08711155752340953 |
| std (sample) | 0.015671842671680352 |

## 目录（本地，不入 Git）

- 训练输出：`outputs/multiseed_6class_cb0p1_scorethr0p05/lora_r16_convnext_base_cb0p1_10k/seed*/`
- 评估输出：`outputs/eval_6class_cb0p1_scorethr0p05/multiseed_lora_r16_convnext_base_cb0p1_10k/seed*/`
