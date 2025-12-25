# 复现记录：RSAR 6-class / LoRA r=16 target=stages2+3（ClassBalanced t=0.1, 10k iters）多 seed（test）

- 日期：2025-12-25
- 代码版本：86c4762
- RUN_TAG：6class_cb0p1_scorethr0p05
- 命令：`DINO_SAR_RUN_TAG=6class_cb0p1_scorethr0p05 bash scripts/run_multiseed_lora_r16_stage23_10k_cb0p1_6class.sh 1`
- 配置：`configs/lora_target_ablation/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_stage23_10kiter_cb0p1.py`
- seeds：`0/1/2`

## 指标（mAP@0.5，test）

| seed | mAP |
|---:|---:|
| 0 | 0.08255305141210556 |
| 1 | 0.08485426753759384 |
| 2 | 0.07791777700185776 |
| mean | 0.08177503198385239 |
| std (sample) | 0.0035330880135008825 |

## 目录（本地，不入 Git）

- 训练输出：`outputs/multiseed_6class_cb0p1_scorethr0p05/lora_r16_stage23_cb0p1_10k/seed*/`
- 评估输出：`outputs/eval_6class_cb0p1_scorethr0p05/multiseed_lora_r16_stage23_cb0p1_10k/seed*/`
