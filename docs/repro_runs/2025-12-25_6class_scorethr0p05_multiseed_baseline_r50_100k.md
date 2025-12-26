# 复现记录：RSAR 6-class / Baseline R50-FPN（100k iters）多 seed（test）

- 日期：2025-12-25
- 代码版本：1e166bc
- RUN_TAG：6class_scorethr0p05
- 命令：`DINO_SAR_RUN_TAG=6class_scorethr0p05 bash scripts/run_multiseed_baseline_r50_100k_6class.sh 1`
- 配置：`configs/baselines/rotated_retinanet_obb_r50_fpn_rsar_le90_100kiter.py`
- seeds：`0/1/2`

## 指标（mAP@0.5，test）

| seed | mAP |
|---:|---:|
| 0 | 0.23659174144268036 |
| 1 | 0.2275192141532898 |
| 2 | 0.22468797862529755 |
| mean | 0.22959964474042258 |
| std (sample) | 0.00621860401328422 |

## 目录（本地，不入 Git）

- 训练输出：`outputs/multiseed_6class_scorethr0p05/baseline_r50_100k/seed*/`
- 评估输出：`outputs/eval_6class_scorethr0p05/multiseed_baseline_r50_100k/seed*/`

