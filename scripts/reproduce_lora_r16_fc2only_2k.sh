#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-1}"

CONFIG="configs/lora_target_ablation/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_fc2only_2kiter.py"
WORK_DIR="outputs/lora_target_ablation/retinanet_timm_convnext_small_lora_r16_fc2only_train_2kiter"
EVAL_DIR="outputs/eval/lora_r16_fc2only_val"

bash scripts/train_baseline.sh "${CONFIG}" "${WORK_DIR}" "${GPUS}"
bash scripts/eval_baseline.sh "${CONFIG}" "${WORK_DIR}/iter_2000.pth" "${GPUS}" "${EVAL_DIR}"

