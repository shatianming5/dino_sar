#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-1}"

CONFIG="configs/dinov3_full/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_full_2kiter_lr0p00025.py"
WORK_DIR="outputs/dinov3_full/retinanet_timm_convnext_small_dinov3_full_train_2kiter_lr0p00025"
EVAL_DIR="outputs/eval/dinov3_full_lr0p00025_val"

bash scripts/train_baseline.sh "${CONFIG}" "${WORK_DIR}" "${GPUS}"
bash scripts/eval_baseline.sh "${CONFIG}" "${WORK_DIR}/iter_2000.pth" "${GPUS}" "${EVAL_DIR}"

