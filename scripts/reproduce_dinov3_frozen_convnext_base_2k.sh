#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-1}"

CONFIG="configs/dinov3_frozen/retinanet_dinov3_timm_convnext_base_fpn_rsar_le90_frozen_2kiter.py"
WORK_DIR="outputs/dinov3_frozen/retinanet_timm_convnext_base_dinov3_frozen_train_2kiter"
EVAL_DIR="outputs/eval/dinov3_frozen_convnext_base_val"

bash scripts/train_baseline.sh "${CONFIG}" "${WORK_DIR}" "${GPUS}"
bash scripts/eval_baseline.sh "${CONFIG}" "${WORK_DIR}/iter_2000.pth" "${GPUS}" "${EVAL_DIR}"

