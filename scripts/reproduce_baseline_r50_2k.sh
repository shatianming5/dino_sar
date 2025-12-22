#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-1}"

CONFIG="configs/baselines/rotated_retinanet_obb_r50_fpn_rsar_le90_2kiter.py"
WORK_DIR="outputs/baselines/retinanet_r50_fpn_rsar_le90_train_2kiter"
EVAL_DIR="outputs/eval/baseline_r50_2k_val"

bash scripts/train_baseline.sh "${CONFIG}" "${WORK_DIR}" "${GPUS}"
bash scripts/eval_baseline.sh "${CONFIG}" "${WORK_DIR}/iter_2000.pth" "${GPUS}" "${EVAL_DIR}"

