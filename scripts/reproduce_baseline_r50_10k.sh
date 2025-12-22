#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-1}"

CONFIG="configs/baselines/rotated_retinanet_obb_r50_fpn_rsar_le90_10kiter.py"
WORK_DIR="outputs/baselines/retinanet_r50_fpn_rsar_le90_train_10kiter"
EVAL_DIR="outputs/eval/retinanet_r50_fpn_rsar_le90_train_10kiter_test"

# Deterministic single-seed run for baseline reproducibility.
SEED="${2:-0}"

bash scripts/train_baseline.sh "${CONFIG}" "${WORK_DIR}" "${GPUS}" "${SEED}"
bash scripts/eval_baseline.sh "${CONFIG}" "${WORK_DIR}/iter_10000.pth" "${GPUS}" "${EVAL_DIR}"

