#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/baselines/rotated_retinanet_obb_r50_fpn_1x_rsar_le90.py}"
WORK_DIR="${2:-outputs/baselines/retinanet_r50_fpn_rsar_le90}"
GPUS="${3:-1}"
SEED="${4:-}"

SEED_ARGS=""
if [[ -n "${SEED}" ]]; then
  SEED_ARGS="--seed ${SEED} --deterministic"
fi

conda run -n dino_sar bash -lc "export PYTHONPATH=\"$(pwd):\$PYTHONPATH\" && mim train mmrotate \"${CONFIG}\" --work-dir \"${WORK_DIR}\" --gpus ${GPUS} ${SEED_ARGS}"
