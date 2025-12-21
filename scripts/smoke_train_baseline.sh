#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/baselines/rotated_retinanet_obb_r50_fpn_rsar_le90_smoke.py}"
WORK_DIR="${2:-outputs/baselines/_smoke_retinanet}"
GPUS="${3:-1}"

conda run -n dino_sar bash -lc "export PYTHONPATH=\"$(pwd):\$PYTHONPATH\" && mim train mmrotate \"${CONFIG}\" --work-dir \"${WORK_DIR}\" --gpus ${GPUS}"

