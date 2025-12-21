#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/baselines/rotated_retinanet_obb_r50_fpn_1x_rsar_le90.py}"
CKPT="${2:-outputs/baselines/retinanet_r50_fpn_rsar_le90/latest.pth}"
GPUS="${3:-1}"
WORK_DIR="${4:-outputs/eval/baseline}"

conda run -n dino_sar bash -lc "export PYTHONPATH=\"$(pwd):\$PYTHONPATH\" && mim test mmrotate \"${CONFIG}\" --checkpoint \"${CKPT}\" --gpus ${GPUS} --eval mAP --work-dir \"${WORK_DIR}\""
