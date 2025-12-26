#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-1}"
SEED="${2:-0}"
RUN_TAG="${DINO_SAR_RUN_TAG:-6class_scorethr0p05}"

CONFIG="configs/dinov3_full/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_full_2kiter_lr0p00025.py"
WORK_DIR="outputs/reproduce_${RUN_TAG}/dinov3_full_2k_lr0p00025/seed${SEED}"
EVAL_DIR="outputs/eval_${RUN_TAG}/reproduce_dinov3_full_2k_lr0p00025/seed${SEED}"

if [[ -f "${WORK_DIR}/iter_2000.pth" ]]; then
  echo "[skip train] Found ${WORK_DIR}/iter_2000.pth"
else
  bash scripts/train_baseline.sh "${CONFIG}" "${WORK_DIR}" "${GPUS}" "${SEED}"
fi

if ls "${EVAL_DIR}"/eval_*.json >/dev/null 2>&1; then
  echo "[skip eval] Found ${EVAL_DIR}/eval_*.json"
else
  bash scripts/eval_baseline.sh "${CONFIG}" "${WORK_DIR}/iter_2000.pth" "${GPUS}" "${EVAL_DIR}"
fi
