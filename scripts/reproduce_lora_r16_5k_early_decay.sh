#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-1}"
SEED="${2:-0}"
RUN_TAG="${DINO_SAR_RUN_TAG:-6class_scorethr0p05}"

CONFIG="configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_5kiter_early_decay.py"
WORK_DIR="outputs/reproduce_${RUN_TAG}/lora_r16_5k_early_decay/seed${SEED}"
EVAL_DIR="outputs/eval_${RUN_TAG}/reproduce_lora_r16_5k_early_decay/seed${SEED}"

if [[ -f "${WORK_DIR}/iter_5000.pth" ]]; then
  echo "[skip train] Found ${WORK_DIR}/iter_5000.pth"
else
  bash scripts/train_baseline.sh "${CONFIG}" "${WORK_DIR}" "${GPUS}" "${SEED}"
fi

if ls "${EVAL_DIR}"/eval_*.json >/dev/null 2>&1; then
  echo "[skip eval] Found ${EVAL_DIR}/eval_*.json"
else
  bash scripts/eval_baseline.sh "${CONFIG}" "${WORK_DIR}/iter_5000.pth" "${GPUS}" "${EVAL_DIR}"
fi
