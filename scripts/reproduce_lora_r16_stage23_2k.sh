#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-1}"
SEED="${2:-0}"
RUN_TAG="${DINO_SAR_RUN_TAG:-6class_scorethr0p05}"

CONFIG="configs/lora_target_ablation/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_stage23_2kiter.py"
WORK_DIR="outputs/reproduce_${RUN_TAG}/lora_r16_stage23_2k/seed${SEED}"
EVAL_DIR="outputs/eval_${RUN_TAG}/reproduce_lora_r16_stage23_2k/seed${SEED}"

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
