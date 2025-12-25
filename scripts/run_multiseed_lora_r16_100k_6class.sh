#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-1}"

CONFIG="configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_100kiter.py"
RUN_TAG="${DINO_SAR_RUN_TAG:-6class}"
BASE_WORK_DIR="outputs/multiseed_${RUN_TAG}/lora_r16_100k"
BASE_EVAL_DIR="outputs/eval_${RUN_TAG}/multiseed_lora_r16_100k"

SEEDS=("0" "1" "2")

for SEED in "${SEEDS[@]}"; do
  WORK_DIR="${BASE_WORK_DIR}/seed${SEED}"
  EVAL_DIR="${BASE_EVAL_DIR}/seed${SEED}"
  echo "=== seed=${SEED} ==="

  if [[ -f "${WORK_DIR}/iter_100000.pth" ]]; then
    echo "[skip train] Found ${WORK_DIR}/iter_100000.pth"
  else
    bash scripts/train_baseline.sh "${CONFIG}" "${WORK_DIR}" "${GPUS}" "${SEED}"
  fi

  if ls "${EVAL_DIR}"/eval_*.json >/dev/null 2>&1; then
    echo "[skip eval] Found ${EVAL_DIR}/eval_*.json"
  else
    bash scripts/eval_baseline.sh "${CONFIG}" "${WORK_DIR}/iter_100000.pth" "${GPUS}" "${EVAL_DIR}"
  fi
done

