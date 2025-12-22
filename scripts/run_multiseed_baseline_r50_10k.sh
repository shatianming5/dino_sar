#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-1}"

CONFIG="configs/baselines/rotated_retinanet_obb_r50_fpn_rsar_le90_10kiter.py"
BASE_WORK_DIR="outputs/multiseed/baseline_r50_10k"
BASE_EVAL_DIR="outputs/eval/multiseed_baseline_r50_10k"

SEEDS=("0" "1" "2")

for SEED in "${SEEDS[@]}"; do
  WORK_DIR="${BASE_WORK_DIR}/seed${SEED}"
  EVAL_DIR="${BASE_EVAL_DIR}/seed${SEED}"
  echo "=== seed=${SEED} ==="

  if [[ -f "${WORK_DIR}/iter_10000.pth" ]]; then
    echo "[skip train] Found ${WORK_DIR}/iter_10000.pth"
  else
    bash scripts/train_baseline.sh "${CONFIG}" "${WORK_DIR}" "${GPUS}" "${SEED}"
  fi

  bash scripts/eval_baseline.sh "${CONFIG}" "${WORK_DIR}/iter_10000.pth" "${GPUS}" "${EVAL_DIR}"
done

