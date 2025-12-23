#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-1}"
SEED="${2:-0}"

CONFIG="configs/lora_target_ablation/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_stage123_10kiter.py"
WORK_DIR="outputs/lora_target_ablation/retinanet_timm_convnext_small_lora_r16_stage123_train_10kiter"
EVAL_DIR="outputs/eval/lora_r16_stage123_10k_test"

bash scripts/train_baseline.sh "${CONFIG}" "${WORK_DIR}" "${GPUS}" "${SEED}"
bash scripts/eval_baseline.sh "${CONFIG}" "${WORK_DIR}/iter_10000.pth" "${GPUS}" "${EVAL_DIR}"

