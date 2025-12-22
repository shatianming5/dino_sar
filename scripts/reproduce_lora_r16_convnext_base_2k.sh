#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-1}"

CONFIG="configs/dinov3_lora/retinanet_dinov3_timm_convnext_base_fpn_rsar_le90_lora_r16_2kiter.py"
WORK_DIR="outputs/dinov3_lora/retinanet_timm_convnext_base_dinov3_lora_r16_train_2kiter"
EVAL_DIR="outputs/eval/dinov3_lora_r16_convnext_base_val"

bash scripts/train_baseline.sh "${CONFIG}" "${WORK_DIR}" "${GPUS}"
bash scripts/eval_baseline.sh "${CONFIG}" "${WORK_DIR}/iter_2000.pth" "${GPUS}" "${EVAL_DIR}"

