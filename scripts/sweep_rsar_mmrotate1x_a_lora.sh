#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RSAR_ROOT_DEFAULT="$(cd "${REPO_ROOT}/../RSAR" && pwd)"
RSAR_ROOT="${RSAR_ROOT:-${RSAR_ROOT_DEFAULT}}"

WAIT_FOR_TMUX_SESSION="${WAIT_FOR_TMUX_SESSION:-}"
if [[ -n "${WAIT_FOR_TMUX_SESSION}" ]]; then
  echo "Waiting for tmux session to finish: ${WAIT_FOR_TMUX_SESSION}"
  while tmux has-session -t "${WAIT_FOR_TMUX_SESSION}" 2>/dev/null; do
    sleep 300
  done
fi

CFG_DIR="${RSAR_ROOT}/configs/dinov3_a_lora"

run() {
  local cfg="$1"
  local work="$2"
  local note="$3"
  "${SCRIPT_DIR}/train_eval_rsar_mmrotate1x_and_push.sh" "${cfg}" "${work}" "${note}"
}

# Control: rotation augmentation + standard LoRA
run "${CFG_DIR}/rotated-retinanet-rbox-le90_dinov3-timm-convnext-small_fpn_1x_rsar_rotaug_lora-r16_bs64_amp.py" \
  "dinov3_rotaug_lora_r16_bs64_amp" \
  "RotAug+LoRA(r16),bs64,amp"

# A-LoRA: angle-conditioned gate g(theta) on LoRA updates
run "${CFG_DIR}/rotated-retinanet-rbox-le90_dinov3-timm-convnext-small_fpn_1x_rsar_a-lora-r16_bs64_amp.py" \
  "dinov3_a_lora_r16_bs64_amp" \
  "A-LoRA(r16,gate),RotAug,bs64,amp"

