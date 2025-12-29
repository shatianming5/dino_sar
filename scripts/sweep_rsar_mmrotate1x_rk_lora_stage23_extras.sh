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

CFG_DIR="${RSAR_ROOT}/configs/dinov3_rk_lora"

run() {
  local cfg="$1"
  local work="$2"
  local note="$3"
  "${SCRIPT_DIR}/train_eval_rsar_mmrotate1x_and_push.sh" "${cfg}" "${work}" "${note}"
}

# Missing placement ablation: backbone(stage2+3)+neck+head
run "${CFG_DIR}/rotated-retinanet-rbox-le90_dinov3-timm-convnext-small_fpn_1x_rsar_rk-lora-dwconv-r16_k4_stage23_bs64_amp_lora-neck-head.py" \
  "dinov3_rk_lora_dwconv_r16_k4_stage23_bs64_amp_lora_neck_head" \
  "RKLoRA(k4,stage23)+LoRA(neck+head),bs64,amp"

