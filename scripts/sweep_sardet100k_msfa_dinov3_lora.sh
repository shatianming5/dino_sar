#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MSFA_ROOT_DEFAULT="$(cd "${REPO_ROOT}/../SARDet_100K/MSFA" && pwd)"
MSFA_ROOT="${MSFA_ROOT:-${MSFA_ROOT_DEFAULT}}"

# Wait for RSAR sweeps to finish first (sessions already exist).
for s in rsar_a_lora_sweep rsar_rk_lora_stage23_extras; do
  if tmux has-session -t "${s}" 2>/dev/null; then
    echo "Waiting for tmux session to finish: ${s}"
    while tmux has-session -t "${s}" 2>/dev/null; do
      sleep 300
    done
  fi
done

if [[ ! -d /home/ubuntu/SARDET/Images/train ]]; then
  echo "Error: dataset not extracted yet: /home/ubuntu/SARDET/Images/train"
  exit 1
fi

CFG_DIR="${MSFA_ROOT}/local_configs/SARDet"

run() {
  local cfg="$1"
  local work="$2"
  local note="$3"
  "${SCRIPT_DIR}/train_eval_sardet100k_msfa_and_push.sh" "${cfg}" "${work}" "${note}"
}

# Baseline: DINOv3 ConvNeXt-S + standard LoRA (r=16)
run "${CFG_DIR}/dinov3_lora/retinanet_dinov3-timm-convnext-small_lora-r16_1x_sardet_bs64_amp.py" \
  "sardet_dinov3_lora_r16_bs64_amp" \
  "SARDet-100K: DINOv3-ConvNeXtS + LoRA(r16),bs64,amp"

# RK-LoRA: apply rotated-kernel LoRA on depthwise conv (stage2+3)
run "${CFG_DIR}/dinov3_rk_lora/retinanet_dinov3-timm-convnext-small_rk-lora-dwconv-r16_k4_stage23_1x_sardet_bs64_amp.py" \
  "sardet_dinov3_rk_lora_r16_k4_stage23_bs64_amp" \
  "SARDet-100K: RKLoRA(k4,stage23,r16),bs64,amp"

