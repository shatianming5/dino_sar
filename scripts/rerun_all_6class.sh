#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-1}"
RUN_TAG="${2:-6class_scorethr0p05}"

RUN_DATE="$(date +%Y-%m-%d)"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
GIT_SHA="$(git rev-parse --short HEAD)"

export DINO_SAR_RUN_TAG="${RUN_TAG}"

LOG_DIR="outputs/_rerun_6class/logs/${RUN_TAG}/${RUN_TS}"
mkdir -p "${LOG_DIR}"

ensure_clean_git() {
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "[ERROR] git working tree not clean. Commit/stash first." >&2
    git status --porcelain >&2
    exit 1
  fi
}

read_map_from_eval_dir() {
  local eval_dir="$1"
  python - <<PY
import glob, json, os, sys
eval_dir = ${eval_dir@Q}
files = sorted(glob.glob(os.path.join(eval_dir, "eval_*.json")))
if not files:
    print("N/A")
    sys.exit(0)
with open(files[-1], "r", encoding="utf-8") as f:
    d = json.load(f)
print(d.get("metric", {}).get("mAP", "N/A"))
PY
}

write_multiseed_report() {
  local out_md="$1"
  local title="$2"
  local cmd="$3"
  local config="$4"
  local base_work_dir="$5"
  local base_eval_dir="$6"

  local m0 m1 m2
  m0="$(read_map_from_eval_dir "${base_eval_dir}/seed0")"
  m1="$(read_map_from_eval_dir "${base_eval_dir}/seed1")"
  m2="$(read_map_from_eval_dir "${base_eval_dir}/seed2")"

  python - <<PY > "${LOG_DIR}/_stats.tmp"
import math
vals = []
for s in [${m0@Q}, ${m1@Q}, ${m2@Q}]:
    try:
        vals.append(float(s))
    except Exception:
        pass
mean = sum(vals) / len(vals) if vals else float("nan")
std = math.sqrt(sum((x-mean)**2 for x in vals)/(len(vals)-1)) if len(vals) > 1 else float("nan")
print("m0", ${m0@Q})
print("m1", ${m1@Q})
print("m2", ${m2@Q})
print("mean", mean)
print("std", std)
PY

  local mean std
  mean="$(rg -n '^mean ' "${LOG_DIR}/_stats.tmp" | awk '{print $2}')"
  std="$(rg -n '^std ' "${LOG_DIR}/_stats.tmp" | awk '{print $2}')"

  mkdir -p "$(dirname "${out_md}")"
  cat > "${out_md}" <<EOF
# ${title}

- 日期：${RUN_DATE}
- 代码版本：${GIT_SHA}
- RUN_TAG：${RUN_TAG}
- 命令：\`${cmd}\`
- 配置：\`${config}\`
- seeds：\`0/1/2\`

## 指标（mAP@0.5，test）

| seed | mAP |
|---:|---:|
| 0 | ${m0} |
| 1 | ${m1} |
| 2 | ${m2} |
| mean | ${mean} |
| std (sample) | ${std} |

## 目录（本地，不入 Git）

- 日志：\`${LOG_DIR}\`
- 训练输出：\`${base_work_dir}/seed*/\`
- 评估输出：\`${base_eval_dir}/seed*/\`
EOF
}

commit_and_push() {
  local msg="$1"
  shift
  git add "$@"
  git commit -m "${msg}"
  GIT_TERMINAL_PROMPT=0 git push origin main
}

ensure_clean_git

echo "[1/2] multiseed LoRA r16 2k (6-class) ..."
bash scripts/run_multiseed_lora_r16_2k_6class.sh "${GPUS}" > "${LOG_DIR}/run_multiseed_lora_r16_2k_6class.log" 2>&1

REPORT1="docs/repro_runs/${RUN_DATE}_${RUN_TAG}_multiseed_lora_r16_2k.md"
write_multiseed_report \
  "${REPORT1}" \
  "复现记录：RSAR 6-class / LoRA r16（2k iters）多 seed（test）" \
  "DINO_SAR_RUN_TAG=${RUN_TAG} bash scripts/run_multiseed_lora_r16_2k_6class.sh ${GPUS}" \
  "configs/dinov3_lora/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_2kiter.py" \
  "outputs/multiseed_${RUN_TAG}/lora_r16_2k" \
  "outputs/eval_${RUN_TAG}/multiseed_lora_r16_2k"

commit_and_push "repro: rsar 6-class multiseed lora r16 2k (${RUN_DATE}, ${RUN_TAG})" "${REPORT1}"

ensure_clean_git

echo "[2/2] multiseed LoRA r16 stage23 10k (6-class) ..."
bash scripts/run_multiseed_lora_r16_stage23_10k_6class.sh "${GPUS}" > "${LOG_DIR}/run_multiseed_lora_r16_stage23_10k_6class.log" 2>&1

REPORT2="docs/repro_runs/${RUN_DATE}_${RUN_TAG}_multiseed_lora_r16_stage23_10k.md"
write_multiseed_report \
  "${REPORT2}" \
  "复现记录：RSAR 6-class / LoRA r16 target=stages2+3（10k iters）多 seed（test）" \
  "DINO_SAR_RUN_TAG=${RUN_TAG} bash scripts/run_multiseed_lora_r16_stage23_10k_6class.sh ${GPUS}" \
  "configs/lora_target_ablation/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_lora_r16_stage23_10kiter.py" \
  "outputs/multiseed_${RUN_TAG}/lora_r16_stage23_10k" \
  "outputs/eval_${RUN_TAG}/multiseed_lora_r16_stage23_10k"

commit_and_push "repro: rsar 6-class multiseed lora r16 stage23 10k (${RUN_DATE}, ${RUN_TAG})" "${REPORT2}"

echo "[DONE] 6-class rerun completed."
