#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MSFA_ROOT_DEFAULT="$(cd "${REPO_ROOT}/../SARDet_100K/MSFA" && pwd)"
MSFA_ROOT="${MSFA_ROOT:-${MSFA_ROOT_DEFAULT}}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config.py> <work_dir_name> [note...]"
  echo "Env: MSFA_ROOT=/path/to/SARDet_100K/MSFA"
  exit 2
fi

CFG="$(realpath "$1")"
WORK_NAME="$2"
shift 2 || true
NOTE="${*:-}"

WORKDIR="${MSFA_ROOT}/work_dirs/${WORK_NAME}"

export PYTHONPATH="${MSFA_ROOT}:${PYTHONPATH:-}"

(cd "${MSFA_ROOT}" && python3 tools/train.py "${CFG}" --work-dir "${WORKDIR}")

CKPT=""
if [[ -f "${WORKDIR}/last_checkpoint" ]]; then
  CKPT_NAME="$(cat "${WORKDIR}/last_checkpoint" | tr -d '\n\r')"
  if [[ -n "${CKPT_NAME}" && -f "${WORKDIR}/${CKPT_NAME}" ]]; then
    CKPT="${WORKDIR}/${CKPT_NAME}"
  fi
fi
if [[ -z "${CKPT}" ]]; then
  CKPT="$(ls -1t "${WORKDIR}"/epoch_*.pth 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "Error: checkpoint not found under: ${WORKDIR}"
  exit 1
fi

exec "${SCRIPT_DIR}/eval_sardet100k_msfa_test_and_push.sh" "${CFG}" "${CKPT}" "${WORK_NAME}" ${NOTE:+${NOTE}}

