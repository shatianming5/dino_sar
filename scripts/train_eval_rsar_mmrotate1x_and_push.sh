#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RSAR_ROOT_DEFAULT="$(cd "${REPO_ROOT}/../RSAR" && pwd)"
RSAR_ROOT="${RSAR_ROOT:-${RSAR_ROOT_DEFAULT}}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <RSAR_config.py> <work_dir_name> [note...]"
  echo "Env: RSAR_ROOT=/path/to/RSAR"
  exit 2
fi

CFG="$(realpath "$1")"
WORK_NAME="$2"
shift 2 || true
NOTE="${*:-}"

WORKDIR="${RSAR_ROOT}/work_dirs/${WORK_NAME}"

export PYTHONPATH="${RSAR_ROOT}:${PYTHONPATH:-}"

(cd "${RSAR_ROOT}" && python3 tools/train.py "${CFG}" --work-dir "${WORKDIR}")

CKPT="${WORKDIR}/epoch_12.pth"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="$(ls -1t "${WORKDIR}"/epoch_*.pth 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "Error: checkpoint not found under: ${WORKDIR}"
  exit 1
fi

exec "${SCRIPT_DIR}/eval_rsar_mmrotate1x_test_and_push.sh" "${CFG}" "${CKPT}" ${NOTE:+${NOTE}}

