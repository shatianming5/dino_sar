#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RSAR_ROOT_DEFAULT="$(cd "${REPO_ROOT}/../RSAR" && pwd)"
RSAR_ROOT="${RSAR_ROOT:-${RSAR_ROOT_DEFAULT}}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <RSAR_config.py> <checkpoint.pth> [note...]"
  echo "Env: RSAR_ROOT=/path/to/RSAR"
  exit 2
fi

CFG="$1"
CKPT="$2"
shift 2 || true
NOTE="${*:-}"

STAMP="$(date -u '+%Y%m%d_%H%M%S')"
WORKDIR="${RSAR_ROOT}/work_dirs/auto_test_${STAMP}"

export PYTHONPATH="${RSAR_ROOT}:${PYTHONPATH:-}"

python3 "${RSAR_ROOT}/tools/test.py" "${CFG}" "${CKPT}" --work-dir "${WORKDIR}"

LOG="$(find "${WORKDIR}" -maxdepth 3 -type f -name '*.log' | sort | tail -n 1)"
if [[ -z "${LOG}" ]]; then
  echo "Error: could not find test log under: ${WORKDIR}"
  exit 1
fi

exec "${SCRIPT_DIR}/record_mmrotate1x_test_and_push.sh" "${LOG}" ${NOTE:+${NOTE}}

