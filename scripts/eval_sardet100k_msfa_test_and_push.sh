#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MSFA_ROOT_DEFAULT="$(cd "${REPO_ROOT}/../SARDet_100K/MSFA" && pwd)"
MSFA_ROOT="${MSFA_ROOT:-${MSFA_ROOT_DEFAULT}}"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <config.py> <checkpoint.pth> <work_dir_name> [note...]"
  echo "Env: MSFA_ROOT=/path/to/SARDet_100K/MSFA"
  exit 2
fi

CFG="$(realpath "$1")"
CKPT="$(realpath "$2")"
WORK_NAME="$3"
shift 3 || true
NOTE="${*:-}"

WORKDIR="${MSFA_ROOT}/work_dirs/${WORK_NAME}"
TESTDIR="${WORKDIR}/test_epoch12"

export PYTHONPATH="${MSFA_ROOT}:${PYTHONPATH:-}"

(cd "${MSFA_ROOT}" && python3 tools/test.py "${CFG}" "${CKPT}" --work-dir "${TESTDIR}")

LOG_PATH="$(ls -1t "${TESTDIR}"/20*/20*.log 2>/dev/null | head -n 1 || true)"
if [[ -z "${LOG_PATH}" || ! -f "${LOG_PATH}" ]]; then
  LOG_PATH="$(find "${TESTDIR}" -name "*.log" -type f | sort | tail -n 1 || true)"
fi
if [[ -z "${LOG_PATH}" || ! -f "${LOG_PATH}" ]]; then
  echo "Error: test log not found under: ${TESTDIR}"
  exit 1
fi

exec "${SCRIPT_DIR}/record_mmdet_coco_test_and_push.sh" "${LOG_PATH}" ${NOTE:+${NOTE}}

