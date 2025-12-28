#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <RSAR_test_log_path> [note...]"
  exit 2
fi

LOG_PATH="$1"
shift || true
NOTE="${*:-}"

OUT="$(
  python3 "${SCRIPT_DIR}/record_mmrotate1x_test.py" \
    --log "${LOG_PATH}" \
    ${NOTE:+--note "${NOTE}"}
)"
echo "${OUT}"

CHANGED="$(echo "${OUT}" | sed -n 's/.*changed=\([^ ]*\).*/\1/p')"
MAP="$(echo "${OUT}" | sed -n 's/.* map=\([^ ]*\).*/\1/p')"
AP50="$(echo "${OUT}" | sed -n 's/.* ap50=\([^ ]*\).*/\1/p')"
AP75="$(echo "${OUT}" | sed -n 's/.* ap75=\([^ ]*\).*/\1/p')"
LOG_REF="$(echo "${OUT}" | sed -n 's/.* log_ref=\([^ ]*\).*/\1/p')"

if [[ "${CHANGED}" != "true" ]]; then
  echo "No README update needed; skipping commit/push."
  exit 0
fi

MSG="mmrotate1x: record test mAP=${MAP} AP50=${AP50} AP75=${AP75} (${LOG_REF})"
exec "${SCRIPT_DIR}/git_commit_push.sh" "${MSG}" README.md
