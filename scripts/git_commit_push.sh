#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MSG="${1:-auto: update $(date -u '+%F %T UTC')}"
shift || true

FILES=("$@")
if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "Usage: $0 \"commit message\" <file1> [file2 ...]"
  exit 2
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: not inside a git repo: ${REPO_ROOT}"
  exit 2
fi

if ! git config user.name >/dev/null; then
  git config user.name "shatianming5"
fi
if ! git config user.email >/dev/null; then
  git config user.email "shatianming5@users.noreply.github.com"
fi

# Stage only the requested files.
git add -- "${FILES[@]}"

if git diff --cached --quiet; then
  echo "No changes to commit for: ${FILES[*]}"
  exit 0
fi

git commit -m "${MSG}"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git push origin "${BRANCH}"
