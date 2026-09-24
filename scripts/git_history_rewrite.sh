#!/usr/bin/env bash
# SAFE Git history rewrite for the quant root repository.
#
# This is DESTRUCTIVE and IRREVERSIBLE: it rewrites every commit hash. Run it
# only in a real shell (not a sandbox), after a full mirror backup, and with
# no remote collaborators. The first argument must be --yes to proceed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d%H%M%S)"
BACKUP="$ROOT/../workspace-bare-backup-$STAMP.git"
FRESH="$ROOT/../workspace-rewrite-fresh-$STAMP.git"

if [[ "${1:-}" != "--yes" ]]; then
  echo "Dry run: this script rewrites Git history destructively."
  echo "Run: bash scripts/git_history_rewrite.sh --yes"
  echo "It will: 1) create a mirror backup, 2) exclude runtime artifacts, 3) report size."
  exit 0
fi

command -v git-filter-repo >/dev/null 2>&1 || python3 -c 'import git_filter_repo' 2>/dev/null || {
  echo "git-filter-repo is required: python3 -m pip install git-filter-repo" >&2
  exit 2
}

cd "$ROOT"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "not a git worktree" >&2; exit 2; }

echo "[1/4] Creating mirror backup: $BACKUP"
git clone --mirror . "$BACKUP" >/dev/null 2>&1

echo "[2/4] Creating fresh clone: $FRESH"
rm -rf "$FRESH"
git clone --no-local --mirror "$BACKUP" "$FRESH" >/dev/null 2>&1

echo "[3/4] Rewriting history (excluding runtime artifacts)"
(
  cd "$FRESH"
  git filter-repo \
    --path data_cache --path data_warehouse --path generated \
    --path logs --path tmp --path tmp_tx --path media \
    --path scripts/ccrd_output --path '.openclaw/tmp' \
    --path-glob '*.parquet' --path-glob '*.sqlite*' --path-glob '*.db' \
    --path-glob '*.npy' --path-glob '*.log' --path-glob '*.pyc' \
    --invert-paths
)

echo "[4/4] Verification"
echo "original-size: $(du -sh "$BACKUP" | cut -f1)"
echo "rewritten-size: $(du -sh "$FRESH" | cut -f1)"
echo "original-head: $(git -C "$ROOT" rev-parse HEAD)"
echo "rewritten-head: $(git --git-dir="$FRESH" rev-parse HEAD)"
echo
echo "Done. Review the rewritten repo, then replace the original:"
echo "  git remote add rewritten $FRESH"
echo "  git fetch rewritten --force"
echo "  git reset --hard rewritten/master"
echo "  git gc --prune=now --aggressive"
echo "Backup is at: $BACKUP"
