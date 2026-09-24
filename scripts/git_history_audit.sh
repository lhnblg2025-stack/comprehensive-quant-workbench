#!/usr/bin/env bash
# Read-only Git history audit and migration helper.
# It never rewrites history automatically. Use the printed backup/rewrite command
# only after reviewing the generated path list and taking an external backup.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

printf 'root=%s\n' "$ROOT"
printf 'objects=%s\n' "$(git count-objects -vH | tr '\n' ';')"
printf '\nLargest tracked paths in current tree:\n'
git ls-files -z | xargs -0 -r du -b 2>/dev/null | sort -nr | head -30 || true
printf '\nTracked generated/data candidates:\n'
git ls-files 'generated/**' 'data_warehouse/**' 'tmp/**' 'logs/**' '*.sqlite*' '*.db' '*.parquet' '*.npy' | sed -n '1,160p'
printf '\nRecommended history rewrite (review first):\n'
printf 'git filter-repo --force --path generated --path data_warehouse --path tmp --path logs --path-glob "*.sqlite*" --path-glob "*.db" --path-glob "*.parquet" --path-glob "*.npy" --invert-paths\n'
printf '\nNote: create a full bare backup before rewriting; every clone must re-fetch the rewritten refs.\n'
