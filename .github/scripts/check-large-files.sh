#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Rejects newly added or renamed files larger than MAX_BYTES (default 1 MiB) unless the
# path matches a line of the allow-list (default .github/large-files-allow.txt;
# exact paths or shell globs, `#` comments). Guards against committing spill
# files, build output or datasets by accident.
#   check-large-files.sh --staged          files staged in the index (pre-commit)
#   check-large-files.sh BASE HEAD         files added between two commits (CI)
set -euo pipefail

max=${MAX_BYTES:-1048576}
allow=${LARGE_FILES_ALLOW:-$(git rev-parse --show-toplevel)/.github/large-files-allow.txt}

if [ "${1:-}" = "--staged" ]; then
  rev=""   # `git cat-file -s :path` reads the index
  list() { git diff --cached --name-only -z --diff-filter=AR; }
elif [ $# -eq 2 ]; then
  rev=$2
  base=$1
  list() { git diff --name-only -z --diff-filter=AR "$base...$rev"; }
else
  echo "usage: $0 --staged | BASE HEAD" >&2
  exit 2
fi

patterns=()
if [ -f "$allow" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%%#*}
    line=$(printf '%s' "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    [ -n "$line" ] && patterns+=("$line")
  done < "$allow"
fi

added=$(mktemp)
trap 'rm -f "$added"' EXIT
list > "$added"   # a failing diff aborts here (set -e) instead of passing silently

bad=0
while IFS= read -r -d '' path; do
  size=$(git cat-file -s "${rev}:${path}" 2>/dev/null) || continue
  [ "$size" -gt "$max" ] || continue
  allowed=0
  for pat in ${patterns[@]+"${patterns[@]}"}; do
    # shellcheck disable=SC2254
    case "$path" in $pat) allowed=1; break ;; esac
  done
  if [ "$allowed" -eq 0 ]; then
    echo "large-files: $path is $size bytes (limit $max); add it to .github/large-files-allow.txt if it is meant to be committed" >&2
    bad=1
  fi
done < "$added"
exit "$bad"
