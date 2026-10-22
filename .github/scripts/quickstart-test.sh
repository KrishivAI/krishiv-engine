#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# P2-T147: runs the README's Quickstart from a clean venv against a built
# wheel, so the documented commands cannot rot.
#
#   quickstart-test.sh DIST [PYTHON]   (DIST holds one krishiv wheel; PYTHON is
#                                       an interpreter or a `uv python` request)
#
# Every fenced block between the `quickstart:begin` and `quickstart:end`
# comments of README.md is run, in order. `bash` blocks run as written, except
# that `pip install krishiv...` installs the wheel (with the same extras)
# because nothing is published. `python` blocks run in one interpreter session
# worth of script, from a scratch directory, with `-I` so only the installed
# wheel can be imported. Other block languages are refused, so a new kind of
# block cannot be added without being tested.
set -euo pipefail
dist=$1
python=${2:-3.11}
repo=$(cd "$(dirname "$0")/../.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

shopt -s nullglob
wheels=("$dist"/krishiv-*.whl)
[ "${#wheels[@]}" -eq 1 ] || { echo "::error::expected one krishiv wheel in $dist, found ${#wheels[@]}"; exit 1; }
wheel=$(cd "$(dirname "${wheels[0]}")" && pwd)/$(basename "${wheels[0]}")

# Blocks of the section: `<n>.<lang>` files holding the block's lines.
awk -v dir="$work" '
  /<!-- quickstart:begin/ { on = 1; next }
  /<!-- quickstart:end/   { on = 0 }
  on && /^```/ {
    if (open) { close(file); open = 0; next }
    lang = substr($0, 4); n++
    file = sprintf("%s/%02d.%s", dir, n, lang); open = 1; next
  }
  on && open { print > file }
' "$repo/README.md"
blocks=("$work"/[0-9]*.*)
[ "${#blocks[@]}" -gt 0 ] || { echo "::error::README.md has no quickstart blocks"; exit 1; }

uv venv --quiet --python "$python" "$work/venv"
py="$work/venv/bin/python"
# The CI interop pins (the same versions every wheel job tests against).
export UV_CONSTRAINT="$repo/python/ci-interop-constraints.txt"
export UV_PYTHON="$py"

# `pip install krishiv[extras]` in a README block: the wheel, same extras.
pip() {
  [ "${1:-}" = install ] || { echo "::error::quickstart bash blocks may only run 'pip install'"; return 1; }
  shift
  local specs=() a
  for a in "$@"; do
    case "$a" in
      krishiv | krishiv\[*) specs+=("$wheel${a#krishiv}") ;;
      *) specs+=("$a") ;;
    esac
  done
  uv pip install --quiet --only-binary :all: "${specs[@]}"
}

: > "$work/quickstart.py"
# shellcheck disable=SC1090 # sourcing the README block is the point
for b in "${blocks[@]}"; do
  case "$b" in
    *.bash) echo "+ README block $(basename "$b")"; (cd "$work" && source "$b") ;;
    *.python) cat "$b" >> "$work/quickstart.py"; echo >> "$work/quickstart.py" ;;
    *) echo "::error::quickstart block $(basename "$b") has an untested language"; exit 1 ;;
  esac
done
grep -q 'pip install' "$work"/*.bash || { echo "::error::the quickstart has no install block"; exit 1; }
grep -q 'analyze=True' "$work/quickstart.py" || { echo "::error::the quickstart lost its EXPLAIN ANALYZE example"; exit 1; }

cd "$work"
"$py" -I quickstart.py
echo "quickstart OK against $(basename "$wheel")"
