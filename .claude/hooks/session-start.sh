#!/usr/bin/env sh
# SessionStart hook: print krishiv-internal STATUS.md and the newest handoff.
# Advisory; always exits 0.
. "$(dirname "$0")/lib.sh" 2>/dev/null || exit 0

engine=$(krishiv_engine_dir) || exit 0
internal=$(krishiv_internal_dir "$engine") || {
  echo "[krishiv] krishiv-internal not found (set KRISHIV_INTERNAL or clone it next to krishiv-engine)."
  exit 0
}

echo "=== Krishiv session start: read $engine/CLAUDE.md, then the files below (skill: krishiv-handoff) ==="
status="$internal/STATUS.md"
if [ -f "$status" ]; then
  echo "--- $status ---"
  head -n 120 "$status"
  n=$(wc -l < "$status")
  [ "$n" -gt 120 ] && echo "... (truncated; $n lines total)"
else
  echo "[krishiv] $status missing."
fi

# Handoff files are named YYYY-MM-DD-<topic>.md, so a name sort is chronological.
latest=$(ls -1 "$internal"/handoffs/*.md 2>/dev/null | LC_ALL=C sort | tail -n 1)
if [ -n "$latest" ]; then
  echo
  echo "--- latest handoff: $latest (first 40 lines) ---"
  head -n 40 "$latest"
else
  echo "[krishiv] no handoffs in $internal/handoffs/."
fi
exit 0
