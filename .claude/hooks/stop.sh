#!/usr/bin/env sh
# Stop hook: remind to update STATUS.md / the active plan / a handoff when
# engine code changed but krishiv-internal/STATUS.md did not. Advisory; exits 0.
. "$(dirname "$0")/lib.sh" 2>/dev/null || exit 0

engine=$(krishiv_engine_dir) || exit 0
internal=$(krishiv_internal_dir "$engine") || exit 0
status="$internal/STATUS.md"
[ -f "$status" ] || exit 0

# Changed (modified, staged or untracked) engine files in code directories.
changed=$(git -C "$engine" status --porcelain --untracked-files=all -- \
  crates xtask python ui tests 2>/dev/null | cut -c4- | sed 's/.* -> //')
[ -n "$changed" ] || exit 0

# STATUS.md has uncommitted edits: the session is keeping it current.
if [ -n "$(git -C "$internal" status --porcelain -- STATUS.md 2>/dev/null)" ]; then
  exit 0
fi

newest=0
for f in $changed; do
  [ -e "$engine/$f" ] || continue
  m=$(krishiv_mtime "$engine/$f")
  [ "$m" -gt "$newest" ] && newest=$m
done
status_m=$(krishiv_mtime "$status")
[ "$status_m" -lt "$newest" ] || exit 0

count=$(printf '%s\n' "$changed" | wc -l | tr -d ' ')
echo "[krishiv] $count engine file(s) changed under crates/ xtask/ python/ ui/ tests/, but $status is older than them."
echo "[krishiv] Before ending: tick the active plan in $internal/plans/, update STATUS.md, and write $internal/handoffs/YYYY-MM-DD-<topic>.md (skill: krishiv-handoff)."
exit 0
