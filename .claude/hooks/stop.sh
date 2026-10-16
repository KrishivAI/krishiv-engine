#!/usr/bin/env sh
# Stop hook: remind to update STATUS.md, the active plan and a handoff when
# engine code (committed on this branch or uncommitted) is newer than
# krishiv-internal/STATUS.md. Advisory: prints {"systemMessage": ...} JSON
# when warning, nothing otherwise; always exits 0 (never 2).
. "$(dirname "$0")/lib.sh" 2>/dev/null || exit 0

engine=$(krishiv_engine_dir) || exit 0
internal=$(krishiv_internal_dir "$engine") || exit 0
status="$internal/STATUS.md"
[ -f "$status" ] || exit 0

# STATUS.md has uncommitted edits: the session is keeping it current.
[ -z "$(git -C "$internal" status --porcelain -- STATUS.md 2>/dev/null)" ] || exit 0

# Reference time: the later of STATUS.md's last commit and its mtime.
ref=$(git -C "$internal" log -1 --format=%ct -- STATUS.md 2>/dev/null)
ref=${ref:-0}
m=$(krishiv_mtime "$status")
[ "$m" -gt "$ref" ] && ref=$m

dirs="crates xtask python ui tests"

# Newest commit on the current branch touching code directories.
# shellcheck disable=SC2086
newest=$(git -C "$engine" log -1 --format=%ct -- $dirs 2>/dev/null)
newest=${newest:-0}
why="commits"

# Uncommitted / untracked code changes (mtime of each changed file).
# shellcheck disable=SC2086
changed=$(git -C "$engine" status --porcelain --untracked-files=all -- $dirs 2>/dev/null \
  | cut -c4- | sed 's/.* -> //')
for f in $changed; do
  [ -e "$engine/$f" ] || continue
  t=$(krishiv_mtime "$engine/$f")
  if [ "$t" -gt "$newest" ]; then newest=$t; why="uncommitted changes"; fi
done

[ "$newest" -gt "$ref" ] || exit 0

# Active plan: the link after "active:" in STATUS.md, else the plans/ dir.
plan=$(sed -n 's/.*active:[^(]*(\(plans\/[^)]*\)).*/\1/p' "$status" | head -n 1)
plan="$internal/${plan:-plans/}"

msg="[krishiv] Engine code ($why under crates/ xtask/ python/ ui/ tests/) is newer than $status. Before ending: tick the active plan ($plan), update STATUS.md, and write $internal/handoffs/YYYY-MM-DD-<topic>.md (skill krishiv-handoff)."
esc=$(printf '%s' "$msg" | sed 's/\\/\\\\/g; s/"/\\"/g' | tr -d '\n\r\t')
printf '{"systemMessage": "%s"}\n' "$esc"
exit 0
