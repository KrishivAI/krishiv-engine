#!/usr/bin/env sh
# Stop hook: remind to update STATUS.md, the active plan and a handoff when
# engine code (committed on this branch or uncommitted) is more than two hours
# newer than the latest of krishiv-internal/STATUS.md and the newest file in
# krishiv-internal/handoffs/ (each by git commit time or mtime). Advisory: prints {"systemMessage": ...} JSON
# when warning, nothing otherwise; always exits 0 (never 2).
. "$(dirname "$0")/lib.sh" 2>/dev/null || exit 0

engine=$(krishiv_engine_dir) || exit 0
internal=$(krishiv_internal_dir "$engine") || exit 0
status="$internal/STATUS.md"
[ -f "$status" ] || exit 0

# STATUS.md has uncommitted edits: the session is keeping it current.
[ -z "$(git -C "$internal" status --porcelain -- STATUS.md 2>/dev/null)" ] || exit 0

# Reference time: the latest of STATUS.md's commit time and mtime, and the
# commit time and mtime of the newest file in handoffs/.
ref=$(git -C "$internal" log -1 --format=%ct -- STATUS.md 2>/dev/null)
ref=${ref:-0}
m=$(krishiv_mtime "$status")
[ "$m" -gt "$ref" ] && ref=$m
h=$(git -C "$internal" log -1 --format=%ct -- handoffs 2>/dev/null)
h=${h:-0}
[ "$h" -gt "$ref" ] && ref=$h
for f in "$internal"/handoffs/*; do
  [ -f "$f" ] || continue
  m=$(krishiv_mtime "$f")
  [ "$m" -gt "$ref" ] && ref=$m
done

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

# Warn only when code is more than 2 hours newer than every reference.
case "$newest$ref" in *[!0-9]*) exit 0 ;; esac
[ "$newest" -gt $((ref + 7200)) ] || exit 0

# Active plan: the link after "active:" in STATUS.md, else the plans/ dir.
plan=$(sed -n 's/.*active:[^(]*(\(plans\/[^)]*\)).*/\1/p' "$status" | head -n 1)
plan="$internal/${plan:-plans/}"

msg="[krishiv] Engine code ($why under crates/ xtask/ python/ ui/ tests/) is more than 2 hours newer than $status and the newest handoff. Before ending: tick the active plan ($plan), update STATUS.md, and write $internal/handoffs/YYYY-MM-DD-<topic>.md (skill krishiv-handoff)."
esc=$(printf '%s' "$msg" | sed 's/\\/\\\\/g; s/"/\\"/g' | tr -d '\n\r\t')
printf '{"systemMessage": "%s"}\n' "$esc"
exit 0
