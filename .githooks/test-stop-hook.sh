#!/usr/bin/env sh
# SPDX-License-Identifier: Apache-2.0
# Shell test for .claude/hooks/stop.sh: warns only when engine code is more than
# 2 hours newer than STATUS.md and the newest handoff; always exits 0.
# Run: sh .githooks/test-stop-hook.sh
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export GIT_AUTHOR_NAME=T GIT_AUTHOR_EMAIL=t@example.com GIT_COMMITTER_NAME=T GIT_COMMITTER_EMAIL=t@example.com
fail() { echo "FAIL: $*" >&2; exit 1; }

mkdir -p "$tmp/hooks" "$tmp/engine/crates" "$tmp/internal/handoffs"
cp "$root/.claude/hooks/stop.sh" "$root/.claude/hooks/lib.sh" "$tmp/hooks/"
git init -q -b main "$tmp/engine"
git init -q -b main "$tmp/internal"
printf 'active: [P3](plans/P3-implementation-plan.md)\n' > "$tmp/internal/STATUS.md"
echo h > "$tmp/internal/handoffs/old.md"
echo c > "$tmp/engine/crates/a.rs"

# Commit with a given epoch (hours ago) for both dates.
commit() { # repo hours_ago
  t=$(( $(date +%s) - $2 * 3600 ))
  git -C "$1" add -A
  GIT_AUTHOR_DATE="$t +0000" GIT_COMMITTER_DATE="$t +0000" git -C "$1" -c commit.gpgsign=false commit -q -m x
}
touch_ago() { touch -d "@$(( $(date +%s) - $2 * 3600 ))" "$1"; }

run() { CLAUDE_PROJECT_DIR="$tmp/engine" KRISHIV_INTERNAL="$tmp/internal" sh "$tmp/hooks/stop.sh"; }

# STATUS and handoff 10 h old, engine code 1 h old: more than 2 h newer -> warn.
touch_ago "$tmp/internal/STATUS.md" 10; touch_ago "$tmp/internal/handoffs/old.md" 10
commit "$tmp/internal" 10
commit "$tmp/engine" 1
out=$(run) || fail "hook exited non-zero"
case "$out" in *systemMessage*"P3-implementation-plan.md"*) echo "ok: warns, links the plan" ;; *) fail "no warning: $out" ;; esac

# A handoff 30 min ago: engine code is no longer 2 h newer -> silent.
echo n > "$tmp/internal/handoffs/new.md"; touch_ago "$tmp/internal/handoffs/new.md" 0
out=$(run) || fail "hook exited non-zero"
[ -z "$out" ] || fail "warned despite a fresh handoff: $out"
echo "ok: a fresh handoff silences it"

# Engine code only 1 h newer than a 2 h-old handoff: inside the window -> silent.
rm "$tmp/internal/handoffs/new.md"
touch_ago "$tmp/internal/STATUS.md" 2; touch_ago "$tmp/internal/handoffs/old.md" 2
git -C "$tmp/internal" -c commit.gpgsign=false commit -q --amend --no-edit --date="@$(( $(date +%s) - 7200 ))" 2>/dev/null || true
out=$(run) || fail "hook exited non-zero"
[ -z "$out" ] || fail "warned inside the 2 h window: $out"
echo "ok: within 2 hours stays silent"
