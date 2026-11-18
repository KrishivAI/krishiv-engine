#!/usr/bin/env sh
# SPDX-License-Identifier: Apache-2.0
# Shell test for .claude/hooks/stop.sh: warns only when engine code is more than
# 2 hours newer than STATUS.md and the newest handoff; always exits 0.
# Run: sh .githooks/test-stop-hook.sh   (also run by `cargo nextest` via xtask).
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

# One clock reading, so every offset below is exact to the second.
now=$(date +%s)
H=3600
# Commit with a given age in seconds for both dates.
commit() { # repo seconds_ago
  t=$(( now - $2 ))
  git -C "$1" add -A
  GIT_AUTHOR_DATE="$t +0000" GIT_COMMITTER_DATE="$t +0000" git -C "$1" -c commit.gpgsign=false commit -q -m x
}
touch_ago() { touch -d "@$(( now - $2 ))" "$1"; } # file seconds_ago

run() { CLAUDE_PROJECT_DIR="$tmp/engine" KRISHIV_INTERNAL="$tmp/internal" sh "$tmp/hooks/stop.sh"; }

# STATUS and handoff 10 h old, engine code 1 h old: more than 2 h newer -> warn.
touch_ago "$tmp/internal/STATUS.md" $((10 * H)); touch_ago "$tmp/internal/handoffs/old.md" $((10 * H))
commit "$tmp/internal" $((10 * H))
commit "$tmp/engine" $H
out=$(run) || fail "hook exited non-zero"
case "$out" in *systemMessage*"P3-implementation-plan.md"*) echo "ok: warns, links the plan" ;; *) fail "no warning: $out" ;; esac

# A fresh handoff (written now): engine code is not 2 h newer -> silent.
echo n > "$tmp/internal/handoffs/new.md"; touch_ago "$tmp/internal/handoffs/new.md" 0
out=$(run) || fail "hook exited non-zero"
[ -z "$out" ] || fail "warned despite a fresh handoff: $out"
echo "ok: a fresh handoff silences it"
rm "$tmp/internal/handoffs/new.md"

# Window edges, exact to the second. `window <engine_age_s> <reference_age_s>`
# builds fresh repos whose reference (STATUS.md and one handoff) and engine code
# are that many seconds old, then runs the hook.
window() {
  rm -rf "$tmp/internal/.git" "$tmp/engine/.git"
  git init -q -b main "$tmp/internal"; git init -q -b main "$tmp/engine"
  touch_ago "$tmp/internal/STATUS.md" "$2"; touch_ago "$tmp/internal/handoffs/old.md" "$2"
  commit "$tmp/internal" "$2"; commit "$tmp/engine" "$1"
  touch_ago "$tmp/engine/crates/a.rs" "$1"
  run
}
out=$(window $H $((H + 7200))) || fail "hook exited non-zero"      # code 7200 s newer
[ -z "$out" ] || fail "warned at exactly 2 h: $out"
echo "ok: 7200 s newer stays silent"
out=$(window $H $((H + 7201))) || fail "hook exited non-zero"      # code 7201 s newer
case "$out" in *systemMessage*) echo "ok: 7201 s newer warns" ;; *) fail "no warning at 7201 s" ;; esac
out=$(window $((4 * H)) $H) || fail "hook exited non-zero"         # code older than the reference
[ -z "$out" ] || fail "warned with code older than the reference: $out"
echo "ok: code older than the reference stays silent"

# No handoffs at all: STATUS.md alone is the reference.
rm -f "$tmp"/internal/handoffs/*
out=$(window $H $((4 * H))) || fail "hook exited non-zero"
case "$out" in *systemMessage*) echo "ok: empty handoffs dir, STATUS.md is the reference" ;; *) fail "no warning without handoffs" ;; esac
