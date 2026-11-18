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

# A fresh handoff (written now): engine code is not 2 h newer -> silent.
echo n > "$tmp/internal/handoffs/new.md"
out=$(run) || fail "hook exited non-zero"
[ -z "$out" ] || fail "warned despite a fresh handoff: $out"
echo "ok: a fresh handoff silences it"
rm "$tmp/internal/handoffs/new.md"

# Window edges: STATUS.md and the handoff are committed and touched H hours ago
# (engine code is 1 h old), so the code is (H-1) h newer... set engine instead:
# reference 1 h old, engine code N hours after it.
window() { # engine_hours_ago ref_hours_ago
  rm -rf "$tmp/internal/.git" "$tmp/engine/.git"
  git init -q -b main "$tmp/internal"; git init -q -b main "$tmp/engine"
  touch_ago "$tmp/internal/STATUS.md" "$2"; touch_ago "$tmp/internal/handoffs/old.md" "$2"
  commit "$tmp/internal" "$2"; commit "$tmp/engine" "$1"
  touch_ago "$tmp/engine/crates/a.rs" "$1"
  run
}
out=$(window 1 3) || fail "hook exited non-zero"   # code 2 h newer: not more than 2 h
[ -z "$out" ] || fail "warned at exactly 2 h: $out"
echo "ok: 2 h newer stays silent"
out=$(window 1 4) || fail "hook exited non-zero"   # code 3 h newer
case "$out" in *systemMessage*) echo "ok: 3 h newer warns" ;; *) fail "no warning 3 h newer" ;; esac
out=$(window 4 1) || fail "hook exited non-zero"   # code older than the reference
[ -z "$out" ] || fail "warned with code older than the reference: $out"
echo "ok: code older than the reference stays silent"

# No handoffs at all: STATUS.md alone is the reference.
rm -f "$tmp"/internal/handoffs/*
out=$(window 1 4) || fail "hook exited non-zero"
case "$out" in *systemMessage*) echo "ok: empty handoffs dir, STATUS.md is the reference" ;; *) fail "no warning without handoffs" ;; esac
