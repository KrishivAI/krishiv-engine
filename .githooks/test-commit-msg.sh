#!/usr/bin/env sh
# SPDX-License-Identifier: Apache-2.0
# Shell test for .githooks/commit-msg: auto sign-off, AI attribution still refused.
# Run: sh .githooks/test-commit-msg.sh   (also run by xtask/tests/policy.rs).
set -eu
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_COMMON_DIR

hooks=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$hooks/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export GIT_AUTHOR_NAME=T GIT_AUTHOR_EMAIL=t@example.com GIT_COMMITTER_NAME=T GIT_COMMITTER_EMAIL=t@example.com

git init -q -b main "$tmp/work"
cd "$tmp/work"
mkdir -p "$tmp/hooks" .github/scripts
cp "$hooks/commit-msg" "$tmp/hooks/commit-msg"
cp "$repo/.github/scripts/check-ai-attribution.sh" .github/scripts/
git config core.hooksPath "$tmp/hooks"
git config commit.gpgsign false

fail() { echo "FAIL: $*" >&2; exit 1; }

echo 1 > f && git add f
git commit -q -m "plain message"
git log -1 --format=%B | grep -q '^Signed-off-by: T <t@example.com>$' \
  || fail "no sign-off added"
echo "ok: sign-off added when missing"

echo 2 >> f
git commit -q -am "own sign-off" -m "Signed-off-by: Other One <other@example.com>"
[ "$(git log -1 --format=%B | grep -c '^Signed-off-by:')" = 2 ] || fail "own sign-off not added next to a foreign one"
git log -1 --format=%B | grep -q '^Signed-off-by: T <t@example.com>$' || fail "own sign-off missing"
echo "ok: foreign sign-off kept, own added"

echo 3 >> f
git commit -q -s -am "with -s"
[ "$(git log -1 --format=%B | grep -c '^Signed-off-by:')" = 1 ] || fail "-s sign-off duplicated"
echo "ok: git commit -s not duplicated"

echo 5 >> f
GIT_COMMITTER_NAME="Someone Else" GIT_COMMITTER_EMAIL=se@example.com git commit -q -am "other committer"
git log -1 --format=%B | grep -q '^Signed-off-by: Someone Else <se@example.com>$' || fail "not signed as the committer"
echo "ok: signed as the committer, not a fixed identity"

echo 4 >> f
if git commit -q -am "x" -m "Generated with Claude Code" 2>/dev/null; then fail "AI attribution allowed"; fi
echo "ok: AI attribution still refused"
