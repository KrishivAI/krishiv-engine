#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Shell test for check-large-files.sh and the pre-commit wiring, in a scratch repo.
# Run: bash .github/scripts/test-check-large-files.sh (also run by `cargo nextest` via xtask).
set -euo pipefail

script="$(cd "$(dirname "$0")" && pwd)/check-large-files.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export GIT_AUTHOR_NAME=T GIT_AUTHOR_EMAIL=t@example.com GIT_COMMITTER_NAME=T GIT_COMMITTER_EMAIL=t@example.com
git init -q -b main "$tmp/w"
cd "$tmp/w"
git config commit.gpgsign false
mkdir -p .github
printf '# none\nallowed/*\n' > .github/large-files-allow.txt
echo x > small && git add . && git commit -q -m base
base=$(git rev-parse HEAD)

fail() { echo "FAIL: $*" >&2; exit 1; }
expect_rejected() { if out=$("$@" 2>&1); then fail "accepted: $*"; fi; case "$out" in *"large-files:"*) ;; *) fail "wrong error: $out" ;; esac; }

head -c 1048577 /dev/zero > big.bin
head -c 1048576 /dev/zero > exact.bin
mkdir allowed && head -c 2000000 /dev/zero > allowed/ok.bin
git add small big.bin exact.bin allowed
expect_rejected bash "$script" --staged
out=$(bash "$script" --staged 2>&1 || true)
case "$out" in *big.bin*) ;; *) fail "big.bin not named" ;; esac
case "$out" in *exact.bin*|*allowed/ok.bin*) fail "exact-limit or allow-listed file flagged: $out" ;; esac
echo "ok: staged oversize file rejected, limit and allow-list respected"

git commit -q -m big --no-verify
expect_rejected bash "$script" "$base" HEAD
echo "ok: PR diff oversize file rejected"

git rm -q --cached big.bin && rm big.bin && git commit -q -m drop
bash "$script" "$base" HEAD || fail "clean diff rejected"
echo "ok: clean diff accepted"

# Modifying an already-committed large file is not a new addition.
# Not allow-listed, so only the add-only filter keeps this accepted.
echo more >> exact.bin && git commit -q -am mod
bash "$script" HEAD~1 HEAD || fail "modified file rejected"
echo "ok: modification accepted"

# A renamed file counts as added (a spill file can arrive as a rename).
git mv exact.bin moved.bin && git commit -q -m mv
expect_rejected bash "$script" HEAD~1 HEAD
echo "ok: renamed oversize file rejected"
git reset -q --hard HEAD~1

# The hook itself.
hooks=$(cd "$(dirname "$script")/../../.githooks" && pwd)
mkdir -p "$tmp/w/.github/scripts" && cp "$script" .github/scripts/
mkdir "$tmp/hooks" && cp "$hooks/pre-commit" "$tmp/hooks/"
git config core.hooksPath "$tmp/hooks"
head -c 1048577 /dev/zero > spill.bin && git add spill.bin
if out=$(SKIP_CLIPPY=1 git commit -m spill 2>&1); then fail "hook let a large file through"; fi
case "$out" in *"large-files:"*) ;; *) fail "hook error: $out" ;; esac
echo "ok: pre-commit hook rejects a large file"
