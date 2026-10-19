#!/usr/bin/env sh
# SPDX-License-Identifier: Apache-2.0
# Shell test for .githooks/pre-push (P2-T134): pushes to a local bare remote.
# Run: sh .githooks/test-pre-push.sh   (also run by `cargo nextest` via xtask).
set -eu

hooks=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export GIT_AUTHOR_NAME=T GIT_AUTHOR_EMAIL=t@example.com GIT_COMMITTER_NAME=T GIT_COMMITTER_EMAIL=t@example.com
unset KRISHIV_ALLOW_MAIN_PUSH || true

git init -q --bare "$tmp/remote.git"
git init -q -b main "$tmp/work"
cd "$tmp/work"
mkdir "$tmp/hooks" && cp "$hooks/pre-push" "$tmp/hooks/pre-push"
git config core.hooksPath "$tmp/hooks"
git config commit.gpgsign false
git remote add origin "$tmp/remote.git"
echo 1 > f && git add f && git commit -q -m one

fail() { echo "FAIL: $*" >&2; exit 1; }
expect_refused() {
  desc=$1; shift
  if out=$("$@" 2>&1); then fail "$desc was allowed"; fi
  case "$out" in *"pre-push: refusing"*) ;; *) fail "$desc was not refused by the hook: $out" ;; esac
  echo "ok: $desc refused by the hook"
}
expect_ok() { desc=$1; shift; "$@" >/dev/null 2>&1 || fail "$desc was refused"; echo "ok: $desc allowed"; }

expect_refused "push to main" git push origin main
expect_refused "push HEAD:main" git push origin HEAD:main

git switch -q -c feature/x
expect_ok "feature push" git push origin feature/x
echo 2 >> f && git commit -q -am two
expect_ok "feature fast-forward push" git push origin feature/x

# Rewrite history locally, then force.
git reset -q --hard HEAD~1
echo 3 >> f && git commit -q -am three
if git push origin feature/x >/dev/null 2>&1; then fail "plain non-ff push was allowed"; fi
echo "ok: plain non-ff push rejected (by git itself)"
expect_refused "--force push" git push --force origin feature/x
expect_refused "--force-with-lease push" git push --force-with-lease origin feature/x
expect_ok "feature delete" git push origin --delete feature/x

KRISHIV_ALLOW_MAIN_PUSH=1 git push -q origin main >/dev/null 2>&1 || fail "override did not allow main push"
echo "ok: override allows main push"
git switch -q -c feature/y
git push -q origin feature/y >/dev/null 2>&1
git reset -q --hard HEAD~0 && echo 4 >> f && git commit -q --amend -am four
KRISHIV_ALLOW_MAIN_PUSH=1 git push -q --force origin feature/y >/dev/null 2>&1 || fail "override did not allow force push"
echo "ok: override allows force push"

# Pinned ruff on changed Python files, with a fake uvx that records its calls.
git switch -q main
mkdir -p python "$tmp/fakebin"
printf '[[package]]\nname = "ruff"\nversion = "9.8.7"\nsource = {}\n' > python/uv.lock
git add python/uv.lock && git commit -q -m lock
KRISHIV_ALLOW_MAIN_PUSH=1 git push -q origin main >/dev/null 2>&1
cat > "$tmp/fakebin/uvx" <<'FAKE'
#!/bin/sh
echo "$*" >> "$FAKE_UVX_LOG"
exit "${FAKE_UVX_RC:-0}"
FAKE
chmod +x "$tmp/fakebin/uvx"
export FAKE_UVX_LOG="$tmp/uvx.log"
git switch -q -c feature/py
echo 'x = 1' > python/a.py && git add python/a.py && git commit -q -m py
: > "$FAKE_UVX_LOG"
KRISHIV_UVX="$tmp/fakebin/uvx" expect_ok "python push with passing ruff" git push origin feature/py
grep -q 'ruff@9.8.7 format --check python/a.py' "$FAKE_UVX_LOG" || fail "format --check not run with the pinned version"
grep -q 'ruff@9.8.7 check python/a.py' "$FAKE_UVX_LOG" || fail "check not run with the pinned version"
echo 'x=2' >> python/a.py && git commit -q -am py2
if out=$(FAKE_UVX_RC=1 KRISHIV_UVX="$tmp/fakebin/uvx" git push origin feature/py 2>&1); then fail "push with ruff problems was allowed"; fi
case "$out" in *"pre-push: ruff 9.8.7 found problems"*) echo "ok: push with ruff problems refused" ;; *) fail "ruff failure not reported: $out" ;; esac
out=$(KRISHIV_UVX="$tmp/nonexistent-uvx" git push origin feature/py 2>&1) || fail "push without uv was refused"
case "$out" in *"uv is not installed, skipping the ruff check"*) echo "ok: missing uv skips with a message" ;; *) fail "no skip message: $out" ;; esac
git switch -q main && git switch -q -c feature/nopy
echo 1 > g && git add g && git commit -q -m nopy
: > "$FAKE_UVX_LOG"
KRISHIV_UVX="$tmp/fakebin/uvx" FAKE_UVX_RC=1 expect_ok "non-Python push does not run ruff" git push origin feature/nopy
[ ! -s "$FAKE_UVX_LOG" ] || fail "ruff ran for a push without Python changes"
echo "pre-push tests passed"
