#!/usr/bin/env sh
# SPDX-License-Identifier: Apache-2.0
# Shell test for .githooks/pre-push (P2-T134 guard, ADR 0029 checks): pushes to a local bare remote.
# Run: sh .githooks/test-pre-push.sh   (also run by `cargo nextest` via xtask).
set -eu

hooks=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export GIT_AUTHOR_NAME=T GIT_AUTHOR_EMAIL=t@example.com GIT_COMMITTER_NAME=T GIT_COMMITTER_EMAIL=t@example.com
unset KRISHIV_ALLOW_MAIN_PUSH SKIP_CLIPPY || true

# A stub `cargo` records each call and fails when CARGO_STUB_FAIL=1, so the
# clippy/deny step is observable without a Rust workspace.
mkdir "$tmp/bin"
cat > "$tmp/bin/cargo" <<'STUB'
#!/usr/bin/env sh
echo "$*" >> "$CARGO_STUB_LOG"
[ "${CARGO_STUB_FAIL:-0}" != 1 ]
STUB
chmod +x "$tmp/bin/cargo"
export PATH="$tmp/bin:$PATH" CARGO_STUB_LOG="$tmp/cargo.log"
: > "$CARGO_STUB_LOG"

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

# clippy/deny run once per push, only when the pushed commits change Rust files.
[ ! -s "$CARGO_STUB_LOG" ] || fail "cargo ran for pushes without Rust changes: $(cat "$CARGO_STUB_LOG")"
echo "ok: no cargo run for non-Rust pushes"
git switch -q -c feature/rust
mkdir -p src && echo 'fn main() {}' > src/main.rs && git add src && git commit -q -m rust
echo x >> f && git commit -q -am more
git push -q origin feature/rust >/dev/null 2>&1 || fail "Rust push was refused"
[ "$(command grep -c '^clippy' "$CARGO_STUB_LOG")" = 1 ] || fail "clippy did not run exactly once: $(cat "$CARGO_STUB_LOG")"
echo "ok: clippy runs once for a push of two commits with Rust changes"
: > "$CARGO_STUB_LOG"
echo 'fn x() {}' >> src/main.rs && git commit -q -am rust2
if CARGO_STUB_FAIL=1 git push -q origin feature/rust >/dev/null 2>&1; then fail "push with failing clippy was allowed"; fi
echo "ok: failing clippy blocks the push"
: > "$CARGO_STUB_LOG"
SKIP_CLIPPY=1 CARGO_STUB_FAIL=1 git push -q origin feature/rust >/dev/null 2>&1 || fail "SKIP_CLIPPY=1 did not skip"
[ ! -s "$CARGO_STUB_LOG" ] || fail "cargo ran under SKIP_CLIPPY=1"
echo "ok: SKIP_CLIPPY=1 skips clippy and deny"
echo "pre-push tests passed"
