#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Tests for check-ai-attribution.sh (message-file and CI range modes) and the
# commit-msg hook wiring, in a scratch repo.
# Run: bash .github/scripts/test-check-ai-attribution.sh (also run by `cargo nextest` via xtask).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
script="$here/check-ai-attribution.sh"
hook="$here/../../.githooks/commit-msg"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
msg="$tmp/msg"

fail() { echo "FAIL: $*" >&2; exit 1; }

# expect_blocked DESC TEXT: the message file must be refused and the offending
# line named in the output.
expect_blocked() {
  local desc=$1 text=$2 out
  printf '%b' "$text" > "$msg"
  if out=$(bash "$script" --message-file "$msg" 2>&1); then fail "$desc was allowed"; fi
  case "$out" in *"ai-attribution: commit message line"*) ;; *) fail "$desc: no offending line named: $out" ;; esac
  echo "ok: blocked $desc"
}

expect_allowed() {
  local desc=$1 text=$2
  printf '%b' "$text" > "$msg"
  bash "$script" --message-file "$msg" >/dev/null 2>&1 || fail "$desc was refused"
  echo "ok: allowed $desc"
}

expect_allowed "plain message" "Fix the parser\n\nSigned-off-by: gopal tathe <44522021+techgopal@users.noreply.github.com>\n"
expect_allowed "human co-author" "Pair fix\n\nSigned-off-by: gopal tathe <44522021+techgopal@users.noreply.github.com>\nCo-Authored-By: Jane Doe <jane@example.com>\n"
expect_allowed "human GitHub noreply co-author" "Pair fix\n\nCo-authored-by: techysneha14 <343708768+shehatathe@users.noreply.github.com>\n"
expect_allowed "human co-author whose name contains a model name" "Note\n\nCo-Authored-By: Museum Ops <ops@example.com>\n"

for name in Claude Anthropic noreply@anthropic.com opencode GPT OpenAI Copilot Gemini Kimi GLM Muse "Meta AI" DeepSeek Qwen; do
  expect_blocked "Co-Authored-By $name" "Subject\n\nCo-Authored-By: $name <x@example.com>\n"
done
expect_blocked "lower-case co-author trailer" "Subject\n\nco-authored-by: claude <x@example.com>\n"
expect_blocked "bot co-author" "Subject\n\nCo-authored-by: github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>\n"
expect_blocked "Claude-Session trailer" "Subject\n\nClaude-Session: https://claude.ai/code/session_x\n"
expect_blocked "lower-case Claude-Session" "Subject\n\nclaude-session: abc\n"
expect_blocked "Generated with line" "Subject\n\nGenerated with [Claude Code](https://claude.com/claude-code)\n"
expect_blocked "lower-case generated with" "Subject\n\ngenerated WITH tool\n"
expect_blocked "robot emoji" "Subject\n\n🤖 done\n"

# Multi-line: the human co-author is fine, the Claude-Session line on line 5 is not.
printf 'Subject line\n\nBody paragraph.\nCo-Authored-By: Jane Doe <jane@example.com>\nClaude-Session: https://claude.ai/code/session_x\n' > "$msg"
if out=$(bash "$script" --message-file "$msg" 2>&1); then fail "multi-line message with a session trailer was allowed"; fi
case "$out" in *"line 5"*"Claude-Session"*) ;; *) fail "multi-line: wrong line named: $out" ;; esac
case "$out" in *"Jane Doe"*) fail "human co-author was flagged: $out" ;; esac
echo "ok: multi-line message names only the offending line"

# CI range mode: every commit in BASE..HEAD, plus the PR title and body.
export GIT_AUTHOR_NAME=T GIT_AUTHOR_EMAIL=t@example.com GIT_COMMITTER_NAME=T GIT_COMMITTER_EMAIL=t@example.com
git init -q -b main "$tmp/w"
cd "$tmp/w"
git config commit.gpgsign false
echo 1 > f && git add f && git commit -q -m base
base=$(git rev-parse HEAD)
echo 2 >> f && git commit -q -am "clean human pair

Co-Authored-By: Jane Doe <jane@example.com>"
head_clean=$(git rev-parse HEAD)
(
  unset PR_TITLE PR_BODY
  bash "$script" "$base" "$head_clean" >/dev/null 2>&1 || fail "clean range refused"
)
echo "ok: clean commit range accepted"

echo 3 >> f && git commit -q -am "bad

Claude-Session: https://claude.ai/code/session_y"
head_bad=$(git rev-parse HEAD)
if out=$(unset PR_TITLE PR_BODY; bash "$script" "$base" "$head_bad" 2>&1); then fail "range with a bad commit was allowed"; fi
case "$out" in *"ai-attribution: commit "*"line 3"*"Claude-Session"*) ;; *) fail "range: wrong commit named: $out" ;; esac
echo "ok: commit in the range with a session trailer rejected"

# A bad commit below a clean one is still found.
echo 4 >> f && git commit -q -am "clean again"
head_top=$(git rev-parse HEAD)
if (unset PR_TITLE PR_BODY; bash "$script" "$base" "$head_top" >/dev/null 2>&1); then fail "bad commit below a clean one was missed"; fi
echo "ok: bad commit below a clean one rejected"

if out=$(PR_TITLE="Add thing 🤖" PR_BODY="" bash "$script" "$base" "$head_clean" 2>&1); then fail "emoji in PR title was allowed"; fi
case "$out" in *"PR title line 1"*) ;; *) fail "PR title not named: $out" ;; esac
echo "ok: PR title with the robot emoji rejected"

if out=$(PR_TITLE="Add thing" PR_BODY=$'Summary\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)' bash "$script" "$base" "$head_clean" 2>&1); then fail "PR body footer was allowed"; fi
case "$out" in *"PR body line 3"*) ;; *) fail "PR body not named: $out" ;; esac
echo "ok: PR body footer rejected"

PR_TITLE="Add thing" PR_BODY="Clean body" bash "$script" "$base" "$head_clean" >/dev/null 2>&1 || fail "clean PR title and body refused"
echo "ok: clean PR title and body accepted"

# The commit-msg hook wiring: a commit with an AI trailer is refused by git itself.
hooks="$tmp/hooks"
mkdir -p "$tmp/w/.githooks" "$tmp/w/.github/scripts"
cp "$hook" "$tmp/w/.githooks/commit-msg"
cp "$script" "$tmp/w/.github/scripts/"
git config core.hooksPath .githooks
echo 5 >> f && git add f
if out=$(git commit -q -m "hooked

Generated with Claude" 2>&1); then fail "commit-msg hook let an attribution through"; fi
case "$out" in *"ai-attribution: commit message line 3"*) ;; *) fail "hook error not shown: $out" ;; esac
echo "ok: commit-msg hook refuses an attributed commit"
git commit -q -m "hooked by a human

Co-Authored-By: Jane Doe <jane@example.com>"
echo "ok: commit-msg hook allows a human co-author"
echo "check-ai-attribution tests passed"
