#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Owner rule: no AI attribution anywhere. Rejects text that carries it:
#   - `Co-Authored-By:` lines naming an AI or assistant (Claude, Anthropic,
#     noreply@anthropic.com, opencode, GPT, OpenAI, Copilot, Gemini, Kimi, GLM,
#     Muse, Meta AI, DeepSeek, Qwen) or a bot (`[bot]` noreply address);
#   - `Claude-Session:` lines, "Generated with", and the robot emoji.
# Matching is case-insensitive and whole-word for names (so "Museum" passes).
# Human co-authors stay allowed.
#   check-ai-attribution.sh --message-file FILE   commit-msg hook
#   check-ai-attribution.sh BASE HEAD             CI: every non-merge commit in
#                                                 BASE..HEAD, plus PR_TITLE and
#                                                 PR_BODY when those are set
set -euo pipefail

coauthor_names='(^|[^a-z])(claude|anthropic|noreply@anthropic\.com|opencode|gpt|openai|copilot|gemini|kimi|glm|muse|meta ai|deepseek|qwen)([^a-z]|$)|\[bot\]'
text_pattern="^[[:space:]]*co-authored-by:.*${coauthor_names}|^[[:space:]]*claude-session:|generated with"
robot='🤖'

# check_text LABEL TEXT: prints each offending line, returns 1 when there is one.
check_text() {
  local label=$1 text=$2 hits
  hits=$({
    printf '%s\n' "$text" | grep -inE "$text_pattern" || true
    printf '%s\n' "$text" | grep -nF "$robot" || true
  })
  [ -z "$hits" ] && return 0
  while IFS= read -r hit; do
    [ -n "$hit" ] && echo "ai-attribution: $label line ${hit%%:*}: ${hit#*:}" >&2
  done <<< "$hits"
  echo "ai-attribution: remove the AI attribution above (no AI attribution anywhere)" >&2
  return 1
}

if [ "${1:-}" = "--message-file" ] && [ $# -eq 2 ]; then
  check_text "commit message" "$(cat "$2")" && echo "ai-attribution: OK (commit message)"
  exit $?
elif [ $# -ne 2 ]; then
  echo "usage: $0 --message-file FILE | BASE HEAD" >&2
  exit 2
fi

base=$1
head=$2
bad=0
while read -r sha; do
  [ -n "$sha" ] || continue
  check_text "commit $(git rev-parse --short "$sha")" "$(git log -1 --format=%B "$sha")" || bad=1
done < <(git rev-list --no-merges "$base..$head")
if [ -n "${PR_TITLE+set}" ]; then
  check_text "PR title" "$PR_TITLE" || bad=1
fi
if [ -n "${PR_BODY+set}" ]; then
  check_text "PR body" "$PR_BODY" || bad=1
fi
[ "$bad" -eq 0 ] || exit 1
echo "ai-attribution: OK (commits $base..$head, PR title and body)"
