#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Prints 1 when the approval label LABEL counts for this PR's head, else 0.
# It counts only when the configured approver applied it after the head
# reached GitHub. The head's arrival is the later of:
#   - the first check suite on the head commit (a normal push creates a new
#     commit, so this is its push time), and
#   - the latest force push on the PR (moving back to an older commit needs a
#     force push, which GitHub records as an event).
# Read-only, so it also works for fork pull requests.
# Env: GH_TOKEN, REPO, PR, HEAD_SHA, APPROVER, HAS_LABEL (true|false).
set -euo pipefail
label="$1"
if [ "${HAS_LABEL:-false}" != "true" ]; then
  echo 0
  exit 0
fi
events=$(gh api "repos/$REPO/issues/$PR/events" --paginate \
  --jq '.[] | [.event, (.label.name // ""), (.actor.login // ""), .created_at] | @tsv')
last=$(printf '%s\n' "$events" | awk -F'\t' -v l="$label" '$1 == "labeled" && $2 == l' | tail -n1)
actor=$(printf '%s' "$last" | cut -f3)
labeled_at=$(printf '%s' "$last" | cut -f4)
forced_at=$(printf '%s\n' "$events" | awk -F'\t' '$1 == "head_ref_force_pushed" {print $4}' | sort | tail -n1)
suite_at=$(gh api "repos/$REPO/commits/$HEAD_SHA/check-suites" \
  --jq '[.check_suites[].created_at] | min // empty')
arrived_at=$(printf '%s\n%s\n' "$suite_at" "$forced_at" | sort | tail -n1)
if [ -z "$APPROVER" ] || [ "$actor" != "$APPROVER" ]; then
  echo "::warning::$label was applied by '$actor', not the approver; ignoring" >&2
  echo 0
elif [ -z "$arrived_at" ] || [[ "$labeled_at" < "$arrived_at" ]]; then
  echo "::warning::$label was applied before the current head arrived ($labeled_at < $arrived_at); re-apply it to approve this head" >&2
  echo 0
else
  echo 1
fi
