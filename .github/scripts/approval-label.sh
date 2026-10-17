#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Prints 1 when the approval label LABEL counts for this PR, else 0.
# It counts only when it is on the PR, the configured approver applied it,
# and it was applied after the head commit was pushed: a push after the
# approval needs a fresh approval.
# Env: GH_TOKEN, REPO, PR, HEAD_SHA, APPROVER, HAS_LABEL (true|false).
set -euo pipefail
label="$1"
if [ "${HAS_LABEL:-false}" != "true" ]; then
  echo 0
  exit 0
fi
last=$(gh api "repos/$REPO/issues/$PR/events" --paginate \
  --jq ".[] | select(.event == \"labeled\" and .label.name == \"$label\") | [.actor.login, .created_at] | @tsv" | tail -n1)
actor=$(printf '%s' "$last" | cut -f1)
labeled_at=$(printf '%s' "$last" | cut -f2)
# When the head commit reached GitHub: the earliest check suite on it.
pushed_at=$(gh api "repos/$REPO/commits/$HEAD_SHA/check-suites" \
  --jq '[.check_suites[].created_at] | min // empty')
if [ -z "$APPROVER" ] || [ "$actor" != "$APPROVER" ]; then
  echo "::warning::$label was applied by '$actor', not the approver; ignoring" >&2
  echo 0
elif [ -z "$pushed_at" ] || [[ "$labeled_at" < "$pushed_at" ]]; then
  echo "::warning::$label was applied before the latest push ($labeled_at < $pushed_at); re-apply it to approve this head" >&2
  echo 0
else
  echo 1
fi
