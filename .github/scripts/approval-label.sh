#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Prints 1 when the approval label LABEL counts for this PR's head, else 0.
# It counts only when the configured approver applied it after the head
# reached GitHub. The head's arrival is the later of:
#   - the first workflow run (or check suite) on the head commit (a normal
#     push creates a new commit, so this is its push time; runs also cover
#     fork pull requests), and
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
# Workflow runs record every pull-request update, fork pushes included; check
# suites are a fallback for a head with no run yet.
run_at=$(gh api "repos/$REPO/actions/runs?head_sha=$HEAD_SHA&per_page=100" \
  --jq '[.workflow_runs[].created_at] | min // empty')
suite_at=$(gh api "repos/$REPO/commits/$HEAD_SHA/check-suites" \
  --jq '[.check_suites[].created_at] | min // empty' 2>/dev/null || true)
pushed_at=$(printf '%s\n%s\n' "$run_at" "$suite_at" | grep -v '^$' | sort | head -n1 || true)
arrived_at=$(printf '%s\n%s\n' "$pushed_at" "$forced_at" | sort | tail -n1)
if [ -z "$APPROVER" ] || [ "$actor" != "$APPROVER" ]; then
  echo "::warning::$label was applied by '$actor', not the approver; ignoring" >&2
  echo 0
# Timestamps have one-second resolution: an approval in the same second as the
# head's arrival does not prove it came after, so it must be strictly later.
elif [ -z "$arrived_at" ] || ! [[ "$labeled_at" > "$arrived_at" ]]; then
  echo "::warning::$label was not applied after the current head arrived ($labeled_at vs $arrived_at); re-apply it to approve this head" >&2
  echo 0
else
  echo 1
fi
