#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Prints 1 when the approval label LABEL counts for this run, else 0.
# Approval is bound to pushes, not timestamps: every push (`synchronize`)
# removes the approval labels (job `reset-approvals` in ci.yml) and a run
# triggered by a push never counts them, so a label always approves the head
# that was current when the approver applied it. It also counts only when the
# configured approver applied it.
# Env: GH_TOKEN, REPO, PR, APPROVER, EVENT_ACTION, HAS_LABEL (true|false).
set -euo pipefail
label="$1"
if [ "${HAS_LABEL:-false}" != "true" ]; then
  echo 0
  exit 0
fi
if [ "${EVENT_ACTION:-}" = "synchronize" ]; then
  echo "::warning::$label is ignored on a push; the approver re-applies it to approve the new head" >&2
  echo 0
  exit 0
fi
actor=$(gh api "repos/$REPO/issues/$PR/events" --paginate \
  --jq ".[] | select(.event == \"labeled\" and .label.name == \"$label\") | .actor.login" | tail -n1)
if [ -n "$APPROVER" ] && [ "$actor" = "$APPROVER" ]; then
  echo 1
else
  echo "::warning::$label was applied by '$actor', not the approver; ignoring" >&2
  echo 0
fi
