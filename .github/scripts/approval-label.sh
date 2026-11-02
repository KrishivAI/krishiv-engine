#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Prints 1 when the approval label LABEL counts for this PR's head, else 0.
#
# It counts only when the configured approver applied it, and either
#   (a) after the head reached GitHub (the head's arrival is the later of the
#       first workflow run or check suite on the head commit, and the latest
#       force push on the PR), or
#   (b) earlier, when the measured values have not grown since: see below.
#
# Staying valid across pushes. With VALUES_FILE set, the run that sees the label
# fresh (case a) records the measured values in a comment on the pull request
# (written by the workflow's own token, marked `krishiv-approval:v1`, bound to
# that application of the label). A later run, on a head that arrived after the
# label, still counts the approval while every measured value that matches
# METRIC_FILTER stays within TOLERANCE_PERCENT (default 5) of the recorded one,
# or with STRICT=true does not grow at all. A value that was not recorded counts
# as 0 (so a new nonzero one is not covered). An approval never
# survives a force push after it was applied, the label being removed, or its
# being applied by someone else. Without VALUES_FILE the rule is (a) alone, and a
# token that cannot write comments (a fork) records nothing, so (a) alone applies.
#
# VALUES_FILE holds one `name value` pair per line (integers).
# Read-only apart from that comment, so it also works for fork pull requests.
# Env: GH_TOKEN, REPO, PR, HEAD_SHA, APPROVER, HAS_LABEL (true|false), and
# optionally VALUES_FILE, METRIC_FILTER (extended regex over names, default
# everything), TOLERANCE_PERCENT, STRICT (true|false), RECORD (false: never write).
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

values_file="${VALUES_FILE:-}"
filter="${METRIC_FILTER:-.}"
strict="${STRICT:-false}"
tolerance="${TOLERANCE_PERCENT:-5}"
if [ "$strict" = "true" ]; then tolerance=0; fi

# The record for this application of the label: the newest comment by the
# workflow's bot that carries the marker, this label and this labeled_at.
# Prints its `name value` lines.
recorded_values() {
  local last="" b64 body
  while read -r b64; do
    body=$(printf '%s' "$b64" | base64 -d)
    if printf '%s\n' "$body" | grep -qx "label=$label" \
      && printf '%s\n' "$body" | grep -qx "labeled_at=$labeled_at"; then
      last=$(printf '%s\n' "$body" | awk '/^<!--/ || /^-->/ || /^label=/ || /^labeled_at=/ {next} NF == 2 && $2 ~ /^[0-9]+$/ {print}')
    fi
  done < <(gh api "repos/$REPO/issues/$PR/comments" --paginate \
    --jq '.[] | select(.user.login == "github-actions[bot]") | select(.body | startswith("<!-- krishiv-approval:v1")) | .body | @base64')
  printf '%s\n' "$last"
}

# Whether the current values stay inside the recorded ones: reads the recorded
# `name value` lines on stdin, compares with VALUES_FILE.
within_record() {
  awk -v filter="$filter" -v tol="$tolerance" -v strict="$strict" '
    NR == FNR { rec[$1] = $2; seen = 1; next }
    $1 ~ filter {
      now = $2 + 0
      if (!($1 in rec)) {
        # Not recorded: nothing there when it was approved, so only a zero is covered.
        if (now == 0) next
        printf "not covered: %s=%s was not recorded\n", $1, $2 > "/dev/stderr"; bad = 1; next
      }
      if (now * 100 > (rec[$1] + 0) * (100 + tol)) {
        printf "grown: %s %s -> %s (more than %s%% over the recorded value)\n", $1, rec[$1], now, tol > "/dev/stderr"; bad = 1
      }
    }
    END { exit (seen && !bad) ? 0 : 1 }
  ' /dev/stdin "$values_file"
}

record_values() {
  [ "${RECORD:-true}" = "false" ] && return 0
  local body
  body="<!-- krishiv-approval:v1
label=$label
labeled_at=$labeled_at
$(awk -v filter="$filter" '$1 ~ filter' "$values_file")
-->
Recorded the measured values for the \`$label\` approval. It stays valid while they do not grow beyond the allowed margin; a new push is checked against them, and removing or re-applying the label starts over."
  gh api "repos/$REPO/issues/$PR/comments" -f body="$body" >/dev/null 2>&1 \
    || echo "::warning::could not record the values for $label (no write access); the approval will not outlast a push" >&2
}

if [ -z "$APPROVER" ] || [ "$actor" != "$APPROVER" ]; then
  echo "::warning::$label was applied by '$actor', not the approver; ignoring" >&2
  echo 0
# Timestamps have one-second resolution: an approval in the same second as the
# head's arrival does not prove it came after, so it must be strictly later.
elif [ -n "$arrived_at" ] && [[ "$labeled_at" > "$arrived_at" ]]; then
  if [ -n "$values_file" ] && [ -s "$values_file" ] && [ -z "$(recorded_values)" ]; then
    record_values
  fi
  echo 1
elif [ -n "$values_file" ] && [ -s "$values_file" ] \
  && { [ -z "$forced_at" ] || [[ "$labeled_at" > "$forced_at" ]]; } \
  && [ -n "$(recorded_values)" ] \
  && recorded_values | within_record; then
  echo "::notice::$label still covers this head: the measured values are within the recorded ones" >&2
  echo 1
else
  echo "::warning::$label was not applied after the current head arrived ($labeled_at vs $arrived_at), and no recorded value covers this head; re-apply it to approve this head" >&2
  echo 0
fi
