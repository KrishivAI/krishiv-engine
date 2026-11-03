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
# that application of the label, to the gate that measured them (APPROVAL_SCOPE:
# the size cap, the budgets and the wheel sizes share the `budget-approved` label
# but not their values) and to the head it measured). Only a head that is still the
# PR's head records, the oldest record of an application wins, and a record counts
# only when the head it names had arrived before the label was applied, so a later
# push cannot replace what the approver saw. A later run, on a head that arrived after the
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
# optionally VALUES_FILE, APPROVAL_SCOPE, METRIC_FILTER (extended regex over names, default
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
# The first time a workflow run or check suite saw commit $1: when it reached GitHub.
arrival_of() {
  local run suite
  run=$(gh api "repos/$REPO/actions/runs?head_sha=$1&per_page=100" \
    --jq '[.workflow_runs[].created_at] | min // empty')
  suite=$(gh api "repos/$REPO/commits/$1/check-suites" \
    --jq '[.check_suites[].created_at] | min // empty' 2>/dev/null || true)
  printf '%s\n%s\n' "$run" "$suite" | grep -v '^$' | sort | head -n1 || true
}
# Workflow runs record every pull-request update, fork pushes included; check
# suites are a fallback for a head with no run yet.
pushed_at=$(arrival_of "$HEAD_SHA")
arrived_at=$(printf '%s\n%s\n' "$pushed_at" "$forced_at" | sort | tail -n1)

values_file="${VALUES_FILE:-}"
filter="${METRIC_FILTER:-.}"
scope="${APPROVAL_SCOPE:-default}"
strict="${STRICT:-false}"
tolerance="${TOLERANCE_PERCENT:-5}"
if [ "$strict" = "true" ]; then tolerance=0; fi

# When the first head other than $1 reached this PR after the label was applied: the
# earliest pull-request workflow run on the PR's branch created after labeled_at
# whose commit is not $1. Prints nothing when there was none.
first_other_push_after_label() {
  local ref
  ref=$(gh api "repos/$REPO/pulls/$PR" --jq '.head.ref' 2>/dev/null || true)
  [ -n "$ref" ] || return 0
  gh api "repos/$REPO/actions/runs?branch=$ref&event=pull_request&per_page=100" --paginate \
    --jq '.workflow_runs[] | [.created_at, .head_sha] | @tsv' 2>/dev/null \
    | awk -F'\t' -v l="$labeled_at" -v h="$1" '$1 > l && $2 != h {print $1}' | sort | head -n1 || true
}

# The record for this application of the label and this gate: the OLDEST comment by the
# workflow's bot that carries the marker, the label, the scope and the labeled_at, whose
# head had arrived before the label was applied, and that was written before any other
# head reached the PR after the label. Prints its `name value` lines.
recorded_values() {
  local created b64 body head arrived later
  while IFS=$'\t' read -r created b64; do
    body=$(printf '%s' "$b64" | base64 -d)
    printf '%s\n' "$body" | grep -qx "label=$label" || continue
    printf '%s\n' "$body" | grep -qx "scope=$scope" || continue
    printf '%s\n' "$body" | grep -qx "labeled_at=$labeled_at" || continue
    head=$(printf '%s\n' "$body" | sed -n 's/^head=//p' | head -n1)
    [ -n "$head" ] || continue
    arrived=$(arrival_of "$head")
    [ -n "$arrived" ] && [[ "$labeled_at" > "$arrived" ]] || continue
    later=$(first_other_push_after_label "$head")
    if [ -n "$later" ] && ! [[ "$created" < "$later" ]]; then continue; fi
    printf '%s\n' "$body" | awk '/^<!--/ || /^-->/ || /^label=/ || /^scope=/ || /^head=/ || /^labeled_at=/ {next} NF == 2 && $2 ~ /^[0-9]+$/ {print}'
    return 0
  done < <(gh api "repos/$REPO/issues/$PR/comments" --paginate \
    --jq '.[] | select(.user.login == "github-actions[bot]") | select(.body | startswith("<!-- krishiv-approval:v1")) | [.created_at, (.body | @base64)] | @tsv')
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
  # A run for a head that is no longer the PR's head (a delayed or re-run job) must not
  # record that head's values.
  if [ "$(gh api "repos/$REPO/pulls/$PR" --jq '.head.sha' 2>/dev/null || true)" != "$HEAD_SHA" ]; then
    return 0
  fi
  local body
  body="<!-- krishiv-approval:v1
label=$label
scope=$scope
head=$HEAD_SHA
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
