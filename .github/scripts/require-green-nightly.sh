#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fails unless the latest completed scheduled or manually dispatched (v0.2.0-publish, owner 2026-11-13) nightly run on main (nightly.yml) is green for the jobs that gate a
# release: every job of the `ci` call (the former scheduled ci.yml run) must have succeeded or been skipped, and the
# run must cover $GITHUB_SHA. The heavy nightly jobs (fuzz, coverage floors, golden ratchet) are reported by their own
# issues and do not block binaries or a TestPyPI rehearsal. Needs GH_TOKEN and GITHUB_REPOSITORY / GITHUB_SHA.
set -euo pipefail

: "${GITHUB_REPOSITORY:?}" "${GITHUB_SHA:?}"

run=$(gh api "repos/$GITHUB_REPOSITORY/actions/workflows/nightly.yml/runs?branch=main&status=completed&per_page=20" \
  --jq '[.workflow_runs[] | select(.event == "schedule" or .event == "workflow_dispatch")][0] | [.id, .head_sha, .html_url] | @tsv')
if [ -z "$run" ] || [ "$run" = "null	null	null" ]; then
  echo "::error::no completed scheduled nightly run on main; run the nightly and retry"
  exit 1
fi
IFS=$'\t' read -r run_id run_sha url <<<"$run"
echo "latest scheduled nightly run: $run_sha ($url)"

jobs=$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$run_id/jobs?per_page=100" --paginate --jq '.jobs[] | select(.name | startswith("ci / ")) | [.name, .conclusion] | @tsv')
if [ -z "$jobs" ]; then
  echo "::error::the scheduled nightly run has no 'ci /' jobs: $url"
  exit 1
fi
bad=$(printf '%s\n' "$jobs" | awk -F'\t' '$2 != "success" && $2 != "skipped" {print $1 " (" $2 ")"}')
if [ -n "$bad" ]; then
  echo "::error::the latest scheduled nightly run on main has failing CI jobs: $url"
  printf '%s\n' "$bad"
  exit 1
fi

status=$(gh api "repos/$GITHUB_REPOSITORY/compare/$GITHUB_SHA...$run_sha" --jq .status)
# v0.2.0-publish (owner 2026-11-13): a publish commit that changes only .github/ on top of a covered parent counts as
# covered, so the packages are built from exactly the parent's sources.
if [ "$status" = diverged ] || [ "$status" = behind ]; then
  parent=$(gh api "repos/$GITHUB_REPOSITORY/commits/$GITHUB_SHA" --jq '.parents | if length == 1 then .[0].sha else "" end')
  if [ -n "$parent" ]; then
    other=$(gh api "repos/$GITHUB_REPOSITORY/compare/$parent...$GITHUB_SHA" --jq '[.files[].filename | select(startswith(".github/") | not)] | length')
    pstatus=$(gh api "repos/$GITHUB_REPOSITORY/compare/$parent...$run_sha" --jq .status)
    if [ "$other" = 0 ] && { [ "$pstatus" = identical ] || [ "$pstatus" = ahead ]; }; then
      echo "$GITHUB_SHA changes only .github/ on top of $parent, which the run covers ($pstatus)"
      status=ahead
    fi
  fi
fi
case "$status" in
  identical | ahead) echo "the scheduled run covers $GITHUB_SHA ($status)" ;;
  *)
    echo "::error::the latest green scheduled run ($run_sha) does not include $GITHUB_SHA ($status); wait for the next nightly run"
    exit 1
    ;;
esac
