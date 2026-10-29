#!/usr/bin/env bash
# Release gate: the latest completed scheduled CI run on main must be green and must cover the
# commit being released (its head is that commit or a later main commit). The merge queue does not
# run Miri, the fuzz-target build or the wheel matrix; the nightly run of main does, so a release
# never goes out on code those checks have not seen in combination.
#
# Usage: require-green-nightly.sh        (env: GH_TOKEN, GITHUB_REPOSITORY, GITHUB_SHA)
# Reads only; it starts no run and publishes nothing.
set -euo pipefail

: "${GITHUB_REPOSITORY:?}" "${GITHUB_SHA:?}"

run=$(gh api "repos/$GITHUB_REPOSITORY/actions/workflows/ci.yml/runs?event=schedule&branch=main&status=completed&per_page=1" \
  --jq '.workflow_runs[0] | [.conclusion, .head_sha, .html_url] | @tsv')
if [ -z "$run" ] || [ "$run" = "null	null	null" ]; then
  echo "::error::no completed scheduled CI run on main; run the nightly CI and retry"
  exit 1
fi
IFS=$'\t' read -r conclusion run_sha url <<<"$run"
echo "latest scheduled CI run: $conclusion at $run_sha ($url)"
if [ "$conclusion" != success ]; then
  echo "::error::the latest scheduled CI run on main did not succeed ($conclusion): $url"
  exit 1
fi
# The run must have seen the release commit: identical, or the release commit is an ancestor.
status=$(gh api "repos/$GITHUB_REPOSITORY/compare/$GITHUB_SHA...$run_sha" --jq .status)
case "$status" in
  identical | ahead) echo "the scheduled run covers $GITHUB_SHA ($status)" ;;
  *)
    echo "::error::the latest green scheduled run ($run_sha) does not include $GITHUB_SHA ($status); wait for the next nightly run"
    exit 1
    ;;
esac
