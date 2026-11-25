#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fails unless the latest completed scheduled nightly run on main (nightly.yml) is green for the jobs that gate a
# release: every job of the `ci` call (the former scheduled ci.yml run) must have succeeded or been skipped, and the
# run must cover $GITHUB_SHA. The heavy nightly jobs (fuzz, coverage floors, golden ratchet) are reported by their own
# issues and do not block binaries or a TestPyPI rehearsal. Needs GH_TOKEN and GITHUB_REPOSITORY / GITHUB_SHA.
set -euo pipefail

: "${GITHUB_REPOSITORY:?}" "${GITHUB_SHA:?}"

# A completed Nightly (heavy) run on exactly $GITHUB_SHA counts whether it was scheduled or dispatched by hand (a release
# branch has no schedule); its `ci /` jobs are checked below like any other run. Otherwise the latest scheduled run on main.
run=$(gh api "repos/$GITHUB_REPOSITORY/actions/workflows/nightly.yml/runs?head_sha=$GITHUB_SHA&status=completed&per_page=1" \
  --jq '.workflow_runs[0] | select(. != null) | [.id, .head_sha, .html_url] | @tsv')
if [ -z "$run" ]; then
  run=$(gh api "repos/$GITHUB_REPOSITORY/actions/workflows/nightly.yml/runs?event=schedule&branch=main&status=completed&per_page=1" \
    --jq '.workflow_runs[0] | select(. != null) | [.id, .head_sha, .html_url] | @tsv')
fi
if [ -z "$run" ]; then
  echo "::error::no completed nightly run on $GITHUB_SHA or scheduled on main; run the nightly and retry"
  exit 1
fi
IFS=$'\t' read -r run_id run_sha url <<<"$run"
echo "nightly run used: $run_sha ($url)"

jobs=$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$run_id/jobs?per_page=100" --paginate --jq '.jobs[] | select(.name | startswith("ci / ")) | [.name, .conclusion] | @tsv')
if [ -z "$jobs" ]; then
  echo "::error::the nightly run has no 'ci /' jobs: $url"
  exit 1
fi
bad=$(printf '%s\n' "$jobs" | awk -F'\t' '$2 != "success" && $2 != "skipped" {print $1 " (" $2 ")"}')
if [ -n "$bad" ]; then
  echo "::error::the nightly run has failing CI jobs: $url"
  printf '%s\n' "$bad"
  exit 1
fi

status=$(gh api "repos/$GITHUB_REPOSITORY/compare/$GITHUB_SHA...$run_sha" --jq .status)
case "$status" in
  identical | ahead) echo "the scheduled run covers $GITHUB_SHA ($status)" ;;
  *)
    # A publish tag (RELEASING.md "Publish tag"): exactly one commit that changes only files under .github/,
    # whose parent the green run covers.
    # The fix commit's own parent must be covered by the green run (an ancestor of, or equal to, run_sha), so the fix
    # may sit on the release commit even after main has moved on. The file list is paginated at 300: refuse longer.
    tag=$(gh api "repos/$GITHUB_REPOSITORY/commits/$GITHUB_SHA" --jq '[(.parents | length), (.parents[0].sha // ""), ([.files[] | (.filename, (.previous_filename // empty)) | select(startswith(".github/") | not)] | length), (.files | length)] | @tsv')
    IFS=$'\t' read -r nparents parent outside nfiles <<<"$tag"
    covered=""
    if [ "$nparents" = 1 ]; then
      covered=$(gh api "repos/$GITHUB_REPOSITORY/compare/$parent...$run_sha" --jq .status)
    fi
    if [ "$nparents" = 1 ] && { [ "$covered" = identical ] || [ "$covered" = ahead ]; } && [ "$outside" = 0 ] && [ "$nfiles" -lt 300 ]; then
      echo "$GITHUB_SHA is one .github-only commit on $parent, which the scheduled run ($run_sha) covers"
    else
      echo "::error::the latest green scheduled run ($run_sha) does not include $GITHUB_SHA ($status); wait for the next nightly run"
      exit 1
    fi
    ;;
esac
