#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fails unless the latest completed scheduled nightly run on main (nightly.yml) is green for the jobs that gate a
# release: every job of the `ci` call (the former scheduled ci.yml run) must have succeeded or been skipped, and the
# run must cover $GITHUB_SHA. The heavy nightly jobs (fuzz, coverage floors, golden ratchet) are reported by their own
# issues and do not block binaries or a TestPyPI rehearsal. Needs GH_TOKEN and GITHUB_REPOSITORY / GITHUB_SHA.
set -euo pipefail

: "${GITHUB_REPOSITORY:?}" "${GITHUB_SHA:?}"

# One-time written waiver (release.yml input `nightly_waiver`, "run=<id> jobs=<a,b> reason=<text>"). Narrow on purpose:
# the named run must be a Nightly (heavy) run of nightly.yml on a release/ branch whose head is an ancestor of (or
# equal to) $GITHUB_SHA, and EVERY job of it that is not named must have succeeded or been skipped. The named jobs may
# fail. The waiver is printed into the job summary. Without the input nothing below changes.
if [ -n "${NIGHTLY_WAIVER:-}" ]; then
  if [[ ! "$NIGHTLY_WAIVER" =~ ^run=([0-9]+)[[:space:]]+jobs=([A-Za-z0-9_.,/() -]+)[[:space:]]+reason=(.+)$ ]]; then
    echo "::error::nightly_waiver must read: run=<id> jobs=<a,b> reason=<text>"
    exit 1
  fi
  w_run=${BASH_REMATCH[1]}; w_jobs=${BASH_REMATCH[2]}; w_reason=${BASH_REMATCH[3]}
  info=$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$w_run" --jq '[.path, .head_branch, .head_sha, .status, .html_url] | @tsv')
  IFS=$'\t' read -r w_path w_branch w_sha w_status w_url <<<"$info"
  [ "$w_path" = ".github/workflows/nightly.yml" ] || { echo "::error::waiver run $w_run is not a nightly.yml run ($w_path)"; exit 1; }
  [ "$w_status" = completed ] || { echo "::error::waiver run $w_run is not completed ($w_status)"; exit 1; }
  case "$w_branch" in release/*) ;; *) echo "::error::waiver run $w_run is on $w_branch, not a release/ branch"; exit 1 ;; esac
  rel=$(gh api "repos/$GITHUB_REPOSITORY/compare/$w_sha...$GITHUB_SHA" --jq .status)
  case "$rel" in identical | ahead) ;; *) echo "::error::waiver run $w_run ($w_sha) is not an ancestor of $GITHUB_SHA ($rel)"; exit 1 ;; esac
  all=$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$w_run/jobs?per_page=100" --paginate --jq '.jobs[] | [.name, .conclusion] | @tsv')
  [ -n "$all" ] || { echo "::error::waiver run $w_run has no jobs"; exit 1; }
  bad=$(printf '%s\n' "$all" | awk -F'\t' -v named="$w_jobs" 'BEGIN { n = split(named, a, ","); for (i = 1; i <= n; i++) skip[a[i]] = 1 }
    !($1 in skip) && $2 != "success" && $2 != "skipped" { print $1 " (" $2 ")" }')
  if [ -n "$bad" ]; then
    echo "::error::jobs outside the waiver are not green in $w_url"
    printf '%s\n' "$bad"
    exit 1
  fi
  {
    echo "### Nightly waiver (one-time, owner-approved)"
    echo "- run: $w_url (head $w_sha, ancestor of $GITHUB_SHA)"
    echo "- waived jobs: $w_jobs"
    echo "- reason: $w_reason"
  } | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"
  exit 0
fi

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
