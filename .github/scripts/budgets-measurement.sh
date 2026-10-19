#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fetches a measurement that ci.yml made for HEAD_SHA into OUT_DIR, waiting for
# its job to finish. By default the budget measurement (job `budgets-measure`,
# artifact `budgets-current`); the wheel sizes use JOB="wheels-targets / sizes"
# ARTIFACT=wheel-sizes SKIP_JOB=wheels-targets. Prints `measured` (artifact
# downloaded) or `skipped` (no Rust/Python change, so nothing was measured).
# Without SKIP_JOB a skipped JOB means nothing was measured; with it only a
# skipped SKIP_JOB does (the called workflow was not run), and a skipped JOB
# (an earlier wheel build failed) is an error. Fails when the measurement failed or did not finish
# within WAIT_SECS. A cancelled run is superseded by a newer one on the same
# head (re-opened PR, re-run), so it keeps waiting.
# Read-only, so it also works for fork pull requests.
# Env: GH_TOKEN, REPO, HEAD_SHA, OUT_DIR, WAIT_SECS (default 4500),
#      POLL_SECS (default 30), JOB, ARTIFACT, SKIP_JOB (see above).
set -euo pipefail
wait_secs="${WAIT_SECS:-4500}"
poll_secs="${POLL_SECS:-30}"
job_name="${JOB:-budgets-measure}"
artifact="${ARTIFACT:-budgets-current}"
skip_job="${SKIP_JOB:-}"
deadline=$((SECONDS + wait_secs))
while :; do
  run=$(gh api "repos/$REPO/actions/workflows/ci.yml/runs?head_sha=$HEAD_SHA&event=pull_request&per_page=1" \
    --jq '.workflow_runs[0].id // empty')
  if [ -n "$run" ]; then
    job=$(gh api "repos/$REPO/actions/runs/$run/jobs?filter=latest&per_page=100" \
      --jq ".jobs[] | select(.name == \"$job_name\" or (\"$skip_job\" != \"\" and .name == \"$skip_job\")) | [.name, .status, (.conclusion // \"\")] | @tsv")
    name=$(printf '%s' "$job" | cut -f1)
    status=$(printf '%s' "$job" | cut -f2)
    conclusion=$(printf '%s' "$job" | cut -f3)
    if [ "$status" = completed ]; then
      case "$conclusion" in
        success)
          if [ "$name" = "$job_name" ]; then
            gh run download "$run" --repo "$REPO" --name "$artifact" --dir "$OUT_DIR" >&2
            echo measured
            exit 0
          fi
          ;;
        skipped)
          if [ -z "$skip_job" ] || [ "$name" = "$skip_job" ]; then
            echo skipped
            exit 0
          fi
          echo "::error::$job_name in CI run $run was skipped (an earlier job failed)" >&2
          exit 1
          ;;
        cancelled) ;;
        *)
          echo "::error::$name in CI run $run concluded '$conclusion'" >&2
          exit 1
          ;;
      esac
    fi
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "::error::no finished $job_name job for $HEAD_SHA after ${wait_secs}s" >&2
    exit 1
  fi
  echo "waiting for $job_name (run: ${run:-none}, status: ${status:-none})" >&2
  sleep "$poll_secs"
done
