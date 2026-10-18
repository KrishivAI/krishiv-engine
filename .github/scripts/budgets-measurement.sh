#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fetches the budget measurement that ci.yml's `budgets-measure` job made for
# HEAD_SHA (artifact `budgets-current`) into OUT_DIR, waiting for that job to
# finish. Prints `measured` (artifact downloaded) or `skipped` (no Rust change,
# so nothing was measured). Fails when the measurement failed or did not finish
# within WAIT_SECS. A cancelled run is superseded by a newer one on the same
# head (re-opened PR, re-run), so it keeps waiting.
# Read-only, so it also works for fork pull requests.
# Env: GH_TOKEN, REPO, HEAD_SHA, OUT_DIR, WAIT_SECS (default 3600),
#      POLL_SECS (default 30).
set -euo pipefail
wait_secs="${WAIT_SECS:-3600}"
poll_secs="${POLL_SECS:-30}"
deadline=$((SECONDS + wait_secs))
while :; do
  run=$(gh api "repos/$REPO/actions/workflows/ci.yml/runs?head_sha=$HEAD_SHA&event=pull_request&per_page=1" \
    --jq '.workflow_runs[0].id // empty')
  if [ -n "$run" ]; then
    job=$(gh api "repos/$REPO/actions/runs/$run/jobs?filter=latest&per_page=100" \
      --jq '.jobs[] | select(.name == "budgets-measure") | [.status, (.conclusion // "")] | @tsv')
    status=$(printf '%s' "$job" | cut -f1)
    conclusion=$(printf '%s' "$job" | cut -f2)
    if [ "$status" = completed ]; then
      case "$conclusion" in
        success)
          gh run download "$run" --repo "$REPO" --name budgets-current --dir "$OUT_DIR" >&2
          echo measured
          exit 0
          ;;
        skipped)
          echo skipped
          exit 0
          ;;
        cancelled) ;;
        *)
          echo "::error::budgets-measure in CI run $run concluded '$conclusion'" >&2
          exit 1
          ;;
      esac
    fi
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "::error::no finished budgets-measure job for $HEAD_SHA after ${wait_secs}s" >&2
    exit 1
  fi
  echo "waiting for budgets-measure (run: ${run:-none}, status: ${status:-none})" >&2
  sleep "$poll_secs"
done
