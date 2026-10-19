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
# Every `gh` call is retried with exponential backoff (RETRY_ATTEMPTS tries,
# RETRY_BASE_SECS doubling: 5, 10, 20, 40) when it fails transiently (HTTP 5xx,
# network or stream errors, timeouts), never past the WAIT_SECS deadline. Real
# errors (404, auth, bad request) fail at once.
# Env: GH_TOKEN, REPO, HEAD_SHA, OUT_DIR, WAIT_SECS (default 4500),
#      POLL_SECS (default 30), RETRY_ATTEMPTS (default 5),
#      RETRY_BASE_SECS (default 5), JOB, ARTIFACT, SKIP_JOB (see above).
set -euo pipefail
wait_secs="${WAIT_SECS:-4500}"
poll_secs="${POLL_SECS:-30}"
job_name="${JOB:-budgets-measure}"
artifact="${ARTIFACT:-budgets-current}"
skip_job="${SKIP_JOB:-}"
retry_attempts="${RETRY_ATTEMPTS:-5}"
retry_base="${RETRY_BASE_SECS:-5}"
for v in retry_attempts retry_base; do
  case "${!v}" in
    '' | *[!0-9]*)
      echo "::error::RETRY_ATTEMPTS and RETRY_BASE_SECS must be non-negative integers (got ${v}='${!v}')" >&2
      exit 1
      ;;
  esac
done
deadline=$((SECONDS + wait_secs))

# gh_retry <gh args>: stdout of the successful attempt is printed; stderr of a
# failed attempt is shown, and the attempt is retried only if it looks transient.
gh_retry() {
  local attempt=1 delay=$retry_base err_file rc
  err_file=$(mktemp)
  while :; do
    rc=0
    gh "$@" 2>"$err_file" || rc=$?
    if [ "$rc" -eq 0 ]; then
      cat "$err_file" >&2
      rm -f "$err_file"
      return 0
    fi
    cat "$err_file" >&2
    if ! grep -Eqi 'HTTP 5[0-9][0-9]|stream error|connection (reset|refused|closed)|timed? ?out|timeout|unexpected EOF|\bEOF\b|TLS|temporary failure|no such host|network is unreachable|server error|bad gateway|service unavailable' "$err_file"; then
      rm -f "$err_file"
      return "$rc"
    fi
    if [ "$attempt" -ge "$retry_attempts" ] || [ $((SECONDS + delay)) -ge "$deadline" ]; then
      echo "::error::gh $1 failed transiently $attempt time(s); giving up (see the error above)" >&2
      rm -f "$err_file"
      return "$rc"
    fi
    echo "gh $1 failed transiently (attempt $attempt/$retry_attempts); retrying in ${delay}s" >&2
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
}
while :; do
  run=$(gh_retry api "repos/$REPO/actions/workflows/ci.yml/runs?head_sha=$HEAD_SHA&event=pull_request&per_page=1" \
    --jq '.workflow_runs[0].id // empty')
  if [ -n "$run" ]; then
    job=$(gh_retry api "repos/$REPO/actions/runs/$run/jobs?filter=latest&per_page=100" \
      --jq ".jobs[] | select(.name == \"$job_name\" or (\"$skip_job\" != \"\" and .name == \"$skip_job\")) | [.name, .status, (.conclusion // \"\")] | @tsv")
    name=$(printf '%s' "$job" | cut -f1)
    status=$(printf '%s' "$job" | cut -f2)
    conclusion=$(printf '%s' "$job" | cut -f3)
    if [ "$status" = completed ]; then
      case "$conclusion" in
        success)
          if [ "$name" = "$job_name" ]; then
            gh_retry run download "$run" --repo "$REPO" --name "$artifact" --dir "$OUT_DIR" >&2
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
