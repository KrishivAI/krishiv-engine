#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Selects the baseline a pull request is gated against: the measurement CI made
# on main for the commit the PR branched from, or failing that for the newest
# main commit before it. Runs in a checkout with full history.
#
# The target commit is `git merge-base BASE_SHA HEAD_SHA` (BASE_SHA itself when
# there is none), so the baseline holds exactly what main had when the PR
# branched or last merged main, and the PR is gated only on the growth it adds.
# Candidates are the push runs of ci.yml on main, newest first; the first whose
# head is the target or an ancestor of it and which still has the artifact
# ARTIFACT wins (a run that failed elsewhere, or is still running, can still
# hold the measurement). When the target's own run has no artifact yet and has
# not finished, the script waits up to WAIT_SECS for it; after that an older
# ancestor is used.
#
# The measurement is copied to OUT_DIR/baseline.json (the file FILE inside the
# artifact). If no run qualifies, the budgets file FALLBACK_PATH as committed
# at BASE_SHA (never the PR's copy) is written there instead. Prints `main` or
# `committed` on stdout, and a note on which baseline it is to the job summary.
# Env: GH_TOKEN, REPO, BASE_SHA, HEAD_SHA, ARTIFACT, FILE, OUT_DIR,
#      FALLBACK_PATH (default budgets.json), WORKFLOW (default ci.yml),
#      BRANCH (default main), LIMIT (default 100 runs), WAIT_SECS (default
#      3000), POLL_SECS (default 30).
set -euo pipefail
for v in GH_TOKEN REPO BASE_SHA HEAD_SHA ARTIFACT FILE OUT_DIR; do
  if [ -z "${!v:-}" ]; then
    echo "::error::$v is required" >&2
    exit 1
  fi
done
fallback_path="${FALLBACK_PATH:-budgets.json}"
workflow="${WORKFLOW:-ci.yml}"
branch="${BRANCH:-main}"
for v in LIMIT:100 WAIT_SECS:3000 POLL_SECS:30; do
  name=${v%%:*}
  val=${!name:-${v##*:}}
  case "$val" in
    '' | *[!0-9]* | ??????????*)
      echo "::error::$name must be a non-negative integer of at most 9 digits (got '$val')" >&2
      exit 1
      ;;
  esac
  printf -v "$name" '%s' "$((10#$val))"
done
deadline=$((SECONDS + WAIT_SECS))
mkdir -p "$OUT_DIR"

note() {
  echo "$1" >&2
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    printf '%s\n\n' "$1" >> "$GITHUB_STEP_SUMMARY"
  fi
}

# retry <cmd...>: three tries, 5 and 10 seconds apart (transient API errors).
retry() {
  local n=1
  until "$@"; do
    if [ "$n" -ge 3 ]; then
      return 1
    fi
    sleep $((5 * n))
    n=$((n + 1))
  done
}

target=$(git merge-base "$BASE_SHA" "$HEAD_SHA" 2>/dev/null) || target=$BASE_SHA
git cat-file -e "$target^{commit}" || {
  echo "::error::$target is not in the checkout (fetch the full history)" >&2
  exit 1
}

has_artifact() { # run id
  local n
  n=$(retry gh api "repos/$REPO/actions/runs/$1/artifacts" \
    --jq '[.artifacts[] | select(.name == env.ARTIFACT and (.expired | not))] | length') || {
    echo "::error::cannot list the artifacts of run $1" >&2
    exit 1
  }
  [ "$n" -gt 0 ]
}

found=""
while :; do
  runs=$(retry gh run list --repo "$REPO" --workflow "$workflow" --branch "$branch" \
    --event push --limit "$LIMIT" --json databaseId,headSha,status \
    --jq '.[] | [.databaseId, .headSha, .status] | @tsv')
  waiting=""
  while IFS=$'\t' read -r id sha status; do
    [ -n "$id" ] || continue
    git cat-file -e "$sha^{commit}" 2>/dev/null || continue
    git merge-base --is-ancestor "$sha" "$target" 2>/dev/null || continue
    if has_artifact "$id"; then
      found="$id $sha"
      break
    fi
    # The target's own run is still going: its measurement is the right one.
    if [ "$sha" = "$target" ] && [ "$status" != completed ] && [ "$SECONDS" -lt "$deadline" ]; then
      waiting=1
      break
    fi
  done <<< "$runs"
  if [ -n "$found" ] || [ -z "$waiting" ]; then
    break
  fi
  echo "waiting for the main run of $target" >&2
  sleep "$POLL_SECS"
done

if [ -n "$found" ]; then
  read -r run sha <<< "$found"
  dl=$(mktemp -d)
  retry gh run download "$run" --repo "$REPO" --name "$ARTIFACT" --dir "$dl" >&2
  if [ ! -f "$dl/$FILE" ]; then
    echo "::error::artifact $ARTIFACT of run $run has no $FILE" >&2
    exit 1
  fi
  cp "$dl/$FILE" "$OUT_DIR/baseline.json"
  rm -rf "$dl"
  if [ "$sha" = "$target" ]; then
    note "Baseline: the measurement of main at ${sha:0:12}, where this PR branched."
  else
    note "Baseline: the measurement of main at ${sha:0:12}, the newest one before ${target:0:12} where this PR branched (main has none for that commit)."
  fi
  echo main
else
  if ! git show "$BASE_SHA:$fallback_path" > "$OUT_DIR/baseline.json"; then
    echo "::error::no main measurement and no $fallback_path at $BASE_SHA" >&2
    exit 1
  fi
  echo "::warning::no main measurement found; using the committed $fallback_path" >&2
  note "Baseline: no main measurement was found, so this compares with the committed $fallback_path at ${BASE_SHA:0:12}, which can be stale."
  echo committed
fi
