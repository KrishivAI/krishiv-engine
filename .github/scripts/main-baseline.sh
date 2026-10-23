#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Selects the baseline a measurement is compared with: the measurement that
# main-measure.yml made on main for exactly the commit TARGET_SHA (the main
# commit the measured tree was built on top of), or, when main has no
# measurement for that commit, for the newest main commit before it. Runs in a
# checkout with the full history of main.
#
# TARGET_SHA is given directly, or read from the file TARGET_SHA_FILE (the
# `measured-base-sha` a pull request's measurement job recorded), so a PR is
# charged only for its own diff. Candidates are the push runs of WORKFLOW on
# BRANCH, newest first; the first whose head is the target or an ancestor of
# it and which still has the artifact ARTIFACT wins. A run that is still going
# for the target itself is waited for (WAIT_SECS); if it has not finished by
# then the script fails (re-run the job later) instead of using an older
# measurement.
#
# The measurement is copied to OUT_DIR/baseline.json (the file FILE inside the
# artifact). The committed budgets file FALLBACK_PATH at BASE_SHA (never the
# PR's copy) is used instead only when no measurement of this kind exists in
# the repository at all (the first runs after this gate was introduced). Once
# one exists, failing to find one for the target is an error with the reason:
# the artifacts are kept 90 days, so a branch that old must merge main.
# Prints `main` or `committed` on stdout, and a note on which baseline it is
# to the job summary.
# Env: GH_TOKEN, REPO, BASE_SHA, TARGET_SHA | TARGET_SHA_FILE, ARTIFACT, FILE,
#      OUT_DIR, FALLBACK_PATH (default budgets.json), WORKFLOW (default
#      main-measure.yml), BRANCH (default main), LIMIT (default 1000 runs),
#      WAIT_SECS (default 600), POLL_SECS (default 30).
set -euo pipefail
for v in GH_TOKEN REPO BASE_SHA ARTIFACT FILE OUT_DIR; do
  if [ -z "${!v:-}" ]; then
    echo "::error::$v is required" >&2
    exit 1
  fi
done
fallback_path="${FALLBACK_PATH:-budgets.json}"
workflow="${WORKFLOW:-main-measure.yml}"
branch="${BRANCH:-main}"
for v in LIMIT:1000 WAIT_SECS:600 POLL_SECS:30; do
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

target=${TARGET_SHA:-}
if [ -z "$target" ] && [ -n "${TARGET_SHA_FILE:-}" ]; then
  if [ ! -f "$TARGET_SHA_FILE" ]; then
    echo "::error::the measurement has no $(basename "$TARGET_SHA_FILE"); re-run the pull request's CI" >&2
    exit 1
  fi
  target=$(tr -d '[:space:]' < "$TARGET_SHA_FILE")
fi
if [[ ! $target =~ ^[0-9a-f]{40}$ ]]; then
  echo "::error::TARGET_SHA (or TARGET_SHA_FILE) must hold a full commit id (got '$target')" >&2
  exit 1
fi

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

# The target may be newer than the checkout (main moved after it).
if ! git cat-file -e "$target^{commit}" 2>/dev/null; then
  git fetch --no-tags -q origin "$target" 2>/dev/null || true
fi
git cat-file -e "$target^{commit}" || {
  echo "::error::$target is not in the checkout and could not be fetched" >&2
  exit 1
}

# live | expired | empty: the state of ARTIFACT in run $1.
artifact_state() {
  local state
  state=$(retry gh api "repos/$REPO/actions/runs/$1/artifacts" \
    --jq '[.artifacts[] | select(.name == env.ARTIFACT) | if .expired then "expired" else "live" end] | sort | first // ""') || {
    echo "::error::cannot list the artifacts of run $1" >&2
    exit 1
  }
  printf '%s' "$state"
}

found="" expired="" pending=""
while :; do
  runs=$(retry gh run list --repo "$REPO" --workflow "$workflow" --branch "$branch" \
    --event push --limit "$LIMIT" --json databaseId,headSha,status \
    --jq '.[] | [.databaseId, .headSha, .status] | @tsv')
  pending=""
  while IFS=$'\t' read -r id sha status; do
    [ -n "$id" ] || continue
    git cat-file -e "$sha^{commit}" 2>/dev/null || continue
    git merge-base --is-ancestor "$sha" "$target" 2>/dev/null || continue
    state=$(artifact_state "$id")
    if [ "$state" = live ]; then
      found="$id $sha"
      break
    fi
    [ "$state" = expired ] && expired=$sha
    # The target's own run is still going: its measurement is the right one.
    if [ "$sha" = "$target" ] && [ "$status" != completed ]; then
      pending=$id
      break
    fi
  done <<< "$runs"
  if [ -n "$found" ] || [ -z "$pending" ]; then
    break
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "::error::the main run $pending for $target (the commit this was measured against) has not finished after ${WAIT_SECS}s; re-run this job when it has" >&2
    exit 1
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
    note "Baseline: the measurement of main at ${sha:0:12}, the commit this was measured against."
  else
    note "Baseline: the measurement of main at ${sha:0:12}, the newest one before ${target:0:12}, the commit this was measured against (main has none for that commit)."
  fi
  echo main
  exit 0
fi

# Nothing usable for the target. Only a repository that has never recorded a
# measurement may use the committed file.
total=$(retry gh api "repos/$REPO/actions/artifacts?name=$ARTIFACT&per_page=1" --jq '.total_count') || {
  echo "::error::cannot list the $ARTIFACT artifacts" >&2
  exit 1
}
if [ "$total" -gt 0 ]; then
  if [ -n "$expired" ]; then
    echo "::error::the main measurement for ${expired:0:12} has expired (artifacts are kept 90 days); merge main into your branch and push to get a fresh baseline" >&2
  else
    echo "::error::no main measurement at or before ${target:0:12} was found among the last $LIMIT main runs; merge main into your branch and push to get a baseline" >&2
  fi
  exit 1
fi
if ! git show "$BASE_SHA:$fallback_path" > "$OUT_DIR/baseline.json"; then
  echo "::error::no main measurement and no $fallback_path at $BASE_SHA" >&2
  exit 1
fi
echo "::warning::no main measurement exists yet; using the committed $fallback_path" >&2
note "Baseline: main has not recorded a measurement yet, so this compares with the committed $fallback_path at ${BASE_SHA:0:12}, which can be stale."
echo committed
