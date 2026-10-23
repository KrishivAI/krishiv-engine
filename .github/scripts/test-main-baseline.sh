#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Shell test for .github/scripts/main-baseline.sh with a fake `gh` on PATH and a
# real scratch git repository. Run: bash .github/scripts/test-main-baseline.sh
# (also run by `cargo nextest` via xtask).
set -euo pipefail

script="$(cd "$(dirname "$0")" && pwd)/main-baseline.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir "$tmp/bin"

# The fake serves what the real calls would print after `--jq`:
#   run list     -> FAKE_RUNS_FILE (one `id<TAB>sha<TAB>status` per line)
#   run artifacts -> live | expired | nothing, from FAKE_ARTIFACTS_FILE;
#                    the repository-wide count is its line count
#   run download -> a file named by FAKE_FILE holding {"run": <id>}
cat > "$tmp/bin/gh" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
case "$1 $2" in
  "run list") cat "$FAKE_RUNS_FILE" ;;
  "api repos/"*"/actions/artifacts?"*) grep -c . "$FAKE_ARTIFACTS_FILE" || true ;;
  "api repos/"*"/artifacts")
    id=${2#repos/*/actions/runs/}
    id=${id%/artifacts}
    # `ID` is a live artifact, `ID:expired` an expired one; none prints nothing.
    if grep -qx "$id" "$FAKE_ARTIFACTS_FILE"; then echo live
    elif grep -qx "$id:expired" "$FAKE_ARTIFACTS_FILE"; then echo expired; fi
    ;;
  "run download")
    id=$3
    shift 3
    while [ $# -gt 0 ]; do
      if [ "$1" = --dir ]; then
        mkdir -p "$2"
        echo "{\"run\": $id}" > "$2/$FAKE_FILE"
      fi
      shift
    done
    ;;
  *) echo "fake gh: unexpected $*" >&2; exit 2 ;;
esac
FAKE
chmod +x "$tmp/bin/gh"
# A fake sleep lets a waiting test finish the pending run.
cat > "$tmp/bin/sleep" <<'FAKE'
#!/usr/bin/env bash
echo "$1" >> "$FAKE_SLEEPS"
if [ -n "${FAKE_FINISH_RUN:-}" ] && [ -f "$FAKE_ARTIFACTS_FILE" ]; then
  echo "$FAKE_FINISH_RUN" >> "$FAKE_ARTIFACTS_FILE"
fi
FAKE
chmod +x "$tmp/bin/sleep"
export FAKE_SLEEPS="$tmp/sleeps" FAKE_RUNS_FILE="$tmp/runs" FAKE_ARTIFACTS_FILE="$tmp/artifacts"
export PATH="$tmp/bin:$PATH" GH_TOKEN=x REPO=o/r ARTIFACT=budgets-main FILE=budgets-current.json
export FAKE_FILE=budgets-current.json OUT_DIR="$tmp/out" POLL_SECS=0

# History: m1 - m2 - m3 (main); the PR branches from m2 (p1), and main moves on to m3.
git init -q "$tmp/repo"
cd "$tmp/repo"
git config user.email t@example.com
git config user.name t
git config commit.gpgsign false
commit() { echo "$1" > f; git add f; git commit -q -m "$1"; git rev-parse HEAD; }
echo '{"committed": true}' > budgets.json
git add budgets.json
m1=$(commit m1)
m2=$(commit m2)
git checkout -q -b pr
p1=$(commit p1)
git checkout -q -
m3=$(commit m3)
git branch -f main "$m3" 2>/dev/null || true
# The PR was measured on top of m2 (the first parent of its merge commit).
export BASE_SHA=$m3 TARGET_SHA=$m2
# A budgets.json the PR rewrote must never be used as the baseline: the
# fallback reads the base commit's file.
git checkout -q pr
echo '{"pr": true}' > budgets.json
git commit -q -am "pr edits budgets.json"

fail() { echo "FAIL: $*" >&2; exit 1; }
tab=$'\t'
case_() { # desc want-stdout runs artifacts [extra env...]
  local desc=$1 want=$2 runs=$3 arts=$4
  shift 4
  printf '%b' "$runs" > "$FAKE_RUNS_FILE"
  printf '%b' "$arts" > "$FAKE_ARTIFACTS_FILE"
  rm -rf "$OUT_DIR" "$FAKE_SLEEPS"
  local got rc=0
  got=$(env "$@" WAIT_SECS=0 bash "$script" 2>"$tmp/stderr") || rc=$?
  if [ "$want" = fail ]; then
    [ "$rc" -ne 0 ] || fail "$desc: expected failure, got '$got'"
  else
    [ "$rc" -eq 0 ] || fail "$desc: exit $rc: $(tail -3 "$tmp/stderr")"
    [ "$got" = "$want" ] || fail "$desc: got '$got', want '$want'"
  fi
  echo "ok: $desc"
}

# The PR was measured against m2: m3's run is newer but is not an ancestor of
# the target, so it must be skipped.
runs="103${tab}${m3}${tab}completed\n102${tab}${m2}${tab}completed\n101${tab}${m1}${tab}completed\n"
case_ "the run of the target is used, not a newer one" main "$runs" "103\n102\n101\n"
[ "$(jq -r .run "$OUT_DIR/baseline.json")" = 102 ] || fail "wrong run downloaded: $(cat "$OUT_DIR/baseline.json")"

case_ "the newest ancestor is used when the target has no run" main \
  "103${tab}${m3}${tab}completed\n101${tab}${m1}${tab}completed\n" "103\n101\n"
[ "$(jq -r .run "$OUT_DIR/baseline.json")" = 101 ] || fail "wrong run downloaded"

case_ "a run without the artifact (failed elsewhere) is skipped" main "$runs" "103\n101\n"
[ "$(jq -r .run "$OUT_DIR/baseline.json")" = 101 ] || fail "wrong run downloaded"

case_ "a failed run that still has the artifact counts" main \
  "102${tab}${m2}${tab}completed\n" "102\n"

case_ "runs unknown to the checkout are skipped" main \
  "104${tab}0000000000000000000000000000000000000001${tab}completed\n101${tab}${m1}${tab}completed\n" "104\n101\n"
[ "$(jq -r .run "$OUT_DIR/baseline.json")" = 101 ] || fail "wrong run downloaded"

# The committed file is the fallback only while main has never recorded one.
case_ "no measurement ever falls back to the base's budgets.json" committed "" ""
grep -q committed "$OUT_DIR/baseline.json" || fail "fallback is not the base's file: $(cat "$OUT_DIR/baseline.json")"
case_ "runs without any artifact fall back" committed "$runs" ""
case_ "fallback without a budgets file fails" fail "" "" FALLBACK_PATH=nope.json
# Once main has measurements, a missing one for the target fails closed.
case_ "only newer measurements: fails closed" fail "103${tab}${m3}${tab}completed\n" "103\n"
grep -q "no main measurement at or before ${m2:0:12}" "$tmp/stderr" || fail "unclear message: $(cat "$tmp/stderr")"
case_ "expired measurement: fails with the reason" fail "$runs" "102:expired\n"
grep -q "has expired" "$tmp/stderr" || fail "unclear message: $(cat "$tmp/stderr")"
grep -q "merge main" "$tmp/stderr" || fail "no advice: $(cat "$tmp/stderr")"

# Summary notes say which baseline was used.
summary="$tmp/summary.md"
rm -f "$summary"
case_ "summary names the main baseline" main "$runs" "102\n" GITHUB_STEP_SUMMARY="$summary"
grep -q "measurement of main at ${m2:0:12}, the commit this was measured against" "$summary" || fail "summary: $(cat "$summary")"
rm -f "$summary"
case_ "summary names the older main baseline" main "101${tab}${m1}${tab}completed\n" "101\n" GITHUB_STEP_SUMMARY="$summary"
grep -q "newest one before ${m2:0:12}" "$summary" || fail "summary: $(cat "$summary")"
rm -f "$summary"
case_ "summary warns about the fallback" committed "" "" GITHUB_STEP_SUMMARY="$summary"
grep -q "can be stale" "$summary" || fail "summary: $(cat "$summary")"

# The target's own run is still going: it is waited for, then used. Past the
# deadline the job fails rather than silently using an older measurement.
pending="102${tab}${m2}${tab}in_progress\n101${tab}${m1}${tab}completed\n"
printf '%b' "$pending" > "$FAKE_RUNS_FILE"
printf '101\n' > "$FAKE_ARTIFACTS_FILE"
rm -rf "$OUT_DIR" "$FAKE_SLEEPS"
got=$(FAKE_FINISH_RUN=102 WAIT_SECS=60 bash "$script" 2>/dev/null)
[ "$got" = main ] || fail "wait: got '$got'"
[ "$(jq -r .run "$OUT_DIR/baseline.json")" = 102 ] || fail "wait: did not use the awaited run"
[ -f "$FAKE_SLEEPS" ] || fail "wait: never slept"
echo "ok: waits for the target's run"
case_ "gives up waiting and fails" fail "$pending" "101\n"
grep -q "has not finished" "$tmp/stderr" || fail "unclear message: $(cat "$tmp/stderr")"
[ ! -f "$FAKE_SLEEPS" ] || fail "WAIT_SECS=0 must not sleep"
# A finished run without the artifact (nothing measured) is skipped, not awaited.
case_ "a completed run without the artifact is skipped" main \
  "102${tab}${m2}${tab}completed\n101${tab}${m1}${tab}completed\n" "101\n"

# The target comes from a file the measurement recorded.
echo "$m2" > "$tmp/measured-base-sha"
case_ "target read from a file" main "$runs" "102\n" TARGET_SHA= TARGET_SHA_FILE="$tmp/measured-base-sha"
[ "$(jq -r .run "$OUT_DIR/baseline.json")" = 102 ] || fail "wrong run downloaded"
case_ "missing target file fails" fail "$runs" "102\n" TARGET_SHA= TARGET_SHA_FILE="$tmp/none"
echo "not-a-sha" > "$tmp/bad-sha"
case_ "malformed target fails" fail "$runs" "102\n" TARGET_SHA= TARGET_SHA_FILE="$tmp/bad-sha"
case_ "no target fails" fail "$runs" "102\n" TARGET_SHA=

# Inputs are validated.
case_ "missing input fails" fail "$runs" "102\n" BASE_SHA=
case_ "bad WAIT_SECS fails" fail "$runs" "102\n" LIMIT=x
case_ "artifact without the file fails" fail "$runs" "102\n" FILE=other.json FAKE_FILE=x.json
echo "all main-baseline checks passed"
