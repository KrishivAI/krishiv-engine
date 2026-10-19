#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Shell test for .github/scripts/budgets-measurement.sh with a fake `gh` on
# PATH. Run: bash .github/scripts/test-budgets-measurement.sh (also run by
# `cargo nextest` via xtask).
set -euo pipefail

script="$(cd "$(dirname "$0")" && pwd)/budgets-measurement.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir "$tmp/bin"
# The fake prints what the real `gh ... --jq` would: the run id for the runs
# query (FAKE_RUN, may be empty), then one `status<TAB>conclusion` line per
# jobs query taken from FAKE_JOBS (the last line repeats). A line is
# `name<TAB>status<TAB>conclusion`; one without a name is the budget job's.
cat > "$tmp/bin/gh" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
# FAKE_FAIL_TIMES calls (of any kind) fail first with FAKE_FAIL_MSG.
f=$(cat "$FAKE_COUNT.fail" 2>/dev/null || echo 0)
if [ "$f" -lt "${FAKE_FAIL_TIMES:-0}" ]; then
  echo $((f + 1)) > "$FAKE_COUNT.fail"
  echo "$FAKE_FAIL_MSG" >&2
  exit 1
fi
case "$1 $2" in
  "api repos/"*"/actions/workflows/ci.yml/runs"*) printf '%s\n' "${FAKE_RUN:-}" ;;
  "api repos/"*"/jobs"*)
    n=$(cat "$FAKE_COUNT" 2>/dev/null || echo 0)
    echo $((n + 1)) > "$FAKE_COUNT"
    total=$(printf '%s\n' "$FAKE_JOBS" | wc -l)
    line=$(( n + 1 > total ? total : n + 1 ))
    printf '%s\n' "$FAKE_JOBS" | sed -n "${line}p" |
      awk 'BEGIN { FS = OFS = "\t" } NF < 3 { $0 = "budgets-measure\t" $0 } { print }'
    ;;
  "run download")
    shift 2
    while [ $# -gt 0 ]; do
      if [ "$1" = --name ]; then name=$2; fi
      if [ "$1" = --dir ]; then mkdir -p "$2"; echo '{}' > "$2/$name.json"; fi
      shift
    done
    ;;
  *) echo "fake gh: unexpected $*" >&2; exit 2 ;;
esac
FAKE
chmod +x "$tmp/bin/gh"
# A fake sleep records its argument so backoff values can be asserted quickly.
printf '#!/usr/bin/env bash\necho "$1" >> "$FAKE_SLEEPS"\n' > "$tmp/bin/sleep"
chmod +x "$tmp/bin/sleep"
export FAKE_SLEEPS="$tmp/sleeps"
export PATH="$tmp/bin:$PATH" REPO=o/r HEAD_SHA=abc POLL_SECS=0 GH_TOKEN=x RETRY_BASE_SECS=0

fail() { echo "FAIL: $*" >&2; exit 1; }
case_() {
  local desc=$1 want=$2 run=$3 jobs=$4 wait=${5:-5}
  rm -rf "$tmp/out" "$tmp/count" "$tmp/count.fail"
  local got rc=0
  got=$(FAKE_RUN="$run" FAKE_JOBS="$jobs" FAKE_COUNT="$tmp/count" OUT_DIR="$tmp/out" \
    WAIT_SECS="$wait" bash "$script" 2>"$tmp/stderr") || rc=$?
  if [ "$want" = fail ]; then
    [ "$rc" -ne 0 ] || fail "$desc: expected failure, got '$got'"
  else
    [ "$rc" -eq 0 ] || fail "$desc: exit $rc: $(tail -3 "$tmp/stderr")"
    [ "$got" = "$want" ] || fail "$desc: got '$got', want '$want'"
  fi
  echo "ok: $desc"
}

tab=$'\t'
case_ "finished measurement is downloaded" measured 7 "completed${tab}success"
[ -f "$tmp/out/budgets-current.json" ] || fail "artifact not downloaded"
case_ "waits while the job runs" measured 7 "queued${tab}
in_progress${tab}
completed${tab}success"
case_ "skipped measurement" skipped 7 "completed${tab}skipped"
case_ "cancelled run keeps waiting" measured 7 "completed${tab}cancelled
completed${tab}success"
case_ "failed measurement fails" fail 7 "completed${tab}failure"
case_ "no run before the deadline fails" fail "" "" 0
case_ "unfinished job at the deadline fails" fail 7 "in_progress${tab}" 0

# Transient gh failures are retried; real errors are not.
err=$(mktemp)
retry_case() {
  local desc=$1 want=$2 times=$3 msg=$4
  FAKE_FAIL_TIMES=$times FAKE_FAIL_MSG=$msg case_ "$desc" "$want" 7 "completed${tab}success"
}
retry_case "HTTP 502 then success" measured 1 "gh: Server Error (HTTP 502)"
retry_case "stream CANCEL then success" measured 2 "stream error: stream ID 1; CANCEL; received from peer"
retry_case "timeout then success" measured 1 "Get https://api.github.com: i/o timeout"
retry_case "persistent 502 fails after retries" fail 99 "gh: Server Error (HTTP 502)"
[ "$(cat "$tmp/count.fail")" = 5 ] || fail "persistent 502: expected 5 attempts, got $(cat "$tmp/count.fail")"
rm -f "$tmp/count.fail"
FAKE_FAIL_TIMES=99 FAKE_FAIL_MSG="gh: Server Error (HTTP 502)" FAKE_RUN=7 FAKE_JOBS="completed${tab}success" \
  FAKE_COUNT="$tmp/count" OUT_DIR="$tmp/out" WAIT_SECS=5 bash "$script" 2>"$err" >/dev/null && fail "persistent 502 passed"
grep -q "failed transiently 5 time" "$err" || fail "persistent 502: unclear message: $(cat "$err")"
retry_case "404 fails immediately" fail 99 "gh: Not Found (HTTP 404)"
[ "$(cat "$tmp/count.fail")" = 1 ] || fail "404 was retried ($(cat "$tmp/count.fail") attempts)"
retry_case "auth error fails immediately" fail 99 "gh: Bad credentials (HTTP 401)"
[ "$(cat "$tmp/count.fail")" = 1 ] || fail "401 was retried"

# Malformed retry settings fail fast instead of silently disabling the cap.
for bad in RETRY_ATTEMPTS=abc RETRY_BASE_SECS=1.5 RETRY_ATTEMPTS=-1 WAIT_SECS=x POLL_SECS=1.5 RETRY_BASE_SECS=12345678901; do
  rc=0
  env FAKE_RUN=7 FAKE_JOBS="completed${tab}success" FAKE_COUNT="$tmp/count" \
    OUT_DIR="$tmp/out" WAIT_SECS=5 "$bad" bash "$script" >/dev/null 2>"$err" || rc=$?
  [ "$rc" -ne 0 ] || fail "$bad was accepted"
  grep -q "must be a non-negative integer" "$err" || fail "$bad: unclear message: $(cat "$err")"
  echo "ok: $bad rejected"
done

# Leading zeros are decimal, not octal (08 must neither crash nor be rejected).
rm -f "$tmp/sleeps"
FAKE_FAIL_TIMES=1 FAKE_FAIL_MSG="gh: Server Error (HTTP 502)" RETRY_BASE_SECS=08 POLL_SECS=00 \
  case_ "leading-zero settings are decimal" measured 7 "completed${tab}success" 60
[ "$(cat "$tmp/sleeps")" = 8 ] || fail "RETRY_BASE_SECS=08 did not back off 8s: $(cat "$tmp/sleeps")"

# The wheel sizes: the job lives in the called workflow, and a skipped
# `wheels-targets` (no Python change) is the only skip that means "nothing".
export JOB="wheels-targets / sizes" ARTIFACT=wheel-sizes SKIP_JOB=wheels-targets
w="wheels-targets / sizes"
case_ "wheel sizes are downloaded" measured 7 "${w}${tab}completed${tab}success"
[ -f "$tmp/out/wheel-sizes.json" ] || fail "wheel-sizes artifact not downloaded"
case_ "wheel sizes: waits for the job" measured 7 "${w}${tab}in_progress${tab}
${w}${tab}completed${tab}success"
case_ "skipped wheels workflow" skipped 7 "wheels-targets${tab}completed${tab}skipped"
case_ "skipped sizes job (a wheel build failed) fails" fail 7 "${w}${tab}completed${tab}skipped"
case_ "failed sizes job fails" fail 7 "${w}${tab}completed${tab}failure"
case_ "no wheels job yet at the deadline fails" fail 7 "" 0
echo "budgets-measurement.sh: all cases passed"
