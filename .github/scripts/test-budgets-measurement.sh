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
# jobs query taken from FAKE_JOBS (the last line repeats).
cat > "$tmp/bin/gh" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
case "$1 $2" in
  "api repos/"*"/actions/workflows/ci.yml/runs"*) printf '%s\n' "${FAKE_RUN:-}" ;;
  "api repos/"*"/jobs"*)
    n=$(cat "$FAKE_COUNT" 2>/dev/null || echo 0)
    echo $((n + 1)) > "$FAKE_COUNT"
    total=$(printf '%s\n' "$FAKE_JOBS" | wc -l)
    line=$(( n + 1 > total ? total : n + 1 ))
    printf '%s\n' "$FAKE_JOBS" | sed -n "${line}p"
    ;;
  "run download")
    shift 2
    while [ $# -gt 0 ]; do
      if [ "$1" = --dir ]; then mkdir -p "$2"; echo '{}' > "$2/budgets-current.json"; fi
      shift
    done
    ;;
  *) echo "fake gh: unexpected $*" >&2; exit 2 ;;
esac
FAKE
chmod +x "$tmp/bin/gh"
export PATH="$tmp/bin:$PATH" REPO=o/r HEAD_SHA=abc POLL_SECS=0 GH_TOKEN=x

fail() { echo "FAIL: $*" >&2; exit 1; }
case_() {
  local desc=$1 want=$2 run=$3 jobs=$4 wait=${5:-5}
  rm -rf "$tmp/out" "$tmp/count"
  local got rc=0
  got=$(FAKE_RUN="$run" FAKE_JOBS="$jobs" FAKE_COUNT="$tmp/count" OUT_DIR="$tmp/out" \
    WAIT_SECS="$wait" bash "$script" 2>/dev/null) || rc=$?
  if [ "$want" = fail ]; then
    [ "$rc" -ne 0 ] || fail "$desc: expected failure, got '$got'"
  else
    [ "$rc" -eq 0 ] || fail "$desc: exit $rc"
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
echo "budgets-measurement.sh: all cases passed"
