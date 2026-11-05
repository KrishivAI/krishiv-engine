#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Shell test for .github/scripts/approval-label.sh and approval-values.sh with a fake `gh` on PATH.
# Run: bash .github/scripts/test-approval-label.sh (also run by `cargo nextest` via xtask).
set -euo pipefail

dir="$(cd "$(dirname "$0")" && pwd)"
script="$dir/approval-label.sh"
values_script="$dir/approval-values.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir "$tmp/bin"

# The fake serves what the real calls print after `--jq`, from JSON fixtures:
#   issues/N/events -> $FAKE_EVENTS, issues/N/comments -> $FAKE_COMMENTS (GET) or appended to
#   $FAKE_POSTS (POST, -f body=...), actions/runs -> $FAKE_RUNS, check-suites -> $FAKE_SUITES.
cat > "$tmp/bin/gh" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
[ "$1" = api ] || exit 2
path=$2
shift 2
filter=""
body=""
post=false
while [ $# -gt 0 ]; do
  case "$1" in
    --jq) filter=$2; shift ;;
    -f) post=true; body=${2#body=}; shift ;;
    *) ;;
  esac
  shift
done
if [ "$post" = true ]; then
  [ "${FAKE_POST_FAILS:-false}" = true ] && exit 1
  printf '%s\n=====\n' "$body" >> "$FAKE_POSTS"
  exit 0
fi
case "$path" in
  */events*) file=$FAKE_EVENTS ;;
  */comments*) file=$FAKE_COMMENTS ;;
  */actions/runs?branch=*) file="$FAKE_RUNS.branch" ;;
  */actions/runs*)
    sha=${path#*head_sha=}
    sha=${sha%%&*}
    file="$FAKE_RUNS.$sha"
    [ -f "$file" ] || file="$FAKE_RUNS.none"
    ;;
  */check-suites*) file=$FAKE_SUITES ;;
  */pulls/*)
    case "$filter" in
      *.head.ref*) echo feature ;;
      *) echo "${FAKE_CURRENT_HEAD:-abc}" ;;
    esac
    exit 0 ;;
  *) exit 2 ;;
esac
jq -r "$filter" "$file"
FAKE
chmod +x "$tmp/bin/gh"
export PATH="$tmp/bin:$PATH"
export GH_TOKEN=x REPO=o/r PR=7 HEAD_SHA=abc APPROVER=maintainer HAS_LABEL=true
export FAKE_EVENTS="$tmp/events.json" FAKE_COMMENTS="$tmp/comments.json" FAKE_RUNS="$tmp/runs.json" \
  FAKE_SUITES="$tmp/suites.json" FAKE_POSTS="$tmp/posts.txt"

fails=0
LABELED=2026-11-02T10:00:00Z

# events LABELER LABELED_AT [FORCED_AT]
events() {
  local extra=""
  [ -n "${3:-}" ] && extra=",{\"event\":\"head_ref_force_pushed\",\"actor\":{\"login\":\"dev\"},\"created_at\":\"$3\"}"
  cat > "$FAKE_EVENTS" <<JSON
[{"event":"labeled","label":{"name":"L"},"actor":{"login":"$1"},"created_at":"$2"}$extra]
JSON
}
# runs HEAD_ARRIVED_AT: when the PR's head (abc) arrived; the earlier head "old" always arrived
# before the label
runs() {
  echo "{\"workflow_runs\":[{\"created_at\":\"$1\"}]}" > "$FAKE_RUNS.abc"
  echo "{\"workflow_runs\":[{\"created_at\":\"2026-11-02T08:00:00Z\"}]}" > "$FAKE_RUNS.old"
  echo "{\"workflow_runs\":[{\"created_at\":\"2026-11-02T10:30:00Z\"}]}" > "$FAKE_RUNS.late"
  echo '{"workflow_runs":[]}' > "$FAKE_RUNS.none"
  # The runs on the PR's branch: the label's own run (on the head it was applied to) and,
  # when the head arrived after the label, the push that brought it.
  if [[ "$1" > "$LABELED" ]]; then
    echo "{\"workflow_runs\":[{\"created_at\":\"2026-11-02T10:00:05Z\",\"head_sha\":\"old\"},{\"created_at\":\"$1\",\"head_sha\":\"abc\"}]}" > "$FAKE_RUNS.branch"
  else
    echo "{\"workflow_runs\":[{\"created_at\":\"$1\",\"head_sha\":\"abc\"}]}" > "$FAKE_RUNS.branch"
  fi
  echo '{"check_suites":[]}' > "$FAKE_SUITES"
}
# comment AUTHOR LABELED_AT NAME VALUE...  (a record as the workflow writes it: scope "default",
# measured on the earlier head "old"); RECORD_SCOPE and RECORD_HEAD override those. Sets the only
# comment; add_comment appends one.
record_json() {
  local author=$1 at=$2
  shift 2
  local lines=""
  while [ $# -ge 2 ]; do lines="$lines$1 $2\n"; shift 2; done
  local body="<!-- krishiv-approval:v1\nlabel=L\nscope=${RECORD_SCOPE:-default}\nhead=${RECORD_HEAD-old}\nlabeled_at=$at\n${lines}-->\nRecorded."
  printf '{"user":{"login":"%s"},"created_at":"%s","body":"%s"}' "$author" "${RECORD_AT:-2026-11-02T10:00:10Z}" "$body"
}
comment() { printf '[%s]\n' "$(record_json "$@")" > "$FAKE_COMMENTS"; }
add_comment() {
  local old
  old=$(sed 's/^\[//; s/\]$//' "$FAKE_COMMENTS")
  printf '[%s,%s]\n' "$old" "$(record_json "$@")" > "$FAKE_COMMENTS"
}
no_comments() { echo '[]' > "$FAKE_COMMENTS"; }
values() { printf '%s\n' "$@" > "$tmp/values"; }

# check NAME EXPECTED [ENV=VALUE...]
check() {
  local name=$1 want=$2
  shift 2
  local got
  got=$(env "$@" "$script" L 2>/dev/null) || got="error"
  if [ "$got" = "$want" ]; then
    echo "ok   $name"
  else
    echo "FAIL $name: expected $want, got $got"
    fails=$((fails + 1))
  fi
}
posts() { [ -f "$FAKE_POSTS" ] && grep -c '^=====$' "$FAKE_POSTS" || echo 0; }

HEAD_BEFORE=2026-11-02T09:00:00Z   # the head arrived before the label
HEAD_AFTER=2026-11-02T11:00:00Z    # a later push

# --- the approval rule without recorded values (as before) ---------------------------------
events maintainer $LABELED; runs $HEAD_BEFORE; no_comments
check "label applied after the head arrived counts" 1
check "no label" 0 HAS_LABEL=false
events someone-else $LABELED
check "label by another actor is ignored" 0
events maintainer $LABELED; runs $HEAD_AFTER
check "a later push ends the approval without recorded values" 0
events maintainer $LABELED; runs $LABELED
check "same-second arrival is not after" 0
check "an empty approver approves nothing" 0 APPROVER=

# --- recording --------------------------------------------------------------------------------
rm -f "$FAKE_POSTS"
events maintainer $LABELED; runs $HEAD_BEFORE; no_comments; values "counted_lines 1000"
check "fresh approval counts and is recorded" 1 VALUES_FILE="$tmp/values"
[ "$(posts)" = 1 ] && grep -q "^labeled_at=$LABELED$" "$FAKE_POSTS" && grep -q '^counted_lines 1000$' "$FAKE_POSTS" \
  && echo "ok   the record carries the label event and the value" || { echo "FAIL the record"; fails=$((fails + 1)); }
rm -f "$FAKE_POSTS"
comment github-actions[bot] $LABELED counted_lines 1000
check "an existing record is not written twice" 1 VALUES_FILE="$tmp/values"
[ "$(posts)" = 0 ] && echo "ok   no second record" || { echo "FAIL second record"; fails=$((fails + 1)); }
rm -f "$FAKE_POSTS"; no_comments
check "no write access: fresh approval still counts" 1 VALUES_FILE="$tmp/values" FAKE_POST_FAILS=true
check "RECORD=false writes nothing" 1 VALUES_FILE="$tmp/values" RECORD=false
[ "$(posts)" = 0 ] && echo "ok   RECORD=false wrote nothing" || { echo "FAIL RECORD=false"; fails=$((fails + 1)); }

# --- staying valid across pushes -----------------------------------------------------------
events maintainer $LABELED; runs $HEAD_AFTER
comment github-actions[bot] $LABELED counted_lines 1000
values "counted_lines 1000"; check "unchanged value after a push" 1 VALUES_FILE="$tmp/values"
values "counted_lines 900";  check "a smaller value after a push" 1 VALUES_FILE="$tmp/values"
values "counted_lines 1050"; check "exactly 5% over still counts" 1 VALUES_FILE="$tmp/values"
values "counted_lines 1051"; check "just over 5% ends it" 0 VALUES_FILE="$tmp/values"
values "counted_lines 1051"; check "a wider tolerance is honoured" 1 VALUES_FILE="$tmp/values" TOLERANCE_PERCENT=10
check "the label removed ends it" 0 VALUES_FILE="$tmp/values" HAS_LABEL=false
values "counted_lines 1000"
check "a metric outside the filter is ignored" 1 VALUES_FILE="$tmp/values" METRIC_FILTER='^other$'
values "counted_lines 1000" "new_metric 5"
check "a metric not recorded is not covered" 0 VALUES_FILE="$tmp/values"
values "counted_lines 1000" "new_metric 0"
check "a metric not recorded but zero is covered" 1 VALUES_FILE="$tmp/values"
comment someone-else $LABELED counted_lines 1000
values "counted_lines 1000"
check "a record by a user (not the workflow) is ignored" 0 VALUES_FILE="$tmp/values"
comment github-actions[bot] 2026-11-01T00:00:00Z counted_lines 1000
check "a record for an earlier application of the label is ignored" 0 VALUES_FILE="$tmp/values"
comment github-actions[bot] $LABELED counted_lines 1000
events maintainer $LABELED 2026-11-02T10:30:00Z; runs 2026-11-02T10:30:00Z
check "a force push after the label ends it" 0 VALUES_FILE="$tmp/values"
events maintainer $LABELED 2026-11-02T09:30:00Z; runs $HEAD_AFTER
check "a force push before the label does not" 1 VALUES_FILE="$tmp/values"
events someone-else $LABELED; runs $HEAD_AFTER
check "a record does not help a label applied by someone else" 0 VALUES_FILE="$tmp/values"
events maintainer $LABELED; runs $HEAD_AFTER; no_comments
check "no record: a later push ends it" 0 VALUES_FILE="$tmp/values"

# --- strict (unsafe): any growth ends it ------------------------------------------------------
comment github-actions[bot] $LABELED unsafe:a 2 unsafe:b 5
values "unsafe:a 2" "unsafe:b 5";  check "strict: same counts" 1 VALUES_FILE="$tmp/values" STRICT=true METRIC_FILTER='^unsafe:'
values "unsafe:a 1" "unsafe:b 5";  check "strict: fewer sites" 1 VALUES_FILE="$tmp/values" STRICT=true METRIC_FILTER='^unsafe:'
values "unsafe:a 3" "unsafe:b 5";  check "strict: one more site ends it" 0 VALUES_FILE="$tmp/values" STRICT=true METRIC_FILTER='^unsafe:'
values "unsafe:a 2" "unsafe:b 5" "unsafe:c 1"
check "strict: a crate with new unsafe ends it" 0 VALUES_FILE="$tmp/values" STRICT=true METRIC_FILTER='^unsafe:'
values "unsafe:a 2" "unsafe:b 5" "unsafe:c 0"
check "strict: a crate with none is fine" 1 VALUES_FILE="$tmp/values" STRICT=true METRIC_FILTER='^unsafe:'
values "unsafe:a 2" "unsafe:b 5"
check "the ratio rule would allow +5% where strict does not" 1 VALUES_FILE="$tmp/values" METRIC_FILTER='^unsafe:'

# --- scopes, forgery and races ---------------------------------------------------------------
events maintainer $LABELED; runs $HEAD_AFTER
comment github-actions[bot] $LABELED counted_lines 1000
values "counted_lines 1000"
check "a record of another scope does not cover this gate" 0 VALUES_FILE="$tmp/values" APPROVAL_SCOPE=wheels
RECORD_SCOPE=wheels comment github-actions[bot] $LABELED counted_lines 1000
check "a record of this scope covers it" 1 VALUES_FILE="$tmp/values" APPROVAL_SCOPE=wheels
comment github-actions[bot] $LABELED counted_lines 1000
add_comment github-actions[bot] $LABELED counted_lines 999999
values "counted_lines 500000"
check "a later record cannot replace the first" 0 VALUES_FILE="$tmp/values"
RECORD_HEAD=late comment github-actions[bot] $LABELED counted_lines 999999
check "a record naming a head that arrived after the label is ignored" 0 VALUES_FILE="$tmp/values"
RECORD_HEAD= comment github-actions[bot] $LABELED counted_lines 999999
check "a record without a head is ignored" 0 VALUES_FILE="$tmp/values"
RECORD_HEAD=unknown comment github-actions[bot] $LABELED counted_lines 999999
check "a record naming an unknown head is ignored" 0 VALUES_FILE="$tmp/values"
# a fake forged record first, the real one after: the first valid one counts, the forged is invalid
RECORD_HEAD=late comment github-actions[bot] $LABELED counted_lines 999999
add_comment github-actions[bot] $LABELED counted_lines 1000
values "counted_lines 500000"
check "a forged first record (late head) does not hide the real one, which is too small" 0 VALUES_FILE="$tmp/values"
values "counted_lines 1000"
check "and the real one still covers" 1 VALUES_FILE="$tmp/values"
# A forged record can name the genuine head; it still comes after the push that replaced it.
RECORD_AT=2026-11-02T11:05:00Z comment github-actions[bot] $LABELED counted_lines 999999
values "counted_lines 500000"
check "a record written after a later push is ignored, whatever head it names" 0 VALUES_FILE="$tmp/values"
RECORD_AT=2026-11-02T10:59:00Z comment github-actions[bot] $LABELED counted_lines 1000
values "counted_lines 1000"
check "a record written before the later push counts" 1 VALUES_FILE="$tmp/values"
# recording only for the PR's current head
rm -f "$FAKE_POSTS"
events maintainer $LABELED; runs $HEAD_BEFORE; no_comments; values "counted_lines 1000"
check "an older head's run still counts the fresh label" 1 VALUES_FILE="$tmp/values" FAKE_CURRENT_HEAD=newer
[ "$(posts)" = 0 ] && echo "ok   but records nothing for a head that is no longer current" || { echo "FAIL stale-head record"; fails=$((fails + 1)); }
rm -f "$FAKE_POSTS"
check "the current head records" 1 VALUES_FILE="$tmp/values"
grep -q '^head=abc$' "$FAKE_POSTS" && grep -q '^scope=default$' "$FAKE_POSTS" \
  && echo "ok   the record names its head and scope" || { echo "FAIL record head/scope"; fails=$((fails + 1)); }

# --- approval-values.sh ----------------------------------------------------------------------
cat > "$tmp/b.json" <<JSON
{"transitive_crates":136,"binary_bytes":26620856,"clean_build_secs":null,"unsafe_sites":{"a":18,"b":2},"wheel_bytes":{"w":1}}
JSON
want=$'transitive_crates 136\nbinary_bytes 26620856\nunsafe:a 18\nunsafe:b 2'
[ "$("$values_script" budgets "$tmp/b.json")" = "$want" ] && echo "ok   budget values" || { echo "FAIL budget values"; fails=$((fails + 1)); }
echo '{"x86":10,"arm":null,"mac":7}' > "$tmp/w.json"
[ "$("$values_script" wheels "$tmp/w.json")" = $'wheel:x86 10\nwheel:mac 7' ] && echo "ok   wheel values" || { echo "FAIL wheel values"; fails=$((fails + 1)); }
echo '{"transitive_crates":1,"binary_bytes":null}' > "$tmp/c.json"
[ "$("$values_script" budgets "$tmp/c.json")" = "transitive_crates 1" ] && echo "ok   unmeasured values are left out" || { echo "FAIL unmeasured"; fails=$((fails + 1)); }

if [ "$fails" -ne 0 ]; then
  echo "$fails check(s) failed"
  exit 1
fi
echo "all checks passed"
