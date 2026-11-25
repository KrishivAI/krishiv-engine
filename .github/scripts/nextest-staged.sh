#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Workspace nextest run in stages, deleting each stage's test executables before the next builds.
# One run builds every test executable at once (about 560; 116 GB of target on 2026-11-24, "No space left on
# device"). Usage: nextest-staged.sh [cargo nextest run args, e.g. --all-features --profile nightly]
# The package selection stays `--workspace`, so feature unification is the one a single run has. A `--test` glob
# matching a target whose `required-features` are off is an error, so the stages name their targets: every
# integration test whose `required-features` the workspace resolve has on, grouped by first letter.
set -euo pipefail
extra=("$@")
meta="${RUNNER_TEMP:-/tmp}/meta.json"
stage() {
  echo "::group::nextest stage ($# args)"
  cargo nextest run --workspace --locked --no-tests=pass "${extra[@]}" "$@"
  echo "::endgroup::"
  find target/debug/deps -maxdepth 1 -type f -perm -u+x ! -name '*.so' -delete
  df -h /
}
df -h /
# krishiv-python is test = false (its cdylib cannot link as a test harness; --lib ignores the flag); its integration
# tests still run in the later stages.
stage --lib --bins --exclude krishiv-python
# shellcheck disable=SC2086
cargo metadata --format-version 1 --locked ${META_ARGS:-} > "$meta"
for range in a-c d-l m-r s t-z; do
  args=$(python3 -I - "$range" "$meta" << 'PY'
import json, re, sys
rng, path = sys.argv[1], sys.argv[2]
meta = json.load(open(path))
on = {n["id"]: set(n["features"]) for n in meta["resolve"]["nodes"]}
names = sorted({t["name"] for p in meta["packages"] if p["id"] in meta["workspace_members"] for t in p["targets"]
                if "test" in t["kind"] and set(t.get("required-features") or []) <= on[p["id"]]
                and re.match("[" + rng + "]", t["name"])})
print(" ".join("--test " + n for n in names))
PY
  )
  [ -n "$args" ] || { echo "no test targets in $range"; continue; }
  # shellcheck disable=SC2086
  stage $args
done
