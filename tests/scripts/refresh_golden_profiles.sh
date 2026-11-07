#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Refresh the committed golden-profile baselines (tests/spark-golden/profiles/<tag>/) against the current tree.
#
# The baselines list the statements that fail on each Spark version's goldens (P3-T090). The nightly
# `golden-profiles` job fails only on a statement that newly fails, so as the engine improves the lists go
# stale: statements that now pass stay listed until a refresh. Run this, review `git diff --stat`, and open
# the PR (a refresh is data only; a statement that newly failed is a regression to fix, not to list: the
# update refuses to write while any failure matches no category rule).
#
#   tests/scripts/refresh_golden_profiles.sh [tag ...]      # default: every directory under profiles/
#
# Needs network the first time per tag (tests/scripts/import_spark_golden.sh fetches the upstream goldens,
# which are not committed). A release build of `golden-report` takes a while; set CARGO_TARGET_DIR to keep it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [ "$#" -gt 0 ]; then
  TAGS=("$@")
else
  mapfile -t TAGS < <(find tests/spark-golden/profiles -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
fi
[ "${#TAGS[@]}" -gt 0 ] || { echo "no profiles under tests/spark-golden/profiles" >&2; exit 1; }

for tag in "${TAGS[@]}"; do
  [ -f "tests/spark-golden/profiles/$tag/known-failures.json" ] \
    || { echo "no baseline for $tag (tests/spark-golden/profiles/$tag/known-failures.json)" >&2; exit 1; }
  if [ ! -d "tests/spark-golden/$tag/inputs" ]; then
    echo "== fetching the $tag goldens"
    bash tests/scripts/import_spark_golden.sh "$tag"
  fi
done

echo "== building golden-report"
cargo build --release -p krishiv-testkit --bin golden-report
BIN="${CARGO_TARGET_DIR:-target}/release/golden-report"

for tag in "${TAGS[@]}"; do
  echo "== $tag"
  "$BIN" --engine session --tag "$tag" --ratchet update \
    --known-failures "tests/spark-golden/profiles/$tag/known-failures.json"
done

echo "== done; review and commit:"
git status --short tests/spark-golden/profiles | head -20
git diff --stat -- tests/spark-golden/profiles | tail -1
