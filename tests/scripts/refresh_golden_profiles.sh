#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Refresh the committed golden-profile baselines (tests/spark-golden/profiles/<tag>/) against the current tree.
#
# The baselines list the statements that fail on each Spark version's goldens (P3-T090). The nightly
# `golden-profiles` job fails only on a statement that newly fails, so as the engine improves the lists go
# stale: statements that now pass stay listed until a refresh. Run this, review `git diff --stat`, and open
# the PR. A refresh is data only: a statement that newly fails is a regression to fix, not to list, so the
# script stops before writing anything for a tag when the run has an unlisted failure (set
# ALLOW_NEW_FAILURES=1 only for a failure that is understood and given a category on purpose).
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

export TZ=UTC  # the cargo test environment forces it; the binary is run directly here

for tag in "${TAGS[@]}"; do
  KF="tests/spark-golden/profiles/$tag/known-failures.json"
  echo "== $tag"
  # `check` exits non-zero when a listed statement now passes (the expected drift), so only its text is used:
  # an unlisted failure is a regression, and it must not be written into the baseline.
  CHECK="$("$BIN" --engine session --tag "$tag" --ratchet check --known-failures "$KF" 2>&1 || true)"
  if grep -q 'fail that the baseline does not list' <<<"$CHECK" && [ "${ALLOW_NEW_FAILURES:-0}" != 1 ]; then
    echo "$tag: statements fail that the baseline does not list (regressions, not refreshed):" >&2
    sed -n '/fail that the baseline does not list/,/now pass/p' <<<"$CHECK" | grep -v 'now pass' | head -40 >&2
    exit 1
  fi
  "$BIN" --engine session --tag "$tag" --ratchet update --known-failures "$KF"
done

echo "== done; review and commit:"
git status --short tests/spark-golden/profiles | head -20
git diff --stat -- tests/spark-golden/profiles | tail -1
