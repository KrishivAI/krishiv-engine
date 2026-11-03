#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Prints the measured values an approval label is recorded against, one `name value` per line
# (integers), for .github/scripts/approval-label.sh (its VALUES_FILE).
#   approval-values.sh budgets FILE   FILE is a budgets measurement (budgets-current.json):
#                                     transitive_crates, binary_bytes, unsafe:<crate>
#   approval-values.sh wheels FILE    FILE is a wheel-sizes map: wheel:<tags>
set -euo pipefail
kind="$1"
file="$2"
case "$kind" in
  budgets)
    jq -r '
      (if .transitive_crates != null then "transitive_crates \(.transitive_crates)" else empty end),
      (if .binary_bytes != null then "binary_bytes \(.binary_bytes)" else empty end),
      ((.unsafe_sites // {}) | to_entries[] | "unsafe:\(.key) \(.value)")
    ' "$file"
    ;;
  wheels)
    jq -r 'to_entries[] | select(.value | type == "number") | "wheel:\(.key) \(.value)"' "$file"
    ;;
  *)
    echo "usage: approval-values.sh budgets|wheels FILE" >&2
    exit 2
    ;;
esac
