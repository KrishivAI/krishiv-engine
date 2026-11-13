#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# `cargo publish --dry-run` for the crates published to crates.io, in the order of
# `.github/scripts/publish-crates.txt` (the same list publish-crates.yml uses). It:
#   * checks that exactly those crates carry `publish = true` and every other crate `publish = false`,
#   * checks that no crate depends (normal or build) on a crate later in the list or outside it,
#   * packages and verifies all of them together (cargo resolves the workspace crates against each
#     other, so no crate needs to be on crates.io yet), stopping before any upload.
#
#   publish-dry-run.sh [--package-only]
#
# `--package-only` runs `cargo package` instead (the same packaging and build verification, no
# `publish` command): for machines whose policy forbids `cargo publish` even with --dry-run.
set -euo pipefail
mode=publish
case "${1:-}" in
  "") ;;
  --package-only) mode=package ;;
  *) echo "usage: publish-dry-run.sh [--package-only]" >&2; exit 2 ;;
esac
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
list=.github/scripts/publish-crates.txt

cargo metadata --no-deps --format-version 1 | python3 -I -c '
import json, sys
order = [line.strip() for line in open(sys.argv[1]) if line.strip()]
pos = {name: i for i, name in enumerate(order)}
bad = []
packages = json.load(sys.stdin)["packages"]
for missing in sorted(set(order) - {p["name"] for p in packages}):
    bad.append(f"{missing}: in the list but not in the workspace")
for p in packages:
    name = p["name"]
    if (p["publish"] != []) != (name in pos):  # `publish = false` shows as []
        bad.append(f"{name}: publish must be {name in pos}")
    if name not in pos:
        continue
    for d in p["dependencies"]:
        dep = d["name"]
        if dep.startswith("krishiv") and d["kind"] != "dev" and (dep not in pos or pos[dep] > pos[name]):
            bad.append(f"{name} depends on {dep}, which is not published before it")
if bad:
    print("\n".join(bad), file=sys.stderr)
    sys.exit(1)
print("publish order ok:", " ".join(order))
' "$list"

args=()
while read -r c; do [ -n "$c" ] && args+=(-p "$c"); done < "$list"
if [ "$mode" = publish ]; then
  cargo publish --dry-run --locked "${args[@]}"
else
  cargo package --locked "${args[@]}"
fi
