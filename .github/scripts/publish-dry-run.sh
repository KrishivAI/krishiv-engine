#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# P2-T147: `cargo publish --dry-run` for the `krishiv` crate and the workspace
# crates it depends on, WITHOUT changing what the repository may publish.
#
# Every crate in this repository carries `publish = false` and path
# dependencies without versions (RELEASING.md): nothing can be published from
# it. A dry run needs the opposite, so it runs on a scratch copy of the tracked
# files in which, for the dependency closure of `krishiv` (all features):
#   * `publish = false` is dropped, and
#   * path dependencies get `version = "=<workspace version>"`.
# That is exactly the change the owner makes to enable a real release. The
# repository itself is not touched. `--dry-run` packages every crate, verifies
# it builds from its packaged files against its (also packaged) workspace
# dependencies, and stops before any upload.
#
#   publish-dry-run.sh [--package-only]
#
# `--package-only` runs `cargo package` instead (the same packaging and build
# verification, no `publish` command): for machines whose policy forbids
# `cargo publish` even with --dry-run.
set -euo pipefail
mode=publish
case "${1:-}" in
  "") ;;
  --package-only) mode=package ;;
  *) echo "usage: publish-dry-run.sh [--package-only]" >&2; exit 2 ;;
esac
repo=$(cd "$(dirname "$0")/../.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cd "$repo"
git ls-files -z | tar --null -T - -cf - | tar -xf - -C "$work"

# The crates `krishiv` needs: itself and every workspace crate it depends on
# through normal or build dependencies, optional ones included.
mapfile -t crates < <(cargo tree -p krishiv --all-features -e normal,build --prefix none \
  | awk '$1 ~ /^krishiv/ { print $1 }' | sort -u)
printf 'dry-run set: %s\n' "${crates[*]}"
[ "${#crates[@]}" -ge 2 ] || { echo "::error::dependency closure of krishiv not found"; exit 1; }
version=$(cargo metadata --no-deps --format-version 1 \
  | python3 -I -c 'import json, sys; print(next(p["version"] for p in json.load(sys.stdin)["packages"] if p["name"] == "krishiv"))')

python3 -I - "$work" "$version" "${crates[@]}" <<'PY'
import re
import sys
from pathlib import Path

root, version, *crates = sys.argv[1:]
# path dependency on a workspace crate, in a single-line inline table
dep = re.compile(r'^(\s*"?krishiv[\w-]*"?\s*=\s*\{[^}]*?path\s*=\s*"[^"]+")(\s*[,}])')
for name in crates:
    manifest = Path(root) / "crates" / name / "Cargo.toml"
    out, table, saw_publish = [], "", False
    for line in manifest.read_text().splitlines():
        header = re.match(r"^\s*\[([^\]]+)\]\s*$", line)
        if header:
            table = header.group(1)
        if table == "package" and re.match(r"^publish\s*=\s*false\s*$", line):
            saw_publish = True
            continue
        # Dev-dependencies are dropped from a published manifest when they
        # have no version; normal and build ones must carry one.
        if table in ("dependencies", "build-dependencies") or (table.endswith(".dependencies") and "dev" not in table):
            line = dep.sub(lambda m: f'{m.group(1)}, version = "={version}"{m.group(2)}', line, count=1)
        out.append(line)
    assert saw_publish, f"{name}: no `publish = false` to drop (the guard moved?)"
    manifest.write_text("\n".join(out) + "\n")
PY

args=()
for c in "${crates[@]}"; do args+=(-p "$c"); done
cd "$work"
# The scratch copy shares the repository's lock file (versions are unchanged).
if [ "$mode" = publish ]; then
  cargo publish --dry-run --locked "${args[@]}"
else
  cargo package --locked "${args[@]}"
fi
