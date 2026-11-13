#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# P3-T074 `docs_examples_run`: runs the Python of the docs quickstart and migration guide and the
# sample notebooks against a built wheel, from a clean venv.
#
#   docs-examples-test.sh DIST [PYTHON]   (DIST holds one krishiv wheel; PYTHON is an interpreter
#                                          or a `uv python` request, default 3.11)
set -euo pipefail
dist=$1
python=${2:-3.11}
repo=$(cd "$(dirname "$0")/../.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

shopt -s nullglob
wheels=("$dist"/krishiv-*.whl)
[ "${#wheels[@]}" -eq 1 ] || { echo "::error::expected one krishiv wheel in $dist, found ${#wheels[@]}"; exit 1; }
wheel=$(cd "$(dirname "${wheels[0]}")" && pwd)/$(basename "${wheels[0]}")

uv venv --quiet --python "$python" "$work/venv"
export UV_PYTHON="$work/venv/bin/python"
export UV_CONSTRAINT="$repo/python/ci-interop-constraints.txt"
# The install line the docs show, with the wheel in place of the registry.
uv pip install --quiet --only-binary :all: "$wheel[pyarrow]"

cd "$repo"
python3 tests/scripts/check_docs_links.py
python3 tests/scripts/gen_migration_guide.py --check || { echo "::error::docs/guides/pyspark-migration.md is stale: run tests/scripts/gen_migration_guide.py"; exit 1; }
# `-I`: only the installed wheel (and the standard library) can be imported; the notebooks add docs/datasets to the path themselves.
TMPDIR="$work" "$work/venv/bin/python" -I tests/scripts/docs_examples_run.py
echo "docs examples OK against $(basename "$wheel") on Python $python"
