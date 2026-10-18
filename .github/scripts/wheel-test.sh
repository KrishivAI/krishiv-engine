#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# P2-T103: installs the krishiv wheel from DIST into a fresh venv of PYTHON and
# runs the Python test suite (python/tests) against it, from outside the source
# tree so the installed wheel is what is tested.
#
#   wheel-test.sh PYTHON DIST FLAVOUR   (FLAVOUR: abi3 | cp314t)
#
# The Arrow consumers the bridge tests use (pyarrow, Polars, DuckDB) are
# installed from binary wheels only; a target they publish no wheel for (e.g.
# musllinux) skips those tests with a notice instead of building them from
# source. The free-threaded flavour checks the GIL stays disabled.
set -euo pipefail
python=$1
dist=$2
flavour=$3
repo=$(cd "$(dirname "$0")/../.." && pwd)
work=$(mktemp -d)
"$python" -m venv "$work/venv"
py="$work/venv/bin/python"
"$py" -m pip install --quiet --upgrade pip
"$py" -m pip install --quiet "$dist"/*.whl pytest
for dep in "pyarrow>=16" "polars>=1.0" "duckdb>=1.1"; do
  if ! "$py" -m pip install --quiet --only-binary :all: "$dep"; then
    echo "::notice::$dep has no wheel for this target ($flavour, $("$py" -c 'import sysconfig; print(sysconfig.get_platform())')); its tests are skipped"
  fi
done
cd "$work"
"$py" - "$flavour" <<'PY'
import sys

import krishiv

print(krishiv.build_info(), sys.version)
assert krishiv.build_info()["allocator"] == "mimalloc"
if sys.argv[1] == "cp314t":
    # Importing the extension must not re-enable the GIL (the module is
    # declared free-threading safe).
    assert not sys._is_gil_enabled(), "importing krishiv enabled the GIL"
PY
if [ "$flavour" = cp314t ]; then
  PYTHON_GIL=0 "$py" -m pytest -q -rs "$repo/python/tests"
else
  "$py" -m pytest -q -rs "$repo/python/tests"
fi
