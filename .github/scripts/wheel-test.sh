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
# source. The free-threaded flavour checks the GIL stays disabled. Before the
# test suite, the acceptance smoke of P2-T103 runs on every wheel:
# `import krishiv; krishiv.session().sql("select 1").toArrow()`.
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
# Versions come from python/ci-interop-constraints.txt, the same pins ci.yml
# and nightly.yml use, so every wheel leg tests against one known set.
for dep in pyarrow polars duckdb; do
  if ! "$py" -m pip install --quiet --only-binary :all: \
      --constraint "$repo/python/ci-interop-constraints.txt" "$dep"; then
    echo "::notice::$dep has no wheel for this target ($flavour, $("$py" -c 'import sysconfig; print(sysconfig.get_platform())')); its tests are skipped"
  fi
done
cd "$work"
"$py" - "$flavour" <<'PY'
import sys

import krishiv

print(krishiv.build_info(), sys.version)
assert krishiv.build_info()["allocator"] == "mimalloc"
# The acceptance smoke: a session, a query, an Arrow table. `toArrow` needs
# pyarrow; a target without a pyarrow wheel (noticed above) runs `collect`.
df = krishiv.session().sql("select 1")
try:
    import pyarrow  # noqa: F401
except ImportError:
    print("::notice::no pyarrow on this target; the smoke runs collect() instead of toArrow()")
    assert [tuple(r) for r in df.collect()] == [(1,)]
else:
    table = df.toArrow()
    assert table.num_rows == 1 and table.column(0).to_pylist() == [1], table
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
