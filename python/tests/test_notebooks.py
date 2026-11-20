# SPDX-License-Identifier: Apache-2.0
"""The notebooks under ``tests/notebooks`` run with nbval: every cell runs in a fresh kernel and its output must
equal the one stored in the notebook (a cell marked ``# NBVAL_IGNORE_OUTPUT`` only has to run).

``notebooks/profile_view.ipynb`` is the notebook of P2-T096: a ``ProfileView`` draws itself through
``_repr_html_``, for the stored profile of TPC-H Q9 and for a query the notebook runs.

The test skips when nbval or ipykernel is not installed, unless ``KRISHIV_REQUIRE_NBVAL=1`` (CI's coverage leg):
then a missing one fails. To store new outputs after a change to the viewer, run the notebook and save it.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

# nbval is a plugin chosen on the command line; the kernel is the interpreter that runs pytest, so the notebook
# imports the package under test.
NBVAL = ("--nbval", "--nbval-current-env")
NOTEBOOKS = sorted((Path(__file__).parent / "notebooks").glob("*.ipynb"))


def test_there_is_a_notebook_to_run() -> None:
    assert NOTEBOOKS


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_the_notebook_runs_to_its_stored_outputs(notebook: Path) -> None:
    pytest.importorskip("krishiv._native")
    for module in ("nbval", "ipykernel"):
        if os.environ.get("KRISHIV_REQUIRE_NBVAL") == "1":
            importlib.import_module(module)
        else:
            pytest.importorskip(module)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *NBVAL, notebook.name],
        cwd=notebook.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
