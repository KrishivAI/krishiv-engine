# SPDX-License-Identifier: Apache-2.0
"""Every public name of the DataFrame API has a docstring (DOC-003, PY-073)."""

from __future__ import annotations

import inspect

import pytest

from krishiv.sql import DataFrame, GroupedData
from krishiv.sql import functions as F
from krishiv.sql.conf import RuntimeConfig
from krishiv.sql.readwriter import DataFrameReader


def _public_methods(cls: type) -> list[str]:
    return [n for n, o in inspect.getmembers(cls) if not n.startswith("_") and callable(o)]


@pytest.mark.parametrize("cls", [DataFrame, GroupedData, DataFrameReader, RuntimeConfig])
def test_public_methods_have_docstrings(cls: type) -> None:
    assert inspect.getdoc(cls), f"{cls.__name__} has no docstring"
    missing = [n for n in _public_methods(cls) if not inspect.getdoc(getattr(cls, n))]
    assert not missing, f"{cls.__name__}: no docstring for {missing}"


def test_public_functions_have_docstrings() -> None:
    names = [
        n
        for n, o in inspect.getmembers(F)
        if not n.startswith("_") and inspect.isfunction(o) and o.__module__ == F.__name__
    ]
    missing = [n for n in names if not inspect.getdoc(getattr(F, n))]
    assert not missing, f"functions without a docstring: {missing}"
