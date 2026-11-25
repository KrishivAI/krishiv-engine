# SPDX-License-Identifier: Apache-2.0
"""Messages from a wheel name the call the user made, and the pandas 3 warning names Krishiv (KN-3)."""

from __future__ import annotations

import sys
import warnings
from types import SimpleNamespace

import pytest

import krishiv


@pytest.mark.parametrize(("call", "name"), [("collect", "collect"), ("show", "show"), ("toArrow", "toArrow")])
def test_missing_pyarrow_names_the_call(monkeypatch: pytest.MonkeyPatch, call: str, name: str) -> None:
    df = krishiv.session().sql("select 1 as a")
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(ImportError, match=f"{name} needs pyarrow"):
        getattr(df, call)()


def test_pandas_3_warning_names_krishiv_and_shows_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from krishiv._udf import pandas_utils

    monkeypatch.setitem(sys.modules, "pandas", SimpleNamespace(__version__="3.0.1"))
    monkeypatch.setattr(pandas_utils, "_warned_pandas_3", [])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(4):
            pandas_utils.require_minimum_pandas_version()
    messages = [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]
    assert len(messages) == 1
    assert "Krishiv" in messages[0]
    assert "PySpark" not in messages[0]
