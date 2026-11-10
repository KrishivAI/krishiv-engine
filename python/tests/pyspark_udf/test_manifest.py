# SPDX-License-Identifier: Apache-2.0
"""The manifest lists every test method of the three adapted PySpark 4.2.0 files, once, and each exclusion says why."""

from __future__ import annotations

import ast
from pathlib import Path

from pyspark_udf.conftest import load_manifest

HERE = Path(__file__).parent
FILES = ["test_udf.py", "test_pandas_udf_scalar.py", "test_arrow_python_udf.py"]
CATEGORIES = {
    "iterator-kind",
    "grouped-kind",
    "jvm-internals",
    "engine-feature",
    "krishiv-bug",
    "krishiv-divergence",
}


def _methods() -> set[tuple[str, str, str]]:
    found = set()
    for file in FILES:
        for node in ast.parse((HERE / file).read_text()).body:
            if isinstance(node, ast.ClassDef):
                for m in node.body:
                    if isinstance(m, ast.FunctionDef) and m.name.startswith("test"):
                        found.add((file, node.name, m.name))
    return found


def test_every_test_method_is_in_the_manifest_exactly_once() -> None:
    entries = load_manifest()
    keys = [(e["file"], e["cls"], e["name"]) for e in entries if "runner" not in e]
    assert len(keys) == len(set(keys)), "a method is listed twice"
    assert set(keys) == _methods()


def test_every_exclusion_names_a_category_and_a_reason() -> None:
    for e in load_manifest():
        assert e["status"] in {"included", "excluded"}, e
        if e["status"] == "excluded":
            assert e["category"] in CATEGORIES, e
            assert len(e["reason"]) > 20, e
        else:
            assert "reason" not in e, e
            assert "category" not in e, e
