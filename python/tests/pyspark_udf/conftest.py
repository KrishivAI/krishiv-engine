# SPDX-License-Identifier: Apache-2.0
"""Applies `manifest.toml` to the adapted PySpark 4.2.0 UDF tests: a test not marked ``included`` is skipped.

A test is identified by ``(class that defines it, method)`` (class names are unique across the three files); an
entry with a ``runner`` also names the test class that runs it, for a test excluded in one runner only.
The manifest is checked for completeness by `test_manifest.py`.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

_HERE = Path(__file__).parent


def load_manifest() -> list[dict[str, str]]:
    with (_HERE / "manifest.toml").open("rb") as f:
        return tomllib.load(f)["test"]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    excluded: dict[tuple[str, str, str], str] = {}
    for e in load_manifest():
        if e["status"] == "excluded":
            excluded[(e["cls"], e["name"], e.get("runner", ""))] = f"excluded ({e['category']}): {e['reason']}"
    for item in items:
        cls = getattr(item, "cls", None)
        if cls is None or not str(item.path).startswith(str(_HERE)):
            continue
        for klass in cls.__mro__:
            reason = excluded.get((klass.__name__, item.name, "")) or excluded.get(
                (klass.__name__, item.name, cls.__name__)
            )
            if reason:
                item.add_marker(pytest.mark.skip(reason=reason))
                break
