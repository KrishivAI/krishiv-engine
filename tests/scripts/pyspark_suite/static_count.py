# SPDX-License-Identifier: Apache-2.0
"""Static (never imported, never executed) test census of a Spark checkout, via `ast`.

This is the draft denominator of the design note: method names starting with `test` in classes
with "Test" in the name, mixin methods inherited through bases defined in the same tree. The
first CI run replaces it with `pytest --collect-only`.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _is_skip(dec: ast.expr) -> bool:
    node = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(node, ast.Attribute):
        return node.attr == "skip"
    return isinstance(node, ast.Name) and node.id == "skip"


class _Class:
    def __init__(self, node: ast.ClassDef):
        self.name = node.name
        self.bases = [
            b.id if isinstance(b, ast.Name) else b.attr for b in node.bases if isinstance(b, (ast.Name, ast.Attribute))
        ]
        self.skipped = any(_is_skip(d) for d in node.decorator_list)
        self.methods = {
            n.name: any(_is_skip(d) for d in n.decorator_list)
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")
        }


def _parse_tree(tests_root: Path) -> dict[Path, dict[str, _Class]]:
    return {
        p: {
            c.name: _Class(c)
            for c in ast.parse(p.read_text(encoding="utf-8"), filename=str(p)).body
            if isinstance(c, ast.ClassDef)
        }
        for p in sorted(tests_root.rglob("*.py"))
    }


def census(connect_dir: Path) -> tuple[set[str], set[str]]:
    """Return (collected ids, ids Spark skips unconditionally) for the test files under `connect_dir`.

    Bases resolve by class name over the whole `pyspark/sql/tests` tree (parity classes inherit the
    mixins of the classic suites), preferring a class from the same file.
    """
    tests_root = connect_dir.parent  # .../sql/tests
    tree = _parse_tree(tests_root)
    global_classes: dict[str, _Class] = {}
    for classes in tree.values():
        for name, c in classes.items():
            global_classes.setdefault(name, c)

    collected: set[str] = set()
    skipped: set[str] = set()
    for path, classes in tree.items():
        if connect_dir not in path.parents or not path.name.startswith("test_"):
            continue
        rel = path.relative_to(connect_dir).as_posix()

        def methods(name: str, seen: frozenset[str] = frozenset(), classes=classes) -> dict[str, bool]:
            c = classes.get(name) or global_classes.get(name)
            if c is None or name in seen:
                return {}
            out: dict[str, bool] = {}
            for b in c.bases:
                out.update(methods(b, seen | {name}))
            out.update(c.methods)
            return out

        for name, c in classes.items():
            if "Test" not in name:
                continue
            for m, skip in methods(name).items():
                tid = f"{rel}::{name}::{m}"
                collected.add(tid)
                if skip or c.skipped:
                    skipped.add(tid)
    return collected, skipped
