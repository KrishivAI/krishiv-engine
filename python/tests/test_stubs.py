# SPDX-License-Identifier: Apache-2.0
"""The type stubs ship with the wheel and match the runtime API.

``test_native_stub_matches_runtime`` compares the stub against the built
extension with the standard library only, so it runs in every test
environment. ``test_mypy_strict_examples`` and ``test_stubtest_native`` run
the mypy checks (``mypy --strict`` over ``examples/`` and ``mypy.stubtest``
against the built extension); they skip when mypy is not installed, unless
``KRISHIV_REQUIRE_MYPY=1`` (the wheel test job sets it), when they fail. The
commands, in the style of the ``python-quality`` job, are in ``README.md``.
"""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

PYTHON_DIR = Path(__file__).resolve().parent.parent


def _installed_package_dir() -> Path:
    """The ``krishiv`` package under test: the installed wheel, not the source tree.

    Resolved without importing the package (which needs the built extension),
    so the check also runs before ``maturin develop``. Under the wheel jobs the
    test files run from outside the source tree, so this is the installed wheel.
    """
    import importlib.util

    spec = importlib.util.find_spec("krishiv")
    assert spec is not None, "krishiv package not found"
    assert spec.origin is not None, "krishiv package not found"
    return Path(spec.origin).resolve().parent


def test_py_typed_marker_is_shipped() -> None:
    assert (_installed_package_dir() / "py.typed").is_file()


def test_native_stub_is_shipped() -> None:
    assert (_installed_package_dir() / "_native.pyi").is_file()


def _stub_methods(tree: ast.Module) -> dict[str, dict[str, ast.FunctionDef]]:
    """Class name to its method definitions, from the stub's syntax tree."""
    found: dict[str, dict[str, ast.FunctionDef]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            found[node.name] = {item.name: item for item in node.body if isinstance(item, ast.FunctionDef)}
    return found


def _stub_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _stub_params(fn: ast.FunctionDef) -> list[str]:
    """Parameter names of a stub definition, without ``self``/``cls``."""
    names = [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]
    return [n for n in names if n not in ("self", "cls")]


def _stub_defaulted(fn: ast.FunctionDef) -> set[str]:
    """Names of the stub parameters that declare a default (``= ...``)."""
    positional = [a.arg for a in fn.args.posonlyargs + fn.args.args]
    names = set(positional[len(positional) - len(fn.args.defaults) :]) if fn.args.defaults else set()
    names.update(a.arg for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults, strict=True) if d is not None)
    return names


def test_native_stub_matches_runtime() -> None:
    krishiv_native = pytest.importorskip("krishiv._native")
    tree = ast.parse((_installed_package_dir() / "_native.pyi").read_text())
    classes = _stub_methods(tree)
    assert set(classes) == {"Session", "DataFrame", "GroupedData", "Column", "WindowSpec", "ArrowStream"}
    for cls_name, methods in classes.items():
        runtime = getattr(krishiv_native, cls_name)
        for name, fn in methods.items():
            if name in ("__new__", "__init__", "__repr__"):
                continue
            assert name in dir(runtime), f"{cls_name}.{name} is in the stub but missing at runtime"
            if any(d.id == "property" for d in fn.decorator_list if isinstance(d, ast.Name)):
                continue
            stub_names = _stub_params(fn)
            try:
                sig = inspect.signature(getattr(runtime, name))
            except (TypeError, ValueError):
                continue
            runtime_names = [p for p in sig.parameters if p not in ("self", "cls")]
            assert stub_names == runtime_names, f"{cls_name}.{name}: stub {stub_names} != runtime {runtime_names}"
            stub_defaults = _stub_defaulted(fn)
            for param in stub_names:
                has_default = sig.parameters[param].default is not inspect.Parameter.empty
                assert (param in stub_defaults) == has_default, f"{cls_name}.{name}.{param}: default mismatch"
    stub_fns = _stub_functions(tree)
    assert set(stub_fns) == {
        "col",
        "lit",
        "map_literal",
        "expr",
        "call_function",
        "count_star",
        "count_distinct",
    }
    for name, fn in stub_fns.items():
        assert callable(getattr(krishiv_native, name)), f"{name} is in the stub but missing at runtime"
        stub_names = _stub_params(fn)
        runtime_names = list(inspect.signature(getattr(krishiv_native, name)).parameters)
        assert stub_names == runtime_names, f"{name}: stub {stub_names} != runtime {runtime_names}"


def _need_mypy() -> None:
    """Skip without mypy, unless CI requires it (``KRISHIV_REQUIRE_MYPY=1``): then fail."""
    if os.environ.get("KRISHIV_REQUIRE_MYPY") == "1":
        import mypy  # noqa: F401  (a missing mypy must fail the job, not skip the check)
    else:
        pytest.importorskip("mypy")


def test_mypy_strict_examples() -> None:
    _need_mypy()
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "examples/basic.py", "examples/dataframe_ops.py"],
        cwd=PYTHON_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_stubtest_native() -> None:
    _need_mypy()
    proc = subprocess.run(
        [sys.executable, "-m", "mypy.stubtest", "krishiv._native", "--concise", "--ignore-disjoint-bases"],
        cwd=PYTHON_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "", proc.stdout
