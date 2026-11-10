# SPDX-License-Identifier: Apache-2.0
"""Snapshot of the public Python API (names and signatures), read from source with `ast`.

Public means no leading underscore, plus `_native.pyi` (the native module's stub, re-exported by the package).
No import, so it needs neither the wheel nor the native build. `--check` diffs against
`python/public-api.txt`; `--write` updates it (a PR that changes the API commits the new file).
"""

import argparse
import ast
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "python" / "krishiv"
SNAPSHOT = ROOT / "python" / "public-api.txt"


def _public(name: str) -> bool:
    return not name.startswith("_") or (name.startswith("__") and name.endswith("__") and name == "__all__")


def _sig(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = ast.unparse(fn.args)
    ret = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
    return f"{'async ' if isinstance(fn, ast.AsyncFunctionDef) else ''}def {fn.name}({args}){ret}"


def _walk(body: list[ast.stmt], prefix: str, out: list[str]) -> None:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _public(node.name) or node.name in {"__init__", "__call__"}:
                out.append(f"{prefix}{_sig(node)}")
        elif isinstance(node, ast.ClassDef) and _public(node.name):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            out.append(f"{prefix}class {node.name}({bases})")
            _walk(node.body, f"{prefix}{node.name}.", out)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and _public(t.id) and prefix == "":
                    out.append(f"{prefix}{t.id}")
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and _public(node.target.id):
            out.append(f"{prefix}{node.target.id}: {ast.unparse(node.annotation)}")


def render() -> str:
    lines: list[str] = []
    files = sorted(p for p in PKG.rglob("*") if p.suffix in {".py", ".pyi"})
    for path in files:
        rel = path.relative_to(PKG)
        if any(part.startswith("_") and part not in {"__init__.py", "_native.pyi"} for part in rel.parts):
            continue
        if "testing" in rel.parts:
            continue
        out: list[str] = []
        _walk(ast.parse(path.read_text()).body, "", out)
        if out:
            lines.append(f"## {rel.as_posix()}")
            lines.extend(sorted(out))
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    args = ap.parse_args()
    text = render()
    if args.write:
        SNAPSHOT.write_text(text)
        return 0
    old = SNAPSHOT.read_text() if SNAPSHOT.exists() else ""
    if old == text:
        return 0
    diff = difflib.unified_diff(old.splitlines(True), text.splitlines(True), "python/public-api.txt", "source")
    sys.stdout.writelines(diff)
    sys.stderr.write(
        "public Python API changed: run `python tests/scripts/python_public_api.py --write` "
        "and commit python/public-api.txt\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
