# SPDX-License-Identifier: Apache-2.0
"""`docs_examples_run` (P3-T074): runs the docs' Python on the installed `krishiv` wheel.

    python tests/scripts/docs_examples_run.py            # run everything
    python tests/scripts/docs_examples_run.py --list     # what would run

- Every ```python (or ```py) block of the quickstart and the migration guide runs, top to bottom, in one
  namespace per page (blocks marked `<!-- no-run -->` on the line before are skipped).
- Every notebook in docs/notebooks runs headless: its code cells execute in order in one namespace
  with the notebook's folder as the working directory. A notebook with stored outputs or execution
  counts fails (they are committed stripped).
Needs `pip install "krishiv[pyarrow]"` in a clean virtualenv; no network.
"""

import json
import os
import pathlib
import re
import sys
import traceback

DOCS = pathlib.Path("docs")
PAGES = [DOCS / "get-started" / "index.md", DOCS / "guides" / "pyspark-migration.md"]
BLOCK = re.compile(r"(<!-- no-run -->\n)?```(?:python|py)\b[^\n]*\n(.*?)```", re.S)


def page_blocks(page: pathlib.Path) -> list[str]:
    return [m.group(2) for m in BLOCK.finditer(page.read_text(encoding="utf-8")) if not m.group(1)]


def notebook_cells(nb: pathlib.Path) -> list[str]:
    data = json.loads(nb.read_text(encoding="utf-8"))
    cells = [c for c in data["cells"] if c["cell_type"] == "code"]
    dirty = [c for c in cells if c.get("outputs") or c.get("execution_count") is not None]
    if dirty:
        raise SystemExit(f"{nb}: outputs or execution counts are stored; strip them")
    return ["".join(c["source"]) for c in cells]


def run(label: str, sources: list[str], cwd: pathlib.Path | None = None) -> bool:
    ns: dict = {"__name__": "__docs__"}
    before = pathlib.Path.cwd()
    if cwd:
        os.chdir(cwd)
    try:
        for i, src in enumerate(sources, 1):
            try:
                exec(compile(src, f"{label}#{i}", "exec"), ns)
            except Exception:
                print(f"FAIL {label} block {i}", file=sys.stderr)
                traceback.print_exc()
                return False
    finally:
        os.chdir(before)
    print(f"ok   {label} ({len(sources)} blocks)")
    return True


def jobs() -> list[tuple[str, list[str], pathlib.Path | None]]:
    found = [(str(p), page_blocks(p), None) for p in PAGES]
    return found + [(str(n), notebook_cells(n), n.parent) for n in sorted((DOCS / "notebooks").glob("*.ipynb"))]


def main() -> int:
    all_jobs = jobs()
    if "--list" in sys.argv:
        for label, src, _ in all_jobs:
            print(label, len(src))
        return 0
    if "--job" in sys.argv:  # one page or notebook, in this process
        label, src, cwd = all_jobs[int(sys.argv[sys.argv.index("--job") + 1])]
        return 0 if run(label, src, cwd) else 1
    # Each page and notebook gets a fresh interpreter, so no session state leaks between them.
    import subprocess

    codes = [
        subprocess.run([sys.executable, "-I", __file__, "--job", str(i)], check=False).returncode
        for i in range(len(all_jobs))
    ]
    return 0 if not any(codes) else 1


if __name__ == "__main__":
    sys.exit(main())
