"""A db-benchmark ``time.csv`` as a table next to the published solutions (P3-T094 step 1).

Reads the ``time.csv`` the ``krishiv`` solution wrote (``dbbench_solution``) and the upstream repository's published
``time.csv`` (any file in that layout; nothing is downloaded here). For each task, data set and question it shows the
second run's time (the benchmark's headline: ``run == 2``) per solution, a ``*`` where the solution spilled
(``on_disk == TRUE``), ``FAIL`` where the question has no row, and Krishiv's time as a ratio to each other solution's.

    python -m krishiv_bench.dbbench_report --ours time.csv --published upstream-time.csv [--data G1_1e7_1e2_0_0]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

HEADLINE_RUN = "2"
OURS = "krishiv"


def read_times(path: Path, batch: str | None = None) -> dict[tuple[str, str, str, str], tuple[float, bool]]:
    """``(task, data, question, solution) -> (seconds, spilled)`` for the headline run.

    Where a solution has several rows for one key (several batches), the newest by ``timestamp`` wins; ``batch``
    limits the file to one ``batch`` value.
    """
    newest: dict[tuple[str, str, str, str], tuple[float, float, bool]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("run") != HEADLINE_RUN or not row.get("time_sec"):
                continue
            if batch is not None and row.get("batch") != batch:
                continue
            key = (row["task"], row["data"], row["question"], row["solution"])
            stamp = float(row.get("timestamp") or 0)
            if key not in newest or stamp >= newest[key][0]:
                newest[key] = (stamp, float(row["time_sec"]), row.get("on_disk", "").upper() == "TRUE")
    return {k: (v[1], v[2]) for k, v in newest.items()}


def build_table(
    ours: dict[tuple[str, str, str, str], tuple[float, bool]],
    published: dict[tuple[str, str, str, str], tuple[float, bool]],
    data: str | None = None,
) -> list[dict[str, object]]:
    """One row per (task, data, question): Krishiv's time and each published solution's time and ratio."""
    solutions = sorted({k[3] for k in published if k[3] != OURS})
    cells: dict[tuple[str, str, str], dict[str, tuple[float, bool]]] = defaultdict(dict)
    for source in (published, ours):
        for (task, d, question, solution), value in source.items():
            if data is None or d == data:
                cells[(task, d, question)][solution] = value
    rows = []
    for (task, d, question), by_solution in sorted(cells.items()):
        mine = by_solution.get(OURS)
        row: dict[str, object] = {
            "task": task,
            "data": d,
            "question": question,
            OURS: mine,
        }
        for s in solutions:
            theirs = by_solution.get(s)
            row[s] = theirs
            row[f"{s}_ratio"] = mine[0] / theirs[0] if mine and theirs and theirs[0] > 0 else None
        rows.append(row)
    return rows


def _fmt(cell: object) -> str:
    if not isinstance(cell, tuple):
        return "FAIL"
    return f"{cell[0]:.2f}" + ("*" if cell[1] else "")


def to_markdown(rows: list[dict[str, object]]) -> str:
    """The table as Markdown: seconds per solution, then Krishiv's ratio to each (below 1 is faster)."""
    if not rows:
        return "No rows.\n"
    solutions = sorted(k[: -len("_ratio")] for k in rows[0] if k.endswith("_ratio"))
    head = ["task", "data", "question", OURS, *solutions, *[f"{s} ratio" for s in solutions]]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [str(r["task"]), str(r["data"]), str(r["question"]), _fmt(r[OURS])]
        cells += [_fmt(r[s]) for s in solutions]
        cells += ["" if r[f"{s}_ratio"] is None else f"{r[f'{s}_ratio']:.2f}" for s in solutions]
        lines.append("| " + " | ".join(cells) + " |")
    failed = sum(1 for r in rows if r[OURS] is None)
    lines.append(
        f"\n{len(rows)} questions, {failed} without a Krishiv row. `*` = spilled to disk; ratio = Krishiv / solution.\n"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Krishiv's db-benchmark time.csv next to the published solutions")
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--published", type=Path, required=True)
    ap.add_argument("--data", help="one data set, e.g. G1_1e7_1e2_0_0")
    ap.add_argument("--batch", help="only this batch of our file")
    args = ap.parse_args(argv)
    ours = {k: v for k, v in read_times(args.ours, args.batch).items() if k[3] == OURS}
    print(to_markdown(build_table(ours, read_times(args.published), args.data)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
