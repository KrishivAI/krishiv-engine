"""A db-benchmark ``time.csv`` as a table next to the published solutions.

Reads the ``time.csv`` the ``krishiv`` solution wrote (``dbbench_solution``) and the upstream repository's published
``time.csv`` (any file in that layout; nothing is downloaded here). For each task, data set and question it shows the
second run's time (the benchmark's headline: ``run == 2``) per solution, a ``*`` where the solution spilled
(``on_disk == TRUE``), ``FAIL`` where the question has no row, and Krishiv's time as a ratio to each other solution's.

    python -m krishiv_bench.dbbench_report --ours time.csv --published upstream-time.csv [--data G1_1e7_1e2_0_0]

``--check <solution>`` compares answers instead, within one ``time.csv`` that holds both solutions' rows from the
same data files (a launcher run of ``krishiv`` and, say, ``duckdb``): for every task, data set, question and run,
``out_rows``, ``out_cols`` and every sum of ``chk`` must agree (sums within a relative 1e-6). It exits 1 on any
difference or on a question only one of the two answered.

    python -m krishiv_bench.dbbench_report --ours time.csv --check duckdb [--batch 1795616032]
"""

from __future__ import annotations

import argparse
import csv
import math
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


# The benchmark writes each ``chk`` sum rounded to three decimals, so two equal sums can differ by one unit of the
# third decimal; a little slack on top for the rounding of each side.
_CHK_ABS_TOL = 1.1e-3


def _same_sum(x: str, y: str) -> bool:
    """Whether two ``chk`` fields are the same sum. A field that is not a number equals only the same text."""
    if x == y:
        return True
    try:
        a, b = float(x), float(y)
    except ValueError:
        return False
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=_CHK_ABS_TOL)


def check_answers(path: Path, other: str, batch: str | None = None, data: str | None = None) -> tuple[int, list[str]]:
    """Compare Krishiv's answers in ``path`` with ``other``'s; returns (answers compared, one line per problem).

    ``batch`` and ``data`` narrow the comparison to one launcher run and one data set. Without ``batch``, only the
    rows of the newest batch (the last one in the file) of each solution, task and data set are read: the launcher
    appends every run to one file, and a question the newest run did not answer must count as missing.
    """
    rows: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    newest: dict[tuple[str, str, str], str | None] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if batch is not None and row.get("batch") != batch:
                continue
            if data is not None and row["data"] != data:
                continue
            group = (row["solution"], row["task"], row["data"])
            if group not in newest or newest[group] != row.get("batch"):
                newest[group] = row.get("batch")
                for stale in [k for k in rows if k[:3] == group]:
                    del rows[stale]
            rows[(*group, row["question"], row["run"])] = row
    keys = sorted({k[1:] for k in rows if k[0] in (OURS, other)})
    problems = []
    for key in keys:
        a, b = rows.get((other, *key)), rows.get((OURS, *key))
        where = " ".join(key)
        if a is None or b is None:
            problems.append(f"{where}: no row from {OURS if b is None else other}")
            continue
        sums_a, sums_b = a["chk"].split(";"), b["chk"].split(";")
        same = (a["out_rows"], a["out_cols"], len(sums_a)) == (b["out_rows"], b["out_cols"], len(sums_b)) and all(
            _same_sum(x, y) for x, y in zip(sums_a, sums_b, strict=True)
        )
        if not same:
            ours = f"{b['out_rows']}x{b['out_cols']} chk {b['chk']}"
            problems.append(f"{where}: {other} {a['out_rows']}x{a['out_cols']} chk {a['chk']}, {OURS} {ours}")
    return len(keys), problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Krishiv's db-benchmark time.csv next to the published solutions")
    ap.add_argument("--ours", type=Path, required=True)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--published", type=Path, help="the published time.csv to print our times next to")
    what.add_argument("--check", metavar="SOLUTION", help="compare our answers with this solution's in --ours")
    ap.add_argument("--data", help="one data set, e.g. G1_1e7_1e2_0_0")
    ap.add_argument("--batch", help="only this batch of our file")
    args = ap.parse_args(argv)
    if args.check:
        compared, problems = check_answers(args.ours, args.check, args.batch, args.data)
        print("\n".join([*problems, f"{compared} answers compared with {args.check}, {len(problems)} differ"]))
        return 1 if problems or not compared else 0
    ours = {k: v for k, v in read_times(args.ours, args.batch).items() if k[3] == OURS}
    print(to_markdown(build_table(ours, read_times(args.published), args.data)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
