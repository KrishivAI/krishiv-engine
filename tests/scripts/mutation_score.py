#!/usr/bin/env python3
"""Mutation score of the nightly `cargo-mutants` run, per area (P2-T107).

Reads the `outcomes.json` files cargo-mutants writes (one per shard) and the area floors in
`tests/mutation-floors.json`, prints a Markdown table (the nightly job puts it in the run summary), writes
`mutation-score.json` (uploaded as an artifact: the tracked history), and exits 1 when an area marked
`enforce` is below its floor.

    score = caught / (caught + missed + timeout)

Unviable mutants (they do not compile) and the baseline run are not counted. A timeout counts against the
score: a test run that hangs on a mutant is a weak test, not a caught mutant.

Usage: mutation_score.py --floors tests/mutation-floors.json --out mutation-score.json \
           kernels=outcomes-kernels-1.json kernels=outcomes-kernels-2.json hash=outcomes-hash.json ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

CAUGHT = "CaughtMutant"
MISSED = "MissedMutant"
TIMEOUT = "Timeout"


def count(outcomes: dict) -> Counter[str]:
    """Outcome summaries of one `outcomes.json`, the baseline scenario excluded."""
    return Counter(o["summary"] for o in outcomes.get("outcomes", []) if o.get("scenario") != "Baseline")


def score(c: Counter[str]) -> float | None:
    """Percentage of tested mutants caught; `None` when none was tested."""
    tested = c[CAUGHT] + c[MISSED] + c[TIMEOUT]
    return None if tested == 0 else 100.0 * c[CAUGHT] / tested


def merge(shards: Sequence[tuple[str, Path]]) -> dict[str, Counter[str]]:
    """Per-area totals over every shard file."""
    areas: dict[str, Counter[str]] = defaultdict(Counter)
    for area, path in shards:
        areas[area].update(count(json.loads(path.read_text())))
    return dict(areas)


def evaluate(areas: dict[str, Counter[str]], floors: dict) -> tuple[list[dict], list[str]]:
    """Result rows (one per area) and the failures of enforced floors."""
    rows, failures = [], []
    # An area with a floor but no input (its job produced nothing) is reported as untested.
    for area in sorted(set(areas) | set(floors.get("areas", {}))):
        c = areas.get(area, Counter())
        s = score(c)
        spec = floors.get("areas", {}).get(area, {})
        floor = spec.get("floor", floors.get("target", 70))
        enforce = bool(spec.get("enforce", False))
        rows.append(
            {
                "area": area,
                "score": None if s is None else round(s, 1),
                "caught": c[CAUGHT],
                "missed": c[MISSED],
                "timeout": c[TIMEOUT],
                "unviable": c["Unviable"],
                "floor": floor,
                "enforce": enforce,
            }
        )
        if enforce and (s is None or s < floor):
            failures.append(f"{area}: {'no mutants tested' if s is None else f'{s:.1f}%'} is below the floor {floor}%")
    return rows, failures


def markdown(rows: list[dict], target: float) -> str:
    out = [
        f"### Mutation score (target {target:g}%)",
        "",
        "| area | score | caught | missed | timeout | unviable | floor | gate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        s = "n/a" if r["score"] is None else f"{r['score']:.1f}%"
        gate = "enforced" if r["enforce"] else "report only"
        cells = [r["area"], s, r["caught"], r["missed"], r["timeout"], r["unviable"], f"{r['floor']}%", gate]
        out.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--floors", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("shards", nargs="+", help="AREA=path/to/outcomes.json")
    args = ap.parse_args(argv)
    shards = []
    for s in args.shards:
        area, _, path = s.partition("=")
        if not path:
            ap.error(f"expected AREA=PATH, got {s!r}")
        shards.append((area, Path(path)))
    floors = json.loads(args.floors.read_text())
    rows, failures = evaluate(merge(shards), floors)
    args.out.write_text(json.dumps({"target": floors.get("target", 70), "areas": rows}, indent=2) + "\n")
    print(markdown(rows, floors.get("target", 70)))
    for f in failures:
        print(f"::error::{f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
