# SPDX-License-Identifier: Apache-2.0
"""CLI: `python -m pyspark_suite {static,report,run}` (from tests/scripts)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import manifest as mf
from .denominator import compute
from .report import dashboard, summarize, write_summary
from .static_count import census

DATA = Path(__file__).resolve().parents[2] / "pyspark"


def _connect_dir(checkout: Path) -> Path:
    return checkout / "python" / "pyspark" / "sql" / "tests" / "connect"


def cmd_static(a: argparse.Namespace) -> None:
    collected, skipped = census(_connect_dir(a.checkout))
    den, remaining = compute(a.version, collected, skipped, set(), mf.load(DATA / "manifest.toml"))
    path = write_summary(summarize(den, remaining, [], source="static"), DATA / "summary")
    print(json.dumps(den.as_dict(), indent=1))
    print(f"wrote {path}")


def cmd_report(_: argparse.Namespace) -> None:
    summaries = [json.loads(p.read_text()) for p in sorted((DATA / "summary").glob("*.json"))]
    (DATA / "DASHBOARD.md").write_text(dashboard(summaries), encoding="utf-8")
    print(f"wrote {DATA / 'DASHBOARD.md'}")


def main() -> None:
    p = argparse.ArgumentParser(prog="pyspark_suite")
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("static", help="ast census of a pinned checkout -> summary with 0 passed")
    s.add_argument("--version", required=True)
    s.add_argument("--checkout", type=Path, required=True, help="sparse checkout of the Spark tag")
    s.set_defaults(fn=cmd_static)
    r = sub.add_parser("report", help="render DASHBOARD.md from the committed summaries")
    r.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
