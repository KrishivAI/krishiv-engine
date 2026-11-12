# SPDX-License-Identifier: Apache-2.0
"""CLI: `python -m pyspark_suite {static,report,run}` (from tests/scripts)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import manifest as mf
from .denominator import compute
from .report import dashboard, summarize, write_summary
from .runner import run_file
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


def cmd_run(a: argparse.Namespace) -> None:
    """Run every non-excluded file against a fresh server each; write the summary with real pass counts."""

    connect = _connect_dir(a.checkout)
    entries = mf.load(DATA / "manifest.toml")
    collected, skipped = census(connect)
    den, remaining = compute(a.version, collected, skipped, set(), entries)
    files = sorted({t.split("::", 1)[0] for t in remaining})
    if a.only:
        files = [f for f in files if any(f.startswith(o) for o in a.only)]
    out_dir = DATA / "results" / a.version
    records = []
    for rel in files:
        print(f"running {rel}", flush=True)
        records += run_file(a.version, a.checkout, rel, out_dir, a.server_cmd, a.port)
    path = write_summary(summarize(den, remaining, records, source="run"), DATA / "summary")
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
    u = sub.add_parser("run", help="run the suite per file against `--server-cmd` (CI sandbox only)")
    u.add_argument("--version", required=True)
    u.add_argument("--checkout", type=Path, required=True)
    u.add_argument("--port", type=int, default=15002)
    u.add_argument("--only", action="append", help="run only files starting with this path (repeatable)")
    u.add_argument("--server-cmd", nargs=argparse.REMAINDER, required=True, help="server command, last option")
    u.set_defaults(fn=cmd_run)
    r = sub.add_parser("report", help="render DASHBOARD.md from the committed summaries")
    r.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
