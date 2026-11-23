# SPDX-License-Identifier: Apache-2.0
"""CLI: `python -m pyspark_suite {static,report,run,ratchet}` (from tests/scripts)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import manifest as mf
from .denominator import compute
from .junit import oracle_skips
from .report import dashboard, ratchet_regressions, summarize, write_summary
from .runner import check_pytest_timeout, run_file, stage_tests
from .static_count import census

DATA = Path(__file__).resolve().parents[2] / "pyspark"


def _connect_dir(checkout: Path) -> Path:
    return checkout / "python" / "pyspark" / "sql" / "tests" / "connect"


def cmd_static(a: argparse.Namespace) -> None:
    if not _connect_dir(a.checkout).is_dir():
        raise SystemExit(f"static: {_connect_dir(a.checkout)} is not a directory (wrong --checkout?)")
    collected, skipped = census(_connect_dir(a.checkout))
    if not collected:
        raise SystemExit("static: no tests found; not overwriting the summary")
    cond = oracle_skips(a.oracle_junit, a.version) if a.oracle_junit else None
    den, remaining = compute(a.version, collected, skipped, set(), mf.load(DATA / "manifest.toml"), cond)
    path = write_summary(summarize(den, remaining, [], source="static"), DATA / "summary")
    print(json.dumps(den.as_dict(), indent=1))
    print(f"wrote {path}")


def cmd_run(a: argparse.Namespace) -> None:
    """Run every non-excluded file against a fresh server each; write the summary with real pass counts."""

    connect = _connect_dir(a.checkout)
    entries = mf.load(DATA / "manifest.toml")
    if not a.server_cmd:
        raise SystemExit("run: --server-cmd needs a command")
    if not connect.is_dir():
        raise SystemExit(f"run: {connect} is not a directory (wrong --checkout?)")
    collected, skipped = census(connect)
    if not collected:
        raise SystemExit(f"run: no tests found under {connect}; not overwriting the summary")
    cond = oracle_skips(a.oracle_junit, a.version) if a.oracle_junit else None
    den, remaining = compute(a.version, collected, skipped, set(), entries, cond)
    files = sorted({t.split("::", 1)[0] for t in remaining})
    if a.only:
        files = [f for f in files if any(f.startswith(o) for o in a.only)]
    if not files:
        raise SystemExit("run: no files selected")
    check_pytest_timeout()
    tests_dir = stage_tests(a.checkout)
    out_dir = (a.out / "results" / a.version) if a.out else DATA / "results" / a.version
    records = []
    for rel in files:
        print(f"running {rel}", flush=True)
        records += run_file(a.version, tests_dir, rel, out_dir, a.server_cmd, a.port)
    # A partial (`--only`) run or one with `--out` must not replace the committed full summary.
    target = a.out or (DATA / "results" if a.only else DATA / "summary")
    summary = summarize(den, remaining, records, source="run")
    path = write_summary(summary, target)
    print(f"wrote {path}")
    if set(summary["file_failures"]) >= set(files):
        raise SystemExit("run: every selected file failed to run (collection error, crash or timeout); no result")


def cmd_report(_: argparse.Namespace) -> None:
    summaries = [json.loads(p.read_text()) for p in sorted((DATA / "summary").glob("*.json"))]
    (DATA / "DASHBOARD.md").write_text(dashboard(summaries), encoding="utf-8")
    print(f"wrote {DATA / 'DASHBOARD.md'}")


def cmd_ratchet(a: argparse.Namespace) -> None:
    """Fail when a fresh run lost a pass the committed summary had, or fell under `--min-rate` when it is given.

    A version with no committed summary (3.5.7 before its first commit) has an empty pass list.
    """

    committed = DATA / "summary" / f"{a.version}.json"
    if committed.exists():
        previous = json.loads(committed.read_text())
    else:
        print(f"{a.version}: no committed summary; treating the pass list as empty")
        previous = {"passed_ids": []}
    current = json.loads(a.current.read_text())
    if current["version"] != a.version:
        raise SystemExit(f"ratchet: {a.current} is for {current['version']}, not {a.version}")
    if current.get("source") != "run":
        raise SystemExit("ratchet: the current summary is not from a `run`")
    if a.min_rate is not None and current.get("counts", {}).get("skip", 0) > 0:
        # `conditional_skips` (excluded by the oracle) and `counts.skip` (left in the run) describe different tests, so
        # a nonzero exclusion does not show that no JVM-only skip is left. Until the oracle's per-id list is checked
        # against the run's skips (with the denominator freeze), a floor is refused while any skip remains.
        raise SystemExit(
            "ratchet: --min-rate is refused while the run still has skipped tests: a JVM-only skip left in the "
            "denominator would lower the rate; exclude them through `run --oracle-junit` first"
        )
    lost = ratchet_regressions(previous, current)
    print(
        f"{a.version}: {current['passed']} passed ({current['pass_rate']}%), {len(lost)} lost since the committed list"
    )
    for t in lost:
        print(f"  lost: {t}")
    failed = bool(lost)
    if a.min_rate is not None and current["pass_rate"] < a.min_rate:
        print(f"  pass rate {current['pass_rate']}% is under the {a.min_rate}% gate")
        failed = True
    if failed and not a.report_only:
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(prog="pyspark_suite")
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("static", help="ast census of a pinned checkout -> summary with 0 passed")
    s.add_argument("--version", required=True)
    s.add_argument("--checkout", type=Path, required=True, help="sparse checkout of the Spark tag")
    s.add_argument(
        "--oracle-junit", type=Path, help="dir of JUnit XML from a Spark oracle run: its skips leave the denominator"
    )
    s.set_defaults(fn=cmd_static)
    u = sub.add_parser("run", help="run the suite per file against `--server-cmd` (CI sandbox only)")
    u.add_argument("--version", required=True)
    u.add_argument("--checkout", type=Path, required=True)
    u.add_argument("--port", type=int, default=15002)
    u.add_argument(
        "--oracle-junit", type=Path, help="dir of JUnit XML from a Spark oracle run: its skips leave the denominator"
    )
    u.add_argument("--out", type=Path, help="write the summary and JUnit files here, never over the committed ones")
    u.add_argument("--only", action="append", help="run only files starting with this path (repeatable)")
    u.add_argument("--server-cmd", nargs=argparse.REMAINDER, required=True, help="server command, last option")
    u.set_defaults(fn=cmd_run)
    r = sub.add_parser("report", help="render DASHBOARD.md from the committed summaries")
    r.set_defaults(fn=cmd_report)
    t = sub.add_parser("ratchet", help="fail on lost passes against the committed summary (and `--min-rate`)")
    t.add_argument("--version", required=True)
    t.add_argument("--current", type=Path, required=True, help="summary written by a fresh `run`")
    t.add_argument("--min-rate", type=float, help="blocking pass-rate floor in percent")
    t.add_argument("--report-only", action="store_true", help="print, never fail (3.5.7)")
    t.set_defaults(fn=cmd_ratchet)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
