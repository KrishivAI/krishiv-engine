# SPDX-License-Identifier: Apache-2.0
"""CLI: `python -m pyspark_suite {static,collect,run,ratchet,report}` (from tests/scripts)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import manifest as mf
from .collect import CollectError, check_drift, collect_ids
from .denominator import compute
from .junit import oracle_skips
from .report import dashboard, ratchet_regressions, summarize, write_summary
from .runner import check_pytest_timeout, run_file
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


def _committed(version: str) -> dict | None:
    path = DATA / "summary" / f"{version}.json"
    return json.loads(path.read_text()) if path.exists() else None


def cmd_collect(a: argparse.Namespace) -> None:
    """`pytest --collect-only` counts; exit 1 on drift from the committed summary and manifest unless `--update`."""

    connect = _connect_dir(a.checkout)
    if not connect.is_dir():
        raise SystemExit(f"collect: {connect} is not a directory (wrong --checkout?)")
    entries = mf.load(DATA / "manifest.toml")
    census_ids, skipped = census(connect)
    try:
        collected = collect_ids(a.checkout)
    except CollectError as e:
        raise SystemExit(f"collect: {e}") from e
    drift = check_drift(a.version, collected, census_ids, _committed(a.version), entries)
    cond = oracle_skips(a.oracle_junit, a.version) if a.oracle_junit else None
    den, remaining = compute(a.version, collected, skipped & collected, set(), entries, cond)
    print(json.dumps(den.as_dict(), indent=1))
    print("\n".join(drift.lines()))
    prev = _committed(a.version)
    if a.update and prev and prev.get("source") == "run":
        # A collected summary has no pass list: replacing a run summary would turn the ratchet off silently.
        raise SystemExit(f"collect: {a.version}'s committed summary is a run; refresh it with `run`, not --update")
    if a.update:
        print(f"wrote {write_summary(summarize(den, remaining, [], source='collected'), DATA / 'summary')}")
    elif drift.drifted:
        raise SystemExit(f"collect: {a.version} drifted from the committed summary/manifest (rerun with --update)")


def cmd_run(a: argparse.Namespace) -> None:
    """Run every non-excluded file against a fresh server each; write the summary with real pass counts."""

    connect = _connect_dir(a.checkout)
    entries = mf.load(DATA / "manifest.toml")
    if not a.server_cmd:
        raise SystemExit("run: --server-cmd needs a command")
    if not connect.is_dir():
        raise SystemExit(f"run: {connect} is not a directory (wrong --checkout?)")
    census_ids, skipped = census(connect)
    if not census_ids:
        raise SystemExit(f"run: no tests found under {connect}; not overwriting the summary")
    try:
        collected = collect_ids(a.checkout)
    except CollectError as e:
        raise SystemExit(f"run: {e}") from e
    drift = check_drift(a.version, collected, census_ids, _committed(a.version), entries)
    if drift.drifted and not a.allow_drift:
        raise SystemExit("run: collection drifted from the committed summary/manifest:\n" + "\n".join(drift.lines()))
    skipped &= collected
    cond = oracle_skips(a.oracle_junit, a.version) if a.oracle_junit else None
    den, remaining = compute(a.version, collected, skipped, set(), entries, cond)
    files = sorted({t.split("::", 1)[0] for t in remaining})
    if a.only:
        files = [f for f in files if any(f.startswith(o) for o in a.only)]
    if not files:
        raise SystemExit("run: no files selected")
    check_pytest_timeout()
    out_dir = DATA / "results" / a.version
    records = []
    for rel in files:
        print(f"running {rel}", flush=True)
        records += run_file(a.version, a.checkout, rel, out_dir, a.server_cmd, a.port)
    # A partial (`--only`) run must not replace the committed full summary.
    target = DATA / "results" if a.only else DATA / "summary"
    # Runtime skips that are the client mode's (remote-only, "Requires JVM access") are decided by the client
    # environment, not the server: they leave the denominator as conditional skips, as the oracle's would.
    run_cond = {r.test_id for r in records if r.conditional_skip}
    den, remaining = compute(a.version, collected, skipped, set(), entries, (cond or set()) | run_cond)
    summary = summarize(
        den,
        remaining,
        records,
        source="run",
        conditional_source="oracle+krishiv-run" if cond else "krishiv-run",
        oracle_run=bool(a.oracle_junit),
    )
    path = write_summary(summary, target)
    print(f"wrote {path}")


def cmd_ratchet(a: argparse.Namespace) -> None:
    """Fail when a test in the previous summary's pass list is not in the current one."""

    prev = json.loads(a.previous.read_text())
    cur = json.loads(a.current.read_text())
    if prev["version"] != cur["version"]:
        raise SystemExit(f"ratchet: versions differ ({prev['version']} vs {cur['version']})")
    if cur["source"] != "run":
        raise SystemExit(f"ratchet: {a.current} is a {cur['source']} summary, not a run")
    lost = ratchet_regressions(prev, cur)
    print(f"ratchet {cur['version']}: {len(prev.get('passed_ids', []))} passed before, {cur['passed']} now")
    if lost:
        print("\n".join(f"  regressed: {t}" for t in lost))
        raise SystemExit(f"ratchet: {len(lost)} test(s) that passed no longer do")


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
    u.add_argument("--allow-drift", action="store_true", help="run although the collection differs from the summary")
    u.add_argument("--only", action="append", help="run only files starting with this path (repeatable)")
    u.add_argument("--server-cmd", nargs=argparse.REMAINDER, required=True, help="server command, last option")
    u.set_defaults(fn=cmd_run)
    c = sub.add_parser("collect", help="pytest --collect-only counts; fails on drift from the summary/manifest")
    c.add_argument("--version", required=True)
    c.add_argument("--checkout", type=Path, required=True)
    c.add_argument("--oracle-junit", type=Path, help="dir of JUnit XML from a Spark oracle run")
    c.add_argument("--update", action="store_true", help="write the collected summary instead of failing on drift")
    c.set_defaults(fn=cmd_collect)
    k = sub.add_parser("ratchet", help="fail if a test that passed in the previous summary does not pass now")
    k.add_argument("--previous", type=Path, required=True)
    k.add_argument("--current", type=Path, required=True)
    k.set_defaults(fn=cmd_ratchet)
    r = sub.add_parser("report", help="render DASHBOARD.md from the committed summaries")
    r.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
