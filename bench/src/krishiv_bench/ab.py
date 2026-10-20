"""Build-vs-build A/B: two Krishiv runners (or one runner with two ``--conf`` sets), interleaved ABBA per query.

Same protocol and statistics as ``abba`` (which this module reuses: lock, load recording, bootstrap CIs, summary):
  1. Correctness first: A and B each run the query once; their rows must match (``compare.rows_match``,
     type-aware, ordered when the query has a top-level ORDER BY). A mismatch, an error or a timeout fails the
     query; it is never timed and is excluded from the ratios.
  2. Timing: ``reps`` blocks of A B B A (``2 * reps`` samples per side). Ratio = median(B) / median(A), so a
     value below 1 means B is faster. The spread of a side is (max - min) / median of its samples.
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
from pathlib import Path
from typing import Any

from . import compare
from .abba import LOAD_INDICATIVE, bench_lock, bootstrap_ci, discover_tables, loadavg, summarize
from .adapters import Dataset
from .adapters.krishiv import KrishivEngine, RunnerCrashed, RunnerError
from .paths import data_root, dataset_dir, fmt_sf, results_dir
from .runner import git_info, load_queries, machine_info

PATTERN = ("a", "b", "b", "a")  # ABBA


def parse_conf(items: list[str] | None) -> list[tuple[str, str]]:
    """``["k=v", ...]`` -> ``[("k", "v")]``; the value may contain ``=``."""
    out = []
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"expected key=value, got {item!r}")
        out.append((key, value))
    return out


def spread(samples: list[float]) -> float:
    """(max - min) / median of one side's samples."""
    return (max(samples) - min(samples)) / statistics.median(samples)


def _side_record(label: str, eng: KrishivEngine, sha: str | None) -> dict:
    info = eng.runner_build_info()  # None for runners that predate --build-info
    if sha is None and info:
        sha = next((str(info[k]) for k in ("git_sha", "git_commit", "commit") if info.get(k)), None)
    return {
        "label": label,
        "runner": str(eng.binary),
        "version": eng.version(),
        "git_sha": sha,
        "conf": [f"{k}={v}" for k, v in eng.conf],
        "build_info": info,
        "config": eng.config(),
    }


def _fail(entry: dict, reason: str) -> dict:
    return {**entry, "status": "failed", "reason": reason}


def one_query(q, a, b, reps: int) -> dict:
    """Correctness check, then ``reps`` ABBA blocks. ``a``/``b`` need ``run`` and ``timed_run`` (see ``Engine``)."""
    entry: dict = {}
    load_before = loadavg()
    rows: dict[str, list] = {}
    for side, eng in (("a", a), ("b", b)):
        try:
            rows[side] = eng.run(q)
        except RunnerError as e:
            return _fail(entry, f"{side} error: {str(e)[:300]}")
        except RunnerCrashed as e:
            return _fail(entry, f"{side} crashed/timed out: {e}")
    entry["rows"] = len(rows["a"])
    ok, msg = compare.rows_match(rows["a"], rows["b"], compare.has_top_level_order_by(q.sql))
    if not ok:
        return _fail(entry, f"A and B differ: {msg[:300]}")
    engines = {"a": a, "b": b}
    times: dict[str, list[float]] = {"a": [], "b": []}
    for _ in range(reps):
        for side in PATTERN:
            try:
                n, secs = engines[side].timed_run(q)
                if n != entry["rows"]:
                    raise RunnerError(f"timed run returned {n} rows, expected {entry['rows']}")
            except (RunnerError, RunnerCrashed) as e:
                return _fail(entry, f"{side} timed run: {str(e)[:300]}")
            times[side].append(secs)
    ta, tb = times["a"], times["b"]
    lo, hi = bootstrap_ci(tb, ta)
    entry.update(
        status="ok",
        a_s=ta,
        b_s=tb,
        a_median_s=statistics.median(ta),
        b_median_s=statistics.median(tb),
        a_spread=spread(ta),
        b_spread=spread(tb),
        ratio=statistics.median(tb) / statistics.median(ta),
        ratio_ci95=[lo, hi],
        load_before=load_before,
        load_after=loadavg(),
    )
    print(
        f"  {q.id}: A {entry['a_median_s'] * 1000:9.1f} ms  B {entry['b_median_s'] * 1000:9.1f} ms"
        f"  B/A x{entry['ratio']:.2f}  load {load_before:.1f}->{entry['load_after']:.1f}"
    )
    return entry


def markdown(record: dict) -> str:
    s = record["summary"]
    lines = [
        "| query | A ms | B ms | B/A | 95% CI | spread A | spread B | load |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for q, e in record["queries"].items():
        if e["status"] == "ok":
            lines.append(
                f"| {q} | {e['a_median_s'] * 1000:.1f} | {e['b_median_s'] * 1000:.1f} | {e['ratio']:.2f} "
                f"| {e['ratio_ci95'][0]:.2f}-{e['ratio_ci95'][1]:.2f} | {e['a_spread']:.0%} | {e['b_spread']:.0%} "
                f"| {max(e['load_before'], e['load_after']):.1f} |"
            )
        else:
            lines.append(f"| {q} | failed | - | - | {e['reason'][:80].replace('|', '/')} | - | - | - |")
    lines.append("")
    if "geomean" in s:
        lines.append(
            f"B/A geomean over {s['queries_ok']}/{s['queries_total']} queries: **{s['geomean']:.3f}x** "
            f"(95% CI {s['geomean_ci95'][0]:.3f}-{s['geomean_ci95'][1]:.3f}; below 1 = B faster); "
            f"worst {s['worst_query']} **{s['worst_ratio']:.2f}x**; "
            f"median spread A {s['median_spread_a']:.0%}, B {s['median_spread_b']:.0%}."
        )
    else:
        lines.append(f"No query passed the A-vs-B correctness check (0/{s['queries_total']}).")
    tag = "INDICATIVE ONLY" if s["indicative"] else "load within bound"
    lines.append(f"Max 1-min load average {s['max_load']:.1f} (bound {LOAD_INDICATIVE:g}): {tag}.")
    return "\n".join(lines)


def run_ab(
    sf: float,
    dataset: str,
    a_runner: str,
    b_runner: str,
    a_conf: list[tuple[str, str]] | None = None,
    b_conf: list[tuple[str, str]] | None = None,
    reps: int = 3,
    only: list[str] | None = None,
    threads: int = 8,
    memory_limit: str = "12g",
    timeout_s: float = 600,
    suite: str = "tpch",
    a_sha: str | None = None,
    b_sha: str | None = None,
) -> Path:
    root = data_root() / dataset if suite == "clickbench" else dataset_dir(dataset, sf)
    tables = discover_tables(root)
    if not tables:
        raise SystemExit(f"no dataset at {root}")
    queries = load_queries(suite, "krishiv", only)
    ds = Dataset(dataset, sf, root, tables)
    a = KrishivEngine(threads=threads, memory_limit=memory_limit.lower(), binary=a_runner, conf=a_conf)
    b = KrishivEngine(threads=threads, memory_limit=memory_limit.lower(), binary=b_runner, conf=b_conf)
    for eng in (a, b):
        eng.timeout_s = timeout_s or None  # 0 disables the timeout
    with bench_lock() as waited:
        started = dt.datetime.now(dt.UTC)
        record: dict[str, Any] = {
            "schema": 1,
            "kind": "ab",
            "suite": suite,
            "started_utc": started.isoformat(),
            "machine": machine_info(),
            "git": git_info(),  # the harness checkout, not the compared builds (see sides.*.git_sha)
            "dataset": {"name": ds.name, "scale_factor": ds.sf, "path": str(ds.root), "tables": ds.tables},
            "reps": reps,
            "pattern": "ABBA (ratio = B / A; below 1 = B faster)",
            "samples_per_side": 2 * reps,
            "timeout_s": timeout_s,
            "cache_policy": "warm: one untimed correctness run per side per query, then timed runs; OS page cache "
            "shared, never dropped; both sides read the same Parquet files",
            "bench_lock_wait_s": waited,
            "load_at_start": loadavg(),
            "queries": {},
        }
        try:
            a.setup(ds)
            b.setup(ds)
            record["sides"] = {"a": _side_record("A", a, a_sha), "b": _side_record("B", b, b_sha)}
            for q in queries:
                record["queries"][q.id] = one_query(q, a, b, reps)
        finally:
            a.teardown()
            b.teardown()
    record["finished_utc"] = dt.datetime.now(dt.UTC).isoformat()
    record["summary"] = summarize(record, "a", "b")
    ok = [e for e in record["queries"].values() if e["status"] == "ok"]
    if ok:
        record["summary"]["median_spread_a"] = statistics.median(e["a_spread"] for e in ok)
        record["summary"]["median_spread_b"] = statistics.median(e["b_spread"] for e in ok)
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{started.strftime('%Y%m%dT%H%M%SZ')}_ab_{suite}_sf{fmt_sf(ds.sf)}.json"
    f.write_text(json.dumps(record, indent=2))
    print(markdown(record))
    print(f"result: {f}")
    return f


__all__ = ["markdown", "one_query", "parse_conf", "run_ab", "spread"]
