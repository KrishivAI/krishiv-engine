"""Interleaved ABBA comparison of Krishiv against DuckDB over the same Parquet (P2-T142).

Protocol per query:
  1. Correctness first: both engines run the query once (this also warms the OS page cache and each engine's
     own caches); Krishiv's rows must equal DuckDB's (``compare.rows_match``: type-aware, ordered when the query
     has a top-level ORDER BY). A mismatch, an engine error or a timeout marks the query ``failed`` and it is
     never timed or counted in the ratios.
  2. Timing: ``reps`` blocks of A B B A (A = Krishiv, B = DuckDB), i.e. ``2 * reps`` samples per engine, so
     drift (thermal, page cache, neighbours) hits both engines alike. Each timed run executes the query to
     completion and materialises every result row; seconds are measured inside the engine process (Krishiv: the
     runner, around parse + plan + execute + collect; DuckDB: ``execute().fetchall()``).
  3. Statistics: per-query ratio = median(Krishiv) / median(DuckDB) with a bootstrap 95% CI (resampling both
     sample sets independently); geomean of the ratios with a bootstrap CI over queries x samples.
  4. ``/proc/loadavg`` (1 minute) is recorded before and after each query; a run whose maximum exceeds
     ``LOAD_INDICATIVE`` is marked indicative only. The whole run holds ``$KRISHIV_DATA/.bench.lock`` (flock).
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import math
import random
import statistics
import threading
import time
from pathlib import Path
from typing import Any

from . import compare
from .adapters import Dataset, Engine
from .adapters.duckdb_engine import DuckDBEngine
from .adapters.krishiv import KrishivEngine, RunnerCrashed, RunnerError
from .paths import data_root, dataset_dir, fmt_sf, results_dir
from .runner import git_info, load_queries, machine_info

LOAD_INDICATIVE = 4.0
BOOTSTRAP_ITERS = 2000
PATTERN = ("krishiv", "duckdb", "duckdb", "krishiv")  # ABBA


def loadavg() -> float:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError):
        return math.nan


def bootstrap_ci(
    a: list[float], b: list[float], iters: int = BOOTSTRAP_ITERS, seed: int = 20261019
) -> tuple[float, float]:
    """95% CI of median(a) / median(b), resampling each side independently."""
    rng = random.Random(seed)
    ratios = sorted(
        statistics.median(rng.choices(a, k=len(a))) / statistics.median(rng.choices(b, k=len(b))) for _ in range(iters)
    )
    return ratios[int(0.025 * iters)], ratios[int(0.975 * iters) - 1]


def geomean(xs: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(x) for x in xs))


def geomean_ci(per_query: list[tuple[list[float], list[float]]], iters: int = BOOTSTRAP_ITERS, seed: int = 7):
    """95% CI of the geomean of per-query median ratios, resampling the samples of every query."""
    rng = random.Random(seed)
    gms = sorted(
        geomean(
            [
                statistics.median(rng.choices(a, k=len(a))) / statistics.median(rng.choices(b, k=len(b)))
                for a, b in per_query
            ]
        )
        for _ in range(iters)
    )
    return gms[int(0.025 * iters)], gms[int(0.975 * iters) - 1]


class _Interrupt:
    """Interrupts a DuckDB query after ``timeout`` seconds."""

    def __init__(self, eng: Engine, timeout: float | None):
        self.timer = threading.Timer(timeout, eng.cancel) if timeout else None

    def __enter__(self):
        if self.timer:
            self.timer.start()

    def __exit__(self, *exc):
        if self.timer:
            self.timer.cancel()


def _duck_timed(duck: Engine, query, timeout: float | None) -> float:
    with _Interrupt(duck, timeout):
        t0 = time.perf_counter()
        duck.run(query)
        return time.perf_counter() - t0


def run_abba(
    sf: float,
    dataset: str = "tpch-duckdb",
    reps: int = 3,
    only: list[str] | None = None,
    threads: int = 8,
    memory_limit: str = "12g",
    timeout_s: float = 600,
) -> Path:
    root = dataset_dir(dataset, sf)
    tables = sorted(p.stem for p in root.glob("*.parquet"))
    if not tables:
        raise SystemExit(f"no dataset at {root}")
    queries = load_queries("tpch", "krishiv", only)
    ds = Dataset(dataset, sf, root, tables)
    lock_path = data_root() / ".bench.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        t_wait = time.perf_counter()
        fcntl.flock(lock, fcntl.LOCK_EX)
        waited = time.perf_counter() - t_wait
        print(f"bench lock acquired (waited {waited:.0f}s)")
        return _run_locked(ds, queries, reps, threads, memory_limit, timeout_s, waited)


def _run_locked(ds, queries, reps, threads, memory_limit, timeout_s, lock_wait_s) -> Path:
    krishiv = KrishivEngine(threads=threads, memory_limit=memory_limit.lower())
    duck = DuckDBEngine(threads=threads, memory_limit=memory_limit.upper())
    krishiv.timeout_s = timeout_s or None  # 0 disables the timeout
    started = dt.datetime.now(dt.UTC)
    load_start = loadavg()
    record: dict[str, Any] = {
        "schema": 1,
        "kind": "abba",
        "started_utc": started.isoformat(),
        "machine": machine_info(),
        "git": git_info(),
        "engines": {
            "duckdb": {"version": duck.version(), "config": duck.config()},
        },
        "dataset": {"name": ds.name, "scale_factor": ds.sf, "path": str(ds.root), "tables": ds.tables},
        "reps": reps,
        "pattern": "ABBA (A=krishiv, B=duckdb)",
        "samples_per_engine": 2 * reps,
        "timeout_s": timeout_s,
        "cache_policy": "warm: one untimed correctness run per engine per query, then timed runs; OS page cache "
        "shared, never dropped; both engines read the same Parquet files, no table is loaded or cached up front",
        "bench_lock_wait_s": lock_wait_s,
        "load_at_start": load_start,
        "queries": {},
    }
    duck.setup(ds)
    krishiv.setup(ds)
    record["engines"]["krishiv"] = {"version": krishiv.version(), "config": krishiv.config()}
    try:
        for q in queries:
            record["queries"][q.id] = _one_query(q, krishiv, duck, reps, timeout_s)
    finally:
        krishiv.teardown()
        duck.teardown()
    record["finished_utc"] = dt.datetime.now(dt.UTC).isoformat()
    record["summary"] = summarize(record)
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{started.strftime('%Y%m%dT%H%M%SZ')}_abba_tpch_sf{fmt_sf(ds.sf)}.json"
    f.write_text(json.dumps(record, indent=2))
    print(markdown(record))
    print(f"result: {f}")
    return f


def _one_query(q, krishiv: KrishivEngine, duck: Engine, reps: int, timeout_s: float) -> dict:
    entry: dict = {}
    load_before = loadavg()
    try:
        with _Interrupt(duck, timeout_s):
            drows = duck.run(q)
    except Exception as e:
        return {"status": "failed", "reason": f"duckdb {type(e).__name__}: {str(e)[:300]}"}
    try:
        krows = krishiv.run(q)
    except RunnerError as e:
        return {"status": "failed", "reason": f"krishiv error: {str(e)[:300]}"}
    except RunnerCrashed as e:
        return {"status": "failed", "reason": f"krishiv crashed/timed out: {e}"}
    ordered = compare.has_top_level_order_by(q.sql)
    ok, msg = compare.rows_match(krows, drows, ordered)
    entry["rows"] = len(drows)
    if not ok:
        return {**entry, "status": "failed", "reason": f"wrong result: {msg[:300]}"}
    times: dict[str, list[float]] = {"krishiv": [], "duckdb": []}
    for _ in range(reps):
        for who in PATTERN:
            if who == "krishiv":
                try:
                    rows, secs = krishiv.timed_run(q)
                    if rows != len(drows):
                        raise RunnerError(f"timed run returned {rows} rows, expected {len(drows)}")
                except (RunnerError, RunnerCrashed) as e:
                    return {**entry, "status": "failed", "reason": f"krishiv timed run: {str(e)[:300]}"}
            else:
                try:  # a DuckDB failure (interrupt, out of memory) fails this query, not the whole run
                    secs = _duck_timed(duck, q, timeout_s)
                except Exception as e:
                    return {
                        **entry,
                        "status": "failed",
                        "reason": f"duckdb timed run {type(e).__name__}: {str(e)[:300]}",
                    }
            times[who].append(secs)
    k, d = times["krishiv"], times["duckdb"]
    lo, hi = bootstrap_ci(k, d)
    entry.update(
        status="ok",
        krishiv_s=k,
        duckdb_s=d,
        krishiv_median_s=statistics.median(k),
        duckdb_median_s=statistics.median(d),
        ratio=statistics.median(k) / statistics.median(d),
        ratio_ci95=[lo, hi],
        load_before=load_before,
        load_after=loadavg(),
    )
    print(
        f"  {q.id}: krishiv {entry['krishiv_median_s'] * 1000:9.1f} ms"
        f"  duckdb {entry['duckdb_median_s'] * 1000:8.1f} ms"
        f"  x{entry['ratio']:.2f}  load {entry['load_before']:.1f}->{entry['load_after']:.1f}"
    )
    return entry


def summarize(record: dict) -> dict:
    ok = {q: e for q, e in record["queries"].items() if e["status"] == "ok"}
    failed = {q: e["reason"] for q, e in record["queries"].items() if e["status"] != "ok"}
    loads = [x for e in ok.values() for x in (e["load_before"], e["load_after"]) if not math.isnan(x)]
    loads.append(record["load_at_start"])
    max_load = max((x for x in loads if not math.isnan(x)), default=math.nan)
    s: dict = {
        "queries_total": len(record["queries"]),
        "queries_ok": len(ok),
        "failed": failed,
        "max_load": max_load,
        "indicative": (max_load > LOAD_INDICATIVE) or math.isnan(max_load),
    }
    if ok:
        ratios = {q: e["ratio"] for q, e in ok.items()}
        worst = max(ratios, key=lambda q: ratios[q])
        lo, hi = geomean_ci([(e["krishiv_s"], e["duckdb_s"]) for e in ok.values()])
        s.update(
            geomean=geomean(list(ratios.values())), geomean_ci95=[lo, hi], worst_query=worst, worst_ratio=ratios[worst]
        )
    return s


def markdown(record: dict) -> str:
    s = record["summary"]
    lines = [
        "| query | Krishiv ms | DuckDB ms | ratio | 95% CI | load |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for q, e in record["queries"].items():
        if e["status"] == "ok":
            lines.append(
                f"| {q} | {e['krishiv_median_s'] * 1000:.1f} | {e['duckdb_median_s'] * 1000:.1f} | {e['ratio']:.2f} "
                f"| {e['ratio_ci95'][0]:.2f}-{e['ratio_ci95'][1]:.2f} | {max(e['load_before'], e['load_after']):.1f} |"
            )
        else:
            lines.append(f"| {q} | failed | - | - | {e['reason'][:80].replace('|', '/')} | - |")
    lines.append("")
    if "geomean" in s:
        lines.append(
            f"Geomean over {s['queries_ok']}/{s['queries_total']} queries: **{s['geomean']:.2f}x** "
            f"(95% CI {s['geomean_ci95'][0]:.2f}-{s['geomean_ci95'][1]:.2f}); worst {s['worst_query']} "
            f"**{s['worst_ratio']:.2f}x**."
        )
    else:
        lines.append(f"No query passed the correctness check (0/{s['queries_total']}).")
    tag = "INDICATIVE ONLY" if s["indicative"] else "load within bound"
    lines.append(f"Max 1-min load average {s['max_load']:.1f} (bound {LOAD_INDICATIVE:g}): {tag}.")
    return "\n".join(lines)


def report_from_file(path: Path) -> str:
    return markdown(json.loads(path.read_text()))


__all__ = ["bootstrap_ci", "geomean", "markdown", "report_from_file", "run_abba", "summarize"]
