"""Interleaved ABBA comparison of Krishiv against DuckDB over the same Parquet (P2-T142).

Protocol per query:
  1. Correctness first: both engines run the query once (this also warms the OS page cache and each engine's
     own caches); Krishiv's rows must equal DuckDB's (``compare.rows_match``: type-aware, ordered when the query
     has a top-level ORDER BY). A mismatch, an engine error or a timeout marks the query ``failed`` and it is
     never timed or counted in the ratios. DuckDB runs its own text of each query (``load_queries(suite,
     "duckdb")``) and sorts NULLs as Spark does (first ascending, last descending), Krishiv's semantics.
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
import os
import random
import statistics
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from importlib.metadata import version as dist_version
from pathlib import Path
from typing import Any

from . import compare
from .adapters import Dataset, Engine
from .adapters.duckdb_engine import SPARK_NULL_ORDER, DuckDBEngine
from .adapters.krishiv import KrishivEngine, RunnerCrashed, RunnerError
from .errors import describe_error
from .paths import data_root, dataset_dir, fmt_sf, results_dir
from .refcache import RefCache
from .resources import evict_page_cache
from .runner import cold_record, git_info, load_queries, machine_info

LOAD_INDICATIVE = 4.0


def load_bound(record: dict) -> float:
    """The 1-minute load above which a run is indicative only.

    The run's own threads drive the load average up (``--threads 12`` reaches about 12 on an idle
    machine), so the bound during the run is ``LOAD_INDICATIVE`` of *other* load on top of them,
    counting at most the machine's CPUs. A record without ``threads`` (older results) keeps the
    plain bound.

    This is less sensitive than the plain bound: short queries keep the run's own share of the
    average below ``threads``, and other work of up to ``LOAD_INDICATIVE`` plus that gap then goes
    unnoticed during the run. Other work already there when the run starts is caught by the load
    at start, which is judged against the plain bound.
    """
    threads = record.get("threads")
    if not isinstance(threads, int) or threads <= 0:
        return LOAD_INDICATIVE
    return LOAD_INDICATIVE + min(threads, os.cpu_count() or threads)


def load_line(record: dict) -> str:
    """The report line on the machine load: what was measured, the bound and why it is indicative."""
    s = record["summary"]
    # A record saved before the summary held the bound is rendered as saved, not re-summarized.
    bound = s.get("load_bound", load_bound(record))
    own = bound - LOAD_INDICATIVE
    makeup = (
        f"{LOAD_INDICATIVE:g} of other load plus the run's {own:g} threads"
        if own > 0
        else f"{LOAD_INDICATIVE:g}, as the record has no thread count"
    )
    start = record["load_at_start"]
    verdict = "INDICATIVE ONLY" if s["indicative"] else "load within bound"
    reasons = []
    if math.isnan(start) or start > LOAD_INDICATIVE:
        reasons.append(f"load at start {start:.1f} is not at most {LOAD_INDICATIVE:g}")
    if math.isnan(s["max_load"]) or s["max_load"] > bound:
        reasons.append(f"max load {s['max_load']:.1f} is above the bound")
    why = f" ({'; '.join(reasons)})" if reasons else ""
    return (
        f"Load: at start {start:.1f}, max 1-min average {s['max_load']:.1f}; bound {bound:g} "
        f"({makeup}): {verdict}{why}."
    )


BOOTSTRAP_ITERS = 2000
PATTERN = ("krishiv", "duckdb", "duckdb", "krishiv")  # ABBA


def loadavg() -> float:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError):
        return math.nan


def wait_idle(limit: float = 2.0, max_wait_s: float = 900.0) -> None:
    """Lets the 1-minute load average decay (e.g. after the previous benchmark run) before recording anything."""
    deadline = time.monotonic() + max_wait_s
    while loadavg() > limit and time.monotonic() < deadline:
        time.sleep(15)


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


def discover_tables(root: Path) -> list[str]:
    """Tables under `root`: each `<t>.parquet` file, and each directory of part files.

    A directory (`hits/` or `hits.parquet/`) is one table, so a partitioned download such as
    ClickBench's `hits_0.parquet` … `hits_99.parquet` placed in `hits/` registers as one `hits`
    relation rather than one table per part.
    """
    if not root.is_dir():
        return []
    names: set[str] = set()
    for p in root.iterdir():
        if p.is_file() and p.suffix == ".parquet":
            names.add(p.stem)
        elif p.is_dir() and any(p.glob("*.parquet")):
            names.add(p.name.removesuffix(".parquet"))
    return sorted(names)


@contextmanager
def bench_lock() -> Generator[float]:
    """Holds ``$KRISHIV_DATA/.bench.lock`` (flock) for the whole run; yields the seconds spent waiting."""
    lock_path = data_root() / ".bench.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        t_wait = time.perf_counter()
        fcntl.flock(lock, fcntl.LOCK_EX)
        waited = time.perf_counter() - t_wait
        print(f"bench lock acquired (waited {waited:.0f}s)")
        yield waited


def run_abba(
    sf: float,
    dataset: str = "tpch-duckdb",
    reps: int = 3,
    only: list[str] | None = None,
    threads: int = 8,
    memory_limit: str = "12g",
    timeout_s: float = 600,
    suite: str = "tpch",
    runner: str | None = None,
    reference_cache: bool = True,
    refresh_reference: bool = False,
    cold: bool = False,
) -> Path:
    root = data_root() / dataset if suite == "clickbench" else dataset_dir(dataset, sf)
    tables = discover_tables(root)
    if not tables:
        raise SystemExit(f"no dataset at {root}")
    queries = load_queries(suite, "krishiv", only)
    # DuckDB runs its own text of each query (ClickBench q29 is written per dialect; `<suite>/duckdb/` overrides).
    duck_sql = {q.id: q for q in load_queries(suite, "duckdb", only)}
    queries = [(q, duck_sql.get(q.id, q)) for q in queries]
    ds = Dataset(dataset, sf, root, tables)
    with bench_lock() as waited:
        wait_idle()
        cache = (
            RefCache(refresh=refresh_reference, engine="duckdb", version=dist_version("duckdb"))
            if reference_cache
            else None
        )
        return _run_locked(ds, queries, reps, threads, memory_limit, timeout_s, waited, suite, runner, cache, cold)


def _run_locked(
    ds, queries, reps, threads, memory_limit, timeout_s, lock_wait_s, suite, runner=None, cache=None, cold=False
) -> Path:
    krishiv = KrishivEngine(threads=threads, memory_limit=memory_limit.lower(), binary=runner)
    # Krishiv sorts NULLs as Spark does; the reference must too, or an ORDER BY over a nullable key differs.
    duck = DuckDBEngine(threads=threads, memory_limit=memory_limit.upper(), null_order=SPARK_NULL_ORDER)
    krishiv.timeout_s = timeout_s or None  # 0 disables the timeout
    started = dt.datetime.now(dt.UTC)
    load_start = loadavg()
    record: dict[str, Any] = {
        "schema": 1,
        "kind": "abba",
        "suite": suite,
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
        "cold": cold_record(cold),
        "cache_policy": "cold: the dataset is dropped from the OS page cache before every timed run"
        if cold
        else "warm: one untimed correctness run per engine per query, then timed runs; OS page cache "
        "shared, never dropped; both engines read the same Parquet files, no table is loaded or cached up front",
        "bench_lock_wait_s": lock_wait_s,
        "load_at_start": load_start,
        "threads": threads,
        "queries": {},
    }
    duck.setup(ds)
    krishiv.setup(ds)
    record["engines"]["krishiv"] = {"version": krishiv.version(), "config": krishiv.config()}
    try:
        for q, dq in queries:
            record["queries"][q.id] = _one_query(
                q, krishiv, duck, reps, timeout_s, dq, cache, suite, ds, ds.root if cold else None
            )
    finally:
        krishiv.teardown()
        duck.teardown()
    record["finished_utc"] = dt.datetime.now(dt.UTC).isoformat()
    record["summary"] = summarize(record)
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{started.strftime('%Y%m%dT%H%M%SZ')}_abba_{suite}_sf{fmt_sf(ds.sf)}.json"
    f.write_text(json.dumps(record, indent=2))
    print(markdown(record))
    print(f"result: {f}")
    return f


def _one_query(
    q,
    krishiv: KrishivEngine,
    duck: Engine,
    reps: int,
    timeout_s: float,
    duck_q=None,
    cache: RefCache | None = None,
    suite="",
    ds=None,
    cold_root: Path | None = None,
) -> dict:
    """Correctness, then ABBA timing of ``q``; DuckDB runs ``duck_q`` (its own text of the query), ``q`` if None.

    With a ``cache`` DuckDB's rows come from it when the same query over the same data was answered before by this
    version (the untimed run is skipped; the timed ABBA runs stay live, since interleaving needs both engines on
    the same machine at the same time), and a fresh answer is stored."""
    duck_q = duck_q or q
    entry: dict = {}
    load_before = loadavg()
    akey = cache.answer_key(suite, ds, duck_q) if cache is not None and ds is not None else None
    hit = cache.load(akey) if cache is not None and akey else None
    try:
        if hit:
            drows = hit[1]
            entry["reference_cached_utc"] = hit[0]["created_utc"]
        else:
            with _Interrupt(duck, timeout_s):
                drows = duck.run(duck_q)
            if cache is not None and akey:
                cache.store(akey, drows)
    except Exception as e:
        return {"status": "failed", "reason": f"duckdb {describe_error(e)}"}
    try:
        krows = krishiv.run(q)
    except RunnerError as e:
        return {"status": "failed", "reason": f"krishiv error: {describe_error(e)}"}
    except RunnerCrashed as e:
        return {"status": "failed", "reason": f"krishiv crashed/timed out: {e}"}
    ordered = compare.has_top_level_order_by(q.sql)
    ok, msg = compare.rows_match(krows, drows, ordered, sql=q.sql)
    entry["rows"] = len(drows)
    if not ok:
        return {**entry, "status": "failed", "reason": f"wrong result: {msg[:300]}"}
    times: dict[str, list[float]] = {"krishiv": [], "duckdb": []}
    for _ in range(reps):
        for who in PATTERN:
            if cold_root is not None:
                evict_page_cache(cold_root)
            if who == "krishiv":
                try:
                    rows, secs = krishiv.timed_run(q)
                    if rows != len(drows):
                        raise RunnerError(f"timed run returned {rows} rows, expected {len(drows)}")
                except (RunnerError, RunnerCrashed) as e:
                    return {**entry, "status": "failed", "reason": f"krishiv timed run: {describe_error(e)}"}
            else:
                try:  # a DuckDB failure (interrupt, out of memory) fails this query, not the whole run
                    secs = _duck_timed(duck, duck_q, timeout_s)
                except Exception as e:
                    return {
                        **entry,
                        "status": "failed",
                        "reason": f"duckdb timed run {describe_error(e)}",
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


def summarize(record: dict, a: str = "krishiv", b: str = "duckdb") -> dict:
    """Run summary; ``a``/``b`` name the per-query sample keys (``<a>_s``, ``<b>_s``) of the compared sides."""
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
        "load_bound": load_bound(record),
        # Other work on the machine before the run (nothing of ours is running yet), or more load
        # during it than the run's own threads explain.
        "indicative": (
            math.isnan(max_load)
            or math.isnan(record["load_at_start"])
            or record["load_at_start"] > LOAD_INDICATIVE
            or max_load > load_bound(record)
        ),
    }
    if ok:
        ratios = {q: e["ratio"] for q, e in ok.items()}
        worst = max(ratios, key=lambda q: ratios[q])
        lo, hi = geomean_ci([(e[f"{a}_s"], e[f"{b}_s"]) for e in ok.values()])
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
    lines.append(load_line(record))
    return "\n".join(lines)


def report_from_file(path: Path) -> str:
    return markdown(json.loads(path.read_text()))


__all__ = [
    "bench_lock",
    "bootstrap_ci",
    "geomean",
    "markdown",
    "report_from_file",
    "run_abba",
    "summarize",
    "wait_idle",
]
