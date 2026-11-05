"""Reproduce the reference engines' published results on this machine (P2-T156).

Engines: Krishiv, DuckDB (the reference), Polars, DataFusion, Sail, Spark local[*], and (opt-in) ClickHouse via chdb;
all read the same Parquet, each in its own worker process (``worker.py``) pinned to the same thread count and
memory cap.

Correctness first (every engine vs DuckDB, and vs Krishiv when Krishiv matched DuckDB): one untimed run per
engine and query; an engine whose rows differ, errors or times out is recorded (``mismatch``, ``mismatch-dialect``
for a documented dialect or type difference listed in ``dialect.py``, ``error`` or ``timeout``) for that query and
never timed.

Two timing methods, selected with ``methods``:

* ``ours``: interleaved, correctness-checked. ``reps`` blocks per query; a block runs every engine that passed in
  list order and then in reverse (A B C ... C B A), so each engine gets ``2 * reps`` samples and drift hits all
  engines alike. Per-query ratio = median(engine) / median(DuckDB) with a bootstrap 95% CI; geomean with CI.
* ``theirs``: the published methodology. A fresh process per (engine, query); ``runs`` hot runs in that process,
  the best one is the result (the first run is kept as ``first_s``); a per-run timeout. Runs and timeout come from
  the cited source (``PUBLISHED``). The OS page cache is never dropped (no root), so "cold first" is a
  first-run-in-a-new-process number, not a cold-disk one.

``/proc/loadavg`` is recorded around every query; the run is marked indicative when the load at start (before any
work of ours, so free of the benchmark's own contribution) exceeds ``LOAD_INDICATIVE``.
The run holds ``$KRISHIV_DATA/.bench.lock`` (flock).
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import math
import statistics
import time
from collections.abc import Mapping, MutableMapping
from importlib.metadata import version as dist_version
from pathlib import Path
from typing import Any

from . import compare, result_check
from .abba import LOAD_INDICATIVE, bootstrap_ci, discover_tables, geomean, geomean_ci, loadavg, wait_idle
from .adapters import Dataset, Engine, Query
from .adapters.base import check_query, result_rows
from .adapters.process import EngineCrashed, EngineError, EngineTimeout, ProcessEngine
from .dialect import dialect_reason
from .errors import describe_error, error_fields
from .h2o import is_h2o
from .paths import data_root, dataset_dir, fmt_sf, results_dir
from .refcache import RefCache
from .resources import evict_page_cache
from .runner import cold_record, git_info, load_queries, machine_info

REFERENCE = "duckdb"
DEFAULT_ENGINES = ["krishiv", "duckdb", "polars", "datafusion", "sail", "spark-local"]
SUITE_DATASET = {"tpch": "tpch-duckdb", "tpcds": "tpcds", "clickbench": "clickbench"}

# Published-method parameters. The runs/timeout are the ones P2-T156 specifies per source; where a source does not
# state a per-query timeout the value is ours and marked as such in the report.
PUBLISHED: dict[str, dict[str, Any]] = {
    "tpch": {"runs": 5, "timeout_s": 60.0, "source": "Polars TPC-H benchmark: 5 runs, 60 s limit"},
    "tpcds": {"runs": 5, "timeout_s": 60.0, "source": "Polars TPC-H method applied to TPC-DS: 5 runs, 60 s limit"},
    "clickbench": {"runs": 3, "timeout_s": 300.0, "source": "ClickBench: 3 runs, first cold; 300 s limit is ours"},
    "h2o-groupby": {"runs": 2, "timeout_s": 600.0, "source": "db-benchmark: each question twice; 600 s limit is ours"},
    "h2o-join": {"runs": 2, "timeout_s": 600.0, "source": "db-benchmark: each question twice; 600 s limit is ours"},
    "micro": {"runs": 2, "timeout_s": 300.0, "source": "operator microbenchmarks: ours; no published method"},
}
LIBS = ["duckdb", "polars", "datafusion", "pysail", "pyspark", "pyarrow", "chdb"]


def resolve_dataset(suite: str, dataset: str | None, sf: float) -> Dataset:
    name = dataset or SUITE_DATASET[suite]
    # clickbench, the h2o suites and micro name a directory relative to $KRISHIV_DATA (h2o: h2o/<version>/<label>)
    root = data_root() / name if suite in ("clickbench", "micro") or is_h2o(suite) else dataset_dir(name, sf)
    tables = discover_tables(root)
    if not tables:
        raise SystemExit(f"no dataset at {root}")
    return Dataset(name, sf, root, tables)


def suite_queries(suite: str, ds: Dataset, engine: str, only: list[str] | None) -> list[Query]:
    """The suite's queries for ``engine``; the micro suite derives them from the dataset it runs on."""
    if suite == "micro":
        from . import micro

        return micro.load_queries(ds.root, engine, only)
    return load_queries(suite, engine, only)


def _fail(status: str, e: Exception | str) -> dict:
    if isinstance(e, Exception):
        fields = error_fields(e)
        return {"status": status, "reason": fields["error"], **fields}
    return {"status": status, "reason": e[:1000]}


def _status_of(e: Exception) -> str:
    return "timeout" if isinstance(e, EngineTimeout) else "error"


# Polars' SQL front end does not run every query; these are its errors for a query it cannot plan or execute. Such a
# query is reported "unsupported" with the error, never rewritten.
POLARS_UNSUPPORTED = frozenset(
    {
        "SQLInterfaceError",
        "SQLSyntaxError",
        "UnsuitableSQLError",
        "ColumnNotFoundError",
        "SchemaFieldNotFoundError",
        "InvalidOperationError",
        "ComputeError",
        "SchemaError",
    }
)


def _answer_status(name: str, e: Exception) -> str:
    """Status of an engine whose untimed answer run raised ``e``."""
    if name.startswith("polars") and isinstance(e, EngineError) and str(e).split(":")[0] in POLARS_UNSUPPORTED:
        return "unsupported"
    return _status_of(e)


def setup_engines(engines: MutableMapping[str, Any], ds: Dataset) -> dict[str, dict]:
    """Set up every engine; one that fails is torn down, removed from ``engines`` and returned as a failure verdict.

    A failed reference engine is fatal for the run (nothing can be checked without it), so it raises.
    """
    failed: dict[str, dict] = {}
    for n, e in list(engines.items()):
        try:
            e.setup(ds)
        except (EngineError, EngineTimeout, EngineCrashed) as exc:
            e.teardown()
            if n == REFERENCE:
                raise SystemExit(f"{REFERENCE} setup failed: {exc}") from exc
            failed[n] = _fail(_status_of(exc), f"setup: {describe_error(exc)}")
            del engines[n]
    return failed


def correctness(
    qm: Mapping[str, Query],
    engines: Mapping[str, Engine],
    cache: RefCache | None = None,
    suite: str = "",
    ds: Dataset | None = None,
) -> tuple[dict[str, dict], int]:
    """One untimed run per engine (``qm``: the query as each engine runs it); verdict per engine against DuckDB
    (and Krishiv). Returns (verdicts, row count).

    With a ``cache`` DuckDB's answer is read from it, not run, when the same query over the same data was answered
    before by this version (its timed runs stay live: the engines are interleaved); a fresh answer is stored."""
    rows: dict[str, list] = {}
    verdict: dict[str, dict] = {}
    akey = cache.answer_key(suite, ds, check_query(qm[REFERENCE])) if cache and ds is not None else None
    for name, eng in engines.items():
        if cache and akey and name == REFERENCE and (hit := cache.load(akey)):
            rows[name] = hit[1]
            continue
        try:
            rows[name] = eng.run(check_query(qm[name]))
            if cache and akey and name == REFERENCE:
                cache.store(akey, rows[name])
        except (EngineError, EngineTimeout, EngineCrashed) as e:
            verdict[name] = _fail(_answer_status(name, e), e)
    ref = rows.get(REFERENCE)
    if ref is None:
        for name in rows:
            verdict[name] = _fail("error", "no DuckDB reference result")
        return verdict, 0
    for name, r in rows.items():
        verdict[name] = judge(name, r, ref, qm, rows)
    return verdict, result_rows(qm[REFERENCE], ref)


def judge(name: str, r: list, ref: list, qm: Mapping[str, Query], rows: Mapping[str, list]) -> dict:
    """Verdict of engine ``name``'s rows ``r`` against DuckDB's ``ref`` (and Krishiv's, when ``rows`` has them)."""
    if name == REFERENCE:
        return {"status": "ok"}
    sql = qm[REFERENCE].sql
    ordered = compare.has_top_level_order_by(sql)
    res = compare.compare_rows(r, ref, ordered, sql=sql)
    if res.ok:
        v: dict = {"status": "ok"}
        if res.status != "ok":
            v["note"] = res.message  # e.g. tie-at-limit
    else:
        reason = dialect_reason(name, qm[name].suite, qm[name].id, r, ref)
        v = _fail("mismatch-dialect", f"{reason}: {res.message}") if reason else _fail("mismatch", res.message)
    if res.ok and "krishiv" in rows and name != "krishiv":
        kok, kmsg = compare.rows_match(r, rows["krishiv"], ordered, sql=sql)
        if not kok:  # equal to DuckDB but not to Krishiv: Krishiv itself differs (also flagged on its own verdict)
            v["vs_krishiv"] = kmsg[:200]
    return v


def _note_peak(peaks: dict[str, int], name: str, eng: Engine) -> None:
    """Keeps the highest per-run peak RSS of ``eng`` (its worker process tree) seen for the query."""
    peak = eng.last_peak_rss_bytes
    if peak is not None:
        peaks[name] = max(peaks.get(name, 0), peak)


def _baselines(qm, engines, live, nrows, entry, samples) -> dict[str, dict | None]:
    """One untimed run per engine through the timed path: its checksum is what every timed run must reproduce.
    An engine that fails here is dropped from ``samples``."""
    sums: dict[str, dict | None] = {}
    for name in list(live):
        try:
            n, _ = engines[name].timed_run(qm[name])
            if n != nrows:
                raise EngineError(f"result run returned {n} rows, expected {nrows}")
            sums[name] = engines[name].result_checksum()
        except (EngineError, EngineTimeout, EngineCrashed) as e:
            entry["engines"][name] = _fail(_status_of(e), f"result run: {e}")
            samples.pop(name, None)
    return sums


def _checksummed(q: Query) -> bool:
    """Whether timed runs of ``q`` must reproduce the verified result checksum. Not when a ``LIMIT`` may cut a tie:
    ``compare`` accepts the other tied rows (``tie-at-limit``), so a correct engine may return another set."""
    return compare.derive_limit(q.sql)[0] is None


def check_description(name: str, baseline: dict | None, q: Query | None = None) -> str:
    """How the timed runs of ``name`` were checked against the verified answer, recorded with its timings.

    Only an adapter that returns a result checksum compares more than the row count; Krishiv's ``RUN`` returns only
    rows and seconds, so its checksum is that of a separate, untimed ``DUMP`` of the query.
    """
    if q is not None and not _checksummed(q):
        return "row count only: the query's LIMIT may cut a tie differently from run to run"
    if baseline is None:
        return "row count only: the adapter returns no result checksum"
    if name == "krishiv":
        return "row count of the timed run; checksum of a separate untimed DUMP run (RUN returns only rows and seconds)"
    return "row count and order-independent checksum of the timed run's own result"


def _timed(eng: Engine, q: Query, nrows: int, baseline: dict | None, cold_root: Path | None = None) -> float:
    """One timed run whose full result (row count and checksum) must equal the verified one; returns the seconds."""
    if cold_root is not None:
        evict_page_cache(cold_root)
    n, secs = eng.timed_run(q)
    if n != nrows:
        raise EngineError(f"timed run returned {n} rows, expected {nrows}")
    got = eng.result_checksum()
    if baseline is not None and _checksummed(q) and (got is None or not result_check.same(got, baseline)):
        raise EngineError("the timed run's result checksum differs from the verified run's")
    return secs


def _finish(entry, samples, peaks, cpus, qm, engines, profile) -> None:
    """Per-engine medians, peaks, CPU and the ratios against DuckDB (and one Krishiv profile run)."""
    base = samples.get(REFERENCE)
    for name, s in samples.items():
        e = entry["engines"][name]
        e.update(samples_s=s, median_s=statistics.median(s))
        if name in peaks:
            e["peak_rss_bytes"] = peaks[name]
        if cpus[name]:
            e["cpu_seconds_median"] = statistics.median(cpus[name])
        if base and name != REFERENCE:
            lo, hi = bootstrap_ci(s, base)
            e.update(ratio=e["median_s"] / statistics.median(base), ratio_ci95=[lo, hi])
    if profile and "krishiv" in samples:  # one extra untimed-for-the-ratio run, after all samples
        try:
            entry["engines"]["krishiv"]["profile"] = engines["krishiv"].profile(qm["krishiv"])
        except (EngineError, EngineTimeout, EngineCrashed, NotImplementedError) as e:
            entry["engines"]["krishiv"]["profile_error"] = str(e)[:300]


def ours_query(
    qm: Mapping[str, Query],
    engines: Mapping[str, Engine],
    reps: int,
    timeout_s: float,
    profile: bool = False,
    cache: RefCache | None = None,
    suite: str = "",
    ds: Dataset | None = None,
    cold_root: Path | None = None,
) -> dict:
    load_before = loadavg()
    verdict, nrows = correctness(qm, engines, cache, suite, ds)
    entry: dict[str, Any] = {"rows": nrows, "engines": {n: dict(v) for n, v in verdict.items()}}
    live = [n for n in engines if verdict.get(n, {}).get("status") == "ok"]
    if REFERENCE not in live:
        return entry
    samples: dict[str, list[float]] = {n: [] for n in live}
    peaks: dict[str, int] = {}
    cpus: dict[str, list[float]] = {n: [] for n in live}
    sums = _baselines(qm, engines, live, nrows, entry, samples)
    for name in sums:
        entry["engines"][name]["result_check"] = check_description(name, sums[name], qm[name])
    for _ in range(reps):
        for name in [*live, *reversed(live)]:
            if name not in samples:
                continue
            try:
                secs = _timed(engines[name], qm[name], nrows, sums[name], cold_root)
                samples[name].append(secs)
                _note_peak(peaks, name, engines[name])
                if (cpu := engines[name].last_cpu_seconds) is not None:
                    cpus[name].append(cpu)
            except (EngineError, EngineTimeout, EngineCrashed) as e:
                entry["engines"][name] = _fail(_status_of(e), f"timed run: {e}")
                del samples[name]
    _finish(entry, samples, peaks, cpus, qm, engines, profile)
    entry["load_before"], entry["load_after"] = load_before, loadavg()
    return entry


def sequential_query(
    qm: Mapping[str, Query],
    names: list[str],
    ds: Dataset,
    threads: int,
    memory_gb: int,
    cap_gb: int | None,
    samples_n: int,
    timeout_s: float,
    profile: bool = False,
    cache: RefCache | None = None,
    suite: str = "",
    cold_root: Path | None = None,
) -> tuple[dict, dict[str, dict]]:
    """One query, one engine alive at a time (DuckDB, then Krishiv, then the rest): a fresh worker per engine inside
    its own memory cgroup runs the verified answer, one untimed run through the timed path (its checksum is the
    baseline) and ``samples_n`` timed runs, each of which must give the same full result. Returns the entry and the
    engines' ``info`` (version, config, build).

    With a ``cache`` the reference engine's answer, samples, CPU and peak are read from it when the same query over the
    same data was measured before with this version, threads, memory and sample count, and stored after a fresh run;
    the other engines are always run and checked against the cached rows."""
    load_before = loadavg()
    order = [REFERENCE] + [n for n in ("krishiv",) if n in names and n != REFERENCE]
    order += [n for n in names if n not in order]
    verdicts: dict[str, dict] = {}
    rows: dict[str, list] = {}
    infos: dict[str, dict] = {}
    samples: dict[str, list[float]] = {}
    peaks: dict[str, int] = {}
    cpus: dict[str, list[float]] = {}
    nrows = 0
    for name in order:
        ckey = (
            cache.timing_key(suite, ds, qm[name], threads, memory_gb, samples_n, cap_gb, cold_root is not None)
            if cache and name == REFERENCE
            else None
        )
        hit = cache.load(ckey) if cache and ckey else None
        if hit:
            meta, rows[name] = hit
            infos[name] = meta["info"]
            nrows = result_rows(qm[REFERENCE], rows[name])
            verdicts[name] = {"status": "ok", "result_check": meta["result_check"], "cached_utc": meta["created_utc"]}
            samples[name], cpus[name] = meta["samples_s"], meta["cpu_seconds"]
            if meta.get("peak_rss_bytes") is not None:
                peaks[name] = meta["peak_rss_bytes"]
            continue
        # Polars' default engine runs first; if its worker is killed (out of memory), the engine is run again with
        # Polars' streaming engine ("polars-streaming") and the result says so.
        attempts = [name, "polars-streaming"] if name == "polars" else [name]
        for attempt, run_name in enumerate(attempts):
            eng = ProcessEngine(run_name, threads, memory_gb, timeout_s, cap_gb=cap_gb)
            phase = "answer"
            crashed = False
            try:
                eng.setup(ds)
                infos[name] = {"version": eng.version(), "config": eng.config(), "build": eng.build_info()}
                rows[name] = eng.run(check_query(qm[name]))
                phase = "timing"
                if name == REFERENCE:
                    nrows = result_rows(qm[REFERENCE], rows[name])
                    verdicts[name] = {"status": "ok"}
                elif REFERENCE not in rows:
                    verdicts[name] = _fail("error", "no DuckDB reference result")
                else:
                    verdicts[name] = judge(name, rows[name], rows[REFERENCE], qm, rows)
                if verdicts[name]["status"] != "ok":
                    break
                base_n, _ = eng.timed_run(qm[name])
                if base_n != nrows:
                    raise EngineError(f"result run returned {base_n} rows, expected {nrows}")
                baseline = eng.result_checksum()
                verdicts[name]["result_check"] = check_description(name, baseline, qm[name])
                samples[name], cpus[name] = [], []
                for _ in range(samples_n):
                    samples[name].append(_timed(eng, qm[name], nrows, baseline, cold_root))
                    _note_peak(peaks, name, eng)
                    if (cpu := eng.last_cpu_seconds) is not None:
                        cpus[name].append(cpu)
                if profile and name == "krishiv":  # one more run, after the samples, in the worker that took them
                    try:
                        verdicts[name]["profile"] = eng.profile(qm[name])
                    except (EngineError, EngineTimeout, EngineCrashed, NotImplementedError) as e:
                        verdicts[name]["profile_error"] = str(e)[:300]
                if attempt:
                    verdicts[name]["engine_mode"] = "streaming (the default engine ran out of memory)"
            except (EngineError, EngineTimeout, EngineCrashed) as e:
                crashed = isinstance(e, EngineCrashed)
                status = _answer_status(name, e) if phase == "answer" else _status_of(e)
                verdicts[name] = _fail(status, e)
                samples.pop(name, None)
            finally:
                eng.teardown()
            if not (crashed and attempt + 1 < len(attempts)):
                break
        if cache and ckey and verdicts.get(name, {}).get("status") == "ok" and name in samples:
            cache.store(
                ckey,
                rows[name],
                {
                    "info": infos[name],
                    "result_check": verdicts[name]["result_check"],
                    "samples_s": samples[name],
                    "cpu_seconds": cpus[name],
                    "peak_rss_bytes": peaks.get(name),
                },
            )
        if name not in (REFERENCE, "krishiv"):
            rows.pop(name, None)  # only DuckDB's and Krishiv's rows are compared against later engines
    entry: dict[str, Any] = {"rows": nrows, "engines": {n: dict(v) for n, v in verdicts.items()}}
    _finish(entry, samples, peaks, cpus, qm, {}, False)  # without DuckDB samples: medians and peaks, no ratios
    entry["load_before"], entry["load_after"] = load_before, loadavg()
    return entry, infos


def theirs_query(
    name: str,
    q: Query,
    ds: Dataset,
    threads: int,
    memory_gb: int,
    runs: int,
    timeout_s: float,
    nrows: int | None = None,
) -> dict:
    """Published method: a fresh process, ``runs`` hot runs in it, best of them; per-run timeout.

    ``nrows``: the row count the correctness run produced; a timed run returning another count fails the pair."""
    eng = ProcessEngine(name, threads, memory_gb, timeout_s)
    try:
        eng.setup(ds)
        times: list[float] = []
        peaks: dict[str, int] = {}
        first_sum: dict | None = None
        for i in range(runs):
            n, secs = eng.timed_run(q)
            _note_peak(peaks, name, eng)
            if n != nrows:
                raise EngineError(f"timed run returned {n} rows, expected {nrows}")
            got = eng.result_checksum()  # outside the measured window; every run must give the first run's result
            if i == 0:
                first_sum = got
            elif first_sum is not None and _checksummed(q) and (got is None or not result_check.same(got, first_sum)):
                raise EngineError("a later run's result checksum differs from the first run's")
            times.append(secs)
        res: dict[str, Any] = {"status": "ok", "runs_s": times, "first_s": times[0], "best_s": min(times)}
        res["result_check"] = check_description(name, first_sum, q)
        if name in peaks:
            res["peak_rss_bytes"] = peaks[name]
        return res
    except (EngineError, EngineTimeout, EngineCrashed) as e:
        return _fail(_status_of(e), e)
    finally:
        eng.teardown()


def run_reproduce(
    suite: str,
    sf: float,
    dataset: str | None = None,
    engines: list[str] | None = None,
    methods: tuple[str, ...] = ("ours", "theirs"),
    reps: int = 3,
    only: list[str] | None = None,
    threads: int = 12,
    memory_gb: int = 32,
    timeout_s: float = 600,
    runs: int | None = None,
    their_timeout_s: float | None = None,
    profile: bool = False,
    record_extra: Mapping[str, Any] | None = None,
    sequential: bool = False,
    cap_gb: int | None = None,
    reference_cache: bool = True,
    refresh_reference: bool = False,
    cold: bool = False,
) -> Path:
    names = engines or DEFAULT_ENGINES
    if REFERENCE not in names:
        raise SystemExit(f"--engines must include {REFERENCE} (the correctness reference)")
    if reps < 1 or (runs is not None and runs < 1):
        raise SystemExit("--reps and --runs must be at least 1")
    ds = resolve_dataset(suite, dataset, sf)
    pub = PUBLISHED[suite]
    runs = runs or pub["runs"]
    their_timeout_s = their_timeout_s or pub["timeout_s"]
    cold_root = ds.root if cold else None
    cache = (
        RefCache(refresh=refresh_reference, engine=REFERENCE, version=dist_version(REFERENCE))
        if reference_cache
        else None
    )
    per_engine = {n: {q.id: q for q in suite_queries(suite, ds, n, only)} for n in names}
    queries = [{n: per_engine[n][qid] for n in names} for qid in per_engine[REFERENCE]]
    lock_path = data_root() / ".bench.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        t_wait = time.perf_counter()
        fcntl.flock(lock, fcntl.LOCK_EX)
        waited = time.perf_counter() - t_wait
        print(f"bench lock acquired (waited {waited:.0f}s)")
        wait_idle()
        started = dt.datetime.now(dt.UTC)
        record: dict[str, Any] = {
            "schema": 1,
            "kind": "reproduce",
            "suite": suite,
            "started_utc": started.isoformat(),
            "machine": machine_info(),
            "git": git_info(),
            "libs": {lib: dist_version(lib) for lib in LIBS},
            "dataset": {"name": ds.name, "scale_factor": sf, "path": str(ds.root), "tables": ds.tables},
            "pinned": {"threads": threads, "memory_gb": memory_gb},
            "methods": list(methods),
            "ours": {
                "reps": reps,
                "samples_per_engine": 2 * reps,
                "timeout_s": timeout_s,
                "mode": "sequential: one engine alive at a time" if sequential else "interleaved",
                "worker_cgroup_gb": cap_gb,
            },
            "theirs": {"runs": runs, "timeout_s": their_timeout_s, "source": pub["source"]},
            "cache_policy": (
                "cold: the dataset is dropped from the OS page cache before every timed run of the ours method"
                if cold
                else "OS page cache shared and never dropped; correctness run precedes timing"
            ),
            "cold": cold_record(cold),
            "bench_lock_wait_s": waited,
            "load_at_start": loadavg(),
            "engines": {},
            "queries": {},
            **(record_extra or {}),
        }
        if sequential and "ours" in methods:
            record["setup_failures"] = {}
            for qm in queries:
                qid = qm[REFERENCE].id
                ent, infos = sequential_query(
                    qm, names, ds, threads, memory_gb, cap_gb, 2 * reps, timeout_s, profile, cache, suite, cold_root
                )
                for n, info in infos.items():
                    record["engines"].setdefault(n, info)
                record["queries"][qid] = ent
                print(_line(qid, ent), flush=True)
            names = list(record["engines"])
        else:
            engs: dict[str, Engine] = {n: ProcessEngine(n, threads, memory_gb, timeout_s) for n in names}
            try:
                record["setup_failures"] = setup_engines(engs, ds)
                for n, e in engs.items():
                    record["engines"][n] = {"version": e.version(), "config": e.config(), "build": e.build_info()}
                for qm in queries:
                    qid = qm[REFERENCE].id
                    ent = (
                        ours_query(qm, engs, reps, timeout_s, profile, cache, suite, ds, cold_root)
                        if "ours" in methods
                        else {}
                    )
                    record["queries"][qid] = ent
                    print(_line(qid, ent), flush=True)
            finally:
                for e in engs.values():
                    e.teardown()
        if "theirs" in methods:
            names = [n for n in names if n not in record["setup_failures"]]
            _run_theirs(record, queries, ds, names, threads, memory_gb, runs, their_timeout_s, "ours" in methods)
    record["finished_utc"] = dt.datetime.now(dt.UTC).isoformat()
    record["reference_cache"] = cache.summary() if cache is not None else {"enabled": False}
    record["summary"] = summarize(record)
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{started.strftime('%Y%m%dT%H%M%SZ')}_reproduce_{suite}_sf{fmt_sf(sf)}.json"
    f.write_text(json.dumps(record, indent=2))
    print(markdown(record))
    print(f"result: {f}")
    return f


def _run_theirs(record, queries, ds, names, threads, memory_gb, runs, timeout_s, have_ours) -> None:
    """Time only (engine, query) pairs that passed correctness; without ``ours`` run the check first, untimed."""
    for qm in queries:
        qid = qm[REFERENCE].id
        ent = record["queries"].setdefault(qid, {})
        if not have_ours:
            probes = {n: ProcessEngine(n, threads, memory_gb, 600) for n in names}
            try:
                verdict = setup_engines(probes, ds)
                checked, nrows = correctness(qm, probes)
                verdict.update(checked)
            finally:
                for p in probes.values():
                    p.teardown()
            ent.update(rows=nrows, engines={n: dict(v) for n, v in verdict.items()})
        t = {}
        for n in names:
            if ent.get("engines", {}).get(n, {}).get("status") != "ok":
                continue
            t[n] = theirs_query(n, qm[n], ds, threads, memory_gb, runs, timeout_s, ent["rows"])
            ent["engines"][n]["theirs"] = t[n]
        ent["theirs_load"] = loadavg()
        print("  theirs " + qid + ": " + " ".join(f"{n}={_best(r)}" for n, r in t.items()), flush=True)


def _best(r: dict) -> str:
    return f"{r['best_s'] * 1000:.0f}ms" if r["status"] == "ok" else r["status"]


def _line(qid: str, ent: dict) -> str:
    parts = []
    for n, e in ent.get("engines", {}).items():
        parts.append(f"{n}={e['median_s'] * 1000:.0f}ms" if "median_s" in e else f"{n}={e['status']}")
    return (
        f"  {qid}: "
        + " ".join(parts)
        + f"  load {ent.get('load_before', math.nan):.1f}->{ent.get('load_after', math.nan):.1f}"
    )


def _pairs(record: dict, engine: str, base: str, key: str) -> dict[str, tuple[Any, Any]]:
    """Per-query (engine, base) timings for queries where both are ok: ours -> sample lists, theirs -> best times."""
    out = {}
    for qid, ent in record["queries"].items():
        engs = ent.get("engines", {})
        a, b = engs.get(engine, {}), engs.get(base, {})
        if key == "ours":
            if "samples_s" in a and "samples_s" in b:
                out[qid] = (a["samples_s"], b["samples_s"])
        else:
            ta, tb = a.get("theirs", {}), b.get("theirs", {})
            if ta.get("status") == "ok" and tb.get("status") == "ok":
                out[qid] = (ta["best_s"], tb["best_s"])
    return out


def summarize(record: dict) -> dict:
    names = list(record["engines"])
    loads = [record["load_at_start"]] + [
        x for e in record["queries"].values() for x in (e.get("load_before"), e.get("load_after"), e.get("theirs_load"))
    ]
    max_load = max((x for x in loads if x is not None and not math.isnan(x)), default=math.nan)
    # The 1-minute load average includes the benchmark's own 12-thread work, so only the idle load at the start
    # (taken under the bench lock) decides "indicative"; the maximum is reported for information.
    start = record["load_at_start"]
    s: dict[str, Any] = {"max_load": max_load, "indicative": start > LOAD_INDICATIVE or math.isnan(start)}
    for method in record["methods"]:
        per: dict[str, Any] = {}
        for n in names:
            statuses: dict[str, int] = {}
            for e in record["queries"].values():
                st = e.get("engines", {}).get(n, {}).get("status", "missing")
                statuses[st] = statuses.get(st, 0) + 1
            per[n] = {"statuses": statuses}
            peaks = [p for e in record["queries"].values() if (p := peak_rss(e, n, method)) is not None]
            if peaks:
                per[n]["max_peak_rss_bytes"] = max(peaks)
            if n == REFERENCE:
                continue
            pairs = _pairs(record, n, REFERENCE, method)
            if pairs:
                if method == "ours":
                    ratios = [statistics.median(a) / statistics.median(b) for a, b in pairs.values()]
                    lo, hi = geomean_ci(list(pairs.values()))
                    per[n].update(geomean=geomean(ratios), geomean_ci95=[lo, hi], n=len(ratios))
                else:
                    ratios = [a / b for a, b in pairs.values()]
                    # the Polars post reports sums of query times as well as geomeans
                    sum_ratio = sum(a for a, _ in pairs.values()) / sum(b for _, b in pairs.values())
                    per[n].update(geomean=geomean(ratios), n=len(ratios), sum_ratio=sum_ratio)
                per[n]["worst_query"] = max(zip(pairs, ratios, strict=True), key=lambda t: t[1])[0]
        if "krishiv" in names:
            for n in names:
                if n == "krishiv":
                    continue
                kp = _pairs(record, "krishiv", n, method)
                if kp:
                    rs = [
                        (statistics.median(a) / statistics.median(b)) if method == "ours" else a / b
                        for a, b in kp.values()
                    ]
                    per[n]["krishiv_over_engine_geomean"] = geomean(rs)
                    per[n]["krishiv_over_engine_n"] = len(rs)
        s[method] = per
    return s


def markdown(record: dict) -> str:
    names = list(record["engines"])
    out = [f"### {record['suite']} sf{record['dataset']['scale_factor']}  (threads {record['pinned']['threads']})", ""]
    for n, f in record.get("setup_failures", {}).items():
        out += [f"Setup {f['status']} for {n}, not run: {f['reason']}", ""]
    for method in record["methods"]:
        note = f"; ratio vs {REFERENCE} in parentheses" if method == "ours" else "; best of N hot runs"
        out += [f"**{method}** (ms{note})", ""]
        out += ["| query | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
        for qid, ent in record["queries"].items():
            cells = []
            for n in names:
                e = ent.get("engines", {}).get(n, {})
                if method == "ours" and "median_s" in e:
                    r = f" ({e['ratio']:.2f})" if "ratio" in e else ""
                    cells.append(f"{e['median_s'] * 1000:.0f}{r}")
                elif method == "theirs" and e.get("theirs", {}).get("status") == "ok":
                    cells.append(f"{e['theirs']['best_s'] * 1000:.0f}")
                else:
                    cells.append(
                        e.get("theirs", {}).get("status")
                        if method == "theirs" and "theirs" in e
                        else e.get("status", "-")
                    )
            out.append(f"| {qid} | " + " | ".join(str(c) for c in cells) + " |")
        out.append("")
        out += _peak_table(record, method, names)
        for n, p in record["summary"][method].items():
            if "geomean" in p:
                ci = p.get("geomean_ci95")
                ci_s = f" (95% CI {ci[0]:.2f}-{ci[1]:.2f})" if ci else ""
                if "sum_ratio" in p:
                    ci_s += f"; sum of times {p['sum_ratio']:.2f}x"
                out.append(
                    f"- {n}: geomean {p['geomean']:.2f}x of {REFERENCE} over {p['n']} queries{ci_s}; {p['statuses']}"
                )
            elif n != REFERENCE:
                out.append(f"- {n}: no comparable queries; {p['statuses']}")
        out.append("")
    s = record["summary"]
    tag = "INDICATIVE ONLY" if s["indicative"] else "within bound"
    out.append(
        f"Load at start {record['load_at_start']:.1f} (bound {LOAD_INDICATIVE:g}): {tag}; "
        f"max 1-min load during the run {s['max_load']:.1f} (includes the benchmark itself)."
    )
    return "\n".join(out)


def peak_rss(ent: dict, engine: str, method: str) -> int | None:
    """Peak RSS in bytes of ``engine`` for one query entry and method (``None`` when not recorded)."""
    e = ent.get("engines", {}).get(engine, {})
    return e.get("theirs", {}).get("peak_rss_bytes") if method == "theirs" else e.get("peak_rss_bytes")


def _peak_table(record: dict, method: str, names: list[str]) -> list[str]:
    rows = []
    for qid, ent in record["queries"].items():
        peaks = [peak_rss(ent, n, method) for n in names]
        if any(p is not None for p in peaks):
            cells = ["-" if p is None else f"{p / 2**20:.0f}" for p in peaks]
            rows.append(f"| {qid} | " + " | ".join(cells) + " |")
    if not rows:
        return []
    head = [f"**{method} peak memory** (MiB, RSS of each engine's whole process tree)", ""]
    head += ["| query | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    return [*head, *rows, ""]


def report_from_file(path: Path) -> str:
    return markdown(json.loads(path.read_text()))
