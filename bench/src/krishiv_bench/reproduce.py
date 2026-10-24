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

from . import compare
from .abba import LOAD_INDICATIVE, bootstrap_ci, discover_tables, geomean, geomean_ci, loadavg, wait_idle
from .adapters import Dataset, Engine, Query
from .adapters.base import check_query, result_rows
from .adapters.process import EngineCrashed, EngineError, EngineTimeout, ProcessEngine
from .dialect import dialect_reason
from .errors import describe_error, error_fields
from .h2o import is_h2o
from .paths import data_root, dataset_dir, fmt_sf, results_dir
from .runner import git_info, load_queries, machine_info

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
}
LIBS = ["duckdb", "polars", "datafusion", "pysail", "pyspark", "pyarrow", "chdb"]


def resolve_dataset(suite: str, dataset: str | None, sf: float) -> Dataset:
    name = dataset or SUITE_DATASET[suite]
    # clickbench and the h2o suites name a directory relative to $KRISHIV_DATA (h2o: h2o/<version>/<label>)
    root = data_root() / name if suite == "clickbench" or is_h2o(suite) else dataset_dir(name, sf)
    tables = discover_tables(root)
    if not tables:
        raise SystemExit(f"no dataset at {root}")
    return Dataset(name, sf, root, tables)


def _fail(status: str, e: Exception | str) -> dict:
    if isinstance(e, Exception):
        fields = error_fields(e)
        return {"status": status, "reason": fields["error"], **fields}
    return {"status": status, "reason": e[:1000]}


def _status_of(e: Exception) -> str:
    return "timeout" if isinstance(e, EngineTimeout) else "error"


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


def correctness(qm: Mapping[str, Query], engines: Mapping[str, Engine]) -> tuple[dict[str, dict], int]:
    """One untimed run per engine (``qm``: the query as each engine runs it); verdict per engine against DuckDB
    (and Krishiv). Returns (verdicts, row count)."""
    rows: dict[str, list] = {}
    verdict: dict[str, dict] = {}
    for name, eng in engines.items():
        try:
            rows[name] = eng.run(check_query(qm[name]))
        except (EngineError, EngineTimeout, EngineCrashed) as e:
            verdict[name] = _fail(_status_of(e), e)
    ref = rows.get(REFERENCE)
    if ref is None:
        for name in rows:
            verdict[name] = _fail("error", "no DuckDB reference result")
        return verdict, 0
    sql = qm[REFERENCE].sql
    ordered = compare.has_top_level_order_by(sql)
    for name, r in rows.items():
        if name == REFERENCE:
            verdict[name] = {"status": "ok"}
            continue
        res = compare.compare_rows(r, ref, ordered, sql=sql)
        if res.ok:
            verdict[name] = {"status": "ok"}
            if res.status != "ok":
                verdict[name]["note"] = res.message  # e.g. tie-at-limit
        else:
            reason = dialect_reason(name, qm[name].suite, qm[name].id, r, ref)
            if reason:
                verdict[name] = _fail("mismatch-dialect", f"{reason}: {res.message}")
            else:
                verdict[name] = _fail("mismatch", res.message)
        if res.ok and "krishiv" in rows and name != "krishiv":
            kok, kmsg = compare.rows_match(r, rows["krishiv"], ordered, sql=sql)
            if not kok:  # equal to DuckDB but not to Krishiv: Krishiv itself differs (also flagged on its own verdict)
                verdict[name]["vs_krishiv"] = kmsg[:200]
    return verdict, result_rows(qm[REFERENCE], ref)


def _note_peak(peaks: dict[str, int], name: str, eng: Engine) -> None:
    """Keeps the highest per-run peak RSS of ``eng`` (its worker process tree) seen for the query."""
    peak = eng.last_peak_rss_bytes
    if peak is not None:
        peaks[name] = max(peaks.get(name, 0), peak)


def ours_query(
    qm: Mapping[str, Query], engines: Mapping[str, Engine], reps: int, timeout_s: float, profile: bool = False
) -> dict:
    load_before = loadavg()
    verdict, nrows = correctness(qm, engines)
    entry: dict[str, Any] = {"rows": nrows, "engines": {n: dict(v) for n, v in verdict.items()}}
    live = [n for n in engines if verdict.get(n, {}).get("status") == "ok"]
    if REFERENCE not in live:
        return entry
    samples: dict[str, list[float]] = {n: [] for n in live}
    peaks: dict[str, int] = {}
    cpus: dict[str, list[float]] = {n: [] for n in live}
    for _ in range(reps):
        for name in [*live, *reversed(live)]:
            if name not in samples:
                continue
            try:
                n, secs = engines[name].timed_run(qm[name])
                if n != nrows:
                    raise EngineError(f"timed run returned {n} rows, expected {nrows}")
                samples[name].append(secs)
                _note_peak(peaks, name, engines[name])
                if (cpu := engines[name].last_cpu_seconds) is not None:
                    cpus[name].append(cpu)
            except (EngineError, EngineTimeout, EngineCrashed) as e:
                entry["engines"][name] = _fail(_status_of(e), f"timed run: {e}")
                del samples[name]
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
    entry["load_before"], entry["load_after"] = load_before, loadavg()
    return entry


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
        for _ in range(runs):
            n, secs = eng.timed_run(q)
            _note_peak(peaks, name, eng)
            if n != nrows:
                raise EngineError(f"timed run returned {n} rows, expected {nrows}")
            times.append(secs)
        res: dict[str, Any] = {"status": "ok", "runs_s": times, "first_s": times[0], "best_s": min(times)}
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
    per_engine = {n: {q.id: q for q in load_queries(suite, n, only)} for n in names}
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
            "ours": {"reps": reps, "samples_per_engine": 2 * reps, "timeout_s": timeout_s},
            "theirs": {"runs": runs, "timeout_s": their_timeout_s, "source": pub["source"]},
            "cache_policy": "OS page cache shared and never dropped; correctness run precedes timing",
            "bench_lock_wait_s": waited,
            "load_at_start": loadavg(),
            "engines": {},
            "queries": {},
        }
        engs: dict[str, Engine] = {n: ProcessEngine(n, threads, memory_gb, timeout_s) for n in names}
        try:
            record["setup_failures"] = setup_engines(engs, ds)
            for n, e in engs.items():
                record["engines"][n] = {"version": e.version(), "config": e.config(), "build": e.build_info()}
            for qm in queries:
                qid = qm[REFERENCE].id
                ent = ours_query(qm, engs, reps, timeout_s, profile) if "ours" in methods else {}
                record["queries"][qid] = ent
                print(_line(qid, ent), flush=True)
        finally:
            for e in engs.values():
                e.teardown()
        if "theirs" in methods:
            names = [n for n in names if n not in record["setup_failures"]]
            _run_theirs(record, queries, ds, names, threads, memory_gb, runs, their_timeout_s, "ours" in methods)
    record["finished_utc"] = dt.datetime.now(dt.UTC).isoformat()
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
