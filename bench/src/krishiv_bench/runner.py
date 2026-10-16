"""Suite runner: executes queries against an Engine, records a JSON result."""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import psutil

from .adapters import Dataset, Engine, Query, make_engine
from .normalize import checksum
from .paths import bench_root, dataset_dir, queries_dir, results_dir

SUITE_DATASET = {"tpch": "tpch", "tpcds": "tpcds"}


def machine_info() -> dict:
    cpu = "unknown"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor()
    gov = None
    try:
        gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        pass
    return {
        "cpu_model": cpu,
        "cores_logical": os.cpu_count(),
        "cores_physical": psutil.cpu_count(logical=False),
        "ram_bytes": psutil.virtual_memory().total,
        "ram_available_bytes_at_start": psutil.virtual_memory().available,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu_governor": gov,
        "hostname": platform.node(),
    }


def git_info() -> dict:
    def g(*a):
        return subprocess.run(["git", "-C", str(bench_root()), *a], capture_output=True, text=True).stdout.strip()
    return {"commit": g("rev-parse", "HEAD") or None, "branch": g("rev-parse", "--abbrev-ref", "HEAD") or None,
            "dirty": bool(g("status", "--porcelain", "--", ".", ":!bench/.venv"))}


def load_queries(suite: str, engine: str, only: list[str] | None = None) -> list[Query]:
    """bench/queries/<suite>/qNN.sql; an engine-specific override lives in bench/queries/<suite>/<engine>/qNN.sql."""
    d = queries_dir(suite)
    out = []
    for f in sorted(d.glob("q*.sql")):
        if only and f.stem not in only:
            continue
        ov = d / engine / f.name
        out.append(Query(f.stem, (ov if ov.exists() else f).read_text(), suite))
    return out


def percentile(xs: list[float], p: float) -> float:
    """Nearest-rank percentile. With few iterations p99 == max (documented in the README)."""
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(-(-p / 100 * len(s) // 1)) - 1))
    return s[k]


def load_expected_rows(suite: str, sf: float) -> dict[str, int]:
    f = queries_dir(suite) / f"expected_rows_sf{int(sf) if float(sf).is_integer() else sf}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def run_suite(engine_name: str, suite: str, sf: float, iterations: int = 5, warmup: int = 1,
              only: list[str] | None = None, engine_kwargs: dict | None = None,
              timeout_s: float | None = None) -> Path:
    ds_name = SUITE_DATASET[suite]
    root = dataset_dir(ds_name, sf)
    tables = sorted(p.stem for p in root.glob("*.parquet"))
    if not tables:
        raise SystemExit(f"no dataset at {root}; run `uv run krishiv-datasets {ds_name} --sf {sf}`")
    eng: Engine = make_engine(engine_name, **(engine_kwargs or {}))
    expected = load_expected_rows(suite, sf)
    ds = Dataset(ds_name, sf, root, tables)
    queries = load_queries(suite, engine_name, only)

    started = dt.datetime.now(dt.timezone.utc)
    record = {
        "schema": 1, "started_utc": started.isoformat(), "machine": machine_info(),
        "git": {"repo": "krishiv-engine", **git_info()},
        "engine": {"name": engine_name, "version": eng.version(), "config": eng.config()},
        "dataset": {"name": ds_name, "scale_factor": sf, "path": str(root), "tables": tables},
        "suite": suite, "iterations": iterations, "warmup": warmup, "queries": {},
    }
    eng.setup(ds)
    try:
        for q in queries:
            entry: dict = {"iterations": iterations}
            times, rows, meas = [], None, []
            try:
                for i in range(warmup + iterations):
                    eng.begin_measure()
                    t0 = time.perf_counter()
                    rows = eng.run(q)
                    dtm = time.perf_counter() - t0
                    m = eng.end_measure()
                    if i >= warmup:
                        times.append(dtm)
                        meas.append(m)
            except Exception as e:  # record, keep going
                entry["error"] = f"{type(e).__name__}: {str(e)[:500]}"
                record["queries"][q.id] = entry
                print(f"  {q.id}: ERROR {entry['error'][:120]}")
                continue
            entry.update(
                times_s=times, p50_s=statistics.median(times), p99_s=percentile(times, 99),
                min_s=min(times), max_s=max(times), mean_s=statistics.fmean(times),
                throughput_qps=len(times) / sum(times),
                rows_returned=len(rows), result_checksum=checksum(rows),
                expected_rows=expected.get(q.id),
                rows_match_expected=(len(rows) == expected[q.id]) if q.id in expected else None,
                peak_rss_bytes=_max([m.peak_rss_bytes for m in meas]),
                cpu_seconds_median=_med([m.cpu_seconds for m in meas]),
                spill_bytes=_max([m.spill_bytes for m in meas]),
            )
            flag = "" if entry["rows_match_expected"] in (True, None) else "  ROWCOUNT MISMATCH"
            print(f"  {q.id}: p50={entry['p50_s']*1000:8.1f} ms rows={len(rows)}{flag}")
            record["queries"][q.id] = entry
    finally:
        eng.teardown()
    record["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{started.strftime('%Y%m%dT%H%M%SZ')}_{engine_name}_{suite}_sf{int(sf) if float(sf).is_integer() else sf}.json"
    f.write_text(json.dumps(record, indent=2))
    return f


def _max(xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None
