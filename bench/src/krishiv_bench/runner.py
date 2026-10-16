"""Suite runner: executes queries against an Engine, records a JSON result."""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import psutil

from .adapters import Dataset, Engine, Query, make_engine
from . import compare
from .paths import bench_root, data_root, dataset_dir, fmt_sf, queries_dir, results_dir

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
    """Harness == engine repo here. `dirty` covers the whole repo (tracked changes and untracked files)."""
    def g(*a):
        return subprocess.run(["git", "-C", str(bench_root()), *a], capture_output=True, text=True).stdout.strip()
    porcelain = g("status", "--porcelain")
    return {"commit": g("rev-parse", "HEAD") or None, "branch": g("rev-parse", "--abbrev-ref", "HEAD") or None,
            "dirty": bool(porcelain), "dirty_files": len(porcelain.splitlines())}


def dataset_provenance(ds_name: str, sf: float) -> dict:
    man = data_root() / "manifests" / f"{ds_name}-sf{fmt_sf(sf)}.sha256"
    prov = {"manifest_path": str(man), "manifest_sha256": hashlib.sha256(man.read_bytes()).hexdigest() if man.exists() else None}
    if ds_name == "tpch":
        r = subprocess.run(["tpchgen-cli", "--version"], capture_output=True, text=True)
        prov["generator"] = r.stdout.strip() or "tpchgen-cli (unknown version)"
    else:
        import duckdb
        prov["generator"] = f"duckdb {duckdb.__version__} tpcds extension (dsdgen)"
    return prov


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
    f = queries_dir(suite) / f"expected_rows_sf{fmt_sf(sf)}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def run_suite(engine_name: str, suite: str, sf: float, iterations: int = 5, warmup: int = 2,
              only: list[str] | None = None, engine_kwargs: dict | None = None,
              timeout_s: float | None = 600) -> Path:
    ds_name = SUITE_DATASET[suite]
    root = dataset_dir(ds_name, sf)
    tables = sorted(p.stem for p in root.glob("*.parquet"))
    if not tables:
        raise SystemExit(f"no dataset at {root}; run `uv run krishiv-datasets {ds_name} --sf {sf}`")
    eng: Engine = make_engine(engine_name, **(engine_kwargs or {}))
    expected = load_expected_rows(suite, sf)
    answers = {}
    if suite == "tpch":
        try:
            answers = compare.load_tpch_answers(sf)
        except Exception as e:  # e.g. extension not installable offline
            print(f"WARNING: official answers unavailable: {e}")
    ds = Dataset(ds_name, sf, root, tables)
    queries = load_queries(suite, engine_name, only)

    started = dt.datetime.now(dt.timezone.utc)
    sf_s = fmt_sf(sf)
    stem = f"{started.strftime('%Y%m%dT%H%M%SZ')}_{engine_name}_{suite}_sf{sf_s}"
    record = {
        "schema": 2, "started_utc": started.isoformat(), "machine": machine_info(),
        "git": {"repo": "krishiv-engine (harness lives in bench/)", **git_info()},
        "engine": {"name": engine_name, "version": eng.version(), "config": eng.config()},
        "dataset": {"name": ds_name, "scale_factor": sf, "path": str(root), "tables": tables,
                    **dataset_provenance(ds_name, sf)},
        "suite": suite, "iterations": iterations, "warmup": warmup, "timeout_s": timeout_s,
        "answers_available": bool(answers), "queries": {},
    }
    rows_store: dict[str, list] = {}
    MAX_STORED_ROWS = 1_000_000
    eng.setup(ds)
    pool = cf.ThreadPoolExecutor(max_workers=1)
    try:
        for q in queries:
            entry: dict = {"iterations": iterations}
            times, rows, meas = [], None, []
            try:
                for i in range(warmup + iterations):
                    eng.begin_measure()
                    m = None
                    try:
                        t0 = time.perf_counter()
                        fut = pool.submit(eng.run, q)
                        try:
                            rows = fut.result(timeout=timeout_s)
                        except cf.TimeoutError:
                            eng.cancel()
                            try:
                                fut.result(timeout=30)
                            except Exception:
                                pass
                            pool.shutdown(wait=False, cancel_futures=True)
                            pool = cf.ThreadPoolExecutor(max_workers=1)
                            raise TimeoutError(f"query exceeded {timeout_s}s")
                        dtm = time.perf_counter() - t0
                    finally:
                        m = eng.end_measure()
                    if i >= warmup:
                        times.append(dtm)
                        meas.append(m)
            except Exception as e:  # record, keep going
                entry["error"] = f"{type(e).__name__}: {str(e)[:500]}"
                record["queries"][q.id] = entry
                print(f"  {q.id}: ERROR {entry['error'][:120]}")
                continue
            ordered = compare.has_top_level_order_by(q.sql)
            entry.update(
                times_s=times, p50_s=statistics.median(times), p99_s=percentile(times, 99),
                min_s=min(times), max_s=max(times), mean_s=statistics.fmean(times),
                throughput_qps=len(times) / sum(times),
                rows_returned=len(rows), has_order_by=ordered,
                expected_rows=expected.get(q.id),
                rows_match_expected=(len(rows) == expected[q.id]) if q.id in expected else None,
                peak_rss_bytes=_max([m.peak_rss_bytes for m in meas]),
                baseline_rss_bytes=_max([m.baseline_rss_bytes for m in meas]),
                cpu_seconds_median=_med([m.cpu_seconds for m in meas]),
                spill_bytes=_max([m.spill_bytes for m in meas]),
            )
            if q.id in answers:
                ok, msg = compare.answer_check(rows, answers[q.id], ordered)
                entry["answer_check"] = {"status": "pass" if ok else "fail", "detail": msg}
            else:
                entry["answer_check"] = {"status": "not_checked", "detail": "no official answer available"}
            if len(rows) <= MAX_STORED_ROWS:
                rows_store[q.id] = rows
                entry["rows_stored"] = True
            else:
                entry["rows_stored"] = False
            ac = entry["answer_check"]
            flag = "" if ac["status"] != "fail" else f"  ANSWER MISMATCH: {ac['detail'][:100]}"
            print(f"  {q.id}: p50={entry['p50_s']*1000:8.1f} ms rows={len(rows)} answer={ac['status']}{flag}")
            record["queries"][q.id] = entry
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        eng.teardown()
    record["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    compare.save_rows(out / f"{stem}.rows.json.gz", rows_store)
    record["rows_file"] = f"{stem}.rows.json.gz"
    f = out / f"{stem}.json"
    f.write_text(json.dumps(record, indent=2))
    return f


def _max(xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None
