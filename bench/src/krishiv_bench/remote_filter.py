"""The ``remote-filter`` suite: one selective scan over a FineWeb-Edu sample, written back as zstd Parquet.

Workload: the FineWeb-Edu ``sample/10BT`` Parquet files (Hugging Face dataset ``HuggingFaceFW/fineweb-edu``),
the rows with ``int_score >= 4 AND token_count >= 4000 AND language_score >= 0.95``, written to zstd Parquet.
Sources are read remotely over HTTPS from the dataset's resolve URLs, or from a local mirror directory when
one is given (``krishiv-datasets remote-filter`` downloads the mirror). Engines: Krishiv (the bench runner),
DuckDB (``read_parquet`` + ``COPY ... (FORMAT parquet, COMPRESSION zstd)``) and Polars
(``scan_parquet(...).filter(...).sink_parquet(..., compression="zstd")``).
The Krishiv arm writes the runner's native output directory with the runner's default Parquet codec because
the runner's DUMP protocol carries no per-write compression option; the codec each engine wrote is recorded,
so the write cost is not like for like across engines.

Correctness is the row count, the exact sum of ``token_count`` and a digest of the sorted ``id`` values of what
each engine wrote. The reference is DuckDB's result of the validation run (or ``expected.json`` when it has
counts, which the operator records with ``write_expected``); every other engine must match it, and a wrong
result is never timed. The local mirror keeps a sha256 manifest of its files (``write_manifest``).

Each engine runs in a fresh process (``remote_filter_child``), optionally inside a systemd scope with
``MemoryMax`` and no swap, the way the ``memcap`` suite gates memory: a run is ``finished``, ``killed`` (out of
memory), ``timeout`` or ``failed``, and a kill is a recorded result, never a crash of the harness. Each scope
(uncapped, then each ``--memory-limit``) runs one validation pass and ``reps`` timed repetitions with the engine
order rotated per repetition (ABBA-style), and the result JSON holds per-engine medians, a bootstrap CI of
each engine's wall time against DuckDB's, peak RSS and pass/fail.

Timing hooks: the workload reports a single timed span (``query_s``). ``phase_buckets`` maps an optional
per-operator profile onto ``scan``/``filter``/``write`` buckets for ledger writers; without one the buckets
stay empty and the total carries the span (this harness records no per-operator split for this workload).
When a memory cap is set, the Krishiv engine runs with 75% of the cap as its own ``memory_limit``.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from .paths import data_root, results_dir
from .sizes import parse_size

SUITE = "remote-filter"
ENGINES = ["krishiv", "duckdb", "polars"]
TABLE = "fw"
FILTER_SQL = "int_score >= 4 AND token_count >= 4000 AND language_score >= 0.95"
HF_DATASET = "HuggingFaceFW/fineweb-edu"
HF_REVISION = "main"  # resolved to a commit sha at listing time; the sha is what a mirror records
HF_SUBDIR = "sample/10BT"
APPROX_SIZE = "~28 GB"
EXPECTED_FILE = "expected.json"
DEFAULT_REPS = 5
THREADS = "8"
CHILD = "krishiv_bench.remote_filter_child"


def describe() -> str:
    """One paragraph about the workload (used by the CLI help and the README example)."""
    return (
        f"selective scan of the FineWeb-Edu {HF_SUBDIR} Parquet ({HF_DATASET}): "
        f"rows with {FILTER_SQL}, written to zstd Parquet"
    )


# ---------------------------------------------------------------- dataset layout
def default_mirror() -> Path:
    return data_root() / "remote-filter" / "sample-10BT"


def mirror_table_files(mirror: Path) -> list[Path]:
    """The Parquet of the ``fw`` table: ``fw.parquet``, or the part files of ``fw/``, else any Parquet below."""
    single = mirror / f"{TABLE}.parquet"
    if single.is_file():
        return [single]
    table_dir = mirror / TABLE
    if table_dir.is_dir():
        parts = sorted(table_dir.rglob("*.parquet"))
        if parts:
            return parts
    fallback = sorted(mirror.rglob("*.parquet"))
    if fallback:
        return fallback
    raise FileNotFoundError(
        f"{mirror} holds no Parquet for the {TABLE!r} table: uv run krishiv-datasets remote-filter --yes"
    )


def write_expected(directory: Path, rows: int | None, sum_tokens: int | None, files: list[str]) -> Path:
    """``expected.json``: the row and token counts once a run has recorded them (None while unknown)."""
    f = directory / EXPECTED_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"rows": rows, "sum_tokens": sum_tokens, "files": files}, indent=2) + "\n")
    return f


def load_expected(directory: Path) -> dict | None:
    """The expected counts, or None when the mirror has no recorded counts yet."""
    f = Path(directory) / EXPECTED_FILE
    if not f.is_file():
        return None
    with contextlib.suppress(ValueError):
        rec = json.loads(f.read_text())
        if isinstance(rec, dict):
            return rec
    return None


def compare_results(ref: dict, got: dict) -> str | None:
    """None when the (row count, token sum, id digest) triples are identical; else the first differing field.

    A key the reference does not know (None) is not compared, so an ``expected.json`` with counts only still works."""
    for key in ("rows", "sum_tokens", "ids_digest"):
        if ref.get(key) is None:
            continue
        if ref.get(key) != got.get(key):
            return f"{key}: reference {ref.get(key)!r}, got {got.get(key)!r}"
    return None


# ---------------------------------------------------------------- download (opt-in)
def list_remote_files(timeout_s: float = 60) -> list[tuple[str, str]]:
    """``(resolve URL, relative path)`` of the Parquet files under ``HF_SUBDIR``, via the Hub API (no new deps)."""
    url = f"https://huggingface.co/api/datasets/{HF_DATASET}"
    req = urllib.request.Request(url, headers={"User-Agent": "krishiv-bench"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            meta = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"cannot list {HF_DATASET} ({e}); pass explicit files to fetch_remote_filter") from e
    out = []
    try:
        revision = meta.get("sha") or HF_REVISION  # the commit the listing describes, so files and counts agree
        siblings = meta.get("siblings", [])
        for sib in siblings:
            name = sib.get("rfilename", "")
            if name.startswith(HF_SUBDIR) and name.endswith(".parquet"):
                out.append((f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{revision}/{name}", name))
    except (AttributeError, TypeError) as e:
        raise RuntimeError(f"unexpected Hub listing for {HF_DATASET} ({e})") from e
    if not out:
        raise RuntimeError(f"no Parquet files under {HF_SUBDIR} in {HF_DATASET}")
    return sorted(out)


def fetch_remote_filter(
    out: Path | None = None,
    yes: bool = False,
    files: list[tuple[str, str]] | None = None,
) -> Path:
    """Downloads the mirror to ``out`` (default under ``$KRISHIV_DATA``) as ``fw/`` part files + manifest.

    Opt-in: without ``yes`` it only warns about the download size (``APPROX_SIZE``) and stops. Finished files
    are kept; an interrupted download resumes from its ``.part`` file.
    """
    if not yes:
        raise SystemExit(f"remote-filter download is {APPROX_SIZE}; re-run with --yes to download.")
    dest = out or default_mirror()
    table_dir = dest / TABLE
    table_dir.mkdir(parents=True, exist_ok=True)
    pairs = files if files is not None else list_remote_files()
    if not pairs:
        raise ValueError("no files to download")
    for url, rel in pairs:
        target = table_dir / Path(rel).name
        if target.exists():
            print(f"[remote-filter] {target} exists, skipping")
            continue
        # Into a .part file that curl resumes; renamed only when complete, so an interrupted
        # download never leaves a truncated file that looks finished.
        part = target.with_name(target.name + ".part")
        print(f"[remote-filter] {url} -> {target}")
        subprocess.run(["curl", "-fL", "-C", "-", "-o", str(part), url], check=True)
        part.rename(target)
    from .datasets import write_manifest

    print("manifest:", write_manifest(dest, "remote-filter-sample-10BT"))
    if not (dest / EXPECTED_FILE).is_file():
        write_expected(dest, None, None, sorted(f.name for f in table_dir.glob("*.parquet")))
    return dest


# ---------------------------------------------------------------- timing buckets for ledger writers
def phase_buckets(child: dict, profile: dict | None = None) -> dict:
    """``scan``/``filter``/``write`` buckets plus the ``total`` span.

    The workload reports one timed span (``query_s``); only an explicit per-operator ``profile`` (a mapping
    with an ``operators`` list of ``{"name", "seconds"}``) fills the buckets, by case-insensitive name match.
    Otherwise the buckets are None and ``total_s`` carries the span.
    """
    buckets: dict[str, float | None] = {
        "scan_s": None,
        "filter_s": None,
        "write_s": None,
        "total_s": child.get("query_s"),
    }
    if not isinstance(profile, dict):
        return buckets
    ops = profile.get("operators")
    if not isinstance(ops, list):
        return buckets
    for op in ops:
        if not isinstance(op, dict):
            continue
        name = str(op.get("name", "")).lower()
        secs = op.get("seconds")
        if not isinstance(secs, (int, float)) or isinstance(secs, bool):
            continue
        for key, needle in (("scan_s", "scan"), ("filter_s", "filter"), ("write_s", "write")):
            if needle in name:
                buckets[key] = (buckets[key] or 0.0) + float(secs)
    return buckets


def ledger_entry(rec: dict) -> dict:
    """The per-engine row a gap ledger would store: status, pass/fail, buckets, peak RSS."""
    phases = rec.get("phases") or {}
    return {
        "engine": rec.get("engine"),
        "status": rec.get("status"),
        "pass": rec.get("pass"),
        "scan_s": phases.get("scan_s"),
        "filter_s": phases.get("filter_s"),
        "write_s": phases.get("write_s"),
        "total_s": phases.get("total_s"),
        "peak_rss_bytes": rec.get("peak_rss_bytes"),
    }


def krishiv_memory_limit(cap: str | None) -> str:
    """The Krishiv engine ``memory_limit`` for a run: 75% of the cgroup cap, else the adapter default."""
    if cap is None:
        from .adapters.krishiv import KrishivEngine

        default = inspect.signature(KrishivEngine.__init__).parameters["memory_limit"].default
        if not isinstance(default, str):
            raise TypeError(f"the Krishiv adapter's memory_limit default is not a size string: {default!r}")
        return default
    cap_bytes = parse_size(cap)
    return f"{(cap_bytes * 3 // 4) // (1 << 20)}m"


# ---------------------------------------------------------------- one run
def read_tail(path: Path, lines: int = 5) -> str:
    return " | ".join(path.read_text(errors="replace").strip().splitlines()[-lines:])[:500]


def run_one(
    engine: str,
    data_arg: str,
    cap: str | None,
    timeout_s: float | None,
    work: Path,
    _popen=subprocess.Popen,
    _find_cgroup=None,
    _monitor=None,
) -> dict:
    """One fresh-process run, optionally under a cgroup memory cap. Never raises for engine failures."""
    from .memcap import classify, find_cgroup, monitor, read_result, scope_command

    _find_cgroup = _find_cgroup or find_cgroup
    _monitor = _monitor or monitor
    if engine not in ENGINES:
        raise ValueError(f"engines must be a non-empty subset of {', '.join(ENGINES)} (got {engine!r})")
    cap_bytes = parse_size(cap) if cap else None
    out, scratch = work / "out", work / "scratch"
    work.mkdir(parents=True, exist_ok=True)
    stdout_f, stderr_f = work / "stdout", work / "stderr"
    argv = [sys.executable, "-m", CHILD, engine, data_arg, str(out), str(scratch)]
    if engine == "krishiv" and cap is not None:
        argv.append(krishiv_memory_limit(cap))
    env = dict(os.environ, POLARS_MAX_THREADS=THREADS)
    cmd = argv if cap_bytes is None else scope_command(cap_bytes, argv)
    real = _popen is subprocess.Popen
    with stdout_f.open("wb") as so, stderr_f.open("wb") as se:
        # Its own session: an uncapped timeout kills the whole process group (the Krishiv runner is a
        # grandchild), so nothing keeps loading the machine while the next engine is timed.
        proc = _popen(cmd, stdout=so, stderr=se, env=env, stdin=subprocess.DEVNULL, start_new_session=real)
        cgroup = _find_cgroup(proc.pid) if cap_bytes is not None else None
        obs = _monitor(proc, cgroup, None, timeout_s)
        if real and obs.timed_out:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
    result = read_result(stdout_f.read_text(errors="replace"))
    if obs.timed_out:
        status = "timeout"
    elif cap_bytes is not None:
        status = classify(obs, cap_bytes, result is not None)
    elif result is not None and obs.returncode == 0:
        status = "finished"
    else:
        status = "failed"
    rec = {
        "engine": engine,
        "status": status,
        "wall_s": round(obs.wall_s, 3),
        "peak_rss_bytes": obs.rss_peak_bytes,  # replaced by the child's own figure when the monitor has none
        "peak_cgroup_bytes": obs.peak_bytes,
        "returncode": obs.returncode,
        "oom_kills": obs.oom_kills,
    }
    if status == "finished" and result is not None:
        rec |= {
            k: result[k]
            for k in (
                "version",
                "setup_s",
                "query_s",
                "rows_out",
                "sum_tokens",
                "ids_digest",
                "compression",
                "engine_memory_limit",
            )
            if k in result
        }
        if not rec["peak_rss_bytes"] and result.get("peak_rss_bytes"):
            rec["peak_rss_bytes"] = result["peak_rss_bytes"]  # uncapped runs have no cgroup to read it from
        rec["phases"] = phase_buckets(result)
        if engine == "krishiv" and "engine_memory_limit" not in rec:
            rec["engine_memory_limit"] = krishiv_memory_limit(cap)
    else:
        rec["error"] = read_tail(stderr_f) if stderr_f.exists() else ""
    return rec


# ---------------------------------------------------------------- suite
def triple(rec: dict) -> dict:
    """What a finished run is checked on: row count, token sum, id digest."""
    return {"rows": rec.get("rows_out"), "sum_tokens": rec.get("sum_tokens"), "ids_digest": rec.get("ids_digest")}


def judge(ref: dict | None, rec: dict) -> str:
    """The verdict of one finished run against the reference (``None``: no reference exists)."""
    if ref is None:
        return "unverified: no reference result"
    diff = compare_results(ref, triple(rec))
    return "pass" if diff is None else f"fail: {diff}"


def summarize_runs(runs: list[dict]) -> dict:
    """Counts and medians over the finished timed runs of one (scope, engine)."""
    timed = [r for r in runs if r["phase"] == "timed"]
    done = [r for r in timed if r["status"] == "finished"]
    walls = [r["wall_s"] for r in done]
    rss = [r["peak_rss_bytes"] for r in done if r.get("peak_rss_bytes")]
    return {
        "n": len(timed),
        "finished": len(done),
        "killed": sum(r["status"] == "killed" for r in timed),
        "median_wall_s": statistics.median(walls) if walls else None,
        "median_peak_rss_bytes": statistics.median(rss) if rss else None,
        "max_peak_rss_bytes": max((r.get("peak_rss_bytes") or 0 for r in timed), default=None),
    }


def wall_ratio_ci(runs: dict[str, list[dict]], engine: str, base: str = "duckdb") -> dict | None:
    """Median wall time of ``engine`` over ``base`` with a 95% bootstrap CI; None without two finished runs each."""
    from .abba import bootstrap_ci

    def walls(e: str) -> list[float]:
        return [r["wall_s"] for r in runs.get(e, []) if r["phase"] == "timed" and r["status"] == "finished"]

    a, b = walls(engine), walls(base)
    if engine == base or len(a) < 2 or len(b) < 2:
        return None
    lo, hi = bootstrap_ci(a, b)
    return {"ratio": statistics.median(a) / statistics.median(b), "ci95": [lo, hi], "base": base}


def run_scope(
    names: list[str],
    data_arg: str,
    cap: str | None,
    timeout_s: float | None,
    reps: int,
    work_root: Path,
    expected: dict | None,
    run,
) -> dict:
    """One memory scope: a validation pass, then ``reps`` timed repetitions with the engine order rotated."""
    from .memcap import rotated

    runs: dict[str, list[dict]] = {e: [] for e in names}
    verdict: dict[str, str] = {}
    ref: dict | None = dict(expected) if expected and expected.get("rows") is not None else None
    tag = cap or "uncapped"
    for rep in range(reps + 1):
        validation = rep == 0
        order = rotated(names, rep)
        if validation and "duckdb" in order:
            order = ["duckdb"] + [e for e in order if e != "duckdb"]  # the reference runs first
        for engine in order:
            if verdict.get(engine, "").startswith("fail"):
                continue  # a wrong result is never timed
            rec = run(engine, data_arg, cap, timeout_s, work_root / f"{tag}-{rep}-{engine}")
            rec["phase"] = "validation" if validation else "timed"
            rec["rep"] = rep
            if rec["status"] == "finished":
                if validation and engine == "duckdb" and ref is None:
                    ref = triple(rec)
                    v = "reference"
                else:
                    v = judge(ref, rec)
                if v.startswith("fail") or verdict.get(engine, "unverified").startswith("unverified"):
                    verdict[engine] = v
                rec["verdict"] = v
                rec["pass"] = True if v in ("pass", "reference") else (False if v.startswith("fail") else None)
                if v.startswith("fail"):
                    rec["status"] = "mismatch"
            else:
                rec["pass"], rec["verdict"] = None, rec["status"]
                verdict.setdefault(engine, f"unverified: {rec['status']}")
            runs[engine].append(rec)
    return {
        "cap": cap,
        "reference": ref,
        "engines": {
            e: {
                "verdict": verdict.get(e, "unverified: not run"),
                "runs": runs[e],
                "summary": summarize_runs(runs[e]),
                "wall_vs_duckdb": wall_ratio_ci(runs, e),
            }
            for e in names
        },
    }


def run_remote_filter(
    data_arg: str,
    engines: list[str] | None = None,
    caps: list[str | None] | None = None,
    timeout_s: float | None = 1800,
    run=run_one,
    reps: int = DEFAULT_REPS,
) -> Path:
    """Runs each engine over ``data_arg`` (a mirror dir or a JSON source list) in every memory scope in ``caps``
    (``None`` is uncapped) and writes the result JSON, even when a run raises."""
    from .abba import bench_lock, loadavg
    from .runner import git_info, machine_info

    names = list(engines or ENGINES)
    unknown = [e for e in names if e not in ENGINES]
    if unknown or not names:
        raise ValueError(f"engines must be a non-empty subset of {', '.join(ENGINES)} (got {names})")
    if reps < 1:
        raise ValueError(f"reps must be at least 1 (got {reps})")
    scopes = list(caps) if caps else [None]
    cap_bytes = [parse_size(c) if c else None for c in scopes]
    expected: dict | None = None
    if not data_arg.strip().startswith("["):
        with contextlib.suppress(OSError):
            expected = load_expected(Path(data_arg))
    record: dict[str, Any] = {
        "suite": SUITE,
        "kind": SUITE,
        "ts": time.strftime("%Y%m%dT%H%M%S"),
        "input": data_arg,
        "reps": reps,
        "cap_bytes": cap_bytes,
        "machine": machine_info(),
        "git": git_info(),
        "dataset": {"name": f"{HF_DATASET} {HF_SUBDIR}", "expected": expected},
        "scopes": {},
    }
    work_root = data_root() / "tmp" / f"remote-filter-{os.getpid()}"
    try:
        with bench_lock() as waited:
            record["bench_lock_wait_s"] = waited
            record["loadavg_start"] = loadavg()
            try:
                for cap in scopes:
                    record["scopes"][cap or "uncapped"] = run_scope(
                        names, data_arg, cap, timeout_s, reps, work_root, expected, run
                    )
            except BaseException as e:
                record["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                raise
            finally:
                record["loadavg_end"] = loadavg()
    finally:
        import shutil

        shutil.rmtree(work_root, ignore_errors=True)
        # Written even after a failure, so the scopes that finished are not lost.
        every = [r for sc in record["scopes"].values() for e in sc["engines"].values() for r in e["runs"]]
        record["summary"] = {
            "n": len(names) * len(scopes),
            "finished": sum(r["status"] == "finished" for r in every if r["phase"] == "timed"),
            "killed": sum(r["status"] == "killed" for r in every if r["phase"] == "timed"),
            "passed": sum(r.get("pass") is True for r in every if r["phase"] == "timed"),
        }
        out = results_dir()
        out.mkdir(parents=True, exist_ok=True)
        f = out / f"{record['ts']}_{SUITE}.json"
        f.write_text(json.dumps(record, indent=2))
        print(f"result: {f}")
    return f


__all__ = [
    "APPROX_SIZE",
    "ENGINES",
    "FILTER_SQL",
    "HF_DATASET",
    "HF_SUBDIR",
    "SUITE",
    "TABLE",
    "compare_results",
    "default_mirror",
    "describe",
    "fetch_remote_filter",
    "krishiv_memory_limit",
    "ledger_entry",
    "list_remote_files",
    "load_expected",
    "mirror_table_files",
    "phase_buckets",
    "run_one",
    "run_remote_filter",
    "write_expected",
]
