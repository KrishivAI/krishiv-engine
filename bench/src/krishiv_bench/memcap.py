"""The ``memcap`` suite: data-engineering queries under a hard memory cap.

Every run is a fresh process (``memcap_child``) started in a systemd scope with ``MemoryMax`` and no swap, so the
kernel kills the whole process tree above the cap. A run is ``finished`` or ``killed`` (out of memory), ``timeout``
or ``failed`` (anything else); only finished, verified runs are timed. Per run the parent records wall time, the
peak resident memory of the tree (sampled from the scope's ``memory.stat``), the scope's ``memory.peak`` (what the cap
is enforced on; it includes page cache the kernel can reclaim) and the peak size of the engine's scratch
directory (spill: DuckDB's temp directory, Krishiv's spill directory, Spark's local dir; Polars has none).

Correctness comes first: for every query each engine runs once untimed (validation) and its result is
checksummed with DuckDB (row count, integer sums exactly, float sums within 1e-9 relative) against the DuckDB
reference run; an engine whose result differs is never timed on that query.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .memcap_data import load_digest, memcap_dir, parse_rows
from .memcap_queries import CHECKS, QUERIES
from .paths import data_root, results_dir
from .sizes import parse_size

ENGINES = ["krishiv", "duckdb", "polars", "spark"]
REFERENCE = "duckdb"
KILL_PEAK_FRACTION = 0.95  # a SIGKILL with the scope this full is the memory controller's, even if the event was missed
POLL_S = 0.01
SPILL_EVERY = 10  # scratch directory scan every Nth poll
CGROUP_ROOT = Path("/sys/fs/cgroup")
GIB = 1 << 30


# ---------------------------------------------------------------- classification
@dataclass
class Observation:
    returncode: int | None
    timed_out: bool
    peak_bytes: int  # the scope's memory.peak: what the cap is enforced on (page cache included)
    oom_kills: int
    spill_peak_bytes: int | None
    wall_s: float
    rss_peak_bytes: int = 0  # highest sampled resident set of the tree (anon + mapped file + shmem; no page cache)


def classify(obs: Observation, cap_bytes: int, got_result: bool) -> str:
    """``finished`` | ``killed`` | ``timeout`` | ``failed``.

    ``killed`` is the memory cap's doing: the scope's ``oom_kill`` counter moved, or the process died of SIGKILL
    (return code -9 or 137) with the scope's peak within 5% of the cap (the cgroup is removed with its last
    process, so the counter can be gone before the last poll). A run that exits 0 without its result line is
    ``failed``, like any other non-zero exit.
    """
    if obs.timed_out:
        return "timeout"
    if obs.oom_kills > 0:
        return "killed"
    if obs.returncode in (-9, 137) and obs.peak_bytes >= KILL_PEAK_FRACTION * cap_bytes:
        return "killed"
    if obs.returncode == 0 and got_result:
        return "finished"
    return "failed"


class Cgroup:
    """The scope's cgroup-v2 files. Reads return ``None`` once the scope is gone."""

    def __init__(self, path: Path):
        self.path = path

    def _int(self, name: str) -> int | None:
        try:
            return int((self.path / name).read_text().strip())
        except (OSError, ValueError):
            return None

    def peak(self) -> int | None:
        values = [v for v in (self._int("memory.peak"), self._int("memory.current")) if v is not None]
        return max(values) if values else None

    def rss(self) -> int | None:
        """Resident memory of the scope's processes: anonymous + mapped file + shared memory pages (no page cache)."""
        try:
            stat = dict(
                (k, int(v))
                for k, _, v in (ln.partition(" ") for ln in (self.path / "memory.stat").read_text().splitlines())
            )
        except (OSError, ValueError):
            return None
        return stat.get("anon", 0) + stat.get("file_mapped", 0) + stat.get("shmem", 0)

    def oom_kills(self) -> int | None:
        try:
            for line in (self.path / "memory.events").read_text().splitlines():
                key, _, value = line.partition(" ")
                if key == "oom_kill":
                    return int(value)
        except (OSError, ValueError):
            return None
        return None

    def kill_all(self) -> None:
        """SIGKILLs every process in the scope (the engine's JVM and helper children included)."""
        with contextlib.suppress(OSError):
            (self.path / "cgroup.kill").write_text("1")


def dir_bytes(path: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(dirpath, f)).st_size
    return total


def monitor(
    proc,
    cgroup,
    scratch: Path | None,
    timeout_s: float | None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    dir_size: Callable[[Path], int] = dir_bytes,
) -> Observation:
    """Polls ``proc`` (a ``Popen``-like: ``poll()``, ``returncode``, ``kill()``, ``wait()``) until it exits.

    Keeps the highest scope peak and ``oom_kill`` count seen and the scratch directory's peak size. On timeout the
    whole scope is killed. ``cgroup`` is ``None`` when the scope could not be located (peak is then 0).
    """
    t0 = clock()
    peak = oom = rss_peak = tick = 0
    spill: int | None = None
    timed_out = False
    while True:
        rc = proc.poll()
        if cgroup is not None:
            p, o = cgroup.peak(), cgroup.oom_kills()
            peak = max(peak, p or 0)
            oom = max(oom, o or 0)
            rss_peak = max(rss_peak, cgroup.rss() or 0)
        if scratch is not None and tick % SPILL_EVERY == 0:
            spill = max(spill or 0, dir_size(scratch))
        tick += 1
        if rc is not None:
            break
        if timeout_s is not None and clock() - t0 > timeout_s:
            timed_out = True
            if cgroup is not None:
                cgroup.kill_all()
            proc.kill()
            proc.wait()
            rc = proc.returncode
            break
        sleep(POLL_S)
    wall = clock() - t0
    if cgroup is not None:
        cgroup.kill_all()  # stragglers (a JVM that outlived its parent) must not leak into the next run
    return Observation(rc, timed_out, peak, oom, spill, wall, rss_peak)


def scope_command(cap_bytes: int, argv: list[str]) -> list[str]:
    mib = -(-cap_bytes // (1 << 20))
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "-p",
        f"MemoryMax={mib}M",
        "-p",
        "MemorySwapMax=0",
        "--",
        *argv,
    ]


def cgroup_of(pid: int | str) -> str | None:
    """The cgroup-v2 path of a process (``/user.slice/.../run-1234.scope``), or None."""
    try:
        line = Path(f"/proc/{pid}/cgroup").read_text().strip().splitlines()[0]
    except (OSError, IndexError):
        return None
    return line.partition("::")[2] or None


def find_cgroup(pid: int, wait_s: float = 5.0) -> Cgroup | None:
    """The scope's cgroup: ``systemd-run --scope`` moves its own pid into the scope, then execs the command.

    Our own cgroup (this harness may itself run in a scope) never counts, so it is never killed by mistake.
    """
    own = cgroup_of("self")
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        rel = cgroup_of(pid)
        if rel is None:
            return None
        if rel != own and rel.endswith(".scope"):
            return Cgroup(CGROUP_ROOT / rel.lstrip("/"))
        time.sleep(0.005)
    return None


# ---------------------------------------------------------------- one run
def read_result(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT "):
            with contextlib.suppress(ValueError):
                return json.loads(line[7:])
    return None


def run_one(
    engine: str,
    data: Path,
    query: str,
    engine_memory: str,
    cap_bytes: int,
    timeout_s: float | None,
    work: Path,
) -> dict:
    """One fresh-process run under the cap. The result stays in ``work/out`` for the caller to verify and remove."""
    out, scratch = work / "out", work / "scratch"
    work.mkdir(parents=True, exist_ok=True)
    stdout_f, stderr_f = work / "stdout", work / "stderr"
    argv = [sys.executable, "-m", "krishiv_bench.memcap_child", engine, str(data), query, engine_memory]
    argv += [str(out), str(scratch)]
    env = dict(os.environ, POLARS_MAX_THREADS="8")
    with stdout_f.open("wb") as so, stderr_f.open("wb") as se:
        proc = subprocess.Popen(scope_command(cap_bytes, argv), stdout=so, stderr=se, env=env, stdin=subprocess.DEVNULL)
        cgroup = find_cgroup(proc.pid)
        obs = monitor(proc, cgroup, scratch, timeout_s)
    result = read_result(stdout_f.read_text(errors="replace"))
    status = classify(obs, cap_bytes, result is not None)
    rec = {
        "engine": engine,
        "query": query,
        "status": status,
        "wall_s": round(obs.wall_s, 3),
        "peak_rss_bytes": obs.rss_peak_bytes,
        "peak_cgroup_bytes": obs.peak_bytes,
        "spill_bytes": obs.spill_peak_bytes,
        "returncode": obs.returncode,
        "oom_kills": obs.oom_kills,
    }
    if status == "finished" and result is not None:
        rec |= {k: result[k] for k in ("version", "setup_s", "query_s", "rows_out")}
    else:
        tail = stderr_f.read_text(errors="replace").strip().splitlines()[-5:]
        rec["error"] = " | ".join(tail)[:500]
    return rec


# ---------------------------------------------------------------- correctness
def checksum(out: Path, query: str) -> dict:
    """The ``CHECKS`` values of a result directory, computed by DuckDB."""
    import duckdb

    con = duckdb.connect()
    con.execute("SET memory_limit = '2GB'")
    glob = str(out / "**" / "*.parquet").replace("'", "''")
    cur = con.execute(f"SELECT {CHECKS[query]} FROM read_parquet('{glob}')")
    cols = [d[0] for d in cur.description or []]
    row = cur.fetchone()
    return dict(zip(cols, row if row else [None] * len(cols), strict=True))


def compare_checksums(ref: dict, got: dict) -> str | None:
    """None when equal; else the first differing key. Integers must match exactly, floats within 1e-9 relative."""
    for k, a in ref.items():
        b = got.get(k)
        if isinstance(a, float) or isinstance(b, float):
            same = a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)
        else:
            same = a == b
        if not same:
            return f"{k}: reference {a!r}, got {b!r}"
    return None


# ---------------------------------------------------------------- suite
def resolve_queries(tokens: list[str]) -> list[str]:
    """``q1,q3_window_latest`` -> full query ids; an unknown token is an error."""
    out = []
    for t in (x.strip() for x in tokens if x.strip()):
        hit = [q for q in QUERIES if q == t or q.startswith(t + "_")]
        if not hit:
            raise ValueError(f"unknown memcap query {t!r}; available: {', '.join(QUERIES)}")
        out += [q for q in hit if q not in out]
    return out


def rotated(order: list[str], rep: int) -> list[str]:
    """The engine order of repetition ``rep``: rotated left by ``rep`` so no engine is always first or last."""
    k = rep % len(order)
    return order[k:] + order[:k]


def median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def summarize(runs: list[dict]) -> dict:
    """The timed runs of one (engine, query): counts and medians over the finished ones."""
    timed = [r for r in runs if r["phase"] == "timed"]
    done = [r for r in timed if r["status"] == "finished"]
    spills = [r["spill_bytes"] for r in done if r.get("spill_bytes") is not None]
    return {
        "n": len(timed),
        "finished": len(done),
        "killed": sum(r["status"] == "killed" for r in timed),
        "median_wall_s": median([r["wall_s"] for r in done]),
        "median_peak_rss_bytes": median([r["peak_rss_bytes"] for r in done]),
        "max_peak_rss_bytes": max((r["peak_rss_bytes"] for r in timed), default=None),
        "median_spill_bytes": median(spills),
    }


def run_memcap(
    label: str,
    cap: str,
    engines: list[str],
    engine_memory: str = "3g",
    reps: int = 3,
    queries: list[str] | None = None,
    timeout_s: float | None = 1800,
    run: Callable[..., dict] = run_one,
    check: Callable[[Path, str], dict] = checksum,
) -> Path:
    """Runs the suite and writes ``$KRISHIV_DATA/results/<ts>_memcap_<rows>_<cap>_<mem>.json``."""
    from .abba import bench_lock, loadavg
    from .runner import git_info, machine_info

    cap_bytes = parse_size(cap)
    parse_size(engine_memory)
    parse_rows(label)
    if reps < 1:
        raise ValueError(f"--reps must be at least 1 (got {reps})")
    unknown = [e for e in engines if e not in ENGINES]
    if unknown or not engines:
        raise ValueError(f"engines must be a non-empty subset of {', '.join(ENGINES)} (got {engines})")
    qs = queries or QUERIES
    bad = [q for q in qs if q not in QUERIES]
    if bad:
        raise ValueError(f"unknown memcap queries {bad}; available: {', '.join(QUERIES)}")
    data = memcap_dir(label)
    digest = load_digest(data)
    order = [REFERENCE] + [e for e in engines if e != REFERENCE]  # the reference runs first in the validation pass
    record = {
        "suite": "memcap",
        "kind": "memcap",
        "ts": time.strftime("%Y%m%dT%H%M%S"),
        "rows": label.lower(),
        "cap_bytes": cap_bytes,
        "engine_memory": engine_memory,
        "threads": 8,
        "reps": reps,
        "reported_engines": list(engines),
        "machine": machine_info(),
        "git": git_info(),
        "dataset": {"path": str(data), **digest},
        "queries": {},
    }
    work_root = data_root() / "tmp" / f"memcap-{os.getpid()}"
    n = 0

    def go(engine: str, query: str, keep: bool) -> tuple[dict, Path]:
        nonlocal n
        n += 1
        work = work_root / f"{n}-{engine}-{query}"
        rec = run(engine, data, query, engine_memory, cap_bytes, timeout_s, work)
        if not (keep and rec["status"] == "finished"):
            shutil.rmtree(work, ignore_errors=True)
        return rec, work / "out"

    with bench_lock() as waited:
        record["bench_lock_wait_s"] = waited
        record["loadavg_start"] = loadavg()
        try:
            for q in qs:
                record["queries"][q] = run_query(q, order, reps, go, check)
        finally:
            shutil.rmtree(work_root, ignore_errors=True)
        record["loadavg_end"] = loadavg()
    versions = {}
    for per_engine in record["queries"].values():
        for e, d in per_engine["engines"].items():
            for r in d["runs"]:
                if r.get("version"):
                    versions[e] = r["version"]
    record["engine_versions"] = versions
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{record['ts']}_memcap_{record['rows']}_{cap.lower()}_{engine_memory.lower()}.json"
    f.write_text(json.dumps(record, indent=2))
    print(f"result: {f}")
    return f


def run_query(q: str, order: list[str], reps: int, go: Callable, check: Callable) -> dict:
    """Validation pass, then ``reps`` timed repetitions (engine order rotated), for one query."""
    ref: dict | None = None
    verdict: dict[str, str] = {}
    runs: dict[str, list[dict]] = {e: [] for e in order}

    def verify(engine: str, rec: dict, out: Path) -> None:
        nonlocal ref
        try:
            got = check(out, q)
        except Exception as e:
            rec["verdict"] = verdict[engine] = f"unverified: {type(e).__name__}: {str(e)[:200]}"
            return
        finally:
            shutil.rmtree(out.parent, ignore_errors=True)
        if engine == REFERENCE:
            ref = got
            rec["verdict"] = verdict[engine] = "reference"
        elif ref is None:
            rec["verdict"] = verdict[engine] = "unverified: no reference result"
        else:
            diff = compare_checksums(ref, got)
            rec["verdict"] = verdict[engine] = "ok" if diff is None else f"mismatch: {diff}"

    for e in order:
        rec, out = go(e, q, True)
        rec["phase"] = "validation"
        runs[e].append(rec)
        if rec["status"] == "finished":
            verify(e, rec, out)
        else:
            verdict[e] = "unverified: validation run " + rec["status"]
    for rep in range(reps):
        for e in rotated(order, rep):
            if verdict[e].startswith("mismatch"):
                continue  # a wrong result is never timed
            pending = not (verdict[e] in ("ok", "reference") or verdict[e].startswith("mismatch"))
            rec, out = go(e, q, pending)
            rec["phase"] = "timed"
            rec["rep"] = rep
            runs[e].append(rec)
            if rec["status"] != "finished":
                continue
            if pending:
                verify(e, rec, out)
            elif ref is not None and rec["rows_out"] != ref.get("n"):
                rec["verdict"] = verdict[e] = f"mismatch: n: reference {ref.get('n')!r}, got {rec['rows_out']!r}"
            else:
                rec["verdict"] = verdict[e]
            if verdict[e].startswith("mismatch"):
                rec["status"] = "mismatch"
    return {
        "reference": ref,
        "engines": {e: {"verdict": verdict[e], "runs": runs[e], "summary": summarize(runs[e])} for e in order},
    }


# ---------------------------------------------------------------- report
def latest_memcap(label: str, directory: Path | None = None) -> dict[tuple[str, str], tuple[str, dict, dict]]:
    """Newest result per (engine, engine memory) for ``label`` rows: ``(file name, run record, engine's queries)``."""
    best: dict[tuple[str, str], tuple[str, dict, dict]] = {}
    for f in sorted((directory or results_dir()).glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except ValueError:
            continue
        if r.get("kind") != "memcap" or r.get("rows") != label.lower():
            continue
        for e in r["reported_engines"]:
            per_q = {q: d["engines"][e] for q, d in r["queries"].items() if e in d["engines"]}
            best[(e, r["engine_memory"])] = (f.name, r, per_q)
    return best


def _gib(x: float | None) -> str:
    return "-" if x is None else f"{x / GIB:.2f}"


def render_memcap(label: str, directory: Path | None = None) -> str:
    best = latest_memcap(label, directory)
    if not best:
        raise SystemExit(f"no memcap results for {label} under {directory or results_dir()}")
    first = next(iter(best.values()))[1]
    caps = sorted({r["cap_bytes"] for _, r, _ in best.values()})
    L = [f"# memcap report: {label.lower()} rows", ""]
    L += [
        "Fresh process per run under a hard cgroup cap; a run killed above the cap counts as failed, never timed.",
        f"- Machine: {first['machine']['cpu_model']}, {first['machine']['cores_logical']} logical cores, "
        f"{first['machine']['ram_bytes'] / GIB:.0f} GiB RAM, {first['machine']['os']}",
        f"- Cap: {', '.join(f'{c / GIB:.2f} GiB' for c in caps)}; threads {first['threads']}",
        f"- Dataset: `{first['dataset']['path']}`, sales sha256 `{first['dataset']['sales_content_sha256'][:16]}`",
        "- Wall time includes interpreter and session start (and the scope's start). Peak memory is the highest "
        "sampled resident set of the whole process tree (the scope's anonymous + mapped-file + shared pages, "
        "sampled every 10 ms; page cache excluded). Spill is the peak size of the engine's scratch directory "
        "(`-`: none or not measurable).",
        "- Medians are over finished runs; `peak` is `>cap (killed)` when no run finished.",
        "",
        "| Query | Engine | Engine memory | Median wall (s) | Finished | Peak memory (GiB) | Spill (GiB) "
        "| Verdict | Result file |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    queries = [q for q in QUERIES if any(q in per_q for _, _, per_q in best.values())]
    for q in queries:
        for (e, mem), (fname, r, per_q) in sorted(best.items()):
            d = per_q.get(q)
            if d is None:
                continue
            s = d["summary"]
            wall = "-" if s["median_wall_s"] is None else f"{s['median_wall_s']:.2f}"
            if s["finished"]:
                peak = _gib(s["median_peak_rss_bytes"])
            else:
                peak = f">{r['cap_bytes'] / GIB:.1f} (killed)" if s["killed"] else _gib(s["max_peak_rss_bytes"])
            L.append(
                f"| {q} | {e} | {mem} | {wall} | {s['finished']}/{s['n']} | {peak} | {_gib(s['median_spill_bytes'])} "
                f"| {d['verdict']} | `{fname}` |"
            )
    return "\n".join(L) + "\n"


def write_memcap_report(label: str) -> Path:
    md = render_memcap(label)
    out = results_dir() / f"report_memcap_{label.lower()}.md"
    out.write_text(md)
    return out


__all__ = ["ENGINES", "classify", "monitor", "render_memcap", "run_memcap", "write_memcap_report"]
