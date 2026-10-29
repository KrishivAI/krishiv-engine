"""Performance gap ledger: where Krishiv spends more wall time than DuckDB, by operator bucket.

``krishiv-bench ledger --suite tpch|clickbench|tpcds --sf N`` measures, per query:

* **Krishiv**: a plain process (3 ``PROFILE`` runs, the 2nd, warm one is the wall time) and a perf process
  (a fresh runner under ``perf record -F 499 -g --call-graph fp -k CLOCK_MONOTONIC``, 2 ``PROFILE`` runs).
  The harness logs the ``CLOCK_MONOTONIC`` start and end of each run; only the samples inside run 2 are
  attributed. Each sample's call chain is classified by the first frame from the leaf up that matches a bucket
  rule (``BUCKET_RULES``). The run's window is cut into ``BIN_S`` bins; each bin's wall is shared among the
  buckets in proportion to their samples (a single-core stage gets its full wall, not 1/threads), and the split is
  scaled to the plain-process wall. Build the runner with frame pointers for usable stacks:
  ``RUSTFLAGS='-C target-cpu=x86-64-v3 -C force-frame-pointers=yes' CARGO_PROFILE_DIST_STRIP=none
  CARGO_PROFILE_DIST_DEBUG=line-tables-only cargo build --profile dist -p krishiv-bench-runner``.
* **DuckDB**: warm-up plus 3 timed runs (median), then one run with the JSON operator profile. Operator timings
  (summed over threads) map to the same buckets (``DUCKDB_BUCKETS``); a join reports one timing, split into
  build and probe by the row counts of its right (build) and left (probe) children. The median wall is shared in
  proportion to the operator timings.

Gap = Krishiv bucket seconds - DuckDB bucket seconds. The ledger ranks buckets by summed gap and gives, per
bucket, the geomean of Krishiv/DuckDB over the queries if every query's positive gap in that bucket went to zero
(K' = K - max(gap, 0)). Caveats: DuckDB's timings have no scheduler or idle term, so Krishiv's ``other`` bucket is
an upper bound; one plain and one perf process per query means no confidence intervals.

Output: ``$KRISHIV_DATA/results/<ts>_ledger_<suite>_sf<N>.json`` and ``.md``, with the delta against the previous
ledger of the same suite (``--previous FILE`` or the newest earlier one).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .abba import bench_lock, discover_tables, geomean, wait_idle
from .adapters import Dataset
from .adapters.duckdb_engine import DuckDBEngine
from .adapters.krishiv import KrishivEngine, RunnerCrashed, RunnerError
from .errors import describe_error
from .paths import data_root, dataset_dir, fmt_sf, results_dir
from .runner import git_info, load_queries, machine_info

BUCKETS = (
    "join build",
    "aggregate",
    "filter/project",
    "other",
    "join probe",
    "scan/decode",
    "sort/top-N",
    "window",
    "exchange/repartition",
)
BIN_S = 0.02
PERF_FREQ = 499
SCAN_FILTER = "scan-filter"  # in-scan predicates; merged into scan/decode (DuckDB evaluates them in its scan)

# Leaf-up, first frame matching any rule decides; rules are tried in order within a frame. The rules are the
# ones of the 2026-10-24 gap ledger (and of the perf attribution scripts that produced it).
BUCKET_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (b, re.compile(p))
    for b, p in (
        ("window", r"krishiv_exec::window::"),
        ("sort/top-N", r"krishiv_exec::operators::sort|krishiv_exec::operators::limit|TopN|krishiv_exec::.*::sort::"),
        ("join build", r"krishiv_exec::join::build|krishiv_exec::join::grace|HashJoinBuild|NestedLoopJoinBuild"),
        ("join probe", r"krishiv_exec::join::probe|krishiv_exec::join::nested|HashJoinProbe"),
        ("aggregate", r"krishiv_exec::aggregate::"),
        (
            "exchange/repartition",
            # Anchored: a bare `[Ee]xchange` also matches `compare_exchange`, the leaf of every atomic.
            r"krishiv_(exec|shuffle)::.*([Ee]xchange|[Rr]epartition)|krishiv_exec::operators::replicate"
            r"|krishiv_shuffle::",
        ),
        (SCAN_FILTER, r"krishiv_io::.*(row_filter|RowFilter|predicate|runtime_filter)|ArrowPredicate|RowFilter"),
        (
            "filter/project",
            r"krishiv_exec::operators::(filter|project|expr|runtime_filter)|krishiv_exec::planner::eval"
            r"|krishiv_functions::",
        ),
        (
            "scan/decode",
            r"krishiv_io::|krishiv_exec::operators::(file_scan|scan)|^parquet::|<parquet::|ZSTD_|snap::|lz4"
            r"|object_store::|GetResult",
        ),
        ("other", r"krishiv_exec::driver::runqueue_wait"),
    )
)
# join::table / keys / rows are shared by build and probe; with no outer frame to say which, they count as probe.
SHARED_JOIN = re.compile(r"krishiv_exec::join::(table|keys)|krishiv_exec::rows::")

# DuckDB operator type -> bucket (joins are split by ``duckdb_buckets``).
DUCKDB_BUCKETS = {
    "TABLE_SCAN": "scan/decode",
    "READ_PARQUET": "scan/decode",
    "FILTER": "filter/project",
    "PROJECTION": "filter/project",
    "HASH_GROUP_BY": "aggregate",
    "PERFECT_HASH_GROUP_BY": "aggregate",
    "UNGROUPED_AGGREGATE": "aggregate",
    "SIMPLE_AGGREGATE": "aggregate",
    "STREAMING_AGGREGATE": "aggregate",
    "PARTITIONED_AGGREGATE": "aggregate",
    "ORDER_BY": "sort/top-N",
    "TOP_N": "sort/top-N",
    "WINDOW": "window",
    "STREAMING_WINDOW": "window",
}
DUCKDB_JOINS = {
    "HASH_JOIN",
    "LEFT_DELIM_JOIN",
    "RIGHT_DELIM_JOIN",
    "PIECEWISE_MERGE_JOIN",
    "NESTED_LOOP_JOIN",
    "BLOCKWISE_NL_JOIN",
    "CROSS_PRODUCT",
    "IE_JOIN",
    "ASOF_JOIN",
    "POSITIONAL_JOIN",
    "DELIM_JOIN",
}


# ---------------------------------------------------------------------------------------------------- Krishiv side
def classify_frames(frames: Iterable[str]) -> str:
    """Bucket of a call chain given leaf first: the first frame matching a rule decides, else ``other``."""
    frames = list(frames)
    for frame in frames:
        for bucket, pattern in BUCKET_RULES:
            if pattern.search(frame):
                return "scan/decode" if bucket == SCAN_FILTER else bucket
    if any(SHARED_JOIN.search(f) for f in frames):
        return "join probe"
    return "other"


_HEADER = re.compile(r"^\s*(?:\d+\s+)?(\d+\.\d+):")  # `[tid] time:`
_FRAME = re.compile(r"^\s+[0-9a-fA-F]+\s+(.*?)(?:\+0x[0-9a-fA-F]+)?(?:\s+\(.*\))?$")


def parse_perf_script(text: str) -> list[tuple[float, list[str]]]:
    """Samples of ``perf script -F time,ip,sym``: ``(seconds, frames leaf first)`` per sample."""
    samples: list[tuple[float, list[str]]] = []
    cur: tuple[float, list[str]] | None = None
    for line in text.splitlines():
        if not line.strip():
            if cur:
                samples.append(cur)
            cur = None
            continue
        if (m := _HEADER.match(line)) and not line.startswith("\t"):
            if cur:
                samples.append(cur)
            cur = (float(m.group(1)), [])
        elif cur is not None and (f := _FRAME.match(line)):
            cur[1].append(f.group(1))
    if cur:
        samples.append(cur)
    return samples


def wall_split(
    samples: list[tuple[float, list[str]]], t0: float, t1: float, wall_s: float, bin_s: float = BIN_S
) -> dict[str, float]:
    """Wall seconds per bucket: each bin's wall is shared by its samples' buckets, scaled to ``wall_s``.

    Samples outside ``[t0, t1)`` are ignored; a bin without samples is idle (no thread on CPU) and its wall is
    booked to ``other``.
    """
    if t1 <= t0 or wall_s <= 0:
        return {}
    bins: dict[int, Counter[str]] = {}
    for ts, frames in samples:
        if t0 <= ts < t1:
            bins.setdefault(int((ts - t0) / bin_s), Counter())[classify_frames(frames)] += 1
    out: defaultdict[str, float] = defaultdict(float)
    for idx in range(max(1, math.ceil((t1 - t0) / bin_s - 1e-9))):
        width = min(bin_s, t1 - (t0 + idx * bin_s))
        counts = bins.get(idx)
        if not counts:
            out["other"] += width  # no thread on CPU: idle time is booked to `other`
            continue
        total = sum(counts.values())
        for bucket, n in counts.items():
            out[bucket] += width * n / total
    scale = wall_s / (t1 - t0)
    return {b: s * scale for b, s in out.items()}


# ---------------------------------------------------------------------------------------------------- DuckDB side
def _walk(node: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield node
    for ch in node.get("children", []):
        yield from _walk(ch)


def _rows(node: dict[str, Any]) -> float:
    return float(node.get("intermediate_rows", node.get("operator_cardinality", 0)) or 0)


def _timing(node: dict[str, Any]) -> float:
    return float(node.get("timing", node.get("operator_timing", 0.0)) or 0.0)


def _type(node: dict[str, Any]) -> str:
    return str(node.get("type", node.get("operator_type", "")))


def duckdb_buckets(profile: dict[str, Any], wall_s: float) -> dict[str, float]:
    """Wall seconds per bucket from a DuckDB JSON profile: operator timings share the median wall."""
    roots = profile["operator"] if "operator" in profile else profile.get("children", [])
    seconds: defaultdict[str, float] = defaultdict(float)
    for root in roots:
        for node in _walk(root):
            typ, t = _type(node), _timing(node)
            if typ in DUCKDB_JOINS:
                kids = node.get("children", [])
                if len(kids) == 2:  # left = probe, right = build
                    probe, build = _rows(kids[0]), _rows(kids[1])
                    share = build / (build + probe) if build + probe > 0 else 0.5
                else:
                    share = 0.0
                seconds["join build"] += t * share
                seconds["join probe"] += t * (1 - share)
            else:
                seconds[DUCKDB_BUCKETS.get(typ, "other")] += t
    total = sum(seconds.values())
    if wall_s <= 0:
        return {}
    if total <= 0:
        # DuckDB writes no profile for a query it answers from metadata (COUNT(*) over Parquet): book it as scan.
        return {"scan/decode": wall_s} if not roots else {}
    return {b: s / total * wall_s for b, s in seconds.items()}


# ---------------------------------------------------------------------------------------------------- the ledger
def bucket_gaps(krishiv: dict[str, float], duckdb: dict[str, float]) -> dict[str, float]:
    """Krishiv - DuckDB seconds per bucket (every bucket either side has)."""
    return {b: krishiv.get(b, 0.0) - duckdb.get(b, 0.0) for b in BUCKETS}


def summarize(queries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-bucket totals, gaps, shares and the geomean effect of closing each bucket's positive gaps."""
    ok = {q: e for q, e in queries.items() if e.get("status") == "ok"}
    if not ok:
        return {"queries_ok": 0, "buckets": {}, "geomean": None}
    ratios = {q: e["krishiv_s"] / e["duckdb_s"] for q, e in ok.items()}
    base = geomean(list(ratios.values()))
    total_gap = sum(max(0.0, e["gaps"].get(b, 0.0)) for e in ok.values() for b in BUCKETS)
    buckets: dict[str, Any] = {}
    for b in BUCKETS:
        gap = sum(e["gaps"].get(b, 0.0) for e in ok.values())
        pos = sum(max(0.0, e["gaps"].get(b, 0.0)) for e in ok.values())
        reduced = [max(e["krishiv_s"] - max(0.0, e["gaps"].get(b, 0.0)), 1e-9) / e["duckdb_s"] for e in ok.values()]
        buckets[b] = {
            "krishiv_s": sum(e["krishiv_buckets"].get(b, 0.0) for e in ok.values()),
            "duckdb_s": sum(e["duckdb_buckets"].get(b, 0.0) for e in ok.values()),
            "gap_s": gap,
            "share": pos / total_gap if total_gap else 0.0,
            "geomean_if_zero": geomean(reduced),
        }
    return {
        "queries_ok": len(ok),
        "queries_total": len(queries),
        "geomean": base,
        "krishiv_sum_s": sum(e["krishiv_s"] for e in ok.values()),
        "duckdb_sum_s": sum(e["duckdb_s"] for e in ok.values()),
        "buckets": buckets,
        "ratios": ratios,
    }


def _top(gaps: dict[str, float], n: int = 2) -> list[tuple[str, float]]:
    return sorted(((b, g) for b, g in gaps.items() if g > 0), key=lambda x: -x[1])[:n]


def render_markdown(record: dict[str, Any], previous: dict[str, Any] | None = None) -> str:
    """The per-query table, the ranked bucket table and, with ``previous``, the deltas."""
    s = record["summary"]
    suite, sf = record["suite"], record["scale_factor"]
    lines = [f"# Performance gap ledger: Krishiv vs DuckDB ({suite} SF{fmt_sf(sf)})", ""]
    git = record.get("git", {})
    lines.append(
        f"Krishiv `{git.get('commit', 'unknown')}`; DuckDB {record['engines']['duckdb']['version']}; "
        f"{record['threads']} threads; {record['started_utc']}. Method: `krishiv_bench/ledger.py`."
    )
    lines += ["", "## Per query", ""]
    head = "| q | Krishiv s | DuckDB s | ratio | gap s | top bucket 1 (gap s) | top bucket 2 (gap s) |"
    if previous:
        head += " ratio vs previous |"
    lines += [head, "|---|--:|--:|--:|--:|---|---|" + ("--:|" if previous else "")]
    prev_q = (previous or {}).get("queries", {})
    for q, e in record["queries"].items():
        if e.get("status") != "ok":
            lines.append(f"| {q} | failed | - | - | - | {str(e.get('reason', ''))[:80].replace('|', '/')} | |")
            continue
        top = _top(e["gaps"])
        cells = [f"{b} {g:+.3f}" for b, g in top] + [""] * (2 - len(top))
        row = (
            f"| {q} | {e['krishiv_s']:.3f} | {e['duckdb_s']:.3f} | {e['krishiv_s'] / e['duckdb_s']:.2f} "
            f"| {e['krishiv_s'] - e['duckdb_s']:.3f} | {cells[0]} | {cells[1]} |"
        )
        if previous:
            p = prev_q.get(q, {})
            row += (
                f" {e['krishiv_s'] / e['duckdb_s'] - p['krishiv_s'] / p['duckdb_s']:+.2f} |"
                if p.get("status") == "ok"
                else " - |"
            )
        lines.append(row)
    if s["queries_ok"]:
        lines += [
            f"| **sum** | {s['krishiv_sum_s']:.2f} | {s['duckdb_sum_s']:.2f} "
            f"| {s['krishiv_sum_s'] / s['duckdb_sum_s']:.2f} | {s['krishiv_sum_s'] - s['duckdb_sum_s']:.2f} | | |"
            + (" |" if previous else ""),
            "",
            f"Geomean Krishiv/DuckDB **{s['geomean']:.2f}x** over {s['queries_ok']}/{s['queries_total']} queries.",
        ]
    lines += ["", "## Gap per bucket, ranked", ""]
    head = (
        f"| rank | bucket | Krishiv s | DuckDB s | gap s | share of gap | geomean if its positive gap goes to 0 "
        f"(now {s['geomean']:.2f}x) |"
        if s["queries_ok"]
        else "| rank | bucket | Krishiv s | DuckDB s | gap s | share | geomean |"
    )
    if previous:
        head += " gap s vs previous |"
    lines += [head, "|--:|---|--:|--:|--:|--:|--:|" + ("--:|" if previous else "")]
    prev_b = (previous or {}).get("summary", {}).get("buckets", {})
    ranked = sorted(s["buckets"].items(), key=lambda kv: -kv[1]["gap_s"])
    for i, (b, v) in enumerate(ranked, 1):
        row = (
            f"| {i} | {b} | {v['krishiv_s']:.2f} | {v['duckdb_s']:.2f} | {v['gap_s']:+.2f} "
            f"| {v['share'] * 100:.1f}% | {v['geomean_if_zero']:.2f}x |"
        )
        if previous:
            row += f" {v['gap_s'] - prev_b[b]['gap_s']:+.2f} |" if b in prev_b else " - |"
        lines.append(row)
    lines += [
        "",
        "Caveats: DuckDB's operator timings have no scheduler or idle term, so Krishiv's `other` gap is an upper "
        "bound; the CPU-to-wall mapping assumes CPU in a bin costs wall in proportion; one plain and one perf "
        "process per query, no confidence intervals.",
    ]
    return "\n".join(lines) + "\n"


def previous_ledger(suite: str, sf: float, before: str, directory: Path | None = None) -> Path | None:
    """The newest ledger JSON of ``suite``/``sf`` with a timestamp earlier than ``before`` (``%Y%m%dT%H%M%SZ``)."""
    d = directory or results_dir()
    found = sorted(d.glob(f"*_ledger_{suite}_sf{fmt_sf(sf)}.json")) if d.is_dir() else []
    older = [p for p in found if p.name.split("_", 1)[0] < before]
    return older[-1] if older else None


# ---------------------------------------------------------------------------------------------------- measurement
def _mono() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def perf_launcher(data: Path) -> list[str]:
    """``perf record`` command prefix writing ``data``; frame-pointer call graphs on the monotonic clock."""
    return ["perf", "record", "-q", "-F", str(PERF_FREQ), "-g", "--call-graph", "fp", "-k", "CLOCK_MONOTONIC",
            "-o", str(data), "--"]  # fmt: skip


def perf_script(data: Path) -> str:
    out = subprocess.run(
        ["perf", "script", "-i", str(data), "-F", "tid,time,ip,sym", "--no-inline"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def duckdb_profile(duck: DuckDBEngine, query) -> dict[str, Any]:
    """One run with the JSON operator profile of ``query``."""
    con = duck.con
    assert con is not None
    out = Path(tempfile.mkdtemp(prefix="ledger-duck-")) / "profile.json"
    try:
        con.execute("PRAGMA enable_profiling='json'")
        con.execute(f"SET profiling_output='{out}'")
        con.execute(query.sql).fetchall()
        con.execute("PRAGMA disable_profiling")
        # DuckDB writes no profile for a query it answers from metadata alone (no operators).
        return json.loads(out.read_text()) if out.is_file() else {"operator": []}
    finally:
        shutil.rmtree(out.parent, ignore_errors=True)


def measure_krishiv(
    runner: str | None, ds: Dataset, query, threads: int, memory_limit: str, timeout_s: float, plain: KrishivEngine
) -> dict[str, Any]:
    """Wall seconds (plain process, 2nd of 3 runs) and the perf bucket split (fresh process, 2nd of 2 runs)."""
    walls = [plain.profile(query)["seconds"] for _ in range(3)]
    wall = walls[1]
    tmp = Path(tempfile.mkdtemp(prefix="ledger-perf-"))
    data = tmp / "perf.data"
    eng: KrishivEngine | None = None
    try:
        eng = KrishivEngine(threads=threads, memory_limit=memory_limit, binary=runner, launcher=perf_launcher(data))
        eng.timeout_s = timeout_s or None
        eng.setup(ds)
        eng.profile(query)
        t0 = _mono()
        eng.profile(query)
        t1 = _mono()
        eng.teardown()  # QUIT lets perf finalise its file
        buckets = wall_split(parse_perf_script(perf_script(data)), t0, t1, wall)
    finally:
        if eng is not None:
            eng.cancel()
        shutil.rmtree(tmp, ignore_errors=True)
    return {"seconds": wall, "runs": walls, "buckets": buckets}


def measure_duckdb(duck: DuckDBEngine, query, timeout_s: float) -> dict[str, Any]:
    """Median of 3 warm runs and the operator-bucket split of one profiled run."""
    duck.run(query)  # warm-up
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        duck.run(query)
        times.append(time.perf_counter() - t0)
    wall = statistics.median(times)
    return {"seconds": wall, "runs": times, "buckets": duckdb_buckets(duckdb_profile(duck, query), wall)}


def run_ledger(
    suite: str,
    sf: float,
    dataset: str | None = None,
    only: list[str] | None = None,
    threads: int = 12,
    memory_limit: str = "32g",
    timeout_s: float = 600,
    runner: str | None = None,
    previous: str | None = None,
    measure: Callable[..., dict[str, Any]] | None = None,
) -> Path:
    from .cli import ABBA_DATASET

    name = dataset or ABBA_DATASET[suite]
    root = data_root() / name if suite == "clickbench" else dataset_dir(name, sf)
    tables = discover_tables(root)
    if not tables:
        raise SystemExit(f"no dataset at {root}")
    queries = load_queries(suite, "krishiv", only)
    ds = Dataset(name, sf, root, tables)
    with bench_lock():
        wait_idle()
        return _run_locked(ds, suite, queries, threads, memory_limit, timeout_s, runner, previous, measure)


def _run_locked(ds, suite, queries, threads, memory_limit, timeout_s, runner, previous, measure=None) -> Path:
    plain = KrishivEngine(threads=threads, memory_limit=memory_limit.lower(), binary=runner)
    plain.timeout_s = timeout_s or None
    duck = DuckDBEngine(threads=threads, memory_limit=memory_limit.upper())
    started = dt.datetime.now(dt.UTC)
    record: dict[str, Any] = {
        "schema": 1,
        "kind": "ledger",
        "suite": suite,
        "scale_factor": ds.sf,
        "started_utc": started.isoformat(),
        "machine": machine_info(),
        "git": git_info(),
        "threads": threads,
        "memory_limit": memory_limit,
        "engines": {"duckdb": {"version": duck.version(), "config": duck.config()}},
        "queries": {},
    }
    duck.setup(ds)
    plain.setup(ds)
    record["engines"]["krishiv"] = {"version": plain.version(), "config": plain.config()}
    try:
        for q in queries:
            record["queries"][q.id] = _one_query(q, ds, plain, duck, threads, memory_limit, timeout_s, runner, measure)
    finally:
        plain.teardown()
        duck.teardown()
    record["finished_utc"] = dt.datetime.now(dt.UTC).isoformat()
    record["summary"] = summarize(record["queries"])
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    prev_path = Path(previous) if previous else previous_ledger(suite, ds.sf, stamp)
    prev = json.loads(prev_path.read_text()) if prev_path and prev_path.is_file() else None
    record["previous"] = str(prev_path) if prev else None
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"{stamp}_ledger_{suite}_sf{fmt_sf(ds.sf)}"
    base.with_suffix(".json").write_text(json.dumps(record, indent=2))
    base.with_suffix(".md").write_text(render_markdown(record, prev))
    print(base.with_suffix(".md").read_text())
    print(f"result: {base.with_suffix('.json')}\nledger: {base.with_suffix('.md')}")
    return base.with_suffix(".json")


def _one_query(q, ds, plain, duck, threads, memory_limit, timeout_s, runner, measure=None) -> dict[str, Any]:
    kr_fn = measure or measure_krishiv
    try:
        d = measure_duckdb(duck, q, timeout_s)
    except Exception as e:
        return {"status": "failed", "reason": f"duckdb {describe_error(e)}"}
    try:
        k = kr_fn(runner, ds, q, threads, memory_limit.lower(), timeout_s, plain)
    except (RunnerError, RunnerCrashed) as e:
        return {"status": "failed", "reason": f"krishiv {describe_error(e)}"}
    except (subprocess.CalledProcessError, OSError) as e:
        return {"status": "failed", "reason": f"perf: {e}"}
    gaps = bucket_gaps(k["buckets"], d["buckets"])
    print(f"  {q.id}: krishiv {k['seconds']:.3f}s duckdb {d['seconds']:.3f}s x{k['seconds'] / d['seconds']:.2f}")
    return {
        "status": "ok",
        "krishiv_s": k["seconds"],
        "duckdb_s": d["seconds"],
        "krishiv_runs_s": k["runs"],
        "duckdb_runs_s": d["runs"],
        "krishiv_buckets": k["buckets"],
        "duckdb_buckets": d["buckets"],
        "gaps": gaps,
    }


__all__ = [
    "BUCKETS",
    "bucket_gaps",
    "classify_frames",
    "duckdb_buckets",
    "parse_perf_script",
    "previous_ledger",
    "render_markdown",
    "run_ledger",
    "summarize",
    "wall_split",
]
