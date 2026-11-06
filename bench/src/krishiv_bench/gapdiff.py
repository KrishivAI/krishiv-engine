"""``krishiv-bench gapdiff``: where Krishiv's CPU goes, operator kind by operator kind, beside DuckDB's.

For one query (or ``--all``: the whole suite) both engines run once with their operator profiles on the same Parquet
files: Krishiv's ``PLAN`` command (the ``krishiv.query-profile`` v1 document) and DuckDB's
``EXPLAIN (ANALYZE, FORMAT json)`` (the same data as ``PRAGMA enable_profiling='json'``). Operators of both are mapped
to common kinds and summed:

========== =====================================================================================================
kind       Krishiv operators / DuckDB operators
========== =====================================================================================================
scan       ``FileScan``, ``MemoryScan``, ``Range``, ``LateFetch`` / ``*_SCAN`` (decode is part of the scan)
filter     ``Filter`` / ``FILTER``
project    ``Project`` / ``PROJECTION``
join       ``HashJoinBuild`` + ``HashJoinProbe`` + the other joins / every ``*JOIN*``, ``CROSS_PRODUCT``
aggregate  ``PartialAggregate`` + ``HashAggregate`` / ``*GROUP_BY*``, ``*AGGREGATE*``
sort       ``Sort``, ``TopN`` / ``ORDER_BY``, ``TOP_N``
exchange   ``ExchangeRead``, ``ExchangeWrite``, ``ExchangeWriteResult`` / (none: DuckDB shares memory)
other      everything else (``Result``, ``Limit``, ``Window``, ``Expand``, unions, CTEs ...)
========== =====================================================================================================

Krishiv also reports the split DuckDB cannot give: hash join *build* vs *probe*, and *partial* vs *final* aggregate
(a fused aggregate, where the partial, the exchange and the final run as one breaker, is all "final"). DuckDB's join
and aggregate time is one number each, so those sub-rows show Krishiv only.

CPU seconds are summed over threads in both engines (Krishiv ``compute_time_ns`` per operator, DuckDB's operator
timing), so the ratio is CPU against CPU, not wall time against wall time. The Krishiv run has detailed metrics on and
DuckDB's profiler is on: the seconds are not benchmark timings (use ``abba`` for those).

**Parallelism.** From the Krishiv profile, per pipeline: the operators that run only in that pipeline, their summed
``compute_time_ns`` (the pipeline's CPU) and the sum of their ``max_driver_compute_time_ns``. That sum is an upper bound
of the busiest driver's CPU (each operator's own busiest driver), so its share of the pipeline's CPU is the most
unbalanced the pipeline can be: 1/N for N perfectly balanced drivers, 100% for one driver. A pipeline is flagged when
that share is over ``--busy-share`` (default 50%) and its CPU is at least ``--min-cpu-share`` (default 2%) of the
query's CPU: the single-driver merges behind q13/q16/q18. Operators that span several pipelines cannot be split
between them and are reported as unattributed CPU.

Output in ``$KRISHIV_DATA/results/gapdiff/<ts>_<suite>_sf<sf>/``: ``summary.md`` (one row per query with ``--all``,
else the query's own report), ``gapdiff.json``, one ``<query>.md`` per query, and with ``--perf`` the Krishiv run's
folded stacks ``perf/<query>.folded`` (and ``.svg`` when ``inferno-flamegraph`` is installed). Folded files of two
runner builds go straight into ``inferno-diff-folded`` (or use ``flamediff``, which runs both).
"""

from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .explain_diff import PNode, duckdb_tree

# Pipelines with at least this share of the query's CPU are looked at, and flagged when one driver does more than
# BUSY_SHARE of the pipeline's CPU.
MIN_CPU_SHARE = 0.02
BUSY_SHARE = 0.5

# Krishiv operator name -> kind. The join and aggregate kinds are rolled up from their parts.
KRISHIV_KINDS = {
    "FileScan": "scan",
    "MemoryScan": "scan",
    "Range": "scan",
    "LateFetch": "scan",
    "Filter": "filter",
    "Project": "project",
    "HashJoinBuild": "join_build",
    "HashJoinProbe": "join_probe",
    "NestedLoopJoin": "join_other",
    "NestedLoopJoinBuild": "join_other",
    "SortMergeJoin": "join_other",
    "PartialAggregate": "agg_partial",
    "HashAggregate": "agg_final",
    "Sort": "sort",
    "TopN": "sort",
    "ExchangeRead": "exchange",
    "ExchangeWrite": "exchange",
    "ExchangeWriteResult": "exchange",
}
PARENT = {
    "join_build": "join",
    "join_probe": "join",
    "join_other": "join",
    "agg_partial": "aggregate",
    "agg_final": "aggregate",
}
# DuckDB kinds of `explain_diff` -> kinds here.
DUCKDB_KINDS = {"topn": "sort"}
# Report order: (key, label, is a sub-row of the kind above it).
ROWS = [
    ("scan", "scan / decode", False),
    ("filter", "filter", False),
    ("project", "project", False),
    ("join", "hash join (total)", False),
    ("join_build", "build", True),
    ("join_probe", "probe", True),
    ("join_other", "other joins", True),
    ("aggregate", "aggregate (total)", False),
    ("agg_partial", "partial", True),
    ("agg_final", "final / fused", True),
    ("sort", "sort / top-n", False),
    ("exchange", "exchange", False),
    ("other", "other", False),
]
# Kinds DuckDB reports as one number: Krishiv's finer rows have no DuckDB side.
DUCKDB_HAS = {"scan", "filter", "project", "join", "aggregate", "sort", "exchange", "other"}


@dataclass
class Totals:
    """CPU seconds and rows by kind for one engine."""

    cpu: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    rows_in: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    rows_out: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, kind: str, cpu: float, rows_in: int, rows_out: int) -> None:
        for key in (kind, PARENT.get(kind)):
            if key is not None:
                self.cpu[key] += cpu
                self.rows_in[key] += rows_in
                self.rows_out[key] += rows_out

    def total(self) -> float:
        """Query CPU: every top-level kind (a sub-row is inside its parent)."""
        return sum(v for k, v in self.cpu.items() if k not in PARENT)


def krishiv_totals(doc: dict[str, Any]) -> Totals:
    """CPU and rows by kind from the operators of a ``krishiv.query-profile`` v1 document."""
    if doc.get("format") != "krishiv.query-profile" or doc.get("version") != 1:
        raise ValueError("not a krishiv.query-profile v1 document")
    out = Totals()
    for op in doc["operators"]:
        m = op["metrics"]
        out.add(
            KRISHIV_KINDS.get(op["name"], "other"),
            m["compute_time_ns"] / 1e9,
            int(m["rows_in"]),
            int(m["rows_out"]),
        )
    return out


def duckdb_totals(tree: PNode) -> Totals:
    """CPU and rows by kind from a DuckDB plan tree (rows in: what the operator's children produced)."""
    out = Totals()
    for n in tree.walk():
        kind = DUCKDB_KINDS.get(n.kind, n.kind)
        if kind not in DUCKDB_HAS:
            kind = "other"
        out.add(kind, n.seconds, sum(c.rows or 0 for c in n.children), n.rows or 0)
    return out


@dataclass
class KindRow:
    """One line of the side-by-side table."""

    key: str
    label: str
    sub: bool
    krishiv_cpu: float
    duckdb_cpu: float | None
    ratio: float | None
    krishiv_rows_in: int
    krishiv_rows_out: int
    duckdb_rows_in: int | None
    duckdb_rows_out: int | None


def compare(k: Totals, d: Totals) -> list[KindRow]:
    """The table rows: every kind either engine spent time in, in report order."""
    rows = []
    for key, label, sub in ROWS:
        k_cpu, d_cpu = k.cpu.get(key, 0.0), d.cpu.get(key, 0.0)
        has_d = key in DUCKDB_HAS  # DuckDB does not split joins and aggregates into their parts
        if k_cpu == 0 and d_cpu == 0 and k.rows_out.get(key, 0) == 0 and d.rows_out.get(key, 0) == 0:
            continue
        rows.append(
            KindRow(
                key=key,
                label=label,
                sub=sub,
                krishiv_cpu=k_cpu,
                duckdb_cpu=d_cpu if has_d else None,
                ratio=(k_cpu / d_cpu) if has_d and d_cpu > 0 else None,
                krishiv_rows_in=k.rows_in.get(key, 0),
                krishiv_rows_out=k.rows_out.get(key, 0),
                duckdb_rows_in=d.rows_in.get(key, 0) if has_d else None,
                duckdb_rows_out=d.rows_out.get(key, 0) if has_d else None,
            )
        )
    return rows


@dataclass
class PipelineRow:
    """One Krishiv pipeline's CPU and how unevenly its drivers shared it."""

    pipeline: int
    drivers: int
    threads: int
    cpu: float
    busiest_driver_cpu: float
    share: float
    wall: float
    operators: list[str]
    flagged: bool


def pipeline_rows(
    doc: dict[str, Any], min_cpu_share: float = MIN_CPU_SHARE, busy_share: float = BUSY_SHARE
) -> tuple[list[PipelineRow], float]:
    """Per-pipeline parallelism of a profile, and the CPU of operators that span several pipelines.

    The share is the sum of the operators' ``max_driver_compute_time_ns`` over their summed ``compute_time_ns``: an
    upper bound of the busiest driver's share of the pipeline.
    """
    by_pipe: dict[int, list[dict[str, Any]]] = defaultdict(list)
    unattributed = 0.0
    for op in doc["operators"]:
        pipes = op.get("pipelines") or []
        if len(pipes) == 1:
            by_pipe[pipes[0]].append(op)
        else:
            unattributed += op["metrics"]["compute_time_ns"] / 1e9
    total = sum(op["metrics"]["compute_time_ns"] for op in doc["operators"]) / 1e9
    rows = []
    for p in sorted(doc.get("pipelines", []), key=lambda p: p["id"]):
        ops = by_pipe.get(p["id"], [])
        cpu = sum(o["metrics"]["compute_time_ns"] for o in ops) / 1e9
        busiest = sum(o["metrics"]["max_driver_compute_time_ns"] for o in ops) / 1e9
        share = min(1.0, busiest / cpu) if cpu > 0 else 0.0
        names = [
            o["name"]
            for o in sorted(ops, key=lambda o: -o["metrics"]["compute_time_ns"])
            if o["metrics"]["compute_time_ns"]
        ]
        rows.append(
            PipelineRow(
                pipeline=p["id"],
                drivers=int(p["drivers"]),
                threads=int(p["threads"]),
                cpu=cpu,
                busiest_driver_cpu=busiest,
                share=share,
                wall=(p["ended_ns"] - p["started_ns"]) / 1e9,
                operators=names[:4],
                flagged=total > 0 and cpu >= min_cpu_share * total and share > busy_share,
            )
        )
    return rows, unattributed


def _ratio(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}x"


def _secs(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _rows(v: int | None) -> str:
    return "n/a" if v is None else f"{v:,}"


def markdown(
    qid: str,
    kinds: list[KindRow],
    pipes: list[PipelineRow],
    unattributed: float,
    k_total: float,
    d_total: float,
    min_cpu_share: float = MIN_CPU_SHARE,
    busy_share: float = BUSY_SHARE,
) -> str:
    """The report of one query."""
    lines = [
        f"## {qid}",
        "",
        f"CPU (summed over threads): Krishiv {k_total:.3f} s, DuckDB {d_total:.3f} s, "
        f"ratio {_ratio(k_total / d_total if d_total > 0 else None)}, gap {k_total - d_total:+.3f} s.",
        "",
        "| kind | Krishiv CPU s | DuckDB CPU s | ratio | gap s | Krishiv rows in | Krishiv rows out "
        "| DuckDB rows in | DuckDB rows out |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    top = max((r for r in kinds if not r.sub and r.duckdb_cpu is not None), key=_gap, default=None)
    for r in kinds:
        name = f"&nbsp;&nbsp;↳ {r.label}" if r.sub else r.label
        gap = None if r.duckdb_cpu is None else r.krishiv_cpu - r.duckdb_cpu
        mark = " ◀" if r is top and gap is not None and gap > 0 else ""
        lines.append(
            f"| {name}{mark} | {_secs(r.krishiv_cpu)} | {_secs(r.duckdb_cpu)} | {_ratio(r.ratio)} "
            f"| {'n/a' if gap is None else f'{gap:+.3f}'} | {_rows(r.krishiv_rows_in)} | {_rows(r.krishiv_rows_out)} "
            f"| {_rows(r.duckdb_rows_in)} | {_rows(r.duckdb_rows_out)} |"
        )
    flagged = [p for p in pipes if p.flagged]
    lines += [
        "",
        f"### Parallelism ({len(flagged)} pipeline(s) flagged: busiest driver over {busy_share:.0%} of the pipeline's "
        f"CPU, pipeline over {min_cpu_share:.0%} of the query's)",
        "",
    ]
    if not pipes:
        lines.append("The profile lists no pipelines.")
    else:
        lines += [
            "| pipeline | drivers | threads | CPU s | busiest driver CPU s (bound) | share | wall s "
            "| main operators | |",
            "|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for p in pipes:
            lines.append(
                f"| {p.pipeline} | {p.drivers} | {p.threads} | {p.cpu:.3f} | {p.busiest_driver_cpu:.3f} "
                f"| {p.share:.0%} | {p.wall:.3f} | {', '.join(p.operators) or '-'} "
                f"| {'**FLAG**' if p.flagged else ''} |"
            )
    if unattributed > 0:
        lines += ["", f"Unattributed (operators spanning several pipelines): {unattributed:.3f} s CPU."]
    return "\n".join(lines) + "\n"


def _gap(r: KindRow) -> float:
    return r.krishiv_cpu - (r.duckdb_cpu or 0.0)


@dataclass
class QueryResult:
    """What one query contributes to the summary and the JSON."""

    query: str
    krishiv_cpu: float | None = None
    duckdb_cpu: float | None = None
    top_gap_kind: str | None = None
    flagged_pipelines: int | None = None
    error: str | None = None
    kinds: list[KindRow] = field(default_factory=list)
    pipelines: list[PipelineRow] = field(default_factory=list)


def analyse(
    qid: str,
    kdoc: dict[str, Any],
    ddoc: dict[str, Any],
    min_cpu_share: float = MIN_CPU_SHARE,
    busy_share: float = BUSY_SHARE,
) -> tuple[QueryResult, str]:
    """The result and Markdown report of one query from the two engines' profile documents."""
    k, d = krishiv_totals(kdoc), duckdb_totals(duckdb_tree(ddoc))
    kinds = compare(k, d)
    pipes, unattributed = pipeline_rows(kdoc, min_cpu_share, busy_share)
    k_total, d_total = k.total(), d.total()
    top = max((r for r in kinds if not r.sub and r.duckdb_cpu is not None), key=_gap, default=None)
    result = QueryResult(
        query=qid,
        krishiv_cpu=k_total,
        duckdb_cpu=d_total,
        top_gap_kind=top.key if top is not None and _gap(top) > 0 else None,
        flagged_pipelines=sum(p.flagged for p in pipes),
        kinds=kinds,
        pipelines=pipes,
    )
    return result, markdown(qid, kinds, pipes, unattributed, k_total, d_total, min_cpu_share, busy_share)


def summary_table(results: list[QueryResult], suite: str, sf: float, threads: int) -> str:
    """One row per query: CPU of both engines, the ratio, the kind that gains the most and the flagged pipelines."""
    lines = [
        f"# gapdiff {suite} SF{sf:g}, {threads} threads",
        "",
        "CPU seconds summed over threads. `top gap` is the kind where Krishiv spends the most CPU beyond DuckDB; "
        "`flags` counts pipelines whose busiest driver does over half of the pipeline's CPU.",
        "",
        "| query | Krishiv CPU s | DuckDB CPU s | ratio | gap s | top gap | flags |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    ok = [r for r in results if r.error is None and r.krishiv_cpu is not None and r.duckdb_cpu]
    for r in results:
        if r.error is not None or r.krishiv_cpu is None or r.duckdb_cpu is None:
            lines.append(f"| {r.query} | - | - | - | - | not compared: {(r.error or '?')[:80]} | - |")
            continue
        ratio = r.krishiv_cpu / r.duckdb_cpu if r.duckdb_cpu > 0 else None
        lines.append(
            f"| {r.query} | {r.krishiv_cpu:.3f} | {r.duckdb_cpu:.3f} | {_ratio(ratio)} "
            f"| {r.krishiv_cpu - r.duckdb_cpu:+.3f} | {r.top_gap_kind or '-'} | {r.flagged_pipelines} |"
        )
    if ok:
        geo = 1.0
        for r in ok:
            geo *= (r.krishiv_cpu or 0.0) / (r.duckdb_cpu or 1.0)
        lines += [
            "",
            f"Geometric mean of the CPU ratios over {len(ok)} queries: {_ratio(geo ** (1 / len(ok)))}.",
        ]
    return "\n".join(lines) + "\n"


def run_gapdiff(
    suite: str,
    sf: float,
    queries: list[str] | None,
    dataset: str | None = None,
    threads: int = 8,
    memory_limit: str = "12g",
    runner: str | None = None,
    out_dir: Path | None = None,
    timeout_s: float | None = 600,
    min_cpu_share: float = MIN_CPU_SHARE,
    busy_share: float = BUSY_SHARE,
    perf: bool = False,
    perf_runs: int = 1,
) -> Path:
    """Profiles ``queries`` (``None``: the whole suite) on both engines and writes the reports; returns the directory.

    A query that fails on either engine is recorded with its error and the run goes on. ``perf`` also samples the
    Krishiv run under ``perf record -g`` and writes folded stacks.
    """
    from .abba import bench_lock
    from .adapters.duckdb_engine import DuckDBEngine
    from .adapters.krishiv import KrishivEngine
    from .dataset import resolve_dataset
    from .paths import results_dir
    from .runner import load_queries

    ds = resolve_dataset(suite, sf, dataset)
    kq = {q.id: q for q in load_queries(suite, "krishiv", queries)}
    dq = {q.id: q for q in load_queries(suite, "duckdb", queries)}
    ids = queries or sorted(kq, key=lambda i: (len(i), i))
    missing = [q for q in ids if q not in kq or q not in dq]
    if missing:
        raise SystemExit(f"unknown queries for {suite}: {', '.join(missing)}")
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_dir or results_dir() / "gapdiff" / f"{stamp}_{suite}_sf{sf:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[QueryResult] = []
    reports: list[str] = []
    with bench_lock():
        krishiv = KrishivEngine(threads=threads, memory_limit=memory_limit.lower(), binary=runner)
        krishiv.timeout_s = timeout_s or None
        duck = DuckDBEngine(threads=threads, memory_limit=memory_limit.upper())
        try:
            duck.setup(ds)
            krishiv.setup(ds)
            for qid in ids:
                try:
                    kp = krishiv.plan_profile(kq[qid])
                    assert duck.con is not None
                    sql = "EXPLAIN (ANALYZE, FORMAT json) " + dq[qid].sql.rstrip().rstrip(";")
                    row = duck.con.execute(sql).fetchone()
                    if row is None:
                        raise ValueError("DuckDB returned no profile")
                    result, report = analyse(qid, kp["profile"], json.loads(row[1]), min_cpu_share, busy_share)
                except Exception as e:  # one query's failure is recorded, the others still run
                    reason = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
                    result = QueryResult(query=qid, error=reason[:400])
                    report = f"## {qid}\n\nNot compared: {reason[:400]}\n"
                results.append(result)
                reports.append(report)
                (out_dir / f"{qid}.md").write_text(report, encoding="utf-8")
                if perf and result.error is None:
                    _sample(ds, kq[qid], runner, perf_runs, threads, memory_limit.lower(), timeout_s or 0, out_dir)
        finally:
            krishiv.teardown()
            duck.teardown()
            summary = summary_table(results, suite, sf, threads) if len(ids) > 1 else "\n".join(reports)
            (out_dir / "summary.md").write_text(summary, encoding="utf-8")
            (out_dir / "gapdiff.json").write_text(
                json.dumps(
                    {
                        "suite": suite,
                        "sf": sf,
                        "threads": threads,
                        "queries": {r.query: asdict(r) for r in results},
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )
    print("\n".join(reports) if len(ids) == 1 else summary)
    print(f"result: {out_dir}")
    return out_dir


def _sample(
    ds: Any, query: Any, runner: str | None, runs: int, threads: int, memory: str, timeout_s: float, out: Path
) -> None:
    """Writes the folded stacks (and the flamegraph when inferno is installed) of the Krishiv run of ``query``."""
    from .flamediff import render_svg, write_folded

    try:
        stacks = sample_with_perf(runner, ds, query, runs, threads, memory, timeout_s)
    except Exception as e:  # perf not permitted, a runner without frame pointers: the profile is still written
        print(f"--perf for {query.id}: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
        return
    if not stacks:
        print(f"--perf for {query.id}: no samples (is perf permitted, and the runner built with frame pointers?)")
        return
    perf_dir = out / "perf"
    perf_dir.mkdir(exist_ok=True)
    write_folded(perf_dir / f"{query.id}.folded", dict(stacks))
    render_svg(perf_dir / f"{query.id}.folded", perf_dir / f"{query.id}.svg", f"Krishiv {query.id}")


def sample_with_perf(runner: str | None, ds: Any, query: Any, runs: int, threads: int, memory: str, timeout_s: float):
    """Folded stacks of ``runs`` warm Krishiv runs of ``query`` under perf (``flamediff``'s sampler)."""
    from .flamediff import sample_side

    return sample_side(runner, ds, query, runs, threads, memory, timeout_s)


__all__ = [
    "analyse",
    "compare",
    "duckdb_totals",
    "krishiv_totals",
    "pipeline_rows",
    "run_gapdiff",
    "summary_table",
]
