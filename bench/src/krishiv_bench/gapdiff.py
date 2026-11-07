"""``krishiv-bench gapdiff``: where Krishiv's time goes, operator kind by operator kind, beside DuckDB's.

For one query (or ``--all``: the whole suite) both engines run with their operator profiles on the same Parquet files:
Krishiv's ``PLAN`` command (the ``krishiv.query-profile`` v1 document) and DuckDB's ``EXPLAIN (ANALYZE, FORMAT json)``
(the same data as ``PRAGMA enable_profiling='json'``). Operators of both are mapped to common kinds and summed:

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
other      everything else (``Result``, ``Limit``, ``Window``, ``Expand``, unions, CTEs ...), and a ``Sort`` that
           sits directly below a ``Window`` (DuckDB's WINDOW sorts inside the operator, so it is ``other`` there)
========== =====================================================================================================

Krishiv also reports the split DuckDB cannot give: hash join *build* vs *probe*, and *partial* vs *final* aggregate.
A ``HashAggregate`` operator is partial when the plan node it implements is named ``Partial...`` and final otherwise.
Where the planner fuses the partial, the exchange and the final into one breaker, the sink does the whole aggregation
under the partial node and the source only emits the groups under the final node, so the split is by where the engine
books the time, not by algorithm. DuckDB's join and aggregate time is one number each, so those sub-rows show Krishiv
only. The operators that fell into ``other`` are listed by name in the report, so a new operator is visible.

**What the seconds are.** Neither engine's operator time is CPU time. Krishiv's ``compute_time_ns`` is the time a
driver spends inside the operator (monotonic clock, IO waits subtracted) summed over drivers; DuckDB's operator timing
is the profiler's per-thread clock, summed over threads and exclusive of children. A descheduled thread still counts,
so both inflate under load (research measured DuckDB q06 SF10 at 22.4 s of operator time against 3.1 s of CPU at load
27). Each engine's *process CPU* over the profiled run (``/proc`` utime+stime of the runner process, or of the harness
for DuckDB, through the adapters' ``begin_measure``/``end_measure``) is therefore measured too, and the per-kind
**CPU s** columns are the operator seconds scaled to it (each kind's share of its engine's operator seconds times the
engine's process CPU). Work done outside any operator call is not in the operator seconds: DuckDB's sort merge and
finalize events (measured: ORDER BY of 20M rows, 4.6 s of CPU against 2.75 s of operator time), Krishiv's output-spill
tasks and IO prefetch. The report prints that remainder as *not attributed* per engine; scaling spreads it over the
kinds in proportion to their operator seconds, which is a guess for the kind that really owns it (it matters most for
``sort``). Without a process-CPU reading for both engines the columns, the ratio and the top gap fall back to operator
seconds and say so. Scan time is not like for like either: DuckDB's includes waiting for reads, Krishiv's excludes it
(its IO wait is printed beside the scan row), so both engines get ``--warmup`` unprofiled runs first (default 1) and the
second engine does not read a page cache the first one warmed. The seconds are profile numbers, not benchmark timings
(use ``abba`` for those). The process-CPU window covers the ``PLAN`` command (planning, execution and writing the
profile JSON, with detailed metrics on) for Krishiv and the ``EXPLAIN ANALYZE`` call for DuckDB, read from ``/proc`` at
clock-tick resolution (10 ms), so a query of a few tens of milliseconds has a coarse CPU figure. Each query prints the
1-minute load average at its start and end and is marked indicative
when it passes the harness bound (``abba.load_bound``).

**Rows.** Krishiv rows in are summed over the operators that consume (not the source half of a breaker) and rows out
over the operators that produce (not the sink half), so a breaker's sink and source do not count its rows twice. The
two aggregate phases and the exchange's write and read are different stages, so a kind's rows are "summed over
operators", not one node's rows as in DuckDB.

**Parallelism.** From the Krishiv profile, per pipeline: the operators that run only in that pipeline (an operator
lists several pipelines only below a ``UNION ALL`` fan-in; the sink and the source half of a breaker are separate
entries), their summed ``compute_time_ns`` (the pipeline's CPU) and the sum of their ``max_driver_compute_time_ns``,
an upper bound of the busiest driver's work (each operator's own busiest driver). Flags, for pipelines with at least
``--min-cpu-share`` (default 2%) of the query's operator seconds:

* ``serial``: the pipeline runs on one driver;
* ``busy``: with more than 2 drivers, the busiest-driver bound is over ``--busy-share`` (default 50%) of the CPU
  (with 2 drivers the share is at least 50% by construction, so it says nothing);
* ``skew``: more than one driver, the bound is at least 80% of the pipeline's wall time and the parallel efficiency
  (CPU over drivers times wall) is under 50%: the wall time is one driver's chain while the others idle, the shape of
  a serial merge inside a heavy parallel pipeline (a share of the pipeline's CPU hides it).

A second table lists operators inside multi-driver pipelines that ran on a single driver, or whose busiest driver did
twice its fair share (``max * drivers / cpu``), the single-driver merges behind q13/q16/q18.

Output in ``$KRISHIV_DATA/results/gapdiff/<ts>_<suite>_sf<sf>/``: ``summary.md`` (one row per query with ``--all``,
else the query's own report), ``gapdiff.json``, one ``<query>.md`` per query, and with ``--perf`` the Krishiv run's
folded stacks ``perf/<query>.folded`` (and ``.svg`` when ``inferno-flamegraph`` is installed). Folded files of two
runner builds go straight into ``inferno-diff-folded`` (or use ``flamediff``, which runs both).

Not here yet (recorded on the queue item): hardware counters per kind (``perf stat`` events, IPC, instructions per
row), the module split per kind from perf's leaf frames, CPU split by kind from perf samples (not scaled operator
seconds), blocked time per pipeline, and a real multi-driver profile as a fixture for the ``busy`` and ``skew`` flags.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .explain_diff import PNode, duckdb_tree

# Pipelines with at least this share of the query's operator seconds are looked at.
MIN_CPU_SHARE = 0.02
# `busy`: the busiest driver's share of the pipeline's CPU (needs more than 2 drivers).
BUSY_SHARE = 0.5
# `skew`: the bound on the busiest driver's time is at least this share of the wall time, with efficiency below...
SKEW_WALL = 0.8
SKEW_EFFICIENCY = 0.5
# An operator whose busiest driver did this many times its fair share (and ran on at least OPERATOR_DRIVERS drivers).
OPERATOR_SKEW = 2.0
OPERATOR_DRIVERS = 4

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
    ("join", "join (total)", False),
    ("join_build", "build", True),
    ("join_probe", "probe", True),
    ("join_other", "other joins", True),
    ("aggregate", "aggregate (total)", False),
    ("agg_partial", "partial", True),
    ("agg_final", "final (emit, if fused)", True),
    ("sort", "sort / top-n", False),
    ("exchange", "exchange", False),
    ("other", "other", False),
]
# Kinds DuckDB reports as one number: Krishiv's finer rows have no DuckDB side.
DUCKDB_HAS = {"scan", "filter", "project", "join", "aggregate", "sort", "exchange", "other"}


@dataclass
class Totals:
    """Operator seconds and rows by kind for one engine."""

    secs: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    rows_in: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    rows_out: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    io_wait: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    other_names: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, kind: str, secs: float, rows_in: int, rows_out: int, io_wait: float = 0.0) -> None:
        for key in (kind, PARENT.get(kind)):
            if key is not None:
                self.secs[key] += secs
                self.rows_in[key] += rows_in
                self.rows_out[key] += rows_out
                self.io_wait[key] += io_wait

    def total(self) -> float:
        """The operator seconds of the query: every top-level kind (a sub-row is inside its parent)."""
        return sum(v for k, v in self.secs.items() if k not in PARENT)


def _node_names(doc: dict[str, Any]) -> dict[int, str]:
    """Plan node id -> its name (``PartialHashAggregate``: the operator name alone, ``HashAggregate``, does not say)."""
    return {n["id"]: n.get("name", "") for n in doc.get("plan") or [] if n.get("id") is not None}


def _parent_names(doc: dict[str, Any]) -> dict[int, str]:
    """Plan node id -> the name of its parent node (the profile lists the plan in pre-order, children by position)."""
    plan = doc.get("plan") or []
    out: dict[int, str] = {}
    for node in plan:
        for c in node.get("children", []):
            if 0 <= c < len(plan) and plan[c].get("id") is not None:
                out[plan[c]["id"]] = node.get("name", "")
    return out


def krishiv_totals(doc: dict[str, Any]) -> Totals:
    """Operator seconds and rows by kind from the operators of a ``krishiv.query-profile`` v1 document."""
    if doc.get("format") != "krishiv.query-profile" or doc.get("version") != 1:
        raise ValueError("not a krishiv.query-profile v1 document")
    parents, nodes = _parent_names(doc), _node_names(doc)
    out = Totals()
    for op in doc["operators"]:
        m = op["metrics"]
        kind = KRISHIV_KINDS.get(op["name"], "other")
        if kind == "agg_final" and "partial" in nodes.get(op.get("node"), "").lower():
            kind = "agg_partial"  # a partial aggregate runs as a `HashAggregate` operator under a `Partial...` node
        if kind == "sort" and "window" in parents.get(op.get("node"), "").lower():
            kind = "other"  # the sort DuckDB's WINDOW does inside itself
        secs = m["compute_time_ns"] / 1e9
        if kind == "other":
            out.other_names[op["name"]] += secs
        # A breaker's sink and source are separate entries: a row is consumed by the sink and produced by the source.
        out.add(
            kind,
            secs,
            0 if op["role"] == "source" else int(m["rows_in"]),
            0 if op["role"] == "sink" else int(m["rows_out"]),
            m.get("blocked_on_io_ns", 0) / 1e9,
        )
    return out


def duckdb_totals(tree: PNode) -> Totals:
    """Operator seconds and rows by kind from a DuckDB plan tree (rows in: what the operator's children produced)."""
    out = Totals()
    for n in tree.walk():
        kind = DUCKDB_KINDS.get(n.kind, n.kind)
        if kind not in DUCKDB_HAS:
            kind = "other"
        if kind == "other":
            out.other_names[n.name] += n.seconds
        out.add(kind, n.seconds, sum(c.rows or 0 for c in n.children), n.rows or 0)
    return out


def scaled(t: Totals, process_cpu: float | None) -> dict[str, float] | None:
    """Each kind's share of the operator seconds times the process CPU; ``None`` without a CPU reading."""
    total = t.total()
    if process_cpu is None or total <= 0:
        return None
    return {k: v / total * process_cpu for k, v in t.secs.items()}


@dataclass
class KindRow:
    """One line of the side-by-side table."""

    key: str
    label: str
    sub: bool
    krishiv_secs: float
    duckdb_secs: float | None
    krishiv_cpu: float | None
    duckdb_cpu: float | None
    ratio: float | None
    gap: float | None
    krishiv_rows_in: int
    krishiv_rows_out: int
    duckdb_rows_in: int | None
    duckdb_rows_out: int | None
    krishiv_io_wait: float = 0.0


def compare(k: Totals, d: Totals, k_cpu: float | None = None, d_cpu: float | None = None) -> list[KindRow]:
    """The table rows: every kind either engine spent time in, in report order.

    The ratio and gap use the process-CPU-scaled seconds when both engines have a reading, else operator seconds.
    """
    ks, ds = scaled(k, k_cpu), scaled(d, d_cpu)
    use_cpu = ks is not None and ds is not None
    rows = []
    for key, label, sub in ROWS:
        k_s, d_s = k.secs.get(key, 0.0), d.secs.get(key, 0.0)
        if k_s == 0 and d_s == 0 and k.rows_out.get(key, 0) == 0 and d.rows_out.get(key, 0) == 0:
            continue
        has_d = key in DUCKDB_HAS  # DuckDB does not split joins and aggregates into their parts
        kc = ks.get(key, 0.0) if ks is not None else None
        dc = (ds.get(key, 0.0) if ds is not None else None) if has_d else None
        a, b = (kc, dc) if use_cpu else (k_s, d_s if has_d else None)
        rows.append(
            KindRow(
                key=key,
                label=label,
                sub=sub,
                krishiv_secs=k_s,
                duckdb_secs=d_s if has_d else None,
                krishiv_cpu=kc,
                duckdb_cpu=dc,
                ratio=(a / b) if a is not None and b else None,
                gap=(a - b) if a is not None and b is not None else None,
                krishiv_rows_in=k.rows_in.get(key, 0),
                krishiv_rows_out=k.rows_out.get(key, 0),
                duckdb_rows_in=d.rows_in.get(key, 0) if has_d else None,
                duckdb_rows_out=d.rows_out.get(key, 0) if has_d else None,
                krishiv_io_wait=k.io_wait.get(key, 0.0),
            )
        )
    return rows


@dataclass
class PipelineRow:
    """One Krishiv pipeline's time and how unevenly its drivers shared it."""

    pipeline: int
    drivers: int
    threads: int
    secs: float
    busiest_driver_secs: float
    share: float
    wall: float
    busiest_wall: float
    efficiency: float
    operators: list[str]
    flags: list[str]


@dataclass
class OperatorRow:
    """An operator of a multi-driver pipeline that ran on one driver, or whose busiest driver did far too much."""

    pipeline: int
    name: str
    role: str
    drivers: int
    secs: float
    busiest_driver_secs: float
    skew: float
    flag: str


def parallelism(
    doc: dict[str, Any], min_cpu_share: float = MIN_CPU_SHARE, busy_share: float = BUSY_SHARE
) -> tuple[list[PipelineRow], list[OperatorRow], float]:
    """Per-pipeline and per-operator parallelism of a profile, and the seconds of operators that span pipelines."""
    by_pipe: dict[int, list[dict[str, Any]]] = defaultdict(list)
    unattributed = 0.0
    for op in doc["operators"]:
        pipes = op.get("pipelines") or []
        if len(pipes) == 1:
            by_pipe[pipes[0]].append(op)
        else:
            unattributed += op["metrics"]["compute_time_ns"] / 1e9
    total = sum(op["metrics"]["compute_time_ns"] for op in doc["operators"]) / 1e9
    prows: list[PipelineRow] = []
    orows: list[OperatorRow] = []
    for p in sorted(doc.get("pipelines", []), key=lambda p: p["id"]):
        ops = by_pipe.get(p["id"], [])
        secs = sum(o["metrics"]["compute_time_ns"] for o in ops) / 1e9
        busiest = sum(o["metrics"]["max_driver_compute_time_ns"] for o in ops) / 1e9
        share = min(1.0, busiest / secs) if secs > 0 else 0.0
        drivers = int(p["drivers"])
        wall = (p["ended_ns"] - p["started_ns"]) / 1e9
        busiest_wall = min(1.0, busiest / wall) if wall > 0 else 0.0
        efficiency = min(1.0, secs / (drivers * wall)) if drivers > 0 and wall > 0 else 0.0
        names = sorted(ops, key=lambda o: -o["metrics"]["compute_time_ns"])
        material = total > 0 and secs >= min_cpu_share * total
        flags = []
        if material and drivers == 1:
            flags.append("serial")
        if material and drivers > 2 and share > busy_share:
            flags.append("busy")
        if material and drivers > 1 and busiest_wall >= SKEW_WALL and efficiency < SKEW_EFFICIENCY:
            flags.append("skew")
        prows.append(
            PipelineRow(
                pipeline=p["id"],
                drivers=drivers,
                threads=int(p["threads"]),
                secs=secs,
                busiest_driver_secs=busiest,
                share=share,
                wall=wall,
                busiest_wall=busiest_wall,
                efficiency=efficiency,
                operators=[o["name"] for o in names if o["metrics"]["compute_time_ns"]][:4],
                flags=flags,
            )
        )
        if drivers > 1:
            for o in names:
                m = o["metrics"]
                op_secs = m["compute_time_ns"] / 1e9
                if total <= 0 or op_secs < min_cpu_share * total:
                    continue
                op_drivers = int(m["drivers"]) if "drivers" in m else drivers
                skew = (m["max_driver_compute_time_ns"] / 1e9) * op_drivers / op_secs if op_secs > 0 else 0.0
                flag = ""
                if op_drivers == 1:
                    flag = "serial operator in a parallel pipeline"
                elif op_drivers >= OPERATOR_DRIVERS and skew >= OPERATOR_SKEW:
                    flag = "skewed"
                if flag:
                    orows.append(
                        OperatorRow(
                            pipeline=p["id"],
                            name=o["name"],
                            role=o["role"],
                            drivers=op_drivers,
                            secs=op_secs,
                            busiest_driver_secs=m["max_driver_compute_time_ns"] / 1e9,
                            skew=skew,
                            flag=flag,
                        )
                    )
    return prows, orows, unattributed


def _ratio(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}x"


def _secs(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _rows(v: int | None) -> str:
    return "n/a" if v is None else f"{v:,}"


def _gap(r: KindRow) -> float:
    return r.gap if r.gap is not None else float("-inf")


@dataclass
class Basis:
    """What was measured around one engine's profiled run."""

    process_cpu: float | None
    operator_secs: float
    load_start: float | None = None
    load_end: float | None = None
    load_bound: float | None = None

    @property
    def unattributed(self) -> float | None:
        return None if self.process_cpu is None else self.process_cpu - self.operator_secs

    @property
    def load_known(self) -> bool:
        """Whether any load reading was taken (``/proc/loadavg`` can be unreadable)."""
        return any(x is not None for x in (self.load_start, self.load_end))

    @property
    def indicative(self) -> bool:
        loads = [x for x in (self.load_start, self.load_end) if x is not None]
        return self.load_bound is not None and bool(loads) and max(loads) > self.load_bound


def _basis_line(name: str, b: Basis) -> str:
    cpu = "no process CPU reading" if b.process_cpu is None else f"process CPU {b.process_cpu:.3f} s"
    rest = "" if b.unattributed is None else f", not attributed to any operator {b.unattributed:+.3f} s"
    if b.unattributed is not None and b.unattributed < 0:
        rest += " (negative: threads waited for a CPU, so operator seconds exceed CPU; the load inflates them)"
    return f"{name}: {cpu}, operator seconds {b.operator_secs:.3f} s{rest}"


def markdown(
    qid: str,
    kinds: list[KindRow],
    pipes: list[PipelineRow],
    ops: list[OperatorRow],
    unattributed: float,
    k: Basis,
    d: Basis,
    other: dict[str, tuple[float, float]] | None = None,
    min_cpu_share: float = MIN_CPU_SHARE,
    busy_share: float = BUSY_SHARE,
) -> str:
    """The report of one query."""
    use_cpu = k.process_cpu is not None and d.process_cpu is not None
    k_total, d_total = (k.process_cpu, d.process_cpu) if use_cpu else (k.operator_secs, d.operator_secs)
    ratio = (k_total or 0.0) / d_total if d_total else None
    unit = "process CPU" if use_cpu else "operator seconds (no process CPU reading for both engines)"
    lines = [
        f"## {qid}",
        "",
        f"{unit[0].upper()}{unit[1:]}: Krishiv {_secs(k_total)} s, DuckDB {_secs(d_total)} s, ratio {_ratio(ratio)}, "
        f"gap {(k_total or 0.0) - (d_total or 0.0):+.3f} s.",
        "",
        "- " + _basis_line("Krishiv", k),
        "- " + _basis_line("DuckDB", d),
    ]
    for name, b in (("Krishiv", k), ("DuckDB", d)):
        if b.load_bound is None:
            continue
        if not b.load_known:
            lines.append(f"- {name} load average: **unknown** (/proc/loadavg was not readable), bound {b.load_bound:g}")
            continue
        start = "n/a" if b.load_start is None else f"{b.load_start:.1f}"
        end = "n/a" if b.load_end is None else f"{b.load_end:.1f}"
        lines.append(
            f"- {name} load average (1 min) {start} at start, {end} at end, bound {b.load_bound:g}: "
            + ("**INDICATIVE ONLY**" if b.indicative else "within bound")
        )
    cpu_note = (
        "CPU s = the kind's share of the engine's operator seconds times its process CPU"
        if use_cpu
        else "CPU s columns need a process CPU reading for both engines; ratio and gap use operator seconds"
    )
    lines += [
        "",
        f"Operator s = time inside the operator, summed over threads (wall time per thread, not CPU). {cpu_note}.",
        "",
        "| kind | Krishiv op s | DuckDB op s | Krishiv CPU s | DuckDB CPU s | ratio | gap s "
        "| Krishiv rows in | Krishiv rows out | DuckDB rows in | DuckDB rows out |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    top = max((r for r in kinds if not r.sub and r.gap is not None), key=_gap, default=None)
    for r in kinds:
        name = f"&nbsp;&nbsp;↳ {r.label}" if r.sub else r.label
        mark = " ◀" if r is top and r.gap is not None and r.gap > 0 else ""
        lines.append(
            f"| {name}{mark} | {_secs(r.krishiv_secs)} | {_secs(r.duckdb_secs)} | {_secs(r.krishiv_cpu)} "
            f"| {_secs(r.duckdb_cpu)} | {_ratio(r.ratio)} | {'n/a' if r.gap is None else f'{r.gap:+.3f}'} "
            f"| {_rows(r.krishiv_rows_in)} | {_rows(r.krishiv_rows_out)} "
            f"| {_rows(r.duckdb_rows_in)} | {_rows(r.duckdb_rows_out)} |"
        )
    scan = next((r for r in kinds if r.key == "scan"), None)
    if scan is not None and scan.krishiv_io_wait > 0:
        lines += [
            "",
            f"Krishiv's scan seconds exclude {scan.krishiv_io_wait:.3f} s of IO wait; DuckDB's scan time includes its "
            "reads.",
        ]
    if other:
        named = sorted(other.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:8]
        lines += [
            "",
            "In `other`: " + ", ".join(f"{n} (Krishiv {a:.3f} s, DuckDB {b:.3f} s)" for n, (a, b) in named) + ".",
        ]
    flagged = [p for p in pipes if p.flags]
    lines += [
        "",
        f"### Parallelism ({len(flagged)} pipeline(s) flagged, {len(ops)} operator(s) flagged)",
        "",
        f"Pipelines with at least {min_cpu_share:.0%} of the query's operator seconds are flagged `serial` (one "
        f"driver), `busy` (over 2 drivers, busiest driver over {busy_share:.0%} of the CPU) or `skew` (the busiest "
        f"driver fills the wall time while parallel efficiency is under {SKEW_EFFICIENCY:.0%}). The busiest-driver "
        "figure is an upper bound (the sum of each operator's own busiest driver).",
        "",
    ]
    if not pipes:
        lines.append("The profile lists no pipelines.")
    else:
        lines += [
            "| pipeline | drivers | threads | op s | busiest driver s (bound) | share | wall s | busiest / wall "
            "| efficiency | main operators | flags |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for p in pipes:
            lines.append(
                f"| {p.pipeline} | {p.drivers} | {p.threads} | {p.secs:.3f} | {p.busiest_driver_secs:.3f} "
                f"| {p.share:.0%} | {p.wall:.3f} | {p.busiest_wall:.0%} | {p.efficiency:.0%} "
                f"| {', '.join(p.operators) or '-'} | {', '.join(f'**{f}**' for f in p.flags)} |"
            )
    if ops:
        lines += [
            "",
            "| pipeline | operator | role | drivers | op s | busiest driver s | skew | |",
            "|---:|---|---|---:|---:|---:|---:|---|",
        ]
        for o in ops:
            lines.append(
                f"| {o.pipeline} | {o.name} | {o.role} | {o.drivers} | {o.secs:.3f} | {o.busiest_driver_secs:.3f} "
                f"| {o.skew:.1f}x | **{o.flag}** |"
            )
    if unattributed > 0:
        lines += ["", f"Unattributed (operators spanning several pipelines): {unattributed:.3f} s."]
    return "\n".join(lines) + "\n"


@dataclass
class QueryResult:
    """What one query contributes to the summary and the JSON."""

    query: str
    krishiv_cpu: float | None = None
    duckdb_cpu: float | None = None
    basis: str | None = None  # "cpu" (process CPU of both engines) or "operator seconds"
    top_gap_kind: str | None = None
    flagged_pipelines: int | None = None
    indicative: bool = False
    load_known: bool = True
    error: str | None = None
    krishiv: Basis | None = None
    duckdb: Basis | None = None
    kinds: list[KindRow] = field(default_factory=list)
    pipelines: list[PipelineRow] = field(default_factory=list)
    operators: list[OperatorRow] = field(default_factory=list)


def analyse(
    qid: str,
    kdoc: dict[str, Any],
    ddoc: dict[str, Any],
    k_basis: Basis | None = None,
    d_basis: Basis | None = None,
    min_cpu_share: float = MIN_CPU_SHARE,
    busy_share: float = BUSY_SHARE,
) -> tuple[QueryResult, str]:
    """The result and Markdown report of one query from the two engines' profile documents and measurements."""
    k, d = krishiv_totals(kdoc), duckdb_totals(duckdb_tree(ddoc))
    kb = k_basis or Basis(None, k.total())
    db = d_basis or Basis(None, d.total())
    kb.operator_secs, db.operator_secs = k.total(), d.total()
    kinds = compare(k, d, kb.process_cpu, db.process_cpu)
    prows, orows, unattributed = parallelism(kdoc, min_cpu_share, busy_share)
    use_cpu = kb.process_cpu is not None and db.process_cpu is not None
    top = max((r for r in kinds if not r.sub and r.gap is not None), key=_gap, default=None)
    names = sorted(k.other_names.keys() | d.other_names.keys())
    other = {n: (k.other_names.get(n, 0.0), d.other_names.get(n, 0.0)) for n in names}
    result = QueryResult(
        query=qid,
        krishiv_cpu=kb.process_cpu if use_cpu else kb.operator_secs,
        duckdb_cpu=db.process_cpu if use_cpu else db.operator_secs,
        basis="cpu" if use_cpu else "operator seconds",
        top_gap_kind=top.key if top is not None and top.gap is not None and top.gap > 0 else None,
        flagged_pipelines=len({p.pipeline for p in prows if p.flags} | {o.pipeline for o in orows}),
        indicative=kb.indicative or db.indicative,
        load_known=kb.load_known and db.load_known,
        krishiv=kb,
        duckdb=db,
        kinds=kinds,
        pipelines=prows,
        operators=orows,
    )
    report = markdown(qid, kinds, prows, orows, unattributed, kb, db, other, min_cpu_share, busy_share)
    return result, report


def summary_table(results: list[QueryResult], suite: str, sf: float, threads: int) -> str:
    """One row per query: both engines' time, the ratio, the kind that gains the most and the flagged pipelines."""
    lines = [
        f"# gapdiff {suite} SF{sf:g}, {threads} threads",
        "",
        "Process CPU seconds of the profiled run for each engine (`basis` says when a query has operator seconds "
        "instead, which are wall time per thread summed over threads and inflate under load). `top gap` is the "
        "kind where Krishiv spends the most beyond DuckDB; `flags` counts pipelines with a parallelism flag; "
        "`load` marks a run that passed the harness's load bound (or whose load could not be read).",
        "",
        "| query | Krishiv s | DuckDB s | ratio | gap s | basis | top gap | flags | load |",
        "|---|---:|---:|---:|---:|---|---|---:|---|",
    ]
    ok = [r for r in results if r.error is None and r.krishiv_cpu is not None and r.duckdb_cpu]
    for r in results:
        if r.error is not None or r.krishiv_cpu is None or r.duckdb_cpu is None:
            lines.append(f"| {r.query} | - | - | - | - | - | not compared: {(r.error or '?')[:80]} | - | - |")
            continue
        ratio = r.krishiv_cpu / r.duckdb_cpu if r.duckdb_cpu > 0 else None
        lines.append(
            f"| {r.query} | {r.krishiv_cpu:.3f} | {r.duckdb_cpu:.3f} | {_ratio(ratio)} "
            f"| {r.krishiv_cpu - r.duckdb_cpu:+.3f} | {r.basis} | {r.top_gap_kind or '-'} | {r.flagged_pipelines} "
            f"| {'INDICATIVE' if r.indicative else 'ok' if r.load_known else 'load unknown'} |"
        )
    for basis, what in (("cpu", "process CPU"), ("operator seconds", "operator seconds")):
        mine = [r for r in ok if r.basis == basis]
        if not mine:
            continue
        geo = 1.0
        for r in mine:
            geo *= (r.krishiv_cpu or 0.0) / (r.duckdb_cpu or 1.0)
        lines += [
            "",
            f"Geometric mean of the ratios over {len(mine)} queries on {what}: {_ratio(geo ** (1 / len(mine)))}."
            + (" Not comparable with the process CPU ratios." if basis != "cpu" else ""),
        ]
    return "\n".join(lines) + "\n"


def _load(v: float) -> float | None:
    """A load reading, ``None`` when it could not be taken (``loadavg`` returns NaN; NaN is not valid JSON)."""
    return None if v != v else v


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
    warmup: int = 1,
) -> Path:
    """Profiles ``queries`` (``None``: the whole suite) on both engines and writes the reports; returns the directory.

    A query that fails on either engine is recorded with its error and the run goes on. ``warmup`` unprofiled runs of
    each engine come first. ``perf`` samples the Krishiv run of every compared query under ``perf record -g`` once
    both profiling engines are shut down, and writes folded stacks.
    """
    from .abba import bench_lock, load_bound, loadavg
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
    bound = load_bound({"threads": threads})
    results: list[QueryResult] = []
    reports: list[str] = []
    sampled: list[str] = []
    with bench_lock():
        krishiv = KrishivEngine(threads=threads, memory_limit=memory_limit.lower(), binary=runner)
        krishiv.timeout_s = timeout_s or None
        duck = DuckDBEngine(threads=threads, memory_limit=memory_limit.upper())
        try:
            duck.setup(ds)
            krishiv.setup(ds)
            for qid in ids:
                try:
                    if duck.con is None:
                        raise RuntimeError("DuckDB is not set up")
                    sql = "EXPLAIN (ANALYZE, FORMAT json) " + dq[qid].sql.rstrip().rstrip(";")
                    for _ in range(warmup):  # both engines read warm files, whichever runs first
                        krishiv.plan_profile(kq[qid])
                        duck.con.execute(sql).fetchone()
                    load0 = _load(loadavg())
                    krishiv.begin_measure()
                    kp = krishiv.plan_profile(kq[qid])
                    k_meas = krishiv.end_measure()
                    k_load = _load(loadavg())
                    duck.begin_measure()
                    row = duck.con.execute(sql).fetchone()
                    d_meas = duck.end_measure()
                    if row is None:
                        raise ValueError("DuckDB returned no profile")
                    d_load = _load(loadavg())
                    result, report = analyse(
                        qid,
                        kp["profile"],
                        json.loads(row[1]),
                        Basis(k_meas.cpu_seconds, 0.0, load0, k_load, bound),
                        Basis(d_meas.cpu_seconds, 0.0, k_load, d_load, bound),
                        min_cpu_share,
                        busy_share,
                    )
                except Exception as e:  # one query's failure is recorded, the others still run
                    reason = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
                    result = QueryResult(query=qid, error=reason[:400])
                    report = f"## {qid}\n\nNot compared: {reason[:400]}\n"
                results.append(result)
                reports.append(report)
                (out_dir / f"{qid}.md").write_text(report, encoding="utf-8")
                if perf and result.error is None:
                    sampled.append(qid)
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
                        "load_bound": bound,
                        "queries": {r.query: asdict(r) for r in results},
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )
        # The profiling runner and DuckDB are shut down: the sampler's own runner has the box to itself.
        for qid in sampled:
            _sample(ds, kq[qid], runner, perf_runs, threads, memory_limit.lower(), timeout_s or 0, out_dir)
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
    "parallelism",
    "run_gapdiff",
    "scaled",
    "summary_table",
]
