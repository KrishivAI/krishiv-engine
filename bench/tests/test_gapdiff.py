"""gapdiff: the kind mapping of both engines' profiles, the process-CPU scaling, the parallelism flags, the summary
format (a golden) and the end-to-end command."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from krishiv_bench import gapdiff as gd

HERE = Path(__file__).parent
GOLDEN = HERE / "golden" / "gapdiff_summary.md"
# A profile captured from the runner (`PLAN`) over 200,000 fact rows joined to a 1,000-row dimension, filtered,
# grouped, ordered and limited: real operator names, roles, pipelines and plan nodes.
REAL = HERE / "fixtures" / "gapdiff_krishiv_profile.json"


def _op(oid, node, name, role, pipelines, rows_in, rows_out, secs, busiest=None, drivers=None, io=0.0):
    """A profile operator; times are in seconds here and nanoseconds in the document."""
    m = {
        "rows_in": rows_in,
        "rows_out": rows_out,
        "compute_time_ns": round(secs * 1e9),
        "max_driver_compute_time_ns": round((secs if busiest is None else busiest) * 1e9),
        "blocked_on_io_ns": round(io * 1e9),
    }
    if drivers is not None:
        m["drivers"] = drivers
    return {"id": oid, "node": node, "name": name, "role": role, "pipelines": pipelines, "metrics": m}


def _pipe(pid, drivers, threads, start, end):
    return {
        "id": pid,
        "drivers": drivers,
        "threads": threads,
        "started_ns": round(start * 1e9),
        "ended_ns": round(end * 1e9),
    }


def _plan(*nodes):
    """Pre-order plan nodes ``(name, children positions)``; ids are the positions."""
    return [{"id": i, "name": n, "detail": "", "children": list(c)} for i, (n, c) in enumerate(nodes)]


def _doc(ops, pipelines, plan=()):
    return {
        "format": "krishiv.query-profile",
        "version": 1,
        "plan": list(plan),
        "operators": ops,
        "pipelines": pipelines,
    }


def real_doc():
    return json.loads(REAL.read_text(encoding="utf-8"))


# A TPC-H q3 shaped query with the roles the engine emits: a breaker is a sink in the pipeline that feeds it and a
# source in the one that reads it. Pipeline 0 reads lineitem on 8 balanced drivers; 1 builds the customer side on 4;
# 2 is the final aggregate, single driver; 3 sorts and returns on one driver; 5 reads on 12 drivers and then runs a
# 3 s serial finalize merge inside the same pipeline; 6 is a balanced 2-driver pipeline.
K_PLAN = _plan(
    ("TopN", [1]),  # 0
    ("HashAggregate", [2]),  # 1
    ("PartialHashAggregate", [3]),  # 2
    ("HashJoin", [4, 5]),  # 3
    ("Scan", []),  # 4
    ("Scan", []),  # 5
    ("Window", [7]),  # 6
    ("Sort", []),  # 7
)
K_OPS = [
    _op(0, 4, "FileScan", "source", [0], 0, 6_000_000, 8.0, 1.2, 8, io=0.4),
    _op(1, 3, "Filter", "operator", [0], 6_000_000, 3_000_000, 1.0, 0.15, 8),
    _op(2, 3, "HashJoinProbe", "operator", [0], 3_000_000, 30_000, 3.0, 0.4, 8),
    _op(3, 2, "HashAggregate", "sink", [0], 30_000, 0, 2.0, 0.3, 8),
    _op(4, 5, "FileScan", "source", [1], 0, 150_000, 0.5, 0.1, 4),
    _op(5, 3, "HashJoinBuild", "sink", [1], 150_000, 0, 0.2, 0.05, 4),
    _op(6, 1, "HashAggregate", "source", [2], 0, 10_000, 2.0, 2.0, 1),
    _op(7, 0, "TopN", "sink", [2], 10_000, 0, 1.0, 1.0, 1),
    _op(8, 0, "TopN", "source", [3], 0, 10, 0.01, 0.01, 1),
    _op(9, None, "Result", "sink", [3], 10, 0, 0.01, 0.01, 1),
    _op(10, 4, "FileScan", "source", [5], 0, 1_000_000, 10.0, 0.9, 12),
    _op(11, 1, "HashAggregate", "sink", [5], 1_000_000, 0, 3.0, 3.0, 1),
    _op(12, 7, "Sort", "sink", [6], 100_000, 0, 1.0, 0.55, 2),
    _op(13, 4, "ExchangeRead", "source", [1, 2], 0, 10_000, 0.5, 0.5, 1),
]
K_PIPES = [
    _pipe(0, 8, 8, 0.0, 2.0),
    _pipe(1, 4, 4, 0.0, 0.3),
    _pipe(2, 1, 1, 2.0, 5.0),
    _pipe(3, 1, 1, 5.0, 5.01),
    _pipe(5, 12, 12, 0.0, 3.9),
    _pipe(6, 2, 2, 0.0, 0.6),
]


def _d(t, rows, seconds, children=(), **extra):
    return {
        "type": t,
        "intermediate_rows": rows,
        "timing": seconds,
        "extra_info": dict(extra),
        "children": list(children),
    }


D_DOC = {
    "operator": [
        _d(
            "RESULT_COLLECTOR",
            10,
            0.0,
            [
                _d(
                    "TOP_N",
                    10,
                    0.2,
                    [
                        _d(
                            "HASH_GROUP_BY",
                            10_000,
                            1.5,
                            [
                                _d(
                                    "PROJECTION",
                                    30_000,
                                    0.1,
                                    [
                                        _d(
                                            "HASH_JOIN",
                                            30_000,
                                            2.0,
                                            [
                                                _d(
                                                    "FILTER",
                                                    3_000_000,
                                                    0.5,
                                                    [_d("TABLE_SCAN", 6_000_000, 4.0, **{"Table": "lineitem"})],
                                                ),
                                                _d("TABLE_SCAN", 150_000, 0.3, **{"Table": "customer"}),
                                            ],
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                )
            ],
        )
    ]
}


def test_a_real_krishiv_profile_maps_by_operator_role_and_plan_node():
    t = gd.krishiv_totals(real_doc())
    # Two FileScans; the Filter; the join's two halves; the Project operators.
    assert t.secs["scan"] == pytest.approx((276_820 + 10_020_755) / 1e9)
    assert t.secs["filter"] == pytest.approx(4_667_731 / 1e9)
    assert t.secs["join_build"] == pytest.approx(731_015 / 1e9)
    assert t.secs["join_probe"] == pytest.approx(9_926_905 / 1e9)
    assert t.secs["project"] == pytest.approx((48_498 + 1_798) / 1e9)
    # The sink under the plan node `PartialHashAggregate` is a partial aggregate although the operator is named
    # `HashAggregate`; the source under `HashAggregate` is the final one.
    assert t.secs["agg_partial"] == pytest.approx(6_629_583 / 1e9)
    assert t.secs["agg_final"] == pytest.approx(48_379 / 1e9)
    assert t.secs["sort"] == pytest.approx((60_610 + 115_962 + 5_982 + 2_214) / 1e9)  # four TopN halves
    # Every operator of this plan has a kind of its own except the query result.
    assert dict(t.other_names) == {"Result": pytest.approx(2_095 / 1e9)}
    # This runner fused the exchanges into the breakers: no Exchange operator reports time.
    assert "exchange" not in t.secs


def test_rows_are_counted_where_they_are_consumed_and_produced():
    t = gd.krishiv_totals(real_doc())
    # TopN: the sinks consume 10 + 35 rows, the sources produce 10 + 10; counting both halves of each would double.
    assert t.rows_in["sort"] == 10 + 35
    assert t.rows_out["sort"] == 10 + 10
    # Scans produce 1,000 + 200,000 rows and consume nothing.
    assert t.rows_out["scan"] == 201_000
    assert t.rows_in["scan"] == 0
    # The join's build consumes the dimension, its probe the filtered fact rows.
    assert t.rows_in["join"] == 1_000 + 66_667
    assert t.rows_out["join"] == 66_667


def test_a_foreign_document_is_refused_and_an_unknown_operator_is_other():
    doc = _doc([_op(0, 0, "my_udf", "operator", [0], 1, 1, 0.5)], [])
    t = gd.krishiv_totals(doc)
    assert t.secs["other"] == pytest.approx(0.5)
    assert dict(t.other_names) == {"my_udf": pytest.approx(0.5)}
    with pytest.raises(ValueError, match="query-profile"):
        gd.krishiv_totals({"format": "something-else", "version": 1, "operators": []})


def test_a_sort_directly_below_a_window_is_other():
    t = gd.krishiv_totals(_doc(K_OPS, K_PIPES, K_PLAN))
    # Operator 12 is a Sort on node 7, whose parent is the Window node: DuckDB sorts inside WINDOW.
    assert t.other_names.get("Sort") == pytest.approx(1.0)
    assert t.secs["sort"] == pytest.approx(1.0 + 0.01)  # TopN sink + source only


def test_krishiv_totals_roll_up_and_count_each_operator_once():
    t = gd.krishiv_totals(_doc(K_OPS, K_PIPES, K_PLAN))
    assert t.secs["scan"] == pytest.approx(8.0 + 0.5 + 10.0)
    assert t.secs["join"] == pytest.approx(3.0 + 0.2)
    assert t.secs["aggregate"] == pytest.approx(2.0 + 2.0 + 3.0)
    assert t.total() == pytest.approx(sum(o["metrics"]["compute_time_ns"] for o in K_OPS) / 1e9)
    assert t.io_wait["scan"] == pytest.approx(0.4)


def test_duckdb_operators_map_to_the_same_kinds():
    from krishiv_bench.explain_diff import duckdb_tree

    t = gd.duckdb_totals(duckdb_tree(D_DOC))
    assert t.secs["scan"] == pytest.approx(4.3)
    assert t.secs["filter"] == pytest.approx(0.5)
    assert t.secs["join"] == pytest.approx(2.0)
    assert t.secs["aggregate"] == pytest.approx(1.5)
    assert t.secs["project"] == pytest.approx(0.1)
    assert t.secs["sort"] == pytest.approx(0.2)  # TOP_N
    assert "exchange" not in t.secs
    # Rows in are what the children produced: the join read the filtered lineitem rows and customer.
    assert t.rows_in["join"] == 3_000_000 + 150_000
    assert t.rows_out["join"] == 30_000


def test_a_real_duckdb_document_maps(tmp_path: Path):
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    from krishiv_bench.explain_diff import duckdb_tree

    pq.write_table(pa.table({"k": list(range(1000)), "v": list(range(1000))}), tmp_path / "fact.parquet")
    pq.write_table(pa.table({"k": list(range(10)), "x": [i % 2 for i in range(10)]}), tmp_path / "dim.parquet")
    con = duckdb.connect()
    for t in ("fact", "dim"):
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{tmp_path / t}.parquet')")
    row = con.execute(
        "EXPLAIN (ANALYZE, FORMAT json) SELECT dim.x, sum(fact.v) FROM fact JOIN dim USING (k) "
        "WHERE dim.x = 1 GROUP BY dim.x ORDER BY dim.x"
    ).fetchone()
    assert row is not None
    t = gd.duckdb_totals(duckdb_tree(json.loads(row[1])))
    assert {"scan", "join", "aggregate"} <= set(t.secs)
    assert t.rows_out["scan"] >= 10  # both tables were scanned
    assert t.rows_out["aggregate"] == 1


def test_kinds_are_scaled_to_each_engines_process_cpu():
    k = gd.krishiv_totals(_doc(K_OPS, K_PIPES, K_PLAN))
    ks = gd.scaled(k, 12.0)
    assert ks is not None
    assert sum(v for key, v in ks.items() if key not in gd.PARENT) == pytest.approx(12.0)
    assert ks["scan"] == pytest.approx(k.secs["scan"] / k.total() * 12.0)
    assert gd.scaled(k, None) is None
    assert gd.scaled(gd.Totals(), 5.0) is None  # nothing to scale


def test_the_ratio_uses_process_cpu_when_both_engines_have_it_and_operator_seconds_otherwise():
    from krishiv_bench.explain_diff import duckdb_tree

    k, d = gd.krishiv_totals(_doc(K_OPS, K_PIPES, K_PLAN)), gd.duckdb_totals(duckdb_tree(D_DOC))
    with_cpu = {r.key: r for r in gd.compare(k, d, 20.0, 5.0)}
    # Both scaled: Krishiv's scan share of 20 s against DuckDB's scan share of 5 s.
    assert with_cpu["scan"].krishiv_cpu == pytest.approx(k.secs["scan"] / k.total() * 20.0)
    assert with_cpu["scan"].duckdb_cpu == pytest.approx(d.secs["scan"] / d.total() * 5.0)
    scan = with_cpu["scan"]
    assert scan.krishiv_cpu is not None
    assert scan.duckdb_cpu is not None
    assert scan.ratio == pytest.approx(scan.krishiv_cpu / scan.duckdb_cpu)
    # One reading missing: operator seconds decide, and the CPU columns that need a reading are empty.
    without = {r.key: r for r in gd.compare(k, d, 20.0, None)}
    assert without["scan"].duckdb_cpu is None
    assert without["scan"].ratio == pytest.approx(k.secs["scan"] / d.secs["scan"])
    # DuckDB has no side for the finer rows.
    for sub in ("join_build", "join_probe", "agg_partial", "agg_final"):
        assert with_cpu[sub].sub
        assert with_cpu[sub].duckdb_cpu is None
        assert with_cpu[sub].ratio is None
    assert "join_other" not in with_cpu  # a kind neither engine spent time in is left out


def test_pipelines_dominated_by_one_driver_are_flagged():
    rows, _, unattributed = gd.parallelism(_doc(K_OPS, K_PIPES))
    by = {p.pipeline: p for p in rows}
    # p0: 14 s, busiest-driver bound 2.05 s (1.2 + 0.15 + 0.4 + 0.3): balanced over 8 drivers.
    assert by[0].secs == pytest.approx(14.0)
    assert by[0].share == pytest.approx(2.05 / 14.0)
    assert by[0].flags == []
    # p2 runs on one driver and has real weight: serial.
    assert by[2].flags == ["serial"]
    assert by[2].operators == ["HashAggregate", "TopN"]
    # p3 is one driver too, but 0.02 s is below the 2% a pipeline needs to matter.
    assert by[3].flags == []
    # ExchangeRead spans pipelines 1 and 2: its time belongs to neither.
    assert unattributed == pytest.approx(0.5)
    assert by[1].secs == pytest.approx(0.7)


def test_a_two_driver_pipeline_is_not_flagged_for_a_share_that_two_drivers_always_reach():
    rows, _, _ = gd.parallelism(_doc(K_OPS, K_PIPES))
    p6 = next(p for p in rows if p.pipeline == 6)
    assert p6.share == pytest.approx(0.55)  # over 50%, as any 2-driver pipeline is
    assert p6.flags == []


def test_a_serial_merge_inside_a_heavy_parallel_pipeline_is_caught():
    rows, ops, _ = gd.parallelism(_doc(K_OPS, K_PIPES))
    p5 = next(p for p in rows if p.pipeline == 5)
    # 13 s on 12 drivers, but 3 s of it is one driver's merge: the share is only 30%, the wall time is that driver's.
    assert p5.share == pytest.approx((0.9 + 3.0) / 13.0)
    assert p5.busiest_wall == pytest.approx(1.0)
    assert p5.efficiency == pytest.approx(13.0 / (12 * 3.9))
    assert p5.flags == ["skew"]
    merge = [o for o in ops if o.pipeline == 5]
    assert [(o.name, o.role, o.drivers, o.flag) for o in merge] == [
        ("HashAggregate", "sink", 1, "serial operator in a parallel pipeline")
    ]


def test_the_thresholds_are_parameters():
    rows, _, _ = gd.parallelism(_doc(K_OPS, K_PIPES), min_cpu_share=0.0)
    assert {p.pipeline for p in rows if p.flags} == {2, 3, 5}  # p3 now counts
    rows, _, _ = gd.parallelism(_doc(K_OPS, K_PIPES), min_cpu_share=0.5)
    assert not any(p.flags for p in rows)
    rows, _, _ = gd.parallelism(_doc(K_OPS, K_PIPES), busy_share=0.1)
    assert "busy" in next(p for p in rows if p.pipeline == 0).flags  # 15% is over a 10% bar with 8 drivers


def test_a_query_report_names_the_top_gap_the_flags_the_remainder_and_the_load():
    k = gd.Basis(24.0, 0.0, 1.5, 2.5, 16.0)
    d = gd.Basis(8.0, 0.0, 2.5, 30.0, 16.0)
    result, md = gd.analyse("q03", _doc(K_OPS, K_PIPES, K_PLAN), D_DOC, k, d)
    assert result.basis == "cpu"
    assert result.krishiv_cpu == 24.0
    assert result.duckdb_cpu == 8.0
    assert result.indicative  # DuckDB's run ended at load 30
    assert result.flagged_pipelines == 2  # pipeline 2 (serial) and 5 (skew, and its serial operator)
    for part in (
        "## q03",
        "Process CPU: Krishiv 24.000 s, DuckDB 8.000 s, ratio 3.00x, gap +16.000 s.",
        "Krishiv: process CPU 24.000 s, operator seconds 32.220 s, not attributed to any operator -8.220 s (negative:",
        "DuckDB: process CPU 8.000 s, operator seconds 8.600 s",
        "load average (1 min) 2.5 at start, 30.0 at end, bound 16: **INDICATIVE ONLY**",
        "load average (1 min) 1.5 at start, 2.5 at end, bound 16: within bound",
        "&nbsp;&nbsp;↳ build",
        "scan / decode ◀",
        "exclude 0.400 s of IO wait",
        "In `other`: Sort (Krishiv 1.000 s",
        "**serial**",
        "**skew**",
        "**serial operator in a parallel pipeline**",
        "Unattributed (operators spanning several pipelines): 0.500 s.",
    ):
        assert part in md, part
    assert result.top_gap_kind == "scan"


def test_without_process_cpu_the_report_says_it_falls_back_to_operator_seconds():
    result, md = gd.analyse("q03", _doc(K_OPS, K_PIPES, K_PLAN), D_DOC)
    assert result.basis == "operator seconds"
    assert "Operator seconds (no process CPU reading for both engines)" in md
    assert "CPU s columns need a process CPU reading for both engines" in md


def _results() -> list[gd.QueryResult]:
    return [
        gd.QueryResult("q01", 9.0, 3.0, "cpu", "scan", 0),
        gd.QueryResult("q13", 24.5, 8.0, "cpu", "join", 2, indicative=True),
        gd.QueryResult("q15", error="RunnerError: [KRISHIV_UNSUPPORTED_FEATURE] a view over a CTE"),
    ]


def test_summary_format_golden():
    text = gd.summary_table(_results(), "tpch", 100, 12)
    if os.environ.get("GAPDIFF_UPDATE"):
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(text, encoding="utf-8")
    assert text == GOLDEN.read_text(encoding="utf-8")


def test_the_geomean_covers_only_the_queries_that_ran():
    text = gd.summary_table(_results(), "tpch", 100, 12)
    # 3.0x and 3.0625x: geomean sqrt(3 * 3.0625) = 3.03x over the 2 queries that ran, not the failed one.
    assert "over 2 queries on process CPU: 3.03x" in text
    assert "not compared: RunnerError" in text


def _setup_fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ids: list[str]) -> tuple[Path, list[str]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from krishiv_bench.adapters.base import Measure, Query
    from krishiv_bench.adapters.duckdb_engine import DuckDBEngine
    from krishiv_bench.adapters.krishiv import KrishivEngine, RunnerError

    data = tmp_path / "data"
    ds = data / "tpch-duckdb" / "sf1"
    ds.mkdir(parents=True)
    pq.write_table(pa.table({"a": [1, 2, 3]}), ds / "t.parquet")
    monkeypatch.setenv("KRISHIV_DATA", str(data))
    import krishiv_bench.runner as runner

    monkeypatch.setattr(
        runner,
        "load_queries",
        lambda suite, engine, only, *a: [Query(i, "SELECT a FROM t", suite) for i in ids if only is None or i in only],
    )
    profile = _doc([_op(0, 0, "FileScan", "source", [0], 0, 3, 0.25, drivers=1)], [_pipe(0, 1, 1, 0.0, 0.3)])
    calls: list[str] = []

    def plan_profile(self, q):
        calls.append(f"krishiv:{q.id}")
        if q.id == "q02":
            raise RunnerError("[KRISHIV_UNSUPPORTED_FEATURE] something")
        return {"rows": 3, "seconds": 0.5, "profile": profile}

    monkeypatch.setattr(KrishivEngine, "__init__", lambda self, **kw: None)
    monkeypatch.setattr(KrishivEngine, "setup", lambda self, d: None)
    monkeypatch.setattr(KrishivEngine, "teardown", lambda self: calls.append("krishiv:teardown"))
    monkeypatch.setattr(KrishivEngine, "plan_profile", plan_profile)
    monkeypatch.setattr(KrishivEngine, "begin_measure", lambda self: None)
    monkeypatch.setattr(KrishivEngine, "end_measure", lambda self: Measure(cpu_seconds=0.4))
    monkeypatch.setattr(DuckDBEngine, "begin_measure", lambda self: None)
    monkeypatch.setattr(DuckDBEngine, "end_measure", lambda self: Measure(cpu_seconds=0.1))
    return tmp_path / "out", calls


def test_one_query_writes_its_report_and_the_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out, calls = _setup_fakes(tmp_path, monkeypatch, ["q01"])
    gd.run_gapdiff("tpch", 1, ["q01"], dataset="tpch-duckdb", threads=1, out_dir=out)
    assert "## q01" in (out / "summary.md").read_text()
    assert (out / "q01.md").exists()
    raw = json.loads((out / "gapdiff.json").read_text())
    assert raw["suite"] == "tpch"
    assert raw["threads"] == 1
    q = raw["queries"]["q01"]
    assert q["basis"] == "cpu"
    assert q["krishiv_cpu"] == pytest.approx(0.4)  # the measured process CPU, not the 0.25 s of operator time
    assert q["duckdb_cpu"] == pytest.approx(0.1)
    assert q["krishiv"]["operator_secs"] == pytest.approx(0.25)
    assert q["error"] is None
    # One warm-up run, then the profiled one.
    assert calls.count("krishiv:q01") == 2
    with pytest.raises(SystemExit, match="unknown queries"):
        gd.run_gapdiff("tpch", 1, ["q99"], dataset="tpch-duckdb", threads=1, out_dir=tmp_path / "x")


def test_warmup_is_a_parameter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out, calls = _setup_fakes(tmp_path, monkeypatch, ["q01"])
    gd.run_gapdiff("tpch", 1, ["q01"], dataset="tpch-duckdb", threads=1, out_dir=out, warmup=0)
    assert calls.count("krishiv:q01") == 1
    out2 = tmp_path / "out2"
    gd.run_gapdiff("tpch", 1, ["q01"], dataset="tpch-duckdb", threads=1, out_dir=out2, warmup=3)
    assert calls.count("krishiv:q01") == 1 + 4


def test_all_runs_every_query_and_a_failure_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out, _ = _setup_fakes(tmp_path, monkeypatch, ["q01", "q02"])
    gd.run_gapdiff("tpch", 1, None, dataset="tpch-duckdb", threads=1, out_dir=out)
    summary = (out / "summary.md").read_text()
    assert "| q01 |" in summary
    assert "| q02 | - | - | - | - | - | not compared: RunnerError" in summary
    raw = json.loads((out / "gapdiff.json").read_text())
    assert raw["queries"]["q02"]["error"].startswith("RunnerError")
    assert (out / "q02.md").read_text().startswith("## q02\n\nNot compared:")


def test_perf_samples_after_the_profiling_engines_are_shut_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from collections import Counter

    out, calls = _setup_fakes(tmp_path, monkeypatch, ["q01"])

    def sample(*a, **k):
        calls.append("perf")
        return Counter({"main;exec;scan": 7, "main;exec;join": 3})

    monkeypatch.setattr(gd, "sample_with_perf", sample)
    gd.run_gapdiff("tpch", 1, ["q01"], dataset="tpch-duckdb", threads=1, out_dir=out, perf=True)
    assert (out / "perf" / "q01.folded").read_text() == "main;exec;join 3\nmain;exec;scan 7\n"
    assert calls.index("perf") > calls.index("krishiv:teardown")  # one runner at a time on the box

    def broken(*a, **k):
        raise PermissionError("perf_event_paranoid")

    monkeypatch.setattr(gd, "sample_with_perf", broken)
    out2 = tmp_path / "out2"
    gd.run_gapdiff("tpch", 1, ["q01"], dataset="tpch-duckdb", threads=1, out_dir=out2, perf=True)
    assert (out2 / "q01.md").exists()
    assert not (out2 / "perf").exists()


def test_the_geomean_is_kept_per_basis():
    rs = [
        gd.QueryResult("q01", 9.0, 3.0, "cpu", "scan", 0),
        gd.QueryResult("q02", 40.0, 10.0, "operator seconds", "scan", 0),
    ]
    text = gd.summary_table(rs, "tpch", 1, 4)
    assert "over 1 queries on process CPU: 3.00x." in text
    assert "over 1 queries on operator seconds: 4.00x. Not comparable with the process CPU ratios." in text
