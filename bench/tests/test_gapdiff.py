"""gapdiff: the kind mapping of both engines' profiles, the side-by-side table, the parallelism flags, the summary
format (a golden) and the end-to-end command."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from krishiv_bench import gapdiff as gd

GOLDEN = Path(__file__).parent / "golden" / "gapdiff_summary.md"


def _op(oid, node, name, role, pipelines, rows_in, rows_out, cpu, busiest=None):
    """A profile operator; times are in seconds here and nanoseconds in the document."""
    return {
        "id": oid,
        "node": node,
        "name": name,
        "role": role,
        "pipelines": pipelines,
        "metrics": {
            "rows_in": rows_in,
            "rows_out": rows_out,
            "compute_time_ns": round(cpu * 1e9),
            "max_driver_compute_time_ns": round((cpu if busiest is None else busiest) * 1e9),
        },
    }


def _pipe(pid, drivers, threads, start, end):
    return {
        "id": pid,
        "drivers": drivers,
        "threads": threads,
        "started_ns": round(start * 1e9),
        "ended_ns": round(end * 1e9),
    }


def _doc(ops, pipelines):
    return {"format": "krishiv.query-profile", "version": 1, "plan": [], "operators": ops, "pipelines": pipelines}


# TPC-H q3 shaped, with the operator names the engine reports. Pipeline 0 reads lineitem on 8 drivers (balanced),
# 1 builds the customer side, 2 is the single-driver final aggregate + sort, 3 is the result (tiny).
K_OPS = [
    _op(0, 3, "FileScan", "source", [0], 0, 6_000_000, 8.0, 1.2),
    _op(1, 2, "Filter", "operator", [0], 6_000_000, 3_000_000, 1.0, 0.15),
    _op(2, 1, "HashJoinProbe", "operator", [0], 3_000_000, 30_000, 3.0, 0.4),
    _op(3, 4, "PartialAggregate", "operator", [0], 30_000, 10_000, 2.0, 0.3),
    _op(4, 5, "FileScan", "source", [1], 0, 150_000, 0.5, 0.1),
    _op(5, 1, "HashJoinBuild", "sink", [1], 150_000, 0, 0.2, 0.05),
    _op(6, 6, "HashAggregate", "source", [2], 10_000, 10, 2.0, 2.0),
    _op(7, 7, "Sort", "operator", [2], 10, 10, 1.0, 1.0),
    _op(8, 8, "ExchangeRead", "source", [1, 2], 0, 10_000, 0.5, 0.5),
    _op(9, None, "Result", "sink", [3], 10, 0, 0.01, 0.01),
]
K_PIPES = [
    _pipe(0, 8, 8, 0.0, 2.0),
    _pipe(1, 4, 4, 0.0, 0.3),
    _pipe(2, 1, 1, 2.0, 5.0),
    _pipe(3, 1, 1, 5.0, 5.01),
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


def test_krishiv_operator_names_map_to_kinds_and_roll_up():
    t = gd.krishiv_totals(_doc(K_OPS, K_PIPES))
    assert t.cpu["scan"] == pytest.approx(8.5)
    assert t.cpu["filter"] == pytest.approx(1.0)
    assert t.cpu["join_probe"] == pytest.approx(3.0)
    assert t.cpu["join_build"] == pytest.approx(0.2)
    assert t.cpu["join"] == pytest.approx(3.2)
    assert t.cpu["agg_partial"] == pytest.approx(2.0)
    assert t.cpu["agg_final"] == pytest.approx(2.0)
    assert t.cpu["aggregate"] == pytest.approx(4.0)
    assert t.cpu["sort"] == pytest.approx(1.0)
    assert t.cpu["exchange"] == pytest.approx(0.5)
    assert t.cpu["other"] == pytest.approx(0.01)  # Result
    # A sub-row is inside its parent: the total counts each operator once.
    assert t.total() == pytest.approx(8.5 + 1.0 + 3.2 + 4.0 + 1.0 + 0.5 + 0.01)
    assert t.rows_in["join"] == 3_000_000 + 150_000
    assert t.rows_out["scan"] == 6_000_000 + 150_000


def test_an_unknown_krishiv_operator_is_other_and_a_foreign_document_is_refused():
    doc = _doc([_op(0, 0, "my_udf", "operator", [0], 1, 1, 0.5)], [])
    assert gd.krishiv_totals(doc).cpu["other"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="query-profile"):
        gd.krishiv_totals({"format": "something-else", "version": 1, "operators": []})


def test_duckdb_operators_map_to_the_same_kinds():
    from krishiv_bench.explain_diff import duckdb_tree

    t = gd.duckdb_totals(duckdb_tree(D_DOC))
    assert t.cpu["scan"] == pytest.approx(4.3)
    assert t.cpu["filter"] == pytest.approx(0.5)
    assert t.cpu["join"] == pytest.approx(2.0)
    assert t.cpu["aggregate"] == pytest.approx(1.5)
    assert t.cpu["project"] == pytest.approx(0.1)
    assert t.cpu["sort"] == pytest.approx(0.2)  # TOP_N
    assert "exchange" not in t.cpu
    # Rows in are what the children produced: the join read the filtered lineitem rows and customer.
    assert t.rows_in["join"] == 3_000_000 + 150_000
    assert t.rows_out["join"] == 30_000


def test_the_table_has_a_duckdb_side_only_where_duckdb_reports_one_number():
    from krishiv_bench.explain_diff import duckdb_tree

    rows = {r.key: r for r in gd.compare(gd.krishiv_totals(_doc(K_OPS, K_PIPES)), gd.duckdb_totals(duckdb_tree(D_DOC)))}
    assert rows["join"].duckdb_cpu == pytest.approx(2.0)
    assert rows["join"].ratio == pytest.approx(3.2 / 2.0)
    for sub in ("join_build", "join_probe", "agg_partial", "agg_final"):
        assert rows[sub].sub
        assert rows[sub].duckdb_cpu is None
        assert rows[sub].ratio is None
    assert rows["exchange"].duckdb_cpu == 0  # DuckDB has no exchange
    assert rows["exchange"].ratio is None  # and no time to divide by
    assert "join_other" not in rows  # a kind neither engine spent time in is left out


def test_pipelines_dominated_by_one_driver_are_flagged():
    rows, unattributed = gd.pipeline_rows(_doc(K_OPS, K_PIPES))
    by = {p.pipeline: p for p in rows}
    # p0: 14 s CPU, busiest-driver bound 2.05 s (1.2 + 0.15 + 0.4 + 0.3): 15%, balanced over 8 drivers.
    assert by[0].cpu == pytest.approx(14.0)
    assert by[0].share == pytest.approx(2.05 / 14.0)
    assert not by[0].flagged
    # p2: one driver does all of it, and the pipeline has 3 s of 17.2 s of CPU.
    assert by[2].share == pytest.approx(1.0)
    assert by[2].flagged
    assert by[2].operators == ["HashAggregate", "Sort"]
    # p3 is a single driver too, but 0.01 s of CPU is below the 2% a pipeline needs to matter.
    assert by[3].share == pytest.approx(1.0)
    assert not by[3].flagged
    # ExchangeRead spans pipelines 1 and 2: its CPU belongs to neither.
    assert unattributed == pytest.approx(0.5)
    assert by[1].cpu == pytest.approx(0.7)


def test_the_thresholds_are_parameters():
    rows, _ = gd.pipeline_rows(_doc(K_OPS, K_PIPES), min_cpu_share=0.0, busy_share=0.5)
    assert {p.pipeline for p in rows if p.flagged} == {2, 3}
    rows, _ = gd.pipeline_rows(_doc(K_OPS, K_PIPES), min_cpu_share=0.02, busy_share=0.99)
    assert {p.pipeline for p in rows if p.flagged} == {2}  # p3 (100%) is below the CPU floor; p1 is 21%
    rows, _ = gd.pipeline_rows(_doc(K_OPS, K_PIPES), min_cpu_share=0.5, busy_share=0.5)
    assert not any(p.flagged for p in rows)


def test_a_query_report_names_the_top_gap_and_the_flags():
    result, md = gd.analyse("q03", _doc(K_OPS, K_PIPES), D_DOC)
    assert result.flagged_pipelines == 1
    assert result.top_gap_kind == "scan"  # 8.5 s of Krishiv CPU against 4.3 s
    assert "## q03" in md
    assert "scan / decode ◀" in md
    assert "&nbsp;&nbsp;↳ build" in md
    assert "**FLAG**" in md
    assert "1 pipeline(s) flagged" in md
    assert "Unattributed (operators spanning several pipelines): 0.500 s CPU." in md
    assert result.krishiv_cpu == pytest.approx(18.21)
    assert result.duckdb_cpu == pytest.approx(4.3 + 0.5 + 2.0 + 1.5 + 0.1 + 0.2)


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
    assert {"scan", "join", "aggregate"} <= set(t.cpu)
    assert t.rows_out["scan"] >= 10  # both tables were scanned
    assert t.rows_out["aggregate"] == 1


def _results() -> list[gd.QueryResult]:
    return [
        gd.QueryResult("q01", 9.0, 3.0, "scan", 0),
        gd.QueryResult("q13", 24.5, 8.0, "join", 2),
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
    assert "over 2 queries: 3.03x" in text
    assert "not compared: RunnerError" in text


def _setup_fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ids: list[str]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from krishiv_bench.adapters.base import Query
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
    profile = _doc([_op(0, 0, "FileScan", "source", [0], 0, 3, 0.25)], [_pipe(0, 1, 1, 0.0, 0.3)])

    def plan_profile(self, q):
        if q.id == "q02":
            raise RunnerError("[KRISHIV_UNSUPPORTED_FEATURE] something")
        return {"rows": 3, "seconds": 0.5, "profile": profile}

    monkeypatch.setattr(KrishivEngine, "__init__", lambda self, **kw: None)
    monkeypatch.setattr(KrishivEngine, "setup", lambda self, d: None)
    monkeypatch.setattr(KrishivEngine, "teardown", lambda self: None)
    monkeypatch.setattr(KrishivEngine, "plan_profile", plan_profile)
    return tmp_path / "out"


def test_one_query_writes_its_report_and_the_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = _setup_fakes(tmp_path, monkeypatch, ["q01"])
    gd.run_gapdiff("tpch", 1, ["q01"], dataset="tpch-duckdb", threads=1, out_dir=out)
    assert "## q01" in (out / "summary.md").read_text()
    assert (out / "q01.md").exists()
    raw = json.loads((out / "gapdiff.json").read_text())
    assert raw["suite"] == "tpch"
    assert raw["threads"] == 1
    assert raw["queries"]["q01"]["krishiv_cpu"] == pytest.approx(0.25)
    assert raw["queries"]["q01"]["error"] is None
    with pytest.raises(SystemExit, match="unknown queries"):
        gd.run_gapdiff("tpch", 1, ["q99"], dataset="tpch-duckdb", threads=1, out_dir=tmp_path / "x")


def test_all_runs_every_query_and_a_failure_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = _setup_fakes(tmp_path, monkeypatch, ["q01", "q02"])
    gd.run_gapdiff("tpch", 1, None, dataset="tpch-duckdb", threads=1, out_dir=out)
    summary = (out / "summary.md").read_text()
    assert "| q01 |" in summary
    assert "| q02 | - | - | - | - | not compared: RunnerError" in summary
    raw = json.loads((out / "gapdiff.json").read_text())
    assert raw["queries"]["q02"]["error"].startswith("RunnerError")
    assert (out / "q02.md").read_text().startswith("## q02\n\nNot compared:")


def test_perf_writes_folded_stacks_and_a_missing_perf_does_not_fail_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from collections import Counter

    out = _setup_fakes(tmp_path, monkeypatch, ["q01"])
    monkeypatch.setattr(gd, "sample_with_perf", lambda *a, **k: Counter({"main;exec;scan": 7, "main;exec;join": 3}))
    gd.run_gapdiff("tpch", 1, ["q01"], dataset="tpch-duckdb", threads=1, out_dir=out, perf=True)
    assert (out / "perf" / "q01.folded").read_text() == "main;exec;join 3\nmain;exec;scan 7\n"

    def broken(*a, **k):
        raise PermissionError("perf_event_paranoid")

    monkeypatch.setattr(gd, "sample_with_perf", broken)
    out2 = tmp_path / "out2"
    gd.run_gapdiff("tpch", 1, ["q01"], dataset="tpch-duckdb", threads=1, out_dir=out2, perf=True)
    assert (out2 / "q01.md").exists()
    assert not (out2 / "perf").exists()
