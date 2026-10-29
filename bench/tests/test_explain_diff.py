"""explain-diff: profile parsing for both engines, the flags, the report and the end-to-end command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from krishiv_bench import explain_diff as ed


def _op(oid, node, name, role, rows_in, rows_out, ns=1_000_000):
    return {
        "id": oid,
        "node": node,
        "name": name,
        "role": role,
        "metrics": {"rows_in": rows_in, "rows_out": rows_out, "compute_time_ns": ns},
    }


def krishiv_doc(plan, ops):
    return {"format": "krishiv.query-profile", "version": 1, "detailed": True, "plan": plan, "operators": ops}


def _n(i, name, detail, children=()):
    return {"id": i, "name": name, "detail": detail, "children": list(children)}


# Krishiv: ((fact ⋈ dim_a) ⋈ dim_b) under an Expand x5 and an aggregate; dim_b is filtered above its scan.
K_PLAN = [
    _n(0, "HashAggregate", "keys=[a#1]", [1]),
    _n(1, "Exchange", "hashpartitioning(a#1)", [2]),
    _n(2, "Expand", "5 projections", [3]),
    _n(3, "HashJoin", "Inner, [k#2] = [k#3]", [4, 7]),
    _n(4, "HashJoin", "Inner, [j#4] = [j#5]", [5, 6]),
    _n(5, "Scan", "spark_catalog.default.fact [j#4, k#2]"),
    _n(6, "Scan", "spark_catalog.default.dim_a [j#5]"),
    _n(7, "BroadcastExchange", "", [8]),
    _n(8, "Filter", "(x#6 = 1)", [9]),
    _n(9, "Scan", "spark_catalog.default.dim_b [k#3, x#6], PushedFilters: [(x#6 = 1)]"),
]
K_OPS = [
    _op(0, 0, "HashAggregate", "sink", 5_000_000, 0, 9_000_000_000),
    _op(1, 0, "HashAggregate", "source", 0, 10),
    _op(2, 2, "Expand", "operator", 1_000_000, 5_000_000),
    _op(3, 3, "HashJoinBuild", "sink", 50, 0),
    _op(4, 3, "HashJoinProbe", "operator", 2_000_000, 1_000_000),
    _op(5, 4, "HashJoinBuild", "sink", 100, 0),
    _op(6, 4, "HashJoinProbe", "operator", 2_000_000, 2_000_000),
    _op(7, 5, "FileScan", "source", 0, 2_000_000),
    _op(8, 6, "FileScan", "source", 0, 100),
    _op(9, 8, "Filter", "operator", 80_000, 50),
    _op(10, 9, "FileScan", "source", 0, 80_000),
    _op(11, None, "Result", "sink", 10, 0),
]


def _d(t, rows, children=(), est=None, **extra):
    info = dict(extra)
    if est is not None:
        info["Estimated Cardinality"] = str(est)
    return {"type": t, "intermediate_rows": rows, "timing": 0.01, "extra_info": info, "children": list(children)}


# DuckDB: ((fact ⋈ dim_b) ⋈ dim_a), dim_b's filter in its scan, one aggregate.
D_DOC = {
    "operator": [
        _d(
            "RESULT_COLLECTOR",
            0,
            [
                _d(
                    "EXPLAIN_ANALYZE",
                    0,
                    [
                        _d(
                            "HASH_GROUP_BY",
                            10,
                            [
                                _d(
                                    "HASH_JOIN",
                                    1_000_000,
                                    [
                                        _d(
                                            "HASH_JOIN",
                                            1_000_000,
                                            [
                                                _d(
                                                    "TABLE_SCAN",
                                                    900_000,
                                                    est=2_000_000,
                                                    **{"Filename(s)": "/d/fact.parquet"},
                                                ),
                                                _d("TABLE_SCAN", 50, est=60, **{"Filename(s)": "/d/dim_b.parquet"}),
                                            ],
                                            est=900_000,
                                        ),
                                        _d("TABLE_SCAN", 100, **{"Table": "dim_a"}),
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


def test_krishiv_profile_becomes_a_tree():
    k = ed.krishiv_tree(krishiv_doc(K_PLAN, K_OPS))
    assert (k.name, k.kind, k.rows) == ("HashAggregate", "aggregate", 10)
    joins = [n for n in k.walk() if n.kind == "join"]
    assert [(n.tables, n.rows, n.build_rows) for n in joins] == [
        (("dim_a", "dim_b", "fact"), 1_000_000, 50),
        (("dim_a", "fact"), 2_000_000, 100),
    ]
    exchange = k.children[0]
    assert exchange.rows == 5_000_000  # an exchange passes its input's rows on
    assert [n.tables for n in k.walk() if n.kind == "scan"] == [("fact",), ("dim_a",), ("dim_b",)]
    assert "#" not in k.signature  # plan ids do not make two equal subtrees differ
    assert k.seconds == pytest.approx(9.001)
    with pytest.raises(ValueError, match="query-profile"):
        ed.krishiv_tree({"format": "other"})


def test_several_roots_get_one_synthetic_root():
    plan = [_n(0, "Scan", "t [a#1]"), _n(1, "Scan", "u [b#2]")]
    ops = [_op(0, 0, "FileScan", "source", 0, 3), _op(1, 1, "FileScan", "source", 0, 4)]
    k = ed.krishiv_tree(krishiv_doc(plan, ops))
    assert k.name == "Plan"
    assert [c.tables for c in k.children] == [("t",), ("u",)]


def test_duckdb_profile_becomes_a_tree():
    d = ed.duckdb_tree(D_DOC)
    assert d.name == "HASH_GROUP_BY"  # the wrappers are skipped
    inner = d.children[0].children[0]
    assert (inner.tables, inner.est, inner.build_rows) == (("dim_b", "fact"), 900_000, 50)
    assert [n.tables for n in d.walk() if n.kind == "scan"] == [("fact",), ("dim_b",), ("dim_a",)]
    with pytest.raises(ValueError, match="EXPLAIN ANALYZE"):
        ed.duckdb_tree({"operator": []})


def test_duckdb_ctes_resolve_to_their_base_tables_through_nesting():
    inner_cte = _d(
        "CTE",
        0,
        [_d("TABLE_SCAN", 5, **{"Table": "t"}), _d("CTE_SCAN", 5, **{"CTE Index": "1"})],
        **{"Table Index": "1"},
    )
    outer = _d(
        "CTE",
        0,
        [
            _d("PROJECTION", 5, [inner_cte]),
            _d(
                "HASH_JOIN",
                5,
                [_d("CTE_SCAN", 5, **{"CTE Index": "2"}), _d("CTE_SCAN", 5, **{"CTE Index": "2"})],
            ),
        ],
        **{"Table Index": "2"},
    )
    d = ed.duckdb_tree({"operator": [outer]})
    join = next(n for n in d.walk() if n.kind == "join")
    assert join.tables == ("t", "t")
    assert d.tables == ("t", "t")  # a CTE outputs what its consumer outputs


def test_flags_name_join_order_build_side_scans_and_blowups():
    k = ed.krishiv_tree(krishiv_doc(K_PLAN, K_OPS))
    d = ed.duckdb_tree(D_DOC)
    res = ed.diff(k, d)
    text = "\n".join(res.flags)
    assert "join order: Krishiv joins {dim_a, fact} (2,000,000 rows); DuckDB never forms this intermediate" in text
    assert "join order: DuckDB joins {dim_b, fact} (1,000,000 rows); Krishiv does not" in text
    # dim_b: 80,000 rows out of Krishiv's scan, 50 after the Filter above it, 50 out of DuckDB's scan.
    assert "scan of dim_b: Krishiv reads out 80,000 rows, DuckDB 50 (50 after the Filter above it)" in text
    # fact: DuckDB's scan already applied the join's runtime filter; ours returns everything with no Filter above.
    assert "scan of fact: Krishiv reads out 2,000,000 rows, DuckDB 900,000 (no Filter above it" in text
    # The Expand is where rows grow; the Exchange above it only passes them on.
    blowups = [f for f in res.flags if f.startswith("row blow-up")]
    assert len(blowups) == 1
    assert "Expand" in blowups[0]
    assert "from 1,000,000 input rows" in blowups[0]
    assert res.top_time["krishiv"][0][0].startswith("HashAggregate")
    joined = {j["tables"]: j for j in res.joins}
    assert joined["{dim_a, dim_b, fact}"]["krishiv_rows"] == [1_000_000]
    assert joined["{dim_a, dim_b, fact}"]["duckdb_build_side"] == ["{dim_a}"]


def test_matched_joins_compare_rows_and_build_sides():
    plan = [_n(0, "HashJoin", "Inner", [1, 2]), _n(1, "Scan", "big [a#1]"), _n(2, "Scan", "small [b#2]")]
    ops = [
        _op(0, 0, "HashJoinBuild", "sink", 400_000, 0),
        _op(1, 0, "HashJoinProbe", "operator", 10, 90_000),
        _op(2, 1, "FileScan", "source", 0, 10),
        _op(3, 2, "FileScan", "source", 0, 400_000),
    ]
    k = ed.krishiv_tree(krishiv_doc(plan, ops))
    d = ed.duckdb_tree(
        {
            "operator": [
                _d(
                    "HASH_JOIN",
                    3_000,
                    [_d("TABLE_SCAN", 400_000, **{"Table": "small"}), _d("TABLE_SCAN", 10, **{"Table": "big"})],
                )
            ]
        }
    )
    text = "\n".join(ed.diff(k, d).flags)
    assert "join {big, small}: Krishiv 90,000 rows vs DuckDB 3,000" in text
    assert "build side of {big, small}: Krishiv builds {small} (400,000 rows), DuckDB builds {big} (10 rows)" in text


def test_repeated_subtrees_and_repeated_scans_are_counted_once_each():
    sub = [_n(1, "Project", "[a#1]", [2]), _n(2, "Scan", "t [a#1]")]
    plan = [
        _n(0, "Union", "", [1, 3]),
        *sub,
        _n(3, "Project", "[a#7]", [4]),
        _n(4, "Scan", "t [a#7]"),
    ]
    ops = [
        _op(0, 0, "Union", "operator", 20, 20),
        _op(1, 1, "Project", "operator", 10, 10),
        _op(2, 2, "FileScan", "source", 0, 10),
        _op(3, 3, "Project", "operator", 10, 10),
        _op(4, 4, "FileScan", "source", 0, 10),
    ]
    k = ed.krishiv_tree(krishiv_doc(plan, ops))
    d = ed.duckdb_tree({"operator": [_d("PROJECTION", 10, [_d("TABLE_SCAN", 10, **{"Table": "t"})])]})
    flags = ed.diff(k, d).flags
    assert "repeated subtree: Krishiv computes Project over {t} 2 times (10 rows each)" in flags
    assert "table t is scanned 2 times by Krishiv, 1 by DuckDB" in flags


def test_identical_flags_are_merged_with_a_count():
    plan = [_n(0, "Union", "", [1, 2]), _n(1, "Scan", "t [a#1]"), _n(2, "Scan", "t [a#2]")]
    ops = [_op(0, 0, "Union", "operator", 0, 0), _op(1, 1, "FileScan", "source", 0, 50_000)]
    ops.append(_op(2, 2, "FileScan", "source", 0, 50_000))
    k = ed.krishiv_tree(krishiv_doc(plan, ops))
    d = ed.duckdb_tree(
        {
            "operator": [
                _d(
                    "UNION",
                    10,
                    [_d("TABLE_SCAN", 5, **{"Table": "t"}), _d("TABLE_SCAN", 5, **{"Table": "t"})],
                )
            ]
        }
    )
    flags = ed.diff(k, d).flags
    assert any(f.startswith("scan of t:") and f.endswith("(x2)") for f in flags), flags


def test_markdown_report_has_every_section():
    k = ed.krishiv_tree(krishiv_doc(K_PLAN, K_OPS))
    d = ed.duckdb_tree(D_DOC)
    md = ed.markdown("q01", k, d, ed.diff(k, d), 1.5, 0.25)
    for part in (
        "## q01",
        "**Flags**",
        "**Joins**",
        "**Scans**",
        "**Most operator time**",
        "Krishiv plan",
        "est 900,000",
    ):
        assert part in md
    assert "#" not in md.split("Krishiv plan")[1].split("```")[1]  # plan ids are stripped from the tree


def test_real_duckdb_document_parses(tmp_path: Path):
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"k": list(range(100)), "v": list(range(100))}), tmp_path / "fact.parquet")
    pq.write_table(pa.table({"k": list(range(10)), "x": [i % 2 for i in range(10)]}), tmp_path / "dim.parquet")
    con = duckdb.connect()
    for t in ("fact", "dim"):
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{tmp_path / t}.parquet')")
    row = con.execute(
        "EXPLAIN (ANALYZE, FORMAT json) SELECT count(*) FROM fact JOIN dim USING (k) WHERE dim.x = 1"
    ).fetchone()
    assert row is not None
    d = ed.duckdb_tree(json.loads(row[1]))
    join = next(n for n in d.walk() if n.kind == "join")
    assert sorted(join.tables) == ["dim", "fact"]
    assert join.rows == 5
    assert {n.tables[0] for n in d.walk() if n.kind == "scan"} == {"dim", "fact"}


def test_run_explain_diff_writes_the_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from krishiv_bench.adapters.base import Query
    from krishiv_bench.adapters.krishiv import KrishivEngine

    data = tmp_path / "data"
    ds = data / "tpch-duckdb" / "sf1"
    ds.mkdir(parents=True)
    pq.write_table(pa.table({"a": [1, 2, 3]}), ds / "t.parquet")
    monkeypatch.setenv("KRISHIV_DATA", str(data))
    import krishiv_bench.runner as runner

    monkeypatch.setattr(
        runner, "load_queries", lambda suite, engine, only, *a: [Query("q01", "SELECT a FROM t", suite)]
    )
    profile = krishiv_doc([_n(0, "Scan", "t [a#1]")], [_op(0, 0, "FileScan", "source", 0, 3)])
    monkeypatch.setattr(KrishivEngine, "setup", lambda self, d: None)
    monkeypatch.setattr(KrishivEngine, "teardown", lambda self: None)
    monkeypatch.setattr(KrishivEngine, "plan_profile", lambda self, q: {"rows": 3, "seconds": 0.5, "profile": profile})
    monkeypatch.setattr(KrishivEngine, "__init__", lambda self, **kw: None)
    out = ed.run_explain_diff("tpch", 1, ["q01"], dataset="tpch-duckdb", threads=1, out_dir=tmp_path / "out")
    text = out.read_text()
    assert "## q01" in text
    assert "| t | 3 | 3 |" in text
    raw = json.loads(out.with_suffix(".json").read_text())
    assert raw["q01"]["krishiv"]["rows"] == 3
    with pytest.raises(SystemExit, match="unknown queries"):
        ed.run_explain_diff("tpch", 1, ["q99"], dataset="tpch-duckdb", threads=1, out_dir=tmp_path / "out")
