"""db-benchmark suites: query definitions against DuckDB, checksum comparison, the run, the report, the CLI."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow.parquet as pq
import pytest

from krishiv_bench import ab, cli, datasets, h2o, reproduce
from krishiv_bench import h2o_data as hd
from krishiv_bench.adapters import Dataset, Query, make_engine
from krishiv_bench.adapters.base import check_query, result_rows
from krishiv_bench.adapters.duckdb_engine import DuckDBEngine
from krishiv_bench.runner import load_queries

ROWS = 10_000
TINY_J1: dict[str, Any] = {"div_small": 1000, "div_medium": 100}
SUITE_TASK = {"h2o-groupby": "groupby", "h2o-join": "join"}


@pytest.fixture(autouse=True)
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path / "data"))
    return tmp_path / "data"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("h2o")
    return {
        "h2o-groupby": hd.generate("groupby", hd.G1Params(ROWS, 10, 5), base),
        "h2o-join": hd.generate("join", hd.J1Params(ROWS, 5, **TINY_J1), base),
    }


def duck(root: Path) -> DuckDBEngine:
    eng = DuckDBEngine(threads=2)
    eng.setup(h2o.resolve_dataset(root, 0.5))
    return eng


# ---------------------------------------------------------------- query definitions
def test_question_counts_and_checksum_columns():
    assert [q.id for q in h2o.load_queries("h2o-groupby", "duckdb")] == [f"q{i:02d}" for i in range(1, 11)]
    assert [q.id for q in h2o.load_queries("h2o-join", "duckdb")] == [f"q{i:02d}" for i in range(1, 6)]
    assert h2o.load_queries("h2o-join", "polars", ["q03"])[0].id == "q03"
    with pytest.raises(ValueError, match="not an h2o suite"):
        h2o.load_queries("tpch", "duckdb")


def test_runner_load_queries_dispatches_h2o_suites():
    qs = load_queries("h2o-groupby", "krishiv", ["q01"])
    assert [q.suite for q in qs] == ["h2o-groupby"]
    assert qs[0].check_sql
    assert qs[0].sql in qs[0].check_sql


def test_check_sql_wraps_the_question_and_strips_the_semicolon():
    assert h2o.check_sql(" SELECT a FROM t ; ", ["a", "b"]) == (
        "SELECT COUNT(*) AS n_rows, SUM(a) AS a, SUM(b) AS b FROM (SELECT a FROM t) AS ans"
    )


def test_an_engine_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    d = tmp_path / "queries" / "h2o-groupby"
    (d / "polars").mkdir(parents=True)
    (d / "q01.sql").write_text("SELECT id1, SUM(v1) AS v1 FROM x GROUP BY id1")
    (d / "polars" / "q01.sql").write_text("SELECT id1, SUM(v1) AS v1 FROM x GROUP BY 1")
    monkeypatch.setattr(h2o, "queries_dir", lambda suite: tmp_path / "queries" / suite)
    assert h2o.load_queries("h2o-groupby", "duckdb")[0].sql.endswith("GROUP BY id1")
    assert h2o.load_queries("h2o-groupby", "polars")[0].sql.endswith("GROUP BY 1")


def test_a_question_without_checksum_columns_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    d = tmp_path / "h2o-join"
    d.mkdir()
    (d / "q09.sql").write_text("SELECT 1")
    monkeypatch.setattr(h2o, "queries_dir", lambda suite: d)
    with pytest.raises(ValueError, match="q09: no checksum columns"):
        h2o.load_queries("h2o-join", "duckdb")


@pytest.mark.parametrize("suite", ["h2o-groupby", "h2o-join"])
def test_every_question_and_checksum_runs_and_agrees_across_engines(suite: str, tiny: dict[str, Path]):
    """DuckDB is the oracle; Polars and DataFusion (independent implementations) must produce the same checksums."""
    root = tiny[suite]
    ds = h2o.resolve_dataset(root, 0.5)
    engines = {
        n: make_engine(n, **({"threads": 2} if n != "datafusion" else {})) for n in ("duckdb", "polars", "datafusion")
    }
    for e in engines.values():
        e.setup(ds)
    for q in h2o.load_queries(suite, "duckdb"):
        rows = {n: e.run(check_query(q)) for n, e in engines.items()}
        ref = rows["duckdb"]
        assert len(ref) == 1, q.id
        assert len(ref[0]) == 1 + len(h2o.CHK_COLUMNS[suite][q.id]), q.id
        assert ref[0][0] > 0
        for n in ("polars", "datafusion"):
            ok, msg = h2o.compare_checksums(list(rows[n][0]), list(ref[0]))
            assert ok, f"{suite} {q.id} {n}: {msg}"
        n_timed = {name: e.timed_run(q)[0] for name, e in engines.items()}
        assert set(n_timed.values()) == {ref[0][0]}, q.id


def test_groupby_questions_against_hand_computed_answers(tiny: dict[str, Path]):
    x = f"read_parquet('{tiny['h2o-groupby'] / 'x.parquet'}')"
    con = duckdb.connect()
    eng = duck(tiny["h2o-groupby"])
    by_id = {q.id: q for q in h2o.load_queries("h2o-groupby", "duckdb")}
    # q01 against PyArrow's own group-by (an independent implementation)
    tbl = pq.read_table(tiny["h2o-groupby"] / "x.parquet")
    agg = tbl.group_by("id1").aggregate([("v1", "sum")]).to_pylist()
    assert {r["id1"]: r["v1_sum"] for r in agg} == dict(eng.run(by_id["q01"]))
    # q06's median against NumPy for one group
    med = {(r[0], r[1]): r[2] for r in eng.run(by_id["q06"])}
    t = tbl.to_pandas()
    g = t[(t.id4 == 1) & (t.id5 == 1)].v3.dropna()
    assert med[(1, 1)] == pytest.approx(float(np.median(g)))
    # q08: the two largest v3 of every id6 group, NULL v3 excluded
    top2 = eng.run(by_id["q08"])
    per_group: dict = {}
    for g, v in top2:
        per_group.setdefault(g, []).append(v)
    assert all(1 <= len(v) <= 2 for v in per_group.values())
    for g, vals in per_group.items():
        allv = sorted(
            (
                r[0]
                for r in con.execute(
                    f"SELECT v3 FROM {x} WHERE id6 IS NOT DISTINCT FROM ? AND v3 IS NOT NULL", [g]
                ).fetchall()
            ),
            reverse=True,
        )
        assert sorted(vals, reverse=True) == allv[:2]
    # q10: the finest grouping accounts for every row exactly once
    assert sum(r[7] for r in eng.run(by_id["q10"])) == ROWS


def test_join_question_row_counts(tiny: dict[str, Path]):
    root = tiny["h2o-join"]
    eng = duck(root)
    qs = {q.id: q for q in h2o.load_queries("h2o-join", "duckdb")}
    n = {i: eng.timed_run(q)[0] for i, q in qs.items()}
    assert n["q03"] == ROWS  # left join keeps every left row
    assert all(n[i] <= ROWS for i in n)
    con = duckdb.connect()

    def count_matches(rhs: str, key: str) -> int:
        sql = (
            f"SELECT count(*) FROM read_parquet('{root / 'x.parquet'}') x "
            f"JOIN read_parquet('{root / (rhs + '.parquet')}') r ON x.{key} = r.{key}"
        )
        row = con.execute(sql).fetchone()
        assert row is not None
        return row[0]

    assert n["q05"] == count_matches("big", "id3")
    assert n["q01"] == count_matches("small", "id1")
    # matched rows carry the right-hand columns; the unmatched rows of the left join carry NULLs
    assert eng.con is not None
    unmatched = eng.con.execute(f"SELECT count(*) FROM ({qs['q03'].sql}) WHERE v2 IS NULL").fetchone()
    assert unmatched == (ROWS - n["q02"],)


# ---------------------------------------------------------------- checksum comparison
def test_compare_checksums():
    assert h2o.compare_checksums([10, 1.0, 2.5], [10, 1.0000000001, 2.5])[0]
    ok, msg = h2o.compare_checksums([10, 1.0], [11, 1.0])
    assert not ok
    assert "!=" in msg
    ok, msg = h2o.compare_checksums([10, 1.1], [10, 1.0])
    assert not ok
    assert "[10, 1.1]" in msg
    ok, msg = h2o.compare_checksums([10], [10, 1.0])
    assert not ok
    assert "has 1 values, expected 2" in msg
    assert not h2o.compare_checksums([10, None], [10, 1.0])[0]


# ---------------------------------------------------------------- run
def test_run_records_the_checksum_then_the_timings(tiny: dict[str, Path]):
    f, failed = h2o.run_h2o(
        "duckdb", "h2o-join", tiny["h2o-join"], 0.5, iterations=2, warmup=1, engine_kwargs={"threads": 2}
    )
    assert failed == 0
    rec = json.loads(f.read_text())
    assert rec["suite"] == "h2o-join"
    assert rec["dataset"]["name"] == tiny["h2o-join"].name
    assert rec["dataset"]["scale_factor"] == 0.5
    assert rec["dataset"]["content_sha256"]
    assert rec["engine"]["name"] == "duckdb"
    assert rec["failed_queries"] == 0
    assert rec["answer_reference"] == "duckdb"
    assert sorted(rec["queries"]) == [f"q0{i}" for i in range(1, 6)]
    for e in rec["queries"].values():
        assert e["answer_check"]["status"] == "pass"
        assert len(e["times_s"]) == 2
        assert e["chk"] == e["chk_reference"]
        assert e["chk"][0] == e["rows_returned"]
        assert e["p50_s"] > 0
        assert e["min_s"] <= e["p50_s"] <= e["max_s"]
    assert f.name.endswith(f"_duckdb_{tiny['h2o-join'].name}_sf0.5.json")


def test_reference_checksums_are_cached_by_dataset_digest(tiny: dict[str, Path], monkeypatch: pytest.MonkeyPatch):
    root = tiny["h2o-groupby"]
    ds = h2o.resolve_dataset(root, 0.5)
    qs = h2o.load_queries("h2o-groupby", "duckdb", ["q01", "q02"])
    first = h2o.reference_checksums(ds, qs)
    assert (h2o.results_dir() / "h2o-ref").is_dir()

    def no_engine(*a, **k):
        raise AssertionError("the cache should have been used")

    monkeypatch.setattr(h2o, "make_engine", no_engine)
    assert h2o.reference_checksums(ds, qs) == first
    with pytest.raises(AssertionError):  # a question missing from the cache is computed
        h2o.reference_checksums(ds, h2o.load_queries("h2o-groupby", "duckdb", ["q01", "q03"]))


def test_reference_checksums_use_the_duckdb_question_not_the_engine_override(
    tiny: dict[str, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A wrong engine override must fail against the DuckDB reference, not define it."""
    d = tmp_path / "queries" / "h2o-groupby"
    (d / "wrapped").mkdir(parents=True)
    (d / "q01.sql").write_text("SELECT id1, SUM(v1) AS v1 FROM x GROUP BY id1")
    (d / "wrapped" / "q01.sql").write_text("SELECT id1, SUM(v1) + 1 AS v1 FROM x GROUP BY id1")
    monkeypatch.setattr(h2o, "queries_dir", lambda suite: tmp_path / "queries" / suite)
    rec, failed = run_with(monkeypatch, tiny["h2o-groupby"], Wrapped(), ["q01"])
    assert failed == 1
    assert "!=" in rec["queries"]["q01"]["answer_check"]["detail"]


def test_reference_cache_recomputes_when_the_question_changes(tiny: dict[str, Path]):
    ds = h2o.resolve_dataset(tiny["h2o-groupby"], 0.5)
    qs = h2o.load_queries("h2o-groupby", "duckdb", ["q01"])
    first = h2o.reference_checksums(ds, qs)
    altered = Query(
        "q01",
        "SELECT id1, SUM(v1) + 1 AS v1 FROM x GROUP BY id1",
        "h2o-groupby",
        h2o.check_sql("SELECT id1, SUM(v1) + 1 AS v1 FROM x GROUP BY id1", ["v1"]),
    )
    eng = make_engine("duckdb", threads=2)
    eng.setup(ds)
    try:
        (expected,) = eng.run(check_query(altered))
    finally:
        eng.teardown()
    assert h2o.reference_checksums(ds, [altered]) == {"q01": list(expected)}
    assert list(expected) != first["q01"]


class Wrapped(DuckDBEngine):
    """A DuckDB engine that misbehaves on chosen questions."""

    def __init__(self, wrong=(), boom=(), miscount=(), block=()):
        super().__init__(threads=2)
        self.wrong, self.boom, self.miscount, self.block = set(wrong), set(boom), set(miscount), set(block)
        self.cancelled = threading.Event()

    def run(self, query):
        if query.id in self.block:
            self.cancelled.wait()
            raise RuntimeError("cancelled")
        if query.id in self.boom:
            raise RuntimeError("engine exploded")
        rows = super().run(query)
        if query.id in self.wrong:
            return [(rows[0][0], *(v * 2 for v in rows[0][1:]))]
        return rows

    def timed_run(self, query):
        n, secs = super().timed_run(query)
        return (n + 1 if query.id in self.miscount else n), secs

    def cancel(self):
        self.cancelled.set()


def run_with(monkeypatch: pytest.MonkeyPatch, root: Path, eng: Wrapped, only: list[str], timeout_s=600):
    real = h2o.make_engine
    monkeypatch.setattr(h2o, "make_engine", lambda name, **kw: eng if name == "wrapped" else real(name, **kw))
    f, failed = h2o.run_h2o("wrapped", "h2o-groupby", root, 0.5, 1, 0, only, None, timeout_s)
    return json.loads(f.read_text()), failed


def test_a_checksum_mismatch_fails_the_question_and_it_is_never_timed(
    tiny: dict[str, Path], monkeypatch: pytest.MonkeyPatch
):
    rec, failed = run_with(monkeypatch, tiny["h2o-groupby"], Wrapped(wrong={"q02"}), ["q01", "q02", "q03"])
    assert failed == 1
    bad = rec["queries"]["q02"]
    assert bad["answer_check"]["status"] == "fail"
    assert "!=" in bad["answer_check"]["detail"]
    assert "times_s" not in bad
    assert "p50_s" not in bad
    assert rec["queries"]["q01"]["answer_check"]["status"] == "pass"
    assert "p50_s" in rec["queries"]["q03"]
    assert rec["failed_queries"] == 1


def test_an_engine_error_is_recorded_and_the_run_continues(tiny: dict[str, Path], monkeypatch: pytest.MonkeyPatch):
    rec, failed = run_with(monkeypatch, tiny["h2o-groupby"], Wrapped(boom={"q01"}), ["q01", "q02"])
    assert failed == 1
    assert "engine exploded" in rec["queries"]["q01"]["error"]
    assert "p50_s" not in rec["queries"]["q01"]
    assert "p50_s" in rec["queries"]["q02"]


def test_a_timed_run_with_another_row_count_than_the_checksum_fails(
    tiny: dict[str, Path], monkeypatch: pytest.MonkeyPatch
):
    rec, failed = run_with(monkeypatch, tiny["h2o-groupby"], Wrapped(miscount={"q01"}), ["q01"])
    assert failed == 1
    assert "timed run returned" in rec["queries"]["q01"]["error"]


def test_a_timeout_cancels_the_engine_and_fails_only_that_question(
    tiny: dict[str, Path], monkeypatch: pytest.MonkeyPatch
):
    # q02 answers from the cached reference, so the test never depends on engine timing: only q01 can time out.
    ds = h2o.resolve_dataset(tiny["h2o-groupby"], 0.5)
    refs = h2o.reference_checksums(ds, h2o.load_queries("h2o-groupby", "duckdb", ["q02"]))

    class FastQ02(Wrapped):
        """Serve q02 from the cached reference; every other question uses the blocking behaviour."""

        def run(self, query: Query) -> list[tuple]:
            if query.id == "q02":
                return [tuple(refs["q02"])]
            return super().run(query)

        def timed_run(self, query: Query) -> tuple[int, float]:
            if query.id == "q02":
                return (int(refs["q02"][0]), 0.001)
            return super().timed_run(query)

    eng = FastQ02(block={"q01"})
    rec, failed = run_with(monkeypatch, tiny["h2o-groupby"], eng, ["q01", "q02"], timeout_s=0.2)
    assert failed == 1
    assert "exceeded 0.2s" in rec["queries"]["q01"]["error"]
    assert "p50_s" in rec["queries"]["q02"]


def test_run_input_validation(tiny: dict[str, Path], tmp_path: Path):
    with pytest.raises(SystemExit, match="--iterations must be >= 1"):
        h2o.run_h2o("duckdb", "h2o-groupby", tiny["h2o-groupby"], 0.5, iterations=0)
    with pytest.raises(SystemExit, match="no dataset at"):
        h2o.run_h2o("duckdb", "h2o-groupby", tmp_path / "missing", 0.5)
    with pytest.raises(SystemExit, match="no question matches"):
        h2o.run_h2o("duckdb", "h2o-groupby", tiny["h2o-groupby"], 0.5, only=["q99"])


def test_spark_timed_run_counts_engine_side_without_collecting():
    from krishiv_bench.adapters import spark_local

    class _DF:
        def collect(self):  # pragma: no cover - must never be called here
            raise AssertionError("timed_run must not materialise rows in Python")

        def count(self):
            return 7

    class _Spark:
        def __init__(self, seen: list[str]):
            self.seen = seen

        def sql(self, sql: str):
            self.seen.append(sql)
            return _DF()

    for cls in (spark_local.SparkLocalEngine, spark_local.SailEngine):
        seen: list[str] = []
        eng = cls(threads=2)
        eng.spark = _Spark(seen)
        n, secs = eng.timed_run(Query("q01", "SELECT 1", "h2o-groupby"))
        assert (n, seen) == (7, ["SELECT 1"])
        assert secs >= 0


def test_duckdb_timed_run_materialises_only_checked_questions(tiny: dict[str, Path]):
    eng = duck(tiny["h2o-groupby"])
    q = h2o.load_queries("h2o-groupby", "duckdb", ["q01"])[0]
    n, secs = eng.timed_run(q)
    assert n == 10
    assert secs > 0
    assert eng.con is not None
    assert eng.con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'h2o_ans'").fetchone() == (0,)
    plain = Query("p", "SELECT 1 UNION ALL SELECT 2", "tpch")
    assert eng.timed_run(plain)[0] == 2


# ---------------------------------------------------------------- reproduce / ab generalise
def test_check_query_and_result_rows():
    q = Query("q", "SELECT 1", "s", "SELECT COUNT(*) FROM t")
    assert check_query(q).sql == "SELECT COUNT(*) FROM t"
    assert check_query(q).check_sql is None
    assert result_rows(q, [(7, 1.5)]) == 7
    plain = Query("q", "SELECT 1", "s")
    assert check_query(plain) is plain
    assert result_rows(plain, [(1,), (2,)]) == 2


def test_reproduce_correctness_compares_checksums(tiny: dict[str, Path]):
    root = tiny["h2o-groupby"]
    ds = h2o.resolve_dataset(root, 0.5)
    engines = {n: make_engine(n, threads=2) for n in ("duckdb", "polars")}
    for e in engines.values():
        e.setup(ds)
    q = h2o.load_queries("h2o-groupby", "duckdb", ["q03"])[0]
    verdict, nrows = reproduce.correctness({n: q for n in engines}, engines)
    assert {n: v["status"] for n, v in verdict.items()} == {"duckdb": "ok", "polars": "ok"}
    assert nrows == duck(root).timed_run(q)[0]


def test_reproduce_resolves_h2o_dataset_directories(tiny: dict[str, Path], data_root: Path):
    rel = f"h2o/v1/{tiny['h2o-groupby'].name}"
    (data_root / "h2o" / "v1").mkdir(parents=True)
    (data_root / rel).symlink_to(tiny["h2o-groupby"])
    ds = reproduce.resolve_dataset("h2o-groupby", rel, 5.0)
    assert ds.root == data_root / rel
    assert ds.tables == ["x"]
    assert ds.sf == 5.0
    assert reproduce.PUBLISHED["h2o-join"]["runs"] == 2


def test_ab_one_query_checks_then_times_h2o_questions(tiny: dict[str, Path]):
    root = tiny["h2o-groupby"]
    q = h2o.load_queries("h2o-groupby", "duckdb", ["q04"])[0]
    entry = ab.one_query(q, duck(root), duck(root), 1)
    assert entry["status"] == "ok"
    assert entry["rows"] == 10


# ---------------------------------------------------------------- CLI, datasets, report
@pytest.fixture
def small_sizes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(h2o.SIZES, "0.5gb", ROWS)


def test_dataset_root_names_follow_the_upstream_labels(data_root: Path):
    assert h2o.dataset_root("h2o-groupby", "5gb") == data_root / "h2o" / "v1" / "G1_1e8_1e2_0_0"
    assert h2o.dataset_root("h2o-groupby", "50gb", "v2", 10, 5, True) == data_root / "h2o" / "v2" / "G1_1e9_1e1_5_1"
    assert h2o.dataset_root("h2o-join", "0.5gb", na_pct=5) == data_root / "h2o" / "v1" / "J1_1e7_NA_5_0"
    with pytest.raises(ValueError, match="unknown size 'huge'"):
        h2o.dataset_root("h2o-join", "huge")


def test_datasets_cli_generates_verifies_and_refuses_to_overwrite(small_sizes, data_root: Path, capsys):
    datasets.main(["h2o-groupby", "--size", "0.5gb", "--k", "10", "--na", "5"])
    d = data_root / "h2o" / "v1" / "G1_1e4_1e1_5_0"
    assert (d / "x.parquet").is_file()
    assert "content sha256" in capsys.readouterr().out
    datasets.main(["h2o-verify", str(d)])
    assert "verified" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="already exists"):
        datasets.main(["h2o-groupby", "--size", "0.5gb", "--k", "10", "--na", "5"])
    datasets.main(["h2o-groupby", "--size", "0.5gb", "--k", "10", "--na", "5", "--h2o-version", "v2"])
    assert (data_root / "h2o" / "v2" / "G1_1e4_1e1_5_0" / "x.parquet").is_file()
    (d / "x.parquet").write_bytes(b"")
    with pytest.raises(SystemExit, match=r"x: cannot read .*0 bytes"):
        datasets.main(["h2o-verify", str(d)])
    with pytest.raises(SystemExit, match="multiple of k"):
        datasets.main(["h2o-groupby", "--size", "0.5gb", "--k", "7"])


def test_cli_run_and_report_end_to_end(small_sizes, data_root: Path, capsys):
    # the join generator's divisors are fixed for the real sizes, so the tiny table is built directly
    hd.generate("join", hd.J1Params(ROWS, **TINY_J1), data_root / "h2o" / "v1")
    for engine in ("duckdb", "polars"):
        cli.main(
            [
                "run",
                "--engine",
                engine,
                "--suite",
                "h2o-join",
                "--size",
                "0.5gb",
                "--iterations",
                "1",
                "--warmup",
                "0",
                "--duckdb-threads",
                "2",
            ]
        )
    assert "result:" in capsys.readouterr().out
    cli.main(["report", "--suite", "h2o-join", "--size", "0.5gb", "--baseline", "duckdb"])
    out = capsys.readouterr().out.strip().split("report: ")[1]
    md = Path(out).read_text()
    assert "| question | duckdb p50 ms | polars p50 ms | polars/duckdb |" in md
    assert "| q05 |" in md
    assert "every engine's answer checksum equals the DuckDB reference" in md
    assert "Geomean polars" in md


def test_cli_requires_a_size_for_h2o_suites(capsys):
    for argv in (
        ["run", "--engine", "duckdb", "--suite", "h2o-join"],
        ["report", "--suite", "h2o-groupby"],
        ["reproduce", "--suite", "h2o-join"],
        ["ab", "--suite", "h2o-join", "--a-runner", "a", "--b-runner", "b"],
    ):
        with pytest.raises(SystemExit) as e:
            cli.main(argv)
        assert e.value.code == 2
        assert "needs --size" in capsys.readouterr().err


def test_cli_run_exits_non_zero_when_a_question_fails(small_sizes, data_root: Path, monkeypatch: pytest.MonkeyPatch):
    hd.generate("groupby", hd.G1Params(ROWS, 100), data_root / "h2o" / "v1")
    monkeypatch.setattr(h2o, "run_h2o", lambda *a, **k: (Path("r.json"), 2))
    with pytest.raises(SystemExit, match=r"2 question\(s\) failed"):
        cli.main(["run", "--engine", "duckdb", "--suite", "h2o-groupby", "--size", "0.5gb"])


def test_h2o_target_applies_variant_flags_and_dataset_override(small_sizes, data_root: Path):
    import argparse

    ap = argparse.ArgumentParser()
    ns = argparse.Namespace(suite="h2o-groupby", size="0.5gb", h2o_version="v1", k=10, na=5, sorted=True, dataset=None)
    root, rel, gb = cli.h2o_target(ap, ns)
    assert (root.name, rel, gb) == ("G1_1e4_1e1_5_1", "h2o/v1/G1_1e4_1e1_5_1", 0.5)
    ns.dataset = "G1_custom"
    assert cli.h2o_target(ap, ns)[1] == "h2o/v1/G1_custom"
    ns.size = "huge"
    with pytest.raises(SystemExit):
        cli.h2o_target(ap, ns)


def test_render_lists_failures_and_ratios():
    def res(name, p50s, failed=()):
        qs = {q: {"p50_s": t} for q, t in p50s.items()}
        for q in failed:
            qs[q] = {"answer_check": {"status": "fail", "detail": "checksum [1] != reference [2]"}}
        return {
            "suite": "h2o-groupby",
            "iterations": 3,
            "warmup": 1,
            "dataset": {"name": "G1_1e7_1e2_0_0", "content_sha256": "ab" * 32, "generator": "g"},
            "engine": {"name": name, "version": "1"},
            "queries": qs,
            "_file": f"{name}.json",
        }

    md = h2o.render(
        {"duckdb": res("duckdb", {"q01": 0.1, "q02": 0.2}), "krishiv": res("krishiv", {"q01": 0.05}, failed=["q02"])},
        "duckdb",
    )
    assert "| q01 | 100.0 | 50.0 | 0.50x |" in md
    assert "| q02 | 200.0 | FAIL | - |" in md
    assert "- `q02` **krishiv**: checksum [1] != reference [2]" in md
    assert "Geomean krishiv: **0.50x** over 1 questions" in md
    with pytest.raises(SystemExit, match="no results"):
        h2o.render({})


def test_render_lists_a_missing_question_as_skipped_not_all_agree():
    def res(name, p50s):
        return {
            "suite": "h2o-groupby",
            "iterations": 3,
            "warmup": 1,
            "dataset": {"name": "G1_1e7_1e2_0_0", "content_sha256": "ab" * 32, "generator": "g"},
            "engine": {"name": name, "version": "1"},
            "queries": {q: {"p50_s": t} for q, t in p50s.items()},
            "_file": f"{name}.json",
        }

    md = h2o.render(
        {"duckdb": res("duckdb", {"q01": 0.1, "q02": 0.2}), "krishiv": res("krishiv", {"q01": 0.05})},
        "duckdb",
    )
    assert "| q02 | 200.0 | - | - |" in md
    assert "no result" in md
    assert "every engine's answer checksum equals the DuckDB reference" not in md


def test_dataset_provenance_without_a_manifest(tmp_path: Path):
    assert h2o.dataset_provenance(tmp_path)["content_sha256"] is None


def test_resolve_dataset_lists_tables(tiny: dict[str, Path]):
    ds = h2o.resolve_dataset(tiny["h2o-join"], 5.0)
    assert isinstance(ds, Dataset)
    assert ds.tables == ["big", "medium", "small", "x"]
    assert ds.sf == 5.0
