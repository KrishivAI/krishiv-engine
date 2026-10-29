"""Query and data minimisation: text scanning, reductions, ddmin, the oracle and the case writer."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

from krishiv_bench import minimize as mz
from krishiv_bench.adapters.base import Query
from krishiv_bench.adapters.krishiv import RunnerCrashed, RunnerError


def test_mask_blanks_strings_identifiers_and_comments():
    sql = "select 'a''b (' , \"x)\" -- tail (\nfrom /* ( */ t"
    m = mz.mask(sql)
    assert len(m) == len(sql)
    assert "(" not in m
    assert m.count(")") == 0
    assert "select" in m
    assert "from" in m


def test_depths_and_top_level_matches():
    m = mz.mask("select (select 1 from u where a = 1) from t where b = 2")
    hits = mz.top_level(m, "where")
    assert len(hits) == 1
    assert m[hits[0].start() :].startswith("where b")


def test_split_top_respects_parens_and_strings():
    assert mz.split_top("a, f(b, c), 'x,y'", ",") == ["a", "f(b, c)", "'x,y'"]
    assert mz.split_top("a = 1 and (b = 2 and c = 3) AND d = 'and'", "and") == [
        "a = 1",
        "(b = 2 and c = 3)",
        "d = 'and'",
    ]


def cands(sql: str) -> list[str]:
    return list(mz.candidates(sql))


def test_candidates_drop_clauses_and_items():
    sql = "select distinct a, b from t where a > 1 and b < 5 group by a, b having count(*) > 1 order by a limit 3"
    out = cands(sql)
    assert "select distinct a, b from t where a > 1 and b < 5 group by a, b having count(*) > 1 order by a" in out
    norm = [" ".join(c.split()) for c in out]
    assert any("order by" not in c and "limit 3" in c for c in norm)
    assert any("limit" not in c and "order by a" in c for c in norm)
    assert any("having" not in c and "group by a, b" in c for c in norm)
    assert any(" where " not in c and "group by a, b" in c for c in norm)
    assert any(c.lower().startswith("select a, b from") for c in out)  # distinct dropped
    assert any("select distinct b from" in c for c in norm)
    assert any("where b < 5" in c for c in norm)
    assert any("where a > 1 group by" in c for c in norm)
    assert any("group by b" in c for c in norm)


def test_candidates_joins_from_items_setops_and_subqueries():
    j = cands("select a from t join u on t.k = u.k left join v on v.k = t.k where a > 1")
    assert any("join u" not in c and "join v" in c for c in j)
    assert any("join v" not in c and "join u" in c for c in j)
    c = cands("select a from t, u, v")
    assert any(x.replace("  ", " ").endswith("from u, v") or " u, v" in x for x in c)
    s = cands("select a from t union all select b from u")
    assert "select a from t" in s
    assert "select b from u" in s
    q = cands("select a from t where a in (select k from u where k > 1 and k < 9)")
    assert any("NULL" in x for x in q)
    assert any("k > 1" in x and "k < 9" not in x and "(select k from u" in x for x in q)


def test_ddmin_finds_the_failure_inducing_pair():
    items = list(range(20))
    assert sorted(mz.ddmin(items, lambda s: 3 in s and 11 in s)) == [3, 11]
    assert mz.ddmin(list(range(8)), lambda s: 5 in s) == [5]
    assert mz.ddmin([1], lambda s: True) == []  # even a single row can go when nothing is needed
    assert mz.ddmin([1, 2], lambda s: False) == [1, 2]


def test_shrink_query_reaches_a_fixpoint():
    sql = "select a, b, c from t where a > 1 and b < 5 and c = 3 order by a limit 7"
    # The "bug" shows whenever the query still reads column b through the where clause.
    shrunk = mz.shrink_query(sql, lambda s: "b < 5" in s)
    assert "b < 5" in shrunk
    assert "limit" not in shrunk.lower()
    assert "order by" not in shrunk.lower()
    assert "a > 1" not in shrunk
    assert "c = 3" not in shrunk
    assert len(shrunk) < len(sql)
    assert mz.shrink_query(sql, lambda s: False, lambda: True) == sql
    assert mz.shrink_query(sql, lambda s: True, lambda: False) == sql  # no budget: unchanged


def test_referenced_tables_ignores_strings():
    assert mz.referenced_tables("select * from a join b on 1 where x = 'c'", ["a", "b", "c"]) == ["a", "b"]


def _table(n: int) -> pa.Table:
    return pa.table({"k": list(range(n)), "v": [f"s{i}" for i in range(n)]})


def test_slices():
    t = _table(100)
    assert mz.seed_slice(t, 10).num_rows == 10
    assert mz.seed_slice(t, 200) is t
    s = mz.strided_slice(t, 5)
    assert s.column("k").to_pylist() == [0, 20, 40, 60, 80]
    assert mz.strided_slice(t, 200) is t


def test_shrink_rows_keeps_only_the_rows_that_matter():
    tables = {"a": _table(40), "b": _table(10)}

    def interesting(t):
        return 7 in t["a"].column("k").to_pylist() and 3 in t["b"].column("k").to_pylist()

    out = mz.shrink_rows(tables, interesting)
    assert out["a"].column("k").to_pylist() == [7]
    assert out["b"].column("k").to_pylist() == [3]


class FakeEngine:
    """Answers ``select count(*) from t`` and a few other shapes from the tables it was set up with."""

    def __init__(self, bug: bool = False, bug_row: int = 13):
        self.bug = bug
        self.bug_row = bug_row
        self.tables: dict[str, list] = {}
        self.setups = 0

    def setup(self, ds):
        import pyarrow.parquet as pq

        self.setups += 1
        self.tables = {t: pq.read_table(ds.root / f"{t}.parquet").column("k").to_pylist() for t in ds.tables}

    def teardown(self):
        self.tables = {}

    def run(self, q):
        if "boom" in q.sql:
            raise RunnerError("planning failed") if self.bug else ValueError("reference fails")
        if "crash" in q.sql:
            raise RunnerCrashed("died")
        ks = self.tables["t"]
        if "where" in q.sql:
            ks = [k for k in ks if k > 2]
        n = len(ks)
        if self.bug and self.bug_row in self.tables["t"]:  # wrong answer only when row 13 is present
            n += 1
        return [(n,)]


def _oracle(tmp_path: Path, max_runs: int = 1000) -> mz.Oracle:
    return mz.Oracle(FakeEngine(), FakeEngine(bug=True), tmp_path / "d", max_runs)


def test_oracle_outcomes(tmp_path: Path):
    o = _oracle(tmp_path)
    t = {"t": _table(20)}
    assert o.differs("select count(*) from t", t)
    assert not o.differs("select count(*) from t", {"t": _table(10)})
    out = o.outcome("select count(*) from t where k > 0", t)
    assert out["reference"] == [(17,)]
    assert out["krishiv"] == [(18,)]
    boom = o.outcome("select boom", t)
    assert boom["differs"] is False  # the reference failed
    assert o.runs == 4

    # A Krishiv-only error is a difference; a crash too.
    class Err(FakeEngine):
        def run(self, q):
            raise RunnerError("unsupported")

    o2 = mz.Oracle(FakeEngine(), Err(), tmp_path / "d2")
    res = o2.outcome("select count(*) from t", t)
    assert res["differs"] is True
    assert res["krishiv"].startswith("error:")
    assert o.within_budget()
    assert not mz.Oracle(FakeEngine(), FakeEngine(), tmp_path / "d3", 0).within_budget()


def test_oracle_reloads_data_only_when_it_changes(tmp_path: Path):
    o = _oracle(tmp_path)
    t = {"t": _table(20)}
    o.differs("select count(*) from t", t)
    o.differs("select count(*) from t where k > 1", t)
    assert o.krishiv.setups == 1
    o.differs("select count(*) from t", {"t": _table(15)})
    assert o.krishiv.setups == 2


def test_minimize_reduces_query_and_data(tmp_path: Path):
    o = _oracle(tmp_path)
    sql = "select count(*) from t where k > 1 and k < 100 order by 1 limit 5"
    small_sql, small = mz.minimize(sql, {"t": _table(5000), "unused": _table(5)}, o, seed_rows=50)
    assert set(small) == {"t"}
    assert small["t"].column("k").to_pylist() == [13]
    assert "limit" not in small_sql.lower()
    assert "order by" not in small_sql.lower()


def test_minimize_errors(tmp_path: Path):
    o = mz.Oracle(FakeEngine(), FakeEngine(bug=True, bug_row=4999), tmp_path / "e")
    with pytest.raises(ValueError, match="does not reproduce"):
        mz.minimize("select count(*) from t", {"t": _table(5000)}, o, seed_rows=1, whole_rows=10)
    full = mz.Oracle(FakeEngine(), FakeEngine(), tmp_path / "f")
    with pytest.raises(ValueError, match="full data"):
        mz.minimize("select count(*) from t", {"t": _table(20)}, full, full_data=True)


def test_first_slice_escalates_until_the_difference_reproduces(tmp_path: Path):
    # The bug row sits at index 450: the 5- and 50-row slices (head or strided) miss it, the 500-row head has it.
    o = mz.Oracle(FakeEngine(), FakeEngine(bug=True, bug_row=450), tmp_path / "g")
    out = mz._first_slice("select count(*) from t", {"t": _table(5000)}, o, 5, 10)
    assert out["t"].num_rows == 500
    # A small table stays whole next to a sliced one.
    o2 = mz.Oracle(FakeEngine(), FakeEngine(bug=True, bug_row=3), tmp_path / "h")
    both = mz._first_slice("select count(*) from t", {"t": _table(5000), "dim": _table(8)}, o2, 5, 10)
    assert both["dim"].num_rows == 8
    assert both["t"].num_rows == 5


def test_minimize_full_data_shrinks_only_the_query(tmp_path: Path):
    o = _oracle(tmp_path)
    sql = "select count(*) from t where k > 1 and k < 100"
    out_sql, small = mz.minimize(sql, {"t": _table(30)}, o, full_data=True)
    assert small["t"].num_rows == 30
    assert "where" not in out_sql.lower()


def test_sql_literal_and_setup_sql():
    assert [mz.sql_literal(v) for v in (None, True, 3, 1.5, "o'k")] == ["NULL", "true", "3", "1.5", "'o''k'"]
    s = mz.setup_sql({"t": pa.table({"k": [1, 2], "v": ["a", None]})})
    assert "CREATE TEMPORARY VIEW t AS SELECT * FROM VALUES" in s
    assert "(1, 'a')" in s
    assert "(2, NULL)" in s
    assert "AS t(k, v);" in s
    empty = mz.setup_sql({"e": pa.table({"k": pa.array([], pa.int64())})})
    assert "WHERE false" in empty


def test_write_case(tmp_path: Path):
    outcome = {"differs": True, "reference": [(1, "a")], "krishiv": "error: boom"}
    out = mz.write_case(tmp_path / "case", "select 1", {"t": _table(3)}, outcome, "note")
    assert (out / "query.sql").read_text() == "select 1;\n"
    assert (out / "data" / "t.parquet").is_file()
    assert json.loads((out / "expected.json").read_text()) == [[1, "a"]]
    assert json.loads((out / "actual.json").read_text()) == "error: boom"
    assert (out / "README.txt").read_text() == "note\n"
    assert "VALUES" in (out / "setup.sql").read_text()


def test_run_minimize_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import pyarrow.parquet as pq

    from krishiv_bench import adapters
    from krishiv_bench.adapters.base import Query

    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    root = tmp_path / "tpch-duckdb" / "sf1"
    root.mkdir(parents=True)
    pq.write_table(_table(300), root / "t.parquet")
    monkeypatch.setattr("krishiv_bench.paths.dataset_dir", lambda name, sf: root)
    monkeypatch.setattr(
        "krishiv_bench.runner.load_queries",
        lambda s, e, only=None: [Query("q1", "select count(*) from t where k > 1", "tpch")],
    )
    monkeypatch.setattr(adapters, "make_engine", lambda name, **kw: FakeEngine())
    monkeypatch.setattr(mz, "KrishivEngine", lambda **kw: FakeEngine(bug=True))
    out = mz.run_minimize("tpch", "q1", 1, seed_rows=20, out_dir=str(tmp_path / "case"))
    assert (out / "query.sql").is_file()
    assert json.loads((out / "expected.json").read_text())
    monkeypatch.setattr(mz, "KrishivEngine", lambda **kw: FakeEngine())
    with pytest.raises(SystemExit, match="reproduce"):
        mz.run_minimize("tpch", "q1", 1, seed_rows=20, out_dir=str(tmp_path / "case2"))


def test_where_conjuncts_are_halved_before_single_removal():
    sql = "select a from t where a = 1 and b = 2 and c = 3 and d = 4"
    norm = [" ".join(c.split()) for c in mz.candidates(sql)]
    assert "select a from t where a = 1 AND b = 2" in norm
    assert "select a from t where c = 3 AND d = 4" in norm


def test_shrink_query_skips_empty_and_longer_candidates_and_stops_without_budget(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mz, "candidates", lambda s: iter(["", "select a from a_much_longer_table_name"]))
    assert mz.shrink_query("select a from t", lambda s: True) == "select a from t"
    calls = iter([True, False, False])
    monkeypatch.setattr(mz, "candidates", lambda s: iter(["select 1"]))
    assert mz.shrink_query("select a from t", lambda s: True, lambda: next(calls)) == "select a from t"


def test_run_minimize_without_a_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    monkeypatch.setattr("krishiv_bench.paths.dataset_dir", lambda name, sf: tmp_path / "none")
    with pytest.raises(SystemExit, match="no dataset"):
        mz.run_minimize("tpch", "q1", 1)


def test_krishiv_engine_can_be_set_up_again_after_teardown(tmp_path: Path):
    import stat

    from test_abba import FAKE_RUNNER

    from krishiv_bench.adapters.base import Dataset
    from krishiv_bench.adapters.krishiv import KrishivEngine

    binary = tmp_path / "runner"
    binary.write_text(FAKE_RUNNER)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    eng = KrishivEngine(threads=1, memory_limit="1g", binary=binary)
    eng.timeout_s = 30
    ds = Dataset("t", 0, tmp_path, ["t"])
    eng.setup(ds)
    eng.teardown()
    eng.setup(ds)
    assert eng.timed_run(Query("q", "select 1", "t"))[0] == 2
    eng.teardown()
