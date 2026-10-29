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

    timeout_s: float | None = None

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

    def cancel(self):
        pass

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
    assert "(CAST(1 AS BIGINT), 'a')" in s
    assert "(CAST(2 AS BIGINT), NULL)" in s
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
    assert (out / "query.sql").read_text() == "select count(*) from t;\n"
    assert json.loads((out / "expected.json").read_text()) == [[1]]
    assert json.loads((out / "actual.json").read_text()) == [[2]]
    import pyarrow.parquet as pq2

    assert pq2.read_table(out / "data" / "t.parquet").column("k").to_pylist() == [13]
    monkeypatch.setattr(mz, "KrishivEngine", lambda **kw: FakeEngine())
    with pytest.raises(SystemExit, match="does not differ"):
        mz.run_minimize("tpch", "q1", 1, seed_rows=20, out_dir=str(tmp_path / "case2"))
    with pytest.raises(SystemExit, match="reproduce"):
        mz.run_minimize("tpch", "q1", 1, seed_rows=20, out_dir=str(tmp_path / "case3"), pin=False, whole_rows=10)


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


def test_signature_distinguishes_error_classes_from_wrong_rows():
    assert mz.signature({"krishiv": [(1,)]}) == "rows"
    assert mz.signature({"krishiv": [(1,)], "order_only": True}) == "order"
    assert mz.signature({"krishiv": "error: RunnerError: [DIVIDE_BY_ZERO] Division by zero"}) == "error:DIVIDE_BY_ZERO"
    assert mz.signature({"krishiv": "error: something broke"}) == "error"


def test_pin_keeps_the_reduction_on_the_original_difference(tmp_path: Path):
    import pyarrow.parquet as pq

    from krishiv_bench.adapters.base import Dataset

    class Zero(FakeEngine):
        """Krishiv fails with a division error on a tiny table, answers wrongly on a big one."""

        def run(self, q):
            n = len(self.tables["t"])
            if self.bug and n < 20:
                raise RunnerError("[DIVIDE_BY_ZERO] Division by zero")
            return [(n + (1 if self.bug else 0),)]

    root = tmp_path / "full"
    root.mkdir()
    pq.write_table(_table(40), root / "t.parquet")
    pinned = mz.Oracle(FakeEngine(), Zero(bug=True), tmp_path / "p")
    out = pinned.pin(Dataset("d", 1, root, ["t"]), "select count(*) from t")
    assert out["krishiv"] == [(41,)]
    assert pinned.differs("select count(*) from t", {"t": _table(30)})
    assert not pinned.differs("select count(*) from t", {"t": _table(5)})  # a different kind of failure
    loose = mz.Oracle(FakeEngine(), Zero(bug=True), tmp_path / "q")
    assert loose.differs("select count(*) from t", {"t": _table(5)})  # unpinned accepts it
    equal = mz.Oracle(FakeEngine(), FakeEngine(), tmp_path / "r")
    with pytest.raises(ValueError, match="does not differ"):
        equal.pin(Dataset("d", 1, root, ["t"]), "select count(*) from t")


def test_load_tables_reads_small_tables_whole_and_pools_big_ones(tmp_path: Path):
    import pyarrow.parquet as pq

    pq.write_table(_table(50), tmp_path / "dim.parquet")
    pq.write_table(_table(10000), tmp_path / "fact.parquet", row_group_size=500)
    parts = tmp_path / "parts"
    parts.mkdir()
    pq.write_table(_table(30), parts / "a.parquet")
    pq.write_table(_table(30), parts / "b.parquet")
    (tmp_path / "wide").mkdir()
    pq.write_table(_table(3000), tmp_path / "wide" / "x.parquet")
    tables, big = mz.load_tables(tmp_path, ["dim", "fact", "parts", "wide"], whole_rows=1000, pool_rows=1500)
    assert tables["dim"].num_rows == 50
    assert tables["parts"].num_rows == 60
    assert big == {"fact", "wide"}
    assert tables["fact"].num_rows == 1500  # three evenly spaced row groups of 500
    ks = tables["fact"].column("k").to_pylist()
    assert ks[0] == 0
    assert ks[-1] > 5000  # spread over the file, not the head
    assert tables["wide"].num_rows == 3000


def test_a_pure_reordering_has_the_order_signature(tmp_path: Path):
    class Rev(FakeEngine):
        def run(self, q):
            return [(2,), (1,)] if self.bug else [(1,), (2,)]

    o = mz.Oracle(Rev(), Rev(bug=True), tmp_path / "o")
    out = o.outcome("select k from t order by k", {"t": _table(3)})
    assert out["differs"] is True
    assert out["order_only"] is True
    assert mz.signature(out) == "order"
    unordered = o.outcome("select k from t", {"t": _table(3)})
    assert unordered["differs"] is False


def test_the_time_budget_stops_the_shrink_and_keeps_the_best_case(tmp_path: Path):
    now = [0.0]

    def clock() -> float:
        now[0] += 1.0  # every look at the clock costs a second
        return now[0]

    o = mz.Oracle(FakeEngine(), FakeEngine(bug=True), tmp_path / "t", max_runs=10_000, max_seconds=12, clock=clock)
    sql, small = mz.minimize("select count(*) from t where k > 1 and k < 100", {"t": _table(5000)}, o, seed_rows=50)
    assert not o.within_budget()
    assert small["t"].num_rows > 1  # the best so far, not minimal
    assert "select count(*) from t" in sql


def test_a_timed_out_run_is_not_interesting_and_the_pin_uses_a_long_timeout(tmp_path: Path):
    import pyarrow.parquet as pq

    from krishiv_bench.adapters.base import Dataset

    seen: list[float | None] = []

    class Slow(FakeEngine):
        timeout_s = None

        def run(self, q):
            seen.append(self.timeout_s)
            if "slow" in q.sql:
                raise RunnerCrashed("no answer within 30s")
            if "died" in q.sql:
                raise RunnerCrashed("runner exited")
            return super().run(q)

    o = mz.Oracle(FakeEngine(), Slow(bug=True), tmp_path / "s", run_timeout=7)
    assert o.krishiv.timeout_s == 7
    out = o.outcome("select slow from t", {"t": _table(20)})
    assert out["differs"] is False
    assert out["krishiv"].startswith("timeout")
    assert o.outcome("select died from t", {"t": _table(20)})["differs"] is True
    root = tmp_path / "full"
    root.mkdir()
    pq.write_table(_table(20), root / "t.parquet")
    o.pin(Dataset("d", 1, root, ["t"]), "select count(*) from t")
    assert seen[-1] == mz.PIN_TIMEOUT
    assert o.krishiv.timeout_s == 7


def test_shrink_query_does_not_rerun_a_candidate():
    calls: list[str] = []

    def interesting(sql: str) -> bool:
        calls.append(sql)
        return "b < 5" in sql

    mz.shrink_query("select a, b from t where a > 1 and b < 5 order by a limit 7", interesting)
    assert len(calls) == len(set(calls))


def test_tune_timeout_bounds_candidates_by_the_reproducing_run(tmp_path: Path):
    now = [0.0]

    def clock() -> float:
        now[0] += 4.0  # a run takes 4 "seconds" (two clock reads)
        return now[0]

    o = mz.Oracle(FakeEngine(), FakeEngine(bug=True), tmp_path / "u", run_timeout=100, clock=clock)
    o.differs("select count(*) from t", {"t": _table(20)})
    assert o.last_secs > 0
    assert o.last_secs == 4.0
    o.tune_timeout(factor=2, floor=1)
    assert o.run_timeout == 8.0  # twice the reproducing run
    assert o.krishiv.timeout_s == 8.0
    o.last_secs = 0.1
    o.tune_timeout(factor=2, floor=3)
    assert o.run_timeout == 3.0  # the floor
    o.last_secs = 1000
    o.tune_timeout()
    assert o.run_timeout == 3.0  # tuning never raises the timeout


def test_load_rewrites_only_the_tables_that_changed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    written: list[str] = []
    real = mz.pq.write_table
    monkeypatch.setattr(
        mz.pq, "write_table", lambda t, where, *a, **k: (written.append(Path(where).name), real(t, where, *a, **k))[1]
    )
    o = _oracle(tmp_path)
    a, b = _table(20), _table(30)
    o._load({"t": a, "u": b})
    assert sorted(written) == ["t.parquet", "u.parquet"]
    written.clear()
    # Same size and same first rows, different later rows: still a change.
    changed = pa.table({"k": [*range(19), 999], "v": [f"s{i}" for i in range(20)]})
    o._load({"t": changed, "u": b})
    assert o.krishiv.setups == 2
    assert written == ["t.parquet"]  # `u` was not rewritten
    o._load({"t": changed, "u": b})
    assert o.krishiv.setups == 2
    o._load({"t": changed})
    assert not (tmp_path / "d" / "u.parquet").exists()
    assert o.krishiv.tables["t"][-1] == 999


def test_ddmin_stops_when_told_to():
    calls = []

    def test(sub):
        calls.append(len(sub))
        return False

    items = list(range(64))
    assert mz.ddmin(items, test, stop=lambda: len(calls) >= 3) == items
    assert 3 <= len(calls) <= 4  # checked before each subset
    assert mz.ddmin(items, test, stop=lambda: True) == items
    out = mz.shrink_rows({"t": _table(10)}, lambda t: True, stop=lambda: True)
    assert out["t"].num_rows == 10


def test_split_conjuncts_keeps_a_between_whole():
    parts = mz.split_conjuncts(
        "a between 1 and 5 and b = 'x and y' and (c between 2 and 3 and d) and e BETWEEN 1 AND 2"
    )
    assert parts == ["a between 1 and 5", "b = 'x and y'", "(c between 2 and 3 and d)", "e BETWEEN 1 AND 2"]
    out = [" ".join(c.split()) for c in mz.candidates("select a from t where a between 1 and 5 and b = 2")]
    assert "select a from t where b = 2" in out
    assert "select a from t where a between 1 and 5" in out
    assert not any(c[-17:] == "where a between 1" for c in out)


def test_distinct_is_dropped_at_the_masked_offset():
    out = mz.candidates("select distinct a, 'select distinct' from t")
    assert "select a, 'select distinct' from t" in out


def test_refuses_to_delete_an_unrelated_output_directory(tmp_path: Path):
    mz.check_case_dir(tmp_path / "new")  # does not exist: fine
    empty = tmp_path / "empty"
    empty.mkdir()
    mz.check_case_dir(empty)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "precious.txt").write_text("x")
    with pytest.raises(SystemExit, match="refusing"):
        mz.check_case_dir(busy)
    with pytest.raises(SystemExit, match="refusing"):
        mz.write_case(busy, "select 1", {"t": _table(2)}, {"reference": [], "krishiv": []}, "n")
    assert (busy / "precious.txt").exists()
    afile = tmp_path / "file"
    afile.write_text("x")
    with pytest.raises(SystemExit, match="refusing"):
        mz.check_case_dir(afile)
    # A directory with a README.txt and a query.sql of someone's own project is not a previous case.
    project = tmp_path / "project"
    project.mkdir()
    (project / "README.txt").write_text("mine")
    (project / "query.sql").write_text("select 1")
    with pytest.raises(SystemExit, match="refusing"):
        mz.check_case_dir(project)
    case = mz.write_case(tmp_path / "case", "select 1", {"t": _table(2)}, {"reference": [], "krishiv": []}, "n")
    assert (case / mz.CASE_MARKER).is_file()
    mz.write_case(
        case, "select 2", {"t": _table(2)}, {"reference": [], "krishiv": []}, "n"
    )  # a previous case: replaced
    assert (case / "query.sql").read_text() == "select 2;\n"


def test_pool_samples_row_groups_across_all_files(tmp_path: Path):
    import pyarrow.parquet as pq

    parts = tmp_path / "hits"
    parts.mkdir()
    for k in range(4):
        pq.write_table(
            pa.table({"k": list(range(k * 1000, k * 1000 + 1000))}), parts / f"p{k}.parquet", row_group_size=250
        )
    tables, big = mz.load_tables(tmp_path, ["hits"], whole_rows=500, pool_rows=1000)
    assert big == {"hits"}
    ks = tables["hits"].column("k").to_pylist()
    assert len(ks) == 1000
    assert {k // 1000 for k in ks} == {0, 1, 2, 3}  # every file contributes, not just the first


def test_run_minimize_aborts_when_the_final_run_no_longer_differs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import pyarrow.parquet as pq

    from krishiv_bench import adapters
    from krishiv_bench.adapters.base import Query

    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    root = tmp_path / "tpch-duckdb" / "sf1"
    root.mkdir(parents=True)
    pq.write_table(_table(30), root / "t.parquet")
    monkeypatch.setattr("krishiv_bench.paths.dataset_dir", lambda name, sf: root)
    monkeypatch.setattr(
        "krishiv_bench.runner.load_queries", lambda s, e, only=None: [Query("q1", "select count(*) from t", "tpch")]
    )
    monkeypatch.setattr(adapters, "make_engine", lambda name, **kw: FakeEngine())

    class Flaky(FakeEngine):
        """Buggy until healed, then it answers like the reference."""

        healed = False

        def run(self, q):
            if Flaky.healed:
                self.bug = False
            return super().run(q)

    monkeypatch.setattr(mz, "KrishivEngine", lambda **kw: Flaky(bug=True))
    orig = mz.minimize

    def minimize_then_heal(*a, **k):
        out = orig(*a, **k)
        Flaky.healed = True
        return out

    monkeypatch.setattr(mz, "minimize", minimize_then_heal)
    with pytest.raises(SystemExit, match="no longer reproduces"):
        mz.run_minimize("tpch", "q1", 1, seed_rows=20, out_dir=str(tmp_path / "flaky"))
    assert not (tmp_path / "flaky").exists()
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x").write_text("x")
    with pytest.raises(SystemExit, match="refusing"):
        mz.run_minimize("tpch", "q1", 1, out_dir=str(busy))


def test_setup_sql_keeps_the_column_types():
    import datetime as dt
    import decimal

    t = pa.table(
        {
            "i": pa.array([1, None], pa.int32()),
            "d": pa.array([dt.date(2020, 1, 2), None], pa.date32()),
            "ts": pa.array([dt.datetime(2020, 1, 2, 3, 4, 5), None], pa.timestamp("us")),
            "tz": pa.array([dt.datetime(2020, 1, 2, 3, 4, 5, tzinfo=dt.UTC), None], pa.timestamp("us", "UTC")),
            "n": pa.array([decimal.Decimal("1.50"), None], pa.decimal128(5, 2)),
            "f": pa.array([1.5, None], pa.float64()),
            "b": pa.array([b"\x01\xff", None], pa.binary()),
            "s": pa.array(["x", None], pa.string()),
            "ok": pa.array([True, None], pa.bool_()),
        }
    )
    sql = mz.setup_sql({"t": t})
    assert "CAST(1 AS INT)" in sql
    assert "CAST(NULL AS INT)" in sql
    assert "CAST('2020-01-02' AS DATE)" in sql
    assert "CAST('2020-01-02 03:04:05' AS TIMESTAMP_NTZ)" in sql
    assert "AS TIMESTAMP)" in sql
    assert "CAST('1.50' AS DECIMAL(5,2))" in sql
    assert "CAST(1.5 AS DOUBLE)" in sql
    assert "X'01ff'" in sql
    assert "'x'" in sql
    assert "true" in sql
    assert mz.sql_type(pa.int8()) == "TINYINT"
    assert mz.sql_type(pa.int16()) == "SMALLINT"
    assert mz.sql_type(pa.int64()) == "BIGINT"
    assert mz.sql_type(pa.float32()) == "FLOAT"
    assert mz.sql_type(pa.large_binary()) == "BINARY"
    assert mz.sql_type(pa.string()) is None
    empty = mz.setup_sql({"e": pa.table({"d": pa.array([], pa.date32())})})
    assert "CAST(NULL AS DATE)" in empty
    assert "WHERE false" in empty


def test_table_files_handles_every_layout(tmp_path: Path):
    import pyarrow.parquet as pq

    one = _table(3)
    pq.write_table(one, tmp_path / "a.parquet")
    (tmp_path / "b.parquet").mkdir()
    pq.write_table(one, tmp_path / "b.parquet" / "p0.parquet")
    pq.write_table(one, tmp_path / "b.parquet" / "p1.parquet")
    (tmp_path / "c").mkdir()
    pq.write_table(one, tmp_path / "c" / "p0.parquet")
    assert [p.name for p in mz.table_files(tmp_path, "a")] == ["a.parquet"]
    assert [p.name for p in mz.table_files(tmp_path, "b")] == ["p0.parquet", "p1.parquet"]  # hits.parquet/
    assert [p.name for p in mz.table_files(tmp_path, "c")] == ["p0.parquet"]  # hits/
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no Parquet files for table 'empty'"):
        mz.table_files(tmp_path, "empty")
    with pytest.raises(SystemExit, match="no Parquet files for table 'nope'"):
        mz.table_files(tmp_path, "nope")
    tables, big = mz.load_tables(tmp_path, ["a", "b"], whole_rows=1000, pool_rows=100)
    assert tables["b"].num_rows == 6
    assert big == set()


def test_the_cli_passes_minimize_and_ledger_arguments_through(monkeypatch: pytest.MonkeyPatch):
    from krishiv_bench import cli, ledger

    seen: dict[str, tuple] = {}
    monkeypatch.setattr(mz, "run_minimize", lambda *a, **k: seen.__setitem__("minimize", (a, k)))
    monkeypatch.setattr(ledger, "run_ledger", lambda *a, **k: seen.__setitem__("ledger", (a, k)))
    cli.main(["minimize", "--query", "q07", "--sf", "1", "--seed-rows", "50", "--no-pin", "--full-data"])
    args, kwargs = seen["minimize"]
    assert args[:3] == ("tpch", "q07", 1.0)
    assert 50 in args
    assert kwargs["pin"] is False
    assert kwargs["max_seconds"] == 1500
    assert kwargs["run_timeout"] == 30
    cli.main(["ledger", "--suite", "clickbench", "--queries", "q01,q02", "--threads", "4"])
    largs, _ = seen["ledger"]
    assert largs[0] == "clickbench"
    assert largs[3] == ["q01", "q02"]
    assert largs[4] == 4
