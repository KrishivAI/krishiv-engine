"""Operator microbenchmarks: the cases, their checksum queries, verdicts, the parity report, the ``micro`` command."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from krishiv_bench import cli, micro, reproduce
from krishiv_bench import micro_data as md
from krishiv_bench.adapters import Dataset, Engine, Query, make_engine
from krishiv_bench.adapters.base import check_query
from krishiv_bench.adapters.process import EngineError
from krishiv_bench.paths import bench_root

TINY = md.MicroSizes("quick", 20_000, (10, 1_000), (100, 20_000), (100,))


@pytest.fixture
def kd(tmp_path: Path, monkeypatch) -> Path:
    """A KRISHIV_DATA root holding the tiny dataset where ``micro --quick`` looks for it."""
    md.generate(TINY, tmp_path / "micro" / md.GENERATOR_VERSION)
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    monkeypatch.setitem(md.SIZES, "quick", TINY)
    return tmp_path


def dataset(kd: Path) -> Dataset:
    root = kd / micro.dataset_name("quick")
    return Dataset("quick", 1, root, sorted(p.stem for p in root.glob("*.parquet")))


def duck(ds: Dataset):
    eng = make_engine("duckdb", threads=2, memory_limit="2GB")
    eng.setup(ds)
    return eng


# ---------------------------------------------------------------- cases
def test_every_size_set_has_unique_case_ids_in_known_families():
    for sizes in md.SIZES.values():
        cs = micro.cases(sizes)
        assert len({c.id for c in cs}) == len(cs)
        assert {c.family for c in cs} == set(micro.FAMILIES)
        assert all(c.rows_in == sizes.rows for c in cs)


def test_case_ranges_cover_the_required_axes():
    cs = {c.id: c for c in micro.cases(md.SIZES["full"])}
    joins = [c for c in cs.values() if c.family == "join"]
    assert {c.params["build"] for c in joins} == {10**3, 10**5, 10**7, 10**8}
    assert {c.params["key"] for c in joins} == {"int64", "string"}
    assert {c.params["match_pct"] for c in joins} >= {100, 50, 10}
    assert {c.params["zipf_alpha"] for c in joins if "zipf_alpha" in c.params} == set(md.ZIPF_ALPHAS)
    groups = [c for c in cs.values() if c.family == "groupby" and "2key" not in c.id]
    assert {c.params["groups"] for c in groups} == {10**2, 10**3, 10**4, 10**6, 10**8}
    assert {c.params["key"] for c in groups} == {"int", "str"}
    assert {c.params["per_mille"] for c in cs.values() if c.id.startswith("filter_rows")} == set(
        micro.MATERIALISED_PER_MILLE
    )
    assert {"sort_int", "sort_str", "topn_k100_int", "decimal_global", "str_eq", "str_like_contains"} <= set(cs)


def test_ordered_answers_compare_rows_where_they_are_small_and_the_rest_a_checksum():
    for c in micro.cases(md.SIZES["quick"]):
        assert (c.check is None) == (c.family in ("topn", "sort")), c.id
    for c in micro.cases(md.SIZES["default"]):
        assert (c.check is None) == (c.family == "topn"), c.id


def test_a_checksum_sees_values_swapped_between_rows():
    con = duckdb.connect()
    exprs = (micro._int("g"), micro._int("s"), micro._assoc("g", "s"))
    right = con.execute(micro.check_sql("SELECT * FROM (VALUES (1, 10), (2, 20)) t(g, s)", exprs)).fetchone()
    swapped = con.execute(micro.check_sql("SELECT * FROM (VALUES (1, 20), (2, 10)) t(g, s)", exprs)).fetchone()
    assert right is not None
    assert swapped is not None
    assert right[:3] == swapped[:3]
    assert right != swapped


def test_every_group_by_and_sort_checksum_ties_keys_to_values():
    for c in micro.cases(md.SIZES["default"]):
        if c.family in ("groupby", "sort") or (c.family == "decimal" and c.id != "decimal_global"):
            assert any("AS DOUBLE) * CAST" in e for e in c.check or ()), c.id


def test_check_sql_wraps_the_question():
    sql = micro.check_sql("SELECT a, b FROM t GROUP BY a;", ("SUM(a)", "SUM(CAST(b AS DOUBLE))"))
    assert sql == (
        "SELECT COUNT(*) AS n_rows, SUM(a) AS c0, SUM(CAST(b AS DOUBLE)) AS c1 "
        "FROM (SELECT a, b FROM t GROUP BY a) AS ans"
    )


def test_every_case_and_checksum_runs_on_generated_data(kd: Path):
    ds = dataset(kd)
    eng = duck(ds)
    try:
        qs = micro.load_queries(ds.root, "duckdb")
        assert [q.id for q in qs] == [c.id for c in micro.cases(TINY)]
        for q in qs:
            rows = eng.run(check_query(q))
            if q.check_sql:
                assert len(rows) == 1, q.id
                assert rows[0][0] > 0, f"{q.id}: empty answer"
            else:
                assert len(rows) in (100, 1000, TINY.rows), q.id
            n, secs = eng.timed_run(q)
            assert n > 0
            assert secs >= 0
    finally:
        eng.teardown()


def test_checksum_row_counts_follow_the_shape_of_each_case(kd: Path):
    ds = dataset(kd)
    eng = duck(ds)
    try:
        qs = {q.id: q for q in micro.load_queries(ds.root, "duckdb")}

        def n_rows(cid: str) -> int:
            return eng.run(check_query(qs[cid]))[0][0]

        assert n_rows("groupby_int_sum_g1e2") == 100
        assert n_rows("groupby_str_multi_g2e4") == 20_000
        assert n_rows("groupby_2key_g1e2") == 10_000
        assert n_rows("filter_rows_pm100") == TINY.rows // 10
        assert len(eng.run(check_query(qs["sort_int"]))) == TINY.rows  # quick compares a sort's rows in order
        assert n_rows("join_int_b1e3_m100") == 1  # the join's own result is aggregated
        assert eng.run(check_query(qs["join_int_b1e3_m100"]))[0][1] == TINY.rows  # every probe row matches once
        assert eng.run(check_query(qs["join_int_b1e3_m10"]))[0][1] < TINY.rows // 5
    finally:
        eng.teardown()


def test_select_by_id_and_family_and_unknown_names():
    cs = micro.cases(TINY)
    assert [c.id for c in micro.select(cs, ["scan_int", "topn"])] == [
        "scan_int",
        "topn_k100_int",
        "topn_k1000_float",
        "topn_k100_str",
    ]
    assert micro.select(cs, None) == cs
    with pytest.raises(SystemExit, match="unknown case or family: nope"):
        micro.select(cs, ["scan_int", "nope"])


def test_missing_dataset_says_how_to_make_it(tmp_path: Path):
    with pytest.raises(SystemExit, match="krishiv-datasets micro"):
        micro.load_queries(tmp_path, "duckdb")


# ---------------------------------------------------------------- criterion twins
def test_every_twin_is_documented_and_the_existing_ones_exist():
    readme = (bench_root() / "README.md").read_text()
    benches = bench_root().parent / "crates" / "krishiv-exec" / "benches"
    for sizes in md.SIZES.values():
        for c in micro.cases(sizes):
            twin = c.twin.removeprefix("stub: ")
            assert f"`{twin}`" in readme, f"{c.id}: twin {twin} missing from the README"
            if c.twin.startswith("stub: ") or not benches.is_dir():  # not a checkout of the whole repository
                continue
            group, _, name = twin.partition("/")
            src = (benches / f"{group}.rs").read_text()
            assert f'benchmark_group("{group}")' in src, f"{c.id}: no bench group {group}"
            # A parametrized bench spells its id as a format string in the source.
            literal = name
            if name.startswith("decimal_"):
                literal = "decimal_{name}_{keys}_keys"
            elif name.startswith("grouped_sum_"):
                literal = "grouped_sum_{keys}_keys"
            assert literal in src, f"{c.id}: no bench like {twin}"


# ---------------------------------------------------------------- derived numbers
def test_operator_seconds_separate_the_operator_from_the_scan():
    profile = {
        "operators": [
            {"name": "FileScan", "compute_seconds": 0.4},
            {"name": "Filter", "compute_seconds": 0.1},
            {"name": "HashAggregate", "compute_seconds": 0.2},
            {"name": "PartialAggregate", "compute_seconds": 0.05},
        ]
    }
    s = micro.operator_seconds(profile, ("HashAggregate", "PartialAggregate"))
    assert s == {
        "operator": pytest.approx(0.25),
        "scan": pytest.approx(0.4),
        "total": pytest.approx(0.75),
        "blocked": pytest.approx(0.0),
    }
    profile["operators"][0]["blocked_io_seconds"] = 1.5
    assert micro.operator_seconds(profile, ("Sort",))["blocked"] == pytest.approx(1.5)
    assert micro.operator_seconds(profile, ("Sort",))["operator"] == 0
    assert micro.operator_seconds({}, ("Sort",)) == {"operator": None, "scan": None, "total": None, "blocked": None}
    assert micro.operator_seconds(None, ("Sort",))["total"] is None


def _engine_entry(samples: list[float], cpu: float | None = 1.0) -> dict:
    import statistics

    e = {"status": "ok", "samples_s": samples, "median_s": statistics.median(samples)}
    if cpu is not None:
        e["cpu_seconds_median"] = cpu
    return e


def _record(engines: dict[str, dict[str, dict]], failed: dict[tuple[str, str], str] | None = None) -> dict:
    case = {"family": "join", "rows_in": 1_000_000, "operators": ["HashJoinProbe"], "twin": "t", "params": {}}
    cases = {cid: dict(case) for cid in engines}
    queries = {cid: {"engines": dict(per)} for cid, per in engines.items()}
    for (cid, eng), why in (failed or {}).items():
        queries[cid]["engines"][eng] = {"status": "mismatch", "reason": why}
    names = list(next(iter(engines.values())))
    return {
        "engines": {n: {} for n in names},
        "queries": queries,
        "micro": {"size": "quick", "sizes": {"rows": 1000}, "cases": cases, "dataset_content_sha256": "ab" * 32},
        "dataset": {"name": "micro/v1/quick"},
        "pinned": {"threads": 12},
        "ours": {"samples_per_engine": 4},
        "load_at_start": 0.5,
        "summary": {"indicative": False, "max_load": 1.5},
    }


def test_ratios_are_medians_with_a_bootstrap_ci():
    a, b = _engine_entry([0.4, 0.4, 0.4, 0.4]), _engine_entry([0.2, 0.2, 0.2, 0.2])
    ratio = micro._ratio(a, b)
    assert ratio is not None
    assert ratio[0] == pytest.approx(2.0)
    assert ratio[1] == (pytest.approx(2.0), pytest.approx(2.0))
    assert micro._ratio(a, {"status": "mismatch"}) is None
    assert micro._ratio({}, b) is None


def test_render_shows_parity_throughput_cpu_operator_time_and_failures():
    rec = _record(
        {
            "join_a": {
                "krishiv": _engine_entry([0.4, 0.5, 0.4, 0.5], cpu=3.0),
                "duckdb": _engine_entry([0.2, 0.2, 0.2, 0.2], cpu=1.5),
                "polars": _engine_entry([0.1, 0.1, 0.1, 0.1], cpu=None),
            },
            "join_b": {"krishiv": _engine_entry([1.0]), "duckdb": _engine_entry([1.0]), "polars": _engine_entry([2.0])},
        },
        failed={("join_b", "polars"): "checksum [1] != reference [2]"},
    )
    rec["queries"]["join_a"]["engines"]["krishiv"]["profile"] = {
        "operators": [
            {"name": "FileScan", "compute_seconds": 0.5},
            {"name": "HashJoinProbe", "compute_seconds": 1.5},
        ]
    }
    md_text = micro.render(rec)
    assert "| join_a | 450.0 | 200.0 | 100.0 | 2.25x (" in md_text  # wall ms, then ratios to duckdb
    assert "| join_a | 1,000,000 | 2.2 | 5.0 | 10.0 | 3.00 | 1.50 | - |" in md_text
    assert "| join_a | HashJoinProbe | 1.500 | 0.500 | 2.000 | 75% | 0.000 | 1.00 | 450.0 | t |" in md_text
    assert "`join_b` **polars**: checksum [1] != reference [2]" in md_text
    assert "| join_b | 1,000.0 | 1,000.0 | MISMATCH |" in md_text
    assert "| join | 2 |" in md_text  # the family row
    assert "within bound" in md_text
    assert micro.failed_cases(rec) == [("join_b", "polars")]


# ---------------------------------------------------------------- verdicts and the whole command
class Spy(Engine):
    """An in-process engine over DuckDB that counts the timed runs; ``wrong`` makes it answer ``+1`` on a case."""

    name = "krishiv"

    def __init__(self, wrong: set[str] | None = None):
        self.inner = make_engine("duckdb", threads=2, memory_limit="2GB")
        self.wrong = wrong or set()
        self.timed: list[str] = []

    def version(self) -> str:
        return "spy"

    def setup(self, dataset: Dataset) -> None:
        self.inner.setup(dataset)

    def run(self, query: Query) -> list[tuple]:
        rows = self.inner.run(query)
        if query.id in self.wrong:
            return [(rows[0][0] + 1, *rows[0][1:])]
        return rows

    def timed_run(self, query: Query) -> tuple[int, float]:
        self.timed.append(query.id)
        return self.inner.timed_run(query)

    def profile(self, query: Query) -> dict:
        return {"rows": 1, "seconds": 0.0, "operators": [{"name": "FileScan", "compute_seconds": 0.002}]}

    def teardown(self) -> None:
        self.inner.teardown()


def test_a_checksum_mismatch_fails_the_case_and_is_never_timed(kd: Path):
    ds = dataset(kd)
    spy, ref = Spy(wrong={"sort_int"}), duck(ds)
    spy.setup(ds)
    try:
        qm = {q.id: q for q in micro.load_queries(ds.root, "x", ["sort_int", "scan_int"])}
        bad = reproduce.ours_query(
            {n: qm["sort_int"] for n in ("krishiv", "duckdb")}, {"krishiv": spy, "duckdb": ref}, 1, 60
        )
        assert bad["engines"]["krishiv"]["status"] == "mismatch"
        assert "samples_s" not in bad["engines"]["krishiv"]
        assert bad["engines"]["duckdb"]["status"] == "ok"
        assert spy.timed == []
        good = reproduce.ours_query(
            {n: qm["scan_int"] for n in ("krishiv", "duckdb")}, {"krishiv": spy, "duckdb": ref}, 1, 60
        )
        assert good["engines"]["krishiv"]["status"] == "ok"
        assert spy.timed == ["scan_int", "scan_int"]  # A B B A: two samples with one rep
    finally:
        spy.teardown()
        ref.teardown()


def _stub_engines(monkeypatch, wrong: set[str] | None = None) -> list[Spy]:
    """``micro`` with in-process engines: Krishiv is a spy over DuckDB, DuckDB and Polars are the real adapters."""
    made: list[Spy] = []

    def factory(name: str, threads: int, memory_gb: int, timeout_s: float):
        if name == "krishiv":
            made.append(Spy(wrong))
            return made[-1]
        return make_engine(name, threads=threads, memory_limit="2GB" if name == "duckdb" else None)

    monkeypatch.setattr(reproduce, "ProcessEngine", factory)
    return made


def _run_micro(kd: Path, *argv: str) -> Path:
    cli.main(["micro", "--quick", "--reps", "1", "--threads", "2", *argv])
    (result,) = (kd / "results").glob("*_reproduce_micro_*.json")
    return result


def test_quick_runs_end_to_end_and_writes_json_and_parity_table(kd: Path, monkeypatch, capsys):
    ds = dataset(kd)
    spies = _stub_engines(monkeypatch)
    result = _run_micro(kd, "--queries", "scan,filter,topn,decimal,join_int_b1e3_m100,groupby_str_sum_g1e2")
    rec = json.loads(result.read_text())
    assert rec["suite"] == "micro"
    assert rec["micro"]["size"] == "quick"
    assert rec["micro"]["dataset_content_sha256"] == md.read_manifest(ds.root)["content_sha256"]  # ty: ignore[not-subscriptable]
    assert list(rec["queries"]) == list(rec["micro"]["cases"])
    assert len(rec["queries"]) == 2 + 7 + 3 + 3 + 1 + 1
    for cid, ent in rec["queries"].items():
        for eng in ("krishiv", "duckdb", "polars"):
            e = ent["engines"][eng]
            assert e["status"] == "ok", (cid, eng, e)
            assert len(e["samples_s"]) == 2
        assert ent["engines"]["krishiv"]["profile"]["operators"][0]["name"] == "FileScan"
        assert "ratio_ci95" in ent["engines"]["polars"]
    assert {r for r in spies[0].timed} == set(rec["queries"])
    report = result.with_suffix(".md").read_text()
    assert "## Parity (wall time, ms)" in report
    assert "| join_int_b1e3_m100 |" in report
    assert "krishiv/duckdb" in report
    assert f"All {len(rec['queries'])} cases: every engine's answer checksum equals the DuckDB reference." in report
    assert "parity report:" in capsys.readouterr().out


def test_quick_marks_a_wrong_engine_failed_and_exits_non_zero(kd: Path, monkeypatch):
    _stub_engines(monkeypatch, wrong={"scan_int"})
    with pytest.raises(SystemExit, match=r"1 case/engine pair\(s\) failed"):
        _run_micro(kd, "--queries", "scan")
    (result,) = (kd / "results").glob("*_reproduce_micro_*.json")
    rec = json.loads(result.read_text())
    krishiv = rec["queries"]["scan_int"]["engines"]["krishiv"]
    assert krishiv["status"] == "mismatch"
    assert "samples_s" not in krishiv
    assert rec["queries"]["scan_int"]["engines"]["polars"]["status"] == "ok"  # the others are still timed
    assert rec["queries"]["scan_str"]["engines"]["krishiv"]["status"] == "ok"
    report = result.with_suffix(".md").read_text()
    assert "`scan_int` **krishiv**" in report
    assert "MISMATCH" in report


def test_an_engine_that_cannot_start_fails_the_run_and_is_named(kd: Path, monkeypatch):
    class Broken(Spy):
        def setup(self, dataset: Dataset) -> None:
            raise EngineError("runner missing")

    _stub_engines(monkeypatch)
    real = reproduce.ProcessEngine
    monkeypatch.setattr(
        reproduce, "ProcessEngine", lambda name, *a, **k: Broken() if name == "krishiv" else real(name, *a, **k)
    )
    with pytest.raises(SystemExit, match=r"1 case/engine pair\(s\) failed"):
        _run_micro(kd, "--queries", "scan_int")
    (result,) = (kd / "results").glob("*_reproduce_micro_*.json")
    rec = json.loads(result.read_text())
    assert rec["setup_failures"]["krishiv"]["status"] == "error"
    assert rec["queries"]["scan_int"]["engines"]["polars"]["status"] == "ok"
    assert micro.failed_cases(rec) == [("*", "krishiv")]
    assert "**krishiv**: not run, setup failed: " in result.with_suffix(".md").read_text()


def test_micro_without_the_reference_engine_is_refused(kd: Path):
    with pytest.raises(SystemExit, match="must include duckdb"):
        cli.main(["micro", "--quick", "--engines", "krishiv,polars"])


def test_micro_names_the_missing_dataset(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    with pytest.raises(SystemExit):
        cli.main(["micro", "--quick"])


def test_report_renders_the_newest_result(kd: Path, monkeypatch, capsys):
    _stub_engines(monkeypatch)
    result = _run_micro(kd, "--queries", "scan_int")
    result.with_suffix(".md").unlink()
    cli.main(["report", "--suite", "micro"])
    assert result.with_suffix(".md").is_file()
    assert "report:" in capsys.readouterr().out
    result.with_suffix(".md").unlink()
    cli.main(["report", "--suite", "micro", str(result)])
    assert result.with_suffix(".md").is_file()


def test_result_lookup_without_results_says_what_to_run(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    with pytest.raises(SystemExit, match="krishiv-bench micro"):
        micro.latest_result()


def test_a_ratio_over_a_zero_median_is_none():
    assert micro._ratio({"samples_s": [1.0]}, {"samples_s": [0.0]}) is None
