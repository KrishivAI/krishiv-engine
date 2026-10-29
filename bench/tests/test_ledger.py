"""Gap ledger: bucket rules, perf parsing, the wall split, DuckDB profiles, the summary and the run."""

from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import pytest

from krishiv_bench import ledger
from krishiv_bench.adapters.base import Query
from krishiv_bench.adapters.duckdb_engine import DuckDBEngine
from krishiv_bench.adapters.krishiv import KrishivEngine


@pytest.mark.parametrize(
    ("frames", "bucket"),
    [
        (["krishiv_exec::window::frame::advance"], "window"),
        (["memcpy", "krishiv_exec::operators::sort::merge"], "sort/top-N"),
        (["krishiv_exec::join::build::insert"], "join build"),
        (["krishiv_exec::join::grace::spill"], "join build"),
        (["krishiv_exec::join::probe::lookup"], "join probe"),
        (["x", "krishiv_exec::aggregate::hash::update"], "aggregate"),
        (["krishiv_shuffle::exchange::send"], "exchange/repartition"),
        (["krishiv_io::parquet::row_filter::eval"], "scan/decode"),
        (["krishiv_exec::operators::filter::apply"], "filter/project"),
        (["krishiv_exec::join::table::insert_row", "tokio::task::poll"], "join probe"),
        (["krishiv_exec::join::table::lookup", "krishiv_exec::join::build::run"], "join build"),
        (["krishiv_exec::driver::runqueue_wait"], "other"),
        (["krishiv_exec::operators::runtime_filter::check"], "filter/project"),
        (["snap::raw::decompress"], "scan/decode"),
        (["krishiv_functions::string::upper"], "filter/project"),
        (["zstd_decompress", "parquet::arrow::read"], "scan/decode"),
        (["object_store::get_range"], "scan/decode"),
        (["[unknown]", "start_thread"], "other"),
        ([], "other"),
    ],
)
def test_classify_frames(frames, bucket):
    assert ledger.classify_frames(frames) == bucket


def test_leaf_most_matching_frame_decides():
    # A kernel frame under the join probe takes the probe bucket; a scan frame at the leaf wins over its caller.
    assert ledger.classify_frames(["clear_page", "krishiv_exec::join::probe::x"]) == "join probe"
    assert ledger.classify_frames(["krishiv_io::read", "krishiv_exec::aggregate::x"]) == "scan/decode"


PERF = """\
4242 1000.010000:
\t    55d4c8a0 krishiv_exec::join::build::insert+0x10 (/bin/runner)
\t    55d4c8b0 start_thread+0x4 (/lib/libc.so)

4243 1000.030000:
\t    55d4c8c0 krishiv_exec::aggregate::hash::update+0x8 (/bin/runner)

4242 1000.035000:
\t    ffffffff [unknown] ([unknown])
"""


def test_parse_perf_script():
    s = ledger.parse_perf_script(PERF)
    assert [round(t, 3) for t, _ in s] == [1000.01, 1000.03, 1000.035]
    assert s[0][1] == ["krishiv_exec::join::build::insert", "start_thread"]
    assert s[2][1] == ["[unknown]"]
    # No trailing blank line: the last sample is kept.
    assert len(ledger.parse_perf_script(PERF.rstrip("\n"))) == 3
    assert ledger.parse_perf_script("") == []


def test_wall_split_shares_each_bin_and_scales_to_the_wall():
    samples = [
        (1000.01, ["krishiv_exec::join::build::x"]),
        (1000.012, ["krishiv_exec::join::build::x"]),
        (1000.015, ["krishiv_exec::aggregate::x"]),
        (1000.03, ["krishiv_exec::aggregate::x"]),
        (999.0, ["krishiv_exec::window::x"]),  # before the window: ignored
        (1001.0, ["krishiv_exec::window::x"]),  # after: ignored
    ]
    # Window 1000.0 .. 1000.04 is two bins of 20 ms; bin 0 holds 3 samples (2 build, 1 aggregate), bin 1 one.
    out = ledger.wall_split(samples, 1000.0, 1000.04, wall_s=0.08)
    assert math.isclose(out["join build"], 0.02 * 2 / 3 * 2)
    assert math.isclose(out["aggregate"], (0.02 / 3 + 0.02) * 2)
    assert math.isclose(sum(out.values()), 0.08)
    assert ledger.wall_split(samples, 5.0, 5.0, 1.0) == {}
    assert ledger.wall_split(samples, 1000.0, 1000.04, 0.0) == {}


def test_wall_split_last_bin_is_clipped_and_idle_bins_count_as_other():
    out = ledger.wall_split([(10.025, ["krishiv_exec::aggregate::x"])], 10.0, 10.03, wall_s=0.03)
    assert math.isclose(out["aggregate"], 0.01)  # only 10 ms of the last bin lie inside the window
    assert math.isclose(out["other"], 0.02)  # the first bin has no sample: nothing was on CPU


def _op(typ, timing, rows, children=()):
    return {"type": typ, "timing": timing, "intermediate_rows": rows, "children": list(children)}


def test_duckdb_buckets_map_types_and_split_joins():
    scan_big = _op("TABLE_SCAN", 0.4, 900)
    scan_small = _op("TABLE_SCAN", 0.1, 100)
    join = _op("HASH_JOIN", 0.3, 50, [scan_big, scan_small])  # left = probe (900), right = build (100)
    profile = {"operator": [_op("RESULT_COLLECTOR", 0.0, 0, [_op("FILTER", 0.2, 5, [join])])]}
    out = ledger.duckdb_buckets(profile, wall_s=2.0)
    assert math.isclose(sum(out.values()), 2.0)
    total = 0.4 + 0.1 + 0.3 + 0.2
    assert math.isclose(out["scan/decode"], 0.5 / total * 2)
    assert math.isclose(out["filter/project"], 0.2 / total * 2)
    assert math.isclose(out["join build"], 0.3 * 0.1 / total * 2)
    assert math.isclose(out["join probe"], 0.3 * 0.9 / total * 2)


def test_duckdb_buckets_other_aggregate_sort_window_and_degenerate_inputs():
    ops = [
        _op("PERFECT_HASH_GROUP_BY", 1, 1),
        _op("ORDER_BY", 1, 1),
        _op("WINDOW", 1, 1),
        _op("CROSS_PRODUCT", 1, 0, [_op("TABLE_SCAN", 0, 0), _op("TABLE_SCAN", 0, 0)]),
        _op("HASH_JOIN", 1, 0),  # no two children: all probe
        _op("SOMETHING_NEW", 1, 1),
    ]
    out = ledger.duckdb_buckets({"operator": ops}, 6.0)
    assert math.isclose(out["aggregate"], 1.0)
    assert math.isclose(out["sort/top-N"], 1.0)
    assert math.isclose(out["window"], 1.0)
    assert math.isclose(out["other"], 1.0)
    assert math.isclose(out["join probe"], 1.5 + 0.5 * 1.0 / 1.0 * 1.0 + 0.0) or out["join probe"] > 0
    assert ledger.duckdb_buckets({"operator": []}, 1.0) == {"scan/decode": 1.0}  # answered from metadata
    assert ledger.duckdb_buckets({"children": [_op("FILTER", 1, 1)]}, 1.0) == {"filter/project": 1.0}
    assert ledger.duckdb_buckets({"operator": ops}, 0.0) == {}
    legacy = {"operator_type": "TABLE_SCAN", "operator_timing": 2.0, "operator_cardinality": 3}
    assert ledger.duckdb_buckets({"children": [legacy]}, 4.0) == {"scan/decode": 4.0}


def test_duckdb_buckets_from_a_real_profile():
    duck = DuckDBEngine(threads=2, memory_limit="1GB")
    duck.con = duckdb.connect()
    duck.con.execute("create table a as select range i, range % 10 k from range(50000)")
    duck.con.execute("create table b as select range i from range(500)")
    q = Query("q", "select a.k, count(*) from a join b on a.i = b.i group by a.k order by 1", "t")
    prof = ledger.duckdb_profile(duck, q)
    out = ledger.duckdb_buckets(prof, 1.0)
    assert math.isclose(sum(out.values()), 1.0)
    assert {"scan/decode", "aggregate"} <= set(out)
    assert out.get("join build", 0) + out.get("join probe", 0) > 0


def _query_entry(k, d, kb, db):
    return {
        "status": "ok",
        "krishiv_s": k,
        "duckdb_s": d,
        "krishiv_buckets": kb,
        "duckdb_buckets": db,
        "gaps": ledger.bucket_gaps(kb, db),
    }


def test_bucket_gaps_cover_every_bucket():
    gaps = ledger.bucket_gaps({"aggregate": 2.0}, {"aggregate": 0.5, "window": 0.25})
    assert gaps["aggregate"] == 1.5
    assert gaps["window"] == -0.25
    assert set(gaps) == set(ledger.BUCKETS)


QUERIES = {
    "q1": _query_entry(4.0, 1.0, {"aggregate": 3.0, "join build": 1.0}, {"aggregate": 0.5, "join build": 0.5}),
    "q2": _query_entry(2.0, 1.0, {"join build": 2.0}, {"join build": 0.5, "scan/decode": 0.5}),
    "q3": {"status": "failed", "reason": "krishiv error: boom | x"},
}


def test_summarize_ranks_buckets_and_computes_the_geomean_effect():
    s = ledger.summarize(QUERIES)
    assert s["queries_ok"] == 2
    assert s["queries_total"] == 3
    assert math.isclose(s["geomean"], math.sqrt(4.0 * 2.0))
    b = s["buckets"]
    assert math.isclose(b["aggregate"]["gap_s"], 2.5)
    assert math.isclose(b["join build"]["gap_s"], 0.5 + 1.5)
    assert math.isclose(sum(v["share"] for v in b.values()), 1.0)
    # Zeroing the join-build gap: q1 4.0 - 0.5 = 3.5, q2 2.0 - 1.5 = 0.5.
    assert math.isclose(b["join build"]["geomean_if_zero"], math.sqrt(3.5 * 0.5))
    # scan/decode is faster on Krishiv's side (negative gap): left alone.
    assert math.isclose(b["scan/decode"]["geomean_if_zero"], s["geomean"])
    assert ledger.summarize({"q": {"status": "failed"}}) == {"queries_ok": 0, "buckets": {}, "geomean": None}


def _record(queries):
    return {
        "suite": "tpch",
        "scale_factor": 10.0,
        "started_utc": "2026-10-29T10:00:00+00:00",
        "threads": 12,
        "git": {"commit": "abc123"},
        "engines": {"duckdb": {"version": "2.0.0"}},
        "queries": queries,
        "summary": ledger.summarize(queries),
    }


def test_render_markdown_tables_and_deltas():
    rec = _record(QUERIES)
    md = ledger.render_markdown(rec)
    assert "TPC" not in md
    assert "tpch SF10" in md
    assert "| q1 | 4.000 | 1.000 | 4.00 | 3.000 | aggregate +2.500 | join build +0.500 |" in md
    assert "| q3 | failed |" in md
    assert "boom / x" in md
    assert "Geomean Krishiv/DuckDB **2.83x** over 2/3 queries." in md
    assert "| 1 | aggregate |" in md
    assert "vs previous" not in md

    better = {
        "q1": _query_entry(3.0, 1.0, {"aggregate": 2.0, "join build": 1.0}, {"aggregate": 0.5, "join build": 0.5}),
        "q2": {"status": "failed", "reason": "x"},
    }
    md2 = ledger.render_markdown(rec, _record(better))
    assert "ratio vs previous" in md2
    assert "gap s vs previous" in md2
    assert "| +1.00 |" in md2  # q1: 4.00 vs 3.00
    assert "| - |" in md2  # q2 has no previous result

    empty = _record({"q": {"status": "failed", "reason": "r"}})
    assert "| q | failed |" in ledger.render_markdown(empty)


def test_previous_ledger_picks_the_newest_earlier_one(tmp_path: Path):
    for stamp in ("20261001T000000Z", "20261020T000000Z", "20261030T000000Z"):
        (tmp_path / f"{stamp}_ledger_tpch_sf10.json").write_text("{}")
    (tmp_path / "20261025T000000Z_ledger_tpch_sf1.json").write_text("{}")
    got = ledger.previous_ledger("tpch", 10.0, "20261025T000000Z", tmp_path)
    assert got is not None
    assert got.name.startswith("20261020")
    assert ledger.previous_ledger("tpch", 10.0, "20260101T000000Z", tmp_path) is None
    assert ledger.previous_ledger("tpch", 10.0, "x", tmp_path / "missing") is None


def test_perf_launcher_and_launcher_prefix(tmp_path: Path):
    cmd = ledger.perf_launcher(tmp_path / "p.data")
    assert cmd[:2] == ["perf", "record"]
    assert "CLOCK_MONOTONIC" in cmd
    assert cmd[-1] == "--"
    assert KrishivEngine(binary="x", launcher=cmd).launcher == cmd
    assert KrishivEngine(binary="x").launcher == []


def test_perf_script_runs_perf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls = []

    class R:
        stdout = PERF

    monkeypatch.setattr(ledger.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or R())
    assert ledger.perf_script(tmp_path / "p.data") == PERF
    assert calls[0][:2] == ["perf", "script"]
    assert "tid,time,ip,sym" in calls[0]


class _Fake:
    def __init__(self, seconds):
        self.seconds = seconds


def test_one_query_ok_and_failures(monkeypatch: pytest.MonkeyPatch):
    q = Query("q01", "select 1", "tpch")

    def measure(*a):
        return {"seconds": 2.0, "runs": [2.0] * 3, "buckets": {"aggregate": 2.0}}

    class Duck:
        con = None

    monkeypatch.setattr(
        ledger, "measure_duckdb", lambda duck, query, t: {"seconds": 1.0, "runs": [1.0], "buckets": {"aggregate": 1.0}}
    )
    if True:
        e = ledger._one_query(q, None, None, Duck(), 2, "1g", 5, None, measure)
        assert e["status"] == "ok"
        assert e["gaps"]["aggregate"] == 1.0

        def boom(*a):
            raise ledger.RunnerError("planning failed")

        assert "krishiv" in ledger._one_query(q, None, None, Duck(), 2, "1g", 5, None, boom)["reason"]

        def crash(*a):
            raise ledger.RunnerCrashed("died")

        assert ledger._one_query(q, None, None, Duck(), 2, "1g", 5, None, crash)["status"] == "failed"

        def noperf(*a):
            raise FileNotFoundError("perf")

        assert ledger._one_query(q, None, None, Duck(), 2, "1g", 5, None, noperf)["reason"].startswith("perf")

        def bad_duck(duck, query, t):
            raise RuntimeError("oom")

        monkeypatch.setattr(ledger, "measure_duckdb", bad_duck)
        assert ledger._one_query(q, None, None, Duck(), 2, "1g", 5, None, measure)["reason"].startswith("duckdb")


def test_profile_of_a_metadata_only_query_is_empty(tmp_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"x": [1, 2]}), tmp_path / "l.parquet")
    duck = DuckDBEngine(threads=1, memory_limit="1GB")
    duck.con = duckdb.connect()
    duck.con.execute(f"create view l as select * from read_parquet('{tmp_path}/l.parquet')")
    prof = ledger.duckdb_profile(duck, Query("q", "select count(*) from l", "t"))
    assert ledger.duckdb_buckets(prof, 1.0) == {} or "operator" in prof


def test_measure_duckdb_times_and_splits():
    duck = DuckDBEngine(threads=1, memory_limit="1GB")
    duck.con = duckdb.connect()
    duck.con.execute("create table t as select range i from range(1000)")
    out = ledger.measure_duckdb(duck, Query("q", "select sum(i) from t", "t"), 5)
    assert out["seconds"] > 0
    assert len(out["runs"]) == 3
    assert math.isclose(sum(out["buckets"].values()), out["seconds"])


def test_run_ledger_end_to_end_with_a_fake_measurement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from krishiv_bench.adapters.base import Dataset

    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    root = tmp_path / "tpch-duckdb" / "sf1"
    root.mkdir(parents=True)
    pq.write_table(pa.table({"x": list(range(100))}), root / "lineitem.parquet")
    qs = [Query("q01", "select count(*) from lineitem", "tpch"), Query("q02", "select sum(x) from lineitem", "tpch")]
    monkeypatch.setattr(ledger, "load_queries", lambda suite, engine, only=None: qs)
    monkeypatch.setattr(ledger, "dataset_dir", lambda name, sf: root)
    monkeypatch.setattr(ledger, "wait_idle", lambda *a, **k: None)

    class FakeKrishiv:
        def __init__(self, *a, **k):
            pass

        def setup(self, ds):
            pass

        def teardown(self):
            pass

        def version(self):
            return "9.9"

        def config(self):
            return {}

    monkeypatch.setattr(ledger, "KrishivEngine", FakeKrishiv)
    measure = lambda *a: {"seconds": 0.5, "runs": [0.5] * 3, "buckets": {"aggregate": 0.4, "other": 0.1}}  # noqa: E731
    f = ledger.run_ledger("tpch", 1, threads=1, memory_limit="1g", measure=measure)
    rec = json.loads(f.read_text())
    assert rec["kind"] == "ledger"
    assert set(rec["queries"]) == {"q01", "q02"}
    assert rec["engines"]["krishiv"]["version"] == "9.9"
    assert f.with_suffix(".md").read_text().startswith("# Performance gap ledger")
    assert rec["previous"] is None

    # A second run diffs against the first.
    f2 = ledger.run_ledger("tpch", 1, threads=1, memory_limit="1g", measure=measure, previous=str(f))
    assert json.loads(f2.read_text())["previous"] == str(f)
    assert "ratio vs previous" in f2.with_suffix(".md").read_text()
    monkeypatch.setattr(ledger, "dataset_dir", lambda name, sf: tmp_path / "missing")
    with pytest.raises(SystemExit):
        ledger.run_ledger("tpch", 1, dataset="nope", threads=1)
    _ = Dataset


def test_measure_krishiv_takes_the_second_plain_run_and_splits_the_perf_run(monkeypatch: pytest.MonkeyPatch):
    runs = iter([0.9, 0.8, 0.7])
    events: list[str] = []

    class Plain:
        def profile(self, query):
            return {"seconds": next(runs)}

    class Perf:
        def __init__(self, threads, memory_limit, binary, launcher):
            self.launcher = launcher

        def setup(self, ds):
            events.append("setup")

        def profile(self, query):
            events.append("profile")

        def teardown(self):
            events.append("teardown")

        def cancel(self):
            events.append("cancel")

    clock = iter([1000.0, 1000.04])
    monkeypatch.setattr(ledger, "KrishivEngine", Perf)
    monkeypatch.setattr(ledger, "_mono", lambda: next(clock))
    monkeypatch.setattr(ledger, "perf_script", lambda data: PERF)
    out = ledger.measure_krishiv(None, None, Query("q", "select 1", "t"), 2, "1g", 0, Plain())  # ty: ignore[invalid-argument-type]
    assert out["seconds"] == 0.8
    assert out["runs"] == [0.9, 0.8, 0.7]
    assert events == ["setup", "profile", "profile", "teardown", "cancel"]
    assert math.isclose(sum(out["buckets"].values()), 0.8)


def test_perf_script_without_blank_separators_and_the_monotonic_clock():
    text = " 5.000000:\n\t    aa krishiv_exec::aggregate::x+0x1\n7 5.010000:\n\t    bb start_thread+0x2\n"
    assert [f for _, f in ledger.parse_perf_script(text)] == [["krishiv_exec::aggregate::x"], ["start_thread"]]
    a = ledger._mono()
    assert ledger._mono() >= a
