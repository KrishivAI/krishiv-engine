"""Tie-group result comparison and failed-query error text (fixtures only, no engines, no timing)."""

from __future__ import annotations

import io
import json
import os
import sys
import types
from typing import Any

import pytest

from krishiv_bench import adapters, report, worker
from krishiv_bench import compare as c
from krishiv_bench.ab import one_query
from krishiv_bench.adapters.base import Query
from krishiv_bench.adapters.krishiv import RunnerError
from krishiv_bench.errors import MAX_ERROR_CHARS, describe_error, error_fields
from krishiv_bench.runner import order_fields

Q71 = (
    "select i_brand_id, t_hour, t_minute, sum(p) as ext_price from t group by 1, 2, 3 "
    "order by ext_price desc, i_brand_id, t_hour"
)
Q65 = "select s_store_name, i_item_desc, revenue from t order by s_store_name, i_item_desc limit 3"


def cast_any(x: object) -> Any:
    return x


def test_rows_in_a_tie_group_may_come_in_any_order():
    # sort key = column 0; the three rows with key 1 differ in column 1 and are swapped between the engines
    a = [(1, "x"), (1, "y"), (1, "z"), (2, "w")]
    b = [(1, "z"), (1, "x"), (1, "y"), (2, "w")]
    res = c.compare_rows(a, b, True, sort_keys=[0])
    assert (res.ok, res.status, res.mode) == (True, "ok", "tie-groups")
    assert c.rows_match(a, b, True, sort_keys=[0]) == (True, "ok (tie-groups)")


def test_tie_group_with_different_rows_is_a_mismatch():
    a = [(1, "x"), (1, "y"), (2, "w")]
    b = [(1, "x"), (1, "q"), (2, "w")]
    res = c.compare_rows(a, b, True, sort_keys=[0])
    assert (res.ok, res.status) == (False, "mismatch")
    assert res.message == "value mismatch in tie group at rows 0..1 (sort key (1,))"


def test_ordering_violation_still_fails():
    a = [(1, "x"), (2, "y"), (3, "z")]
    b = [(2, "y"), (1, "x"), (3, "z")]
    res = c.compare_rows(a, b, True, sort_keys=[0])
    assert (res.ok, res.status, res.mode) == (False, "mismatch", "tie-groups")
    assert res.message == "ORDER differs at row 0 (same rows as a set)"
    # a violation that also changes values reports the value difference
    res = c.compare_rows(a, [(2, "y"), (1, "x"), (3, "other")], True, sort_keys=[0])
    assert res.message == "value mismatch at row 0: (1, 'x') vs (2, 'y')"


def test_tie_group_swap_across_a_key_boundary_is_not_a_tie():
    # rows 1 and 2 have different keys, so exchanging them is an ordering violation, not a tie reorder
    a = [(1, "a"), (1, "b"), (2, "c")]
    b = [(1, "a"), (2, "c"), (1, "b")]
    assert not c.compare_rows(a, b, True, sort_keys=[0]).ok


def test_limit_boundary_tie_is_reported_not_failed():
    a = [(1, "a"), (2, "b"), (2, "c")]  # LIMIT 3 cut the tie group of key 2; the engines kept different members
    b = [(1, "a"), (2, "b"), (2, "d")]
    res = c.compare_rows(a, b, True, sort_keys=[0], limit=3)
    assert (res.ok, res.status, res.mode) == (True, "tie-at-limit", "tie-groups")
    assert res.message == "tie-at-limit: rows differ only in the tie group cut by LIMIT, rows 1..2 (sort key (2,))"
    # without a cut (fewer rows than the limit) the same difference is a real mismatch
    assert c.compare_rows(a, b, True, sort_keys=[0], limit=4).status == "mismatch"
    assert c.compare_rows(a, b, True, sort_keys=[0]).status == "mismatch"


def test_limit_tolerance_covers_only_the_last_group():
    a = [(1, "a"), (1, "b"), (2, "c")]
    b = [(1, "a"), (1, "x"), (2, "d")]
    res = c.compare_rows(a, b, True, sort_keys=[0], limit=3)
    assert (res.ok, res.status) == (False, "mismatch")
    assert res.message == "value mismatch in tie group at rows 0..1 (sort key (1,))"


def test_limit_does_not_excuse_a_key_difference():
    a = [(1, "a"), (2, "b")]
    b = [(1, "a"), (3, "b")]
    assert c.compare_rows(a, b, True, sort_keys=[0], limit=2).status == "mismatch"


def test_offset_makes_the_first_group_a_boundary_too():
    a = [(1, "a"), (2, "b"), (3, "c")]
    b = [(1, "z"), (2, "b"), (3, "c")]
    sql = "select k, v from t order by k limit 3 offset 5"
    assert c.compare_rows(a, b, True, sql=sql, sort_keys=[0]).status == "tie-at-limit"
    assert c.compare_rows(a, b, True, sort_keys=[0], limit=3).status == "mismatch"


def test_offset_zero_is_not_a_boundary():
    a = [(1, "a"), (2, "b"), (3, "c")]
    b = [(1, "z"), (2, "b"), (3, "c")]
    sql = "select k, v from t order by k limit 3 offset 0"
    res = c.compare_rows(a, b, True, sql=sql, sort_keys=[0])
    assert (res.ok, res.status) == (False, "mismatch")
    # the same query with a real offset relaxes the first group
    res = c.compare_rows(a, b, True, sql=sql.replace("offset 0", "offset 1"), sort_keys=[0])
    assert (res.ok, res.status) == (True, "tie-at-limit")


def test_offset_without_limit_cuts_the_first_group_only():
    sql = "select k, v from t order by k offset 2"
    a = [(1, "a"), (1, "b"), (2, "c"), (3, "d")]
    first = [(1, "a"), (1, "x"), (2, "c"), (3, "d")]  # differs only in the group the offset cut
    res = c.compare_rows(a, first, True, sql=sql, sort_keys=[0])
    assert (res.ok, res.status) == (True, "tie-at-limit")
    assert res.message == "tie-at-limit: rows differ only in the tie group cut by OFFSET, rows 0..1 (sort key (1,))"
    # no LIMIT cut the end, so a difference in the last group is real
    last = [(1, "a"), (1, "b"), (2, "c"), (3, "z")]
    assert c.compare_rows(a, last, True, sql=sql, sort_keys=[0]).status == "mismatch"
    middle = [(1, "a"), (1, "b"), (2, "z"), (3, "d")]
    assert c.compare_rows(a, middle, True, sql=sql, sort_keys=[0]).status == "mismatch"
    # the stored form (no sql): the offset flag alone is enough
    assert c.compare_rows(a, first, True, sort_keys=[0], offset=True).status == "tie-at-limit"
    assert c.compare_rows(a, first, True, sort_keys=[0]).status == "mismatch"


def test_message_names_the_boundary_that_cut_the_group():
    sql = "select k, v from t order by k limit 2 offset 1"
    both = c.compare_rows([(1, "a")] * 2, [(1, "a"), (1, "b")], True, sql=sql, sort_keys=[0])
    assert (
        both.message
        == "tie-at-limit: rows differ only in the tie group cut by LIMIT and OFFSET, rows 0..1 (sort key (1,))"
    )
    two = c.compare_rows([(1, "a"), (2, "b")], [(1, "x"), (2, "y")], True, sql=sql, sort_keys=[0])
    assert two.message == (
        "tie-at-limit: rows differ only in the tie group cut by OFFSET, rows 0..0 (sort key (1,)); "
        "LIMIT, rows 1..1 (sort key (2,))"
    )


def test_offset_is_kept_when_the_limit_is_stored_or_passed_separately():
    a = [(1, "a"), (2, "b"), (3, "c")]
    b = [(1, "z"), (2, "b"), (3, "c")]
    sql = "select k, v from t order by k limit 3 offset 5"
    # a stored limit (no sql) keeps its OFFSET through `offset`
    assert c.compare_rows(a, b, True, sort_keys=[0], limit=3, offset=True).status == "tie-at-limit"
    assert c.rows_match(a, b, True, sort_keys=[0], limit=3, offset=True) == (
        True,
        "tie-at-limit: rows differ only in the tie group cut by OFFSET, rows 0..0 (sort key (1,))",
    )
    # an explicit limit does not drop the OFFSET found in the sql, unless `offset` says otherwise
    assert c.compare_rows(a, b, True, sql=sql, sort_keys=[0], limit=3).status == "tie-at-limit"
    assert c.compare_rows(a, b, True, sql=sql, sort_keys=[0], limit=3, offset=False).status == "mismatch"


def test_derived_sort_keys_outside_the_result_width_fall_back_to_whole_rows():
    a = [(1, "x"), (2, "y")]
    # the select list has three columns, the results two: the derived key (column 2) is not in the rows
    sql = "select k, v, w from t order by w"
    res = c.compare_rows(a, a, True, sql=sql)
    assert (res.ok, res.mode) == (True, "ordered-fallback")
    res = c.compare_rows(a, list(reversed(a)), True, sql=sql)
    assert (res.ok, res.mode, res.message) == (False, "ordered-fallback", "ORDER differs at row 0 (same rows as a set)")
    # keys the caller passes are still validated
    with pytest.raises(ValueError, match="sort key index 2"):
        c.compare_rows(a, a, True, sql=sql, sort_keys=[2])


def test_runner_records_the_ordering_the_report_compares_with():
    assert order_fields("select k, v from t order by k limit 3 offset 5") == {
        "has_order_by": True,
        "sort_keys": [0],
        "limit": 3,
        "has_offset": True,
    }
    assert order_fields("select k from t order by k desc limit 10")["has_offset"] is False
    assert order_fields("select k from t") == {
        "has_order_by": False,
        "sort_keys": None,
        "limit": None,
        "has_offset": False,
    }


def test_unknown_sort_keys_fall_back_to_ordered_comparison_and_say_so():
    a = [(1, "x"), (1, "y")]
    b = [(1, "y"), (1, "x")]
    res = c.compare_rows(a, b, True)
    assert (res.ok, res.mode) == (False, "ordered-fallback")
    assert res.message == "ORDER differs at row 0 (same rows as a set)"
    res = c.compare_rows(a, a, True)
    assert (res.ok, res.mode) == (True, "ordered-fallback")
    assert res.message == "ok (ordered-fallback: sort keys unknown)"
    # SELECT * has no derivable keys either
    assert c.compare_rows(a, b, True, sql="select * from t order by 1").mode == "ordered-fallback"


def test_sort_keys_are_derived_from_the_query_when_not_given():
    a = [(1, "x"), (1, "y")]
    b = [(1, "y"), (1, "x")]
    res = c.compare_rows(a, b, True, sql="select k, v from t order by k")
    assert (res.ok, res.mode) == (True, "tie-groups")


def test_derive_sort_keys():
    assert c.derive_sort_keys(Q71) == [3, 0, 1]
    assert c.derive_sort_keys("select a, b from t order by 2 desc nulls first, 1 asc") == [1, 0]
    assert c.derive_sort_keys("select x.a, b as bee from x order by x.a, bee desc") == [0, 1]
    assert c.derive_sort_keys("select a + 1, b from t order by a + 1") == [0]
    assert c.derive_sort_keys("select a from t") is None
    assert c.derive_sort_keys("select a from t order by b") is None  # key outside the select list
    assert c.derive_sort_keys("select a from t order by 2") is None  # ordinal out of range
    assert c.derive_sort_keys("select * from t order by a") is None
    # a CTE or subquery with its own ORDER BY / commas does not confuse the final one
    sql = "with w as (select a, b from t order by b) select f(a, b) as fb, a from w order by fb limit 5"
    assert c.derive_sort_keys(sql) == [0]
    assert c.derive_sort_keys("select a, 'order by z' as s from t order by a") == [0]


def test_derive_limit():
    assert c.derive_limit(Q65) == (3, False)
    assert c.derive_limit("select a from t order by a limit 10 offset 2") == (10, True)
    assert c.derive_limit("select a from t order by a") == (None, False)
    # OFFSET 0 skips nothing; OFFSET works without a LIMIT
    assert c.derive_limit("select a from t order by a limit 10 offset 0") == (10, False)
    assert c.derive_limit("select a from t order by a offset 2") == (None, True)
    assert c.derive_limit("select a from t order by a offset 0") == (None, False)
    assert c.derive_limit("select a from t order by a limit 3 offset (1 + 1)") == (3, True)
    assert c.derive_limit("select a from (select a from t order by a limit 5) x") == (None, False)


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        ("limit 1 + 1", (None, False)),  # a numeric prefix is not the limit
        ("limit 2 * 3 offset 0", (None, False)),
        ("limit all", (None, False)),
        ("limit 7;", (7, False)),
        ("limit 3 offset 0 + 1", (3, True)),  # not a plain number: it may skip rows
        ("limit 3 offset 1 + 0", (3, True)),
        ("limit 3 offset 0 rows", (3, False)),
        ("limit 3 offset 2 rows", (3, True)),
        ("offset 0 limit 4", (4, False)),
    ],
)
def test_derive_limit_takes_a_number_only_when_it_is_the_whole_clause(tail, expected):
    assert c.derive_limit(f"select a from t order by a {tail}") == expected


def test_limit_expression_is_not_read_as_its_numeric_prefix():
    a = [(1, "a"), (2, "b")]
    b = [(1, "a"), (2, "z")]
    # LIMIT 1 + 1 returns two rows: the result is not known to be cut at one row, so the last group differing is real
    res = c.compare_rows([(1, "a")], [(1, "z")], True, sql="select k, v from t order by k limit 1 + 1", sort_keys=[0])
    assert (res.ok, res.status) == (False, "mismatch")
    res = c.compare_rows(a, b, True, sql="select k, v from t order by k limit 1 + 1", sort_keys=[0])
    assert (res.ok, res.status) == (False, "mismatch")
    res = c.compare_rows(a, b, True, sql="select k, v from t order by k limit 2", sort_keys=[0])
    assert (res.ok, res.status) == (True, "tie-at-limit")


def test_offset_expression_may_skip_rows():
    a = [(1, "a"), (2, "b")]
    b = [(1, "z"), (2, "b")]
    res = c.compare_rows(a, b, True, sql="select k, v from t order by k offset 0 + 1", sort_keys=[0])
    assert (res.ok, res.status) == (True, "tie-at-limit")


def test_distinct_rows_tied_on_the_coarse_sort_key_match_as_a_multiset():
    # None and "" have the same coarse key, so input order must not matter
    a = [(1, None), (1, "")]
    assert c._unordered_diff(a, a[::-1]) is None
    res = c.compare_rows(a, a[::-1], True, sort_keys=[0])
    assert (res.ok, res.status) == (True, "ok")
    res = c.compare_rows(a, a[::-1], False)
    assert (res.ok, res.status) == (True, "ok")
    # a genuinely different multiset in the same coarse run is still found
    assert c._unordered_diff([(1, None), (1, None)], [(1, None), (1, "")]) == 1
    res = c.compare_rows([(1, None), (1, None)], [(1, ""), (1, None)], True, sort_keys=[0])
    assert (res.ok, res.status) == (False, "mismatch")


def test_q65_shape_end_to_end_from_sql():
    # tie group (able, NULL) is reordered; the group cut by LIMIT 3 holds different members in the two engines
    a = [("able", None, 1), ("able", None, 2), ("able", "d", 3)]
    b = [("able", None, 2), ("able", None, 1), ("able", "d", 4)]
    res = c.compare_rows(a, b, True, sql=Q65)
    assert (res.ok, res.status) == (True, "tie-at-limit")
    assert c.compare_rows(a[:2], b[:2], True, sql=Q65).status == "ok"


def test_unordered_comparison_is_unchanged():
    assert c.compare_rows([(1,), (2,)], [(2,), (1,)], False) == c.Comparison(True, "ok", "unordered", "ok")
    res = c.compare_rows([(1,), (2,)], [(1,), (3,)], False)
    assert res.message == "value mismatch (unordered, sorted row 1): (2,) vs (3,)"
    assert c.compare_rows([(1,)], [], True).message == "row count 1 != 0"


def test_input_validation():
    rows = [(1, "x")]
    for bad in ([], [2], [-1], [True], ["0"]):
        with pytest.raises(ValueError, match=r"sort_keys|sort key"):
            c.compare_rows(rows, rows, True, sort_keys=cast_any(bad))
    with pytest.raises(ValueError, match="limit"):
        c.compare_rows(rows, rows, True, sort_keys=[0], limit=-1)
    with pytest.raises(ValueError, match="limit"):
        c.compare_rows(rows, rows, True, sort_keys=[0], limit=True)


# ------------------------------------------------------------ error text
def test_error_fields_truncate_the_message_to_500_chars():
    exc = RunnerError("x" * 900)
    f = error_fields(exc)
    assert f["error_class"] == "RunnerError"
    assert f["error_message"] == "x" * MAX_ERROR_CHARS
    assert f["error"] == "RunnerError: " + "x" * 500
    assert describe_error(ValueError("boom")) == "ValueError: boom"


def test_worker_reply_carries_class_and_message(monkeypatch):
    """Drives ``worker.main`` over real pipes: a failing engine call and an unknown op each get an error reply."""

    class Stub:
        def run(self, query):
            raise RunnerError("z" * 900)

        def teardown(self):
            pass

    requests = io.BytesIO()
    for req in ({"op": "run", "query": None}, {"op": "bogus"}, {"op": "quit"}):
        worker.send(requests, req)
    r, w = os.pipe()
    # `main` rebinds fd 1 and duplicates it for the reply stream: give it a private pipe instead of the real fds
    fake_os = types.SimpleNamespace(environ={}, dup=lambda fd: w, dup2=lambda a, b: None, fdopen=os.fdopen)
    monkeypatch.setattr(worker, "os", fake_os)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(requests.getvalue())))
    monkeypatch.setattr(adapters, "make_engine", lambda name, threads, memory_limit: Stub())
    worker.main(["stub", "1", "1"])
    with os.fdopen(r, "rb") as replies:
        assert worker.recv(replies) == {"error": "RunnerError: " + "z" * MAX_ERROR_CHARS}
        assert worker.recv(replies) == {"error": "ValueError: unknown op 'bogus'"}
    assert fake_os.environ == {"POLARS_MAX_THREADS": "1"}


def test_describe_error_formats_class_and_repr_of_the_message():
    assert describe_error(KeyError("k")) == "KeyError: 'k'"


def test_failed_ab_query_records_the_engine_error_text():
    class Failing:
        def run(self, query):
            raise RunnerError("[DIVIDE_BY_ZERO] " + "y" * 600)

    class Fine:
        def run(self, query):
            return [(1,)]

    entry = one_query(Query("q01", "select 1", "tpch"), Failing(), Fine(), reps=1)
    assert entry["status"] == "failed"
    assert entry["reason"] == "a error: RunnerError: [DIVIDE_BY_ZERO] " + "y" * (500 - len("[DIVIDE_BY_ZERO] "))


def test_report_cross_engine_check_keeps_the_stored_offset(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "results_dir", lambda: tmp_path)

    def result(name: str, rows: list, has_offset: bool) -> dict:
        c.save_rows(tmp_path / f"{name}.rows.gz", {"q01": rows})
        query = {
            "iterations": 1,
            "p50_s": 0.1,
            "p99_s": 0.1,
            "min_s": 0.1,
            "max_s": 0.1,
            "mean_s": 0.1,
            "throughput_qps": 10.0,
            "rows_returned": len(rows),
            "has_order_by": True,
            "sort_keys": [0],
            "limit": 3,
            "has_offset": has_offset,
            "answer_check": {"status": "pass", "detail": ""},
        }
        return {
            "dataset": {"name": "tpch", "scale_factor": 1},
            "machine": {"cpu_model": "cpu", "cores_logical": 1, "ram_bytes": 2**30, "os": "os"},
            "git": {"commit": "abc", "branch": "b", "dirty": False},
            "iterations": 1,
            "warmup": 0,
            "engine": {"version": "1", "config": {}},
            "_file": f"{name}.json",
            "rows_file": f"{name}.rows.gz",
            "queries": {"q01": query},
        }

    a = [(1, "a"), (2, "b"), (3, "c")]
    b = [(1, "z"), (2, "b"), (3, "c")]  # differs only in the first tie group
    with_offset = report.render({"duckdb": result("d1", a, True), "krishiv": result("k1", b, True)})
    line = (
        "- `q01` **duckdb vs krishiv**: tie-at-limit: rows differ only in the tie group cut by OFFSET, "
        "rows 0..0 (sort key (1,))"
    )
    unverified = with_offset.split("**Checks skipped (NOT verified):**")[1]
    assert line in unverified.splitlines()
    assert "**Mismatches / failures:**" not in with_offset
    without = report.render({"duckdb": result("d2", a, False), "krishiv": result("k2", b, False)})
    assert "`q01` **duckdb vs krishiv**: value mismatch in tie group at rows 0..0" in without


def test_summary_lists_the_recorded_error():
    exc = RunnerError("z" * 700)
    result = {
        "dataset": {"name": "tpch", "scale_factor": 1},
        "machine": {"cpu_model": "cpu", "cores_logical": 1, "ram_bytes": 2**30, "os": "os"},
        "git": {"commit": "abc", "branch": "b", "dirty": False},
        "iterations": 1,
        "warmup": 0,
        "engine": {"version": "1", "config": {}},
        "_file": "r.json",
        "queries": {"q01": {"iterations": 1, **error_fields(exc)}},
    }
    assert json.loads(json.dumps(result))["queries"]["q01"]["error_class"] == "RunnerError"
    text = report.render({"krishiv": result})
    assert f"- `q01` **krishiv**: error `RunnerError: {'z' * 500}`" in text
