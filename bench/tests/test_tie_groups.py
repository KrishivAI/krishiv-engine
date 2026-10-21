"""Tie-group result comparison and failed-query error text (fixtures only, no engines, no timing)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from krishiv_bench import compare as c
from krishiv_bench import report, worker
from krishiv_bench.ab import one_query
from krishiv_bench.adapters.base import Query
from krishiv_bench.adapters.krishiv import RunnerError
from krishiv_bench.errors import MAX_ERROR_CHARS, describe_error, error_fields

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
    assert c.derive_limit("select a from (select a from t order by a limit 5) x") == (None, False)


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


def test_worker_reply_carries_class_and_message():
    assert describe_error(KeyError("k")) == "KeyError: 'k'"
    assert worker.describe_error is describe_error


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
