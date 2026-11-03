# SPDX-License-Identifier: Apache-2.0
"""The TPC-H correctness check (`scripts/p2a_correctness.py`) must fail on what it exists to catch."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "p2a_correctness", Path(__file__).resolve().parents[1] / "scripts" / "p2a_correctness.py"
)
assert _SPEC is not None
assert _SPEC.loader is not None
m = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(m)


def diff(got, ref, *, ordered=True, answer=False):
    return m.rows_equal(got, ref, ordered, answer, [0.0])


def test_equal_rows_pass():
    assert diff([["1", "a", "1.50"]], [["1", "a", "1.50"]]) is None


def test_an_integer_count_off_by_one_fails_against_duckdb():
    assert diff([["1478494"]], [["1478493"]]) is not None
    assert diff([["1478494"]], [["1478493"]], answer=True) is not None


def test_decimals_of_the_same_scale_must_be_equal():
    assert diff([["123141078.2284"]], [["123141078.2283"]]) is not None


def test_a_spark_decimal_average_may_differ_from_a_double_within_the_tolerance():
    assert diff([["25.522006"]], [["25.522005853257337"]]) is None
    assert diff([["25.53"]], [["25.522005853257337"]]) is not None
    # The official answer holds the double: the same value at the coarser scale.
    assert diff([["25.522006"]], [["25.522005853257337"]], answer=True) is None


def test_leading_whitespace_in_text_is_part_of_the_value():
    assert diff([["Address"]], [[" Address"]]) is not None
    assert diff([["Address"]], [[" Address"]], answer=True) is not None
    # CHAR padding at the end is not.
    assert diff([["AIR"]], [["AIR  "]]) is None


def test_rows_in_the_wrong_order_fail_unless_they_tie_on_the_keys():
    got, ref = [["1", "a"], ["2", "b"]], [["2", "b"], ["1", "a"]]
    assert diff(got, ref) is not None
    assert diff(got, ref, ordered=False) is None
    # Equal on the ORDER BY key column 1, other column swapped: ties may come in any order.
    tie_got, tie_ref = [["x", "5"], ["y", "5"]], [["y", "5"], ["x", "5"]]
    assert m.rows_equal(tie_got, tie_ref, True, False, [0.0], [1]) is None
    assert m.rows_equal(tie_got, tie_ref, True, False, [0.0], [0]) is not None


def test_a_missing_row_fails():
    assert diff([["1"]], [["1"], ["2"]], ordered=False) is not None


def test_unknown_or_empty_query_selections_are_refused():
    assert m.query_ids(None)[0] == "q01"
    assert len(m.query_ids(None)) == 22
    assert m.query_ids(["q03", "q01"]) == ["q01", "q03"]
    with pytest.raises(SystemExit):
        m.query_ids(["q1"])
    with pytest.raises(SystemExit):
        m.query_ids(["q01", "q99"])
    with pytest.raises(SystemExit):
        m.query_ids([])
