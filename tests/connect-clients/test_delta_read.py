# SPDX-License-Identifier: Apache-2.0
from delta_read import expected_rows, render, summary, tables


def test_render_matches_cast_as_string():
    assert [render(v) for v in (None, True, 3, "a b")] == ["NULL", "true", "3", "a b"]


def test_expected_rows_are_sorted_strings_by_column_name():
    cols, rows = expected_rows({"rows": [{"b": 2, "a": None}, {"b": 1, "a": "x"}]})
    assert cols == ["a", "b"]
    assert rows == [("NULL", "2"), ("x", "1")]


def test_the_committed_oracle_tables_are_found():
    names = tables()
    assert "partitioned" in names
    assert "deletion-vectors" in names


def test_summary_lists_only_failures():
    text = summary(
        {
            "t": {
                "sql": {"status": "pass", "rows": 1},
                "format_load": {"status": "fail", "detail": "boom"},
            }
        }
    )
    assert "1/2 checks passed" in text
    assert "- t / format_load: boom" in text
    assert "sql:" not in text
