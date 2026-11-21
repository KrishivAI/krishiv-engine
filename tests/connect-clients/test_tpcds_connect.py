# SPDX-License-Identifier: Apache-2.0
from tpcds_connect import error_class, statements, summary


def test_error_class_takes_the_first_bracketed_class():
    assert (
        error_class("[UNRESOLVED_COLUMN.WITH_SUGGESTION] A column x cannot be resolved")
        == "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    )
    assert error_class("boom") == "UNCLASSIFIED"


def test_statements_drops_empty_ones():
    assert statements("select 1;\nselect 2;\n") == ["select 1", "select 2"]


def test_summary_groups_errors_and_compares_rows():
    ours = {
        "q01": {"status": "ok", "rows": 3, "seconds": 1.0},
        "q02": {"status": "error", "class": "X", "message": ""},
        "q03": {"status": "ok", "rows": 5, "seconds": 1.0},
    }
    ref = {"q01": {"status": "ok", "rows": 3}, "q03": {"status": "ok", "rows": 6}}
    text = summary(ours, ref)
    assert "2/3 queries ran" in text
    assert "| X | q02 |" in text
    assert "differ from the reference on 1 queries: q03" in text
