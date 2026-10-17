"""Unit tests for the pure (Spark-free) parts of gen_function_matrix.py.

Run: (cd bench && uv run --group dev pytest ../tests/scripts/test_gen_function_matrix.py)
"""

import sys
from pathlib import Path
from typing import cast

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import gen_function_matrix as g


def test_parse_examples_joins_continuations_and_strips_semicolons():
    text = "\n    Examples:\n      > SELECT abs(-1);\n       1\n      > SELECT a,\n        b FROM t;\n       x\n"
    assert g.parse_examples(text) == ["SELECT abs(-1)", "SELECT a, b FROM t"]


def test_split_args_respects_quotes_and_nesting():
    s = "SELECT f(1, 'a,b', g(2, 3), \"x)\")"
    call = g.find_call(s, "f")
    assert call is not None
    o, c = call
    assert [s[a:b] for a, b in g.split_args(s, o + 1, c)] == ["1", "'a,b'", "g(2, 3)", '"x)"']


def test_find_call_skips_quoted_and_qualified_names():
    s = "SELECT 'abs(1)', x.abs(2), abs(3)"
    call = g.find_call(s, "abs")
    assert call is not None
    o, c = call
    assert s[o + 1 : c] == "3"
    assert g.find_call("SELECT 1 + 2", "+") is None


def operands(s, name):
    spans = g.find_infix(s, name)
    assert spans is not None
    return [s[a:b] for a, b in spans]


def test_find_infix_operators():
    s = "SELECT 2 % 1.8"
    assert operands(s, "%") == ["2", "1.8"]
    s = "SELECT 2 <=> 2"
    assert g.find_infix(s, "<") is None
    assert operands(s, "<=>") == ["2", "2"]
    s = "SELECT ! true"
    assert operands(s, "!") == ["true"]
    s = "SELECT true and false"
    assert operands(s, "and") == ["true", "false"]


def test_edge_values_cover_the_adr_edge_set():
    def labels(t):
        return {label for label, _e, _c in g.edge_values(t)}

    assert {"null", "nan", "neg_zero", "pos_inf", "neg_inf"} <= labels("double")
    assert {"decimal38_max_frac", "decimal38_min_int"} <= labels("decimal(10,2)")
    assert {"invalid_utf8", "empty_string"} <= labels("string")
    dst = {label: conf for label, _e, conf in g.edge_values("timestamp")}
    assert dst["dst_gap"] == {g.TZ: g.DST_TZ}
    assert dst["dst_overlap"] == {g.TZ: g.DST_TZ}
    assert g.edge_values("void") == [("null", "NULL", {})]
    assert all(
        label in g.EDGE_CATALOG
        for t in ("int", "float", "binary", "date", "array<int>", "map<string,int>")
        for label in labels(t)
    )


def test_order_by_detection_ignores_window_clauses():
    assert g.has_top_level_order_by("SELECT a FROM t ORDER BY a")
    assert not g.has_top_level_order_by("SELECT rank() OVER (ORDER BY b) FROM t")


def test_value_unstable_detection_and_jvm_identity():
    assert g.calls_value_unstable("SELECT hex(aes_encrypt('a', '0000111122223333'))")
    assert not g.calls_value_unstable("SELECT aes_decrypt(x, 'k'), abs(1)")
    assert g.JVM_IDENTITY.search("Hello World 100 [B@222c02b1")


def test_message_normalisation_strips_stack_and_plan():
    msg = "[X] bad SQLSTATE: 42K09; line 1 pos 7;\n'Project [a#12]\n+- OneRowRelation\n\nJVM stacktrace:\nfoo"
    assert g.norm_message(msg) == "[X] bad SQLSTATE: 42K09; line 1 pos 7"
    assert g.norm_message("col#123L missing") == "col#NL missing"
    assert g.norm_message("Task 2 in stage 3.0 failed (TID 99)") == "Task N in stage N failed (TID N)"
    sql_ctx = "[DIVIDE_BY_ZERO] Division by zero. SQLSTATE: 22012\n== SQL (line 1, position 8) ==\nSELECT 1/0\n  ^^^"
    assert g.norm_message(sql_ctx) == "[DIVIDE_BY_ZERO] Division by zero. SQLSTATE: 22012"
    df_ctx = (
        '[INVALID_ARRAY_INDEX] bad index. SQLSTATE: 22003\n== DataFrame ==\n"cast" was called from\n'
        "/some/where/gen_function_matrix.py:542\n"
    )
    assert g.norm_message(df_ctx) == "[INVALID_ARRAY_INDEX] bad index. SQLSTATE: 22003"
    assert g.norm_message("x called from /a/b.py:12 y") == "x y"


def test_case_ids_are_stable_content_hashes():
    a = g.case_id("abs", "SELECT abs(1)", {g.TZ: "UTC"})
    assert a == g.case_id("abs", "SELECT abs(1)", {g.TZ: "UTC"})
    assert a != g.case_id("abs", "SELECT abs(2)", {g.TZ: "UTC"})
    assert a.startswith("abs#")
    assert len(a) == len("abs#") + 12


def test_agg_rows_variants():
    class Oracle:
        class spark:
            @staticmethod
            def sql(_q):
                class F:
                    class dataType:
                        @staticmethod
                        def simpleString():
                            return "int"

                class S:
                    fields = (F,)

                class D:
                    schema = S

                return D

    oracle = cast("g.TypeOracle", Oracle)
    fn = {"name": "sum", "group": "agg_funcs"}
    out = g.derive_agg_rows(fn, "SELECT sum(col) FROM VALUES (5), (10) AS tab(col)", oracle, set())
    kinds = [k for k, _s, _c in out]
    assert kinds == ["edge:rows_all_null@rows", "edge:rows_int_max@rows", "edge:rows_empty@rows"]
    assert out[2][1] == "SELECT sum(col) FROM VALUES (5), (10) AS tab(col) WHERE false"
    # Multi-column VALUES are not rewritten.
    assert g.derive_agg_rows(fn, "SELECT sum(a) FROM VALUES (1, 2) AS tab(a, b)", oracle, set()) == []


def test_typed_encoding_is_exact_and_distinguishes_null():
    import decimal

    import pyarrow as pa

    assert g.encode_typed(pa.array([1.5, float("nan"), -0.0, None])) == [
        {"f64": "3ff8000000000000", "repr": "1.5"},
        {"f64": "7ff8000000000000", "repr": "nan"},
        {"f64": "8000000000000000", "repr": "-0.0"},
        None,
    ]
    big = decimal.Decimal("-99999999999999999999.999999999999999999")
    assert g.encode_typed(pa.array([big], pa.decimal128(38, 18))) == [
        {"unscaled": "-99999999999999999999999999999999999999", "scale": 18}
    ]
    assert g.encode_typed(pa.array([["null", None]], pa.list_(pa.string()))) == [["null", None]]
    assert g.encode_typed(pa.array([b"\xc3\x28"], pa.binary())) == [{"hex": "C328"}]
    assert g.encode_typed(pa.array([0], pa.timestamp("us", tz="UTC"))) == [{"ts_us": 0, "tz": "UTC"}]
    assert g.encode_typed(pa.array([1], pa.date32())) == [{"days": 1}]
    m = pa.array([[("a", 1)]], pa.map_(pa.string(), pa.int32()))
    assert g.encode_typed(m) == [{"map": [["a", 1]]}]
    st = pa.array([{"a": 1, "b": None}, None], pa.struct([("a", pa.int32()), ("b", pa.string())]))
    assert g.encode_typed(st) == [{"struct": [["a", 1], ["b", None]]}, None]


def test_error_stability_flags():
    class Legacy(Exception):
        def getCondition(self):
            return "_LEGACY_ERROR_TEMP_1210"

    err = g.error_of(Legacy("boom"))
    assert (err["stability"], err["phase"], err["sqlstate"]) == ("unstable", "execution", None)


def test_self_check_rejects_local_paths(tmp_path):
    err = {"error": {"error_class": "X", "message": "called from /home/someone/x.py:1"}}
    leak = {"f": [{"id": "x#1", "ansi_on": err, "ansi_off": {"same_as": "ansi_on"}}]}
    with pytest.raises(SystemExit) as exc:
        g.self_check(tmp_path, leak)
    assert "x#1/ansi_on" in str(exc.value)
    example = {"f": [{"id": "x#2", "ansi_on": {"rows": [["%SystemDrive%/Users/John"]]}, "ansi_off": {}}]}
    g.self_check(tmp_path, example)  # Spark's own example literals are fine
    (tmp_path / "m.json").write_text(f'{{"host": "{g.socket.gethostname()}"}}')
    with pytest.raises(SystemExit) as exc:
        g.self_check(tmp_path, example)
    assert "host" in str(exc.value)
    bad = {"f": [{"id": "x#1", "ansi_on": {"error": {"error_class": "GENERATOR.TIMEOUT"}}, "ansi_off": {}}]}
    with pytest.raises(SystemExit) as exc:
        g.self_check(tmp_path, bad)
    assert "x#1/ansi_on" in str(exc.value)
