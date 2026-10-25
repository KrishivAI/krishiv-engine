#!/usr/bin/env python3
"""Generate the implicit-coercion matrix fixture (P2-T071d unit 1).

Every expectation comes from a REAL Spark 4.2 run (a Spark Connect server),
never from docs or memory. For each form -- binary arithmetic
(`+ - * / % div`), comparisons (`= <> < <= > >= <=>`), `IN` list, `CASE`,
`IF`, `COALESCE`, `NVL`, `NULLIF`, `greatest`, `least`, `concat`, `elt`,
`array(...)`, `map(...)` and `UNION` columns -- every pair of operand types
(all scalar types the engine has in `krishiv-types`, plus one array, one map
and one struct type) as `CAST(NULL AS <type>)` operands is analyzed in both
ANSI modes (`spark.sql.ansi.enabled` on and off): `SELECT typeof(<expr>)`
records the result type, or the error class, SQLSTATE and message parameters
Spark raised. `UNION` columns cannot nest in `typeof`, so those rows run the
`UNION ALL` query itself and record its output column type (or the same
error shape). A few literal spot checks pin the task's examples
(`'1' + 1`, `coalesce('1', 1)`, `array(1, '2')`, `'1' * '2'`,
`greatest('1', 2)`).

Each unit of work (one form in one ANSI mode) runs in its own fresh session;
no tables or views are created and the only config set is
`spark.sql.ansi.enabled` on that session, so nothing outlives the session.
Output is deterministic (fixed form/operand/mode order, sorted keys, no
timestamps, no local paths); the run fails rather than commit bad data
(wrong server version, an operand that does not analyze to its own type, a
spot check that moved, or a local path in the output).

Usage (from repo root; start a Spark 4.2 Connect server on 15012 first):

    uv run --no-project --python 3.12 --with 'pyspark-client==4.2.0' \\
        tests/scripts/gen_coercion_matrix.py --out tests/fixtures/spark-4.2.0/coercion-matrix.json

The client needs `SPARK_HOME` set to any empty directory.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_coercion_matrix.py"
ANSI = "spark.sql.ansi.enabled"
LOCAL_PATH = re.compile(r"/home/|/Users/|[A-Za-z]:\\\\")

# Every operand type of the matrix (Spark DDL) with the `typeof` it must
# analyze to (the self-check that pins the server and the spellings).
OPERANDS: list[tuple[str, str]] = [
    ("VOID", "void"),
    ("BOOLEAN", "boolean"),
    ("TINYINT", "tinyint"),
    ("SMALLINT", "smallint"),
    ("INT", "int"),
    ("BIGINT", "bigint"),
    ("FLOAT", "float"),
    ("DOUBLE", "double"),
    ("STRING", "string"),
    ("BINARY", "binary"),
    ("DECIMAL(10,2)", "decimal(10,2)"),
    ("DATE", "date"),
    ("TIMESTAMP", "timestamp"),
    ("TIMESTAMP_NTZ", "timestamp_ntz"),
    ("ARRAY<INT>", "array<int>"),
    ("MAP<STRING,INT>", "map<string,int>"),
    ("STRUCT<a INT>", "struct<a:int>"),
]

# (form, template): `{L}` and `{R}` are the `CAST(NULL AS <type>)` operands.
# `CASE`/`IF` vary the two value branches over every pair; their condition
# is a fixed boolean literal. `elt` varies the two values over every pair;
# the position is a fixed `1`. `union` is a query, not an expression (it
# cannot nest in `typeof`).
FORMS: list[tuple[str, str]] = [
    (f, "{L} " + op + " {R}")
    for f, op in [
        ("add", "+"),
        ("subtract", "-"),
        ("multiply", "*"),
        ("divide", "/"),
        ("remainder", "%"),
        ("intdiv", "div"),
        ("eq", "="),
        ("noteq", "<>"),
        ("lt", "<"),
        ("lte", "<="),
        ("gt", ">"),
        ("gte", ">="),
        ("eqnull", "<=>"),
    ]
] + [
    ("in", "{L} IN ({R})"),
    ("case", "CASE WHEN TRUE THEN {L} ELSE {R} END"),
    ("if", "if(TRUE, {L}, {R})"),
    ("coalesce", "coalesce({L}, {R})"),
    ("nvl", "nvl({L}, {R})"),
    ("nullif", "nullif({L}, {R})"),
    ("greatest", "greatest({L}, {R})"),
    ("least", "least({L}, {R})"),
    ("concat", "concat({L}, {R})"),
    ("elt", "elt(1, {L}, {R})"),
    ("array", "array({L}, {R})"),
    ("map", "map({L}, {R})"),
    ("union", "SELECT {L} UNION ALL SELECT {R}"),
]

MODES = (("ansi_on", "true"), ("ansi_off", "false"))

# Literal spot checks the fixture must contain: (sql, typeof in ANSI on,
# typeof in ANSI off); `("error", class)` where Spark analyzes an error.
SPOTS: list[tuple[str, object, object]] = [
    ("SELECT typeof('1' + 1)", "bigint", "double"),
    ("SELECT typeof(coalesce('1', 1))", "bigint", "string"),
    ("SELECT typeof(array(1, '2'))", "array<bigint>", "array<string>"),
    ("SELECT typeof('1' * '2')", ("error", "DATATYPE_MISMATCH.BINARY_OP_WRONG_TYPE"), "double"),
    (
        "SELECT typeof(greatest('1', 2))",
        ("error", "DATATYPE_MISMATCH.DATA_DIFF_TYPES"),
        ("error", "DATATYPE_MISMATCH.DATA_DIFF_TYPES"),
    ),
]


def operand_sql(ddl: str) -> str:
    return f"CAST(NULL AS {ddl})"


def record_error(e) -> dict:
    get = getattr
    cond = get(e, "getCondition", None) or get(e, "getErrorClass", None)
    out: dict = {
        "error_class": cond() if cond else "UNCLASSIFIED",
        "sql_state": e.getSqlState() if hasattr(e, "getSqlState") else None,
    }
    params = e.getMessageParameters() if hasattr(e, "getMessageParameters") else {}
    out["message_parameters"] = dict(sorted((params or {}).items()))
    return out


def analyze_expr(spark, expr: str) -> dict:
    """`SELECT typeof(<expr>)`: the result type, or Spark's error."""
    from pyspark.errors import PySparkException

    try:
        value = spark.sql(f"SELECT typeof({expr})").collect()[0][0]
    except PySparkException as e:
        return {"ok": False, **record_error(e)}
    assert isinstance(value, str), expr
    return {"ok": True, "type": value}


def analyze_union(spark, query: str) -> dict:
    """The `UNION ALL` query: its output column type, or Spark's error."""
    from pyspark.errors import PySparkException

    try:
        fields = spark.sql(query).schema.fields
    except PySparkException as e:
        return {"ok": False, **record_error(e)}
    assert len(fields) == 1, query
    return {"ok": True, "type": fields[0].dataType.simpleString()}


def self_check_operands(spark) -> None:
    for ddl, want in OPERANDS:
        got = analyze_expr(spark, operand_sql(ddl))
        if got != {"ok": True, "type": want}:
            raise SystemExit(f"operand self-check failed for {ddl!r}: {got!r} (want type {want!r})")


def check_spots(spark, mode: str, want_idx: int) -> list[dict]:
    """The task's literal spot checks in one mode; returns their rows."""
    rows = []
    for i, (sql, want_on, want_off) in enumerate(SPOTS):
        want = want_on if want_idx == 0 else want_off
        expr = sql[len("SELECT typeof(") : -len(")")]
        got = analyze_expr(spark, expr)
        if isinstance(want, tuple):
            _, cls = want
            if got.get("ok") or got.get("error_class") != cls:
                raise SystemExit(f"spot check failed for {sql!r} in {mode}: {got!r}")
            rows.append({"id": f"spot/{i}/{mode}", "form": "spot", "expr": expr, "sql": sql, "mode": mode, **got})
        elif got != {"ok": True, "type": want}:
            raise SystemExit(f"spot check failed for {sql!r} in {mode}: {got!r} (want type {want!r})")
        else:
            rows.append({"id": f"spot/{i}/{mode}", "form": "spot", "expr": expr, "sql": sql, "mode": mode, **got})
    return rows


def run_unit(remote: str, form: str, template: str, mode: str, ansi: str) -> list[dict]:
    """Every operand pair of one form in one ANSI mode, in a fresh session."""
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.remote(remote).create()
    try:
        if spark.version != "4.2.0":
            raise SystemExit(f"expected Spark 4.2.0, found {spark.version}")
        spark.conf.set(ANSI, ansi)
        self_check_operands(spark)
        rows = []
        for left, _ in OPERANDS:
            for right, _ in OPERANDS:
                rid = f"{form}/{left}/{right}/{mode}"
                if form == "union":
                    query = template.format(L=operand_sql(left), R=operand_sql(right))
                    rows.append(
                        {
                            "id": rid,
                            "form": form,
                            "left": left,
                            "right": right,
                            "mode": mode,
                            "sql": query,
                            **analyze_union(spark, query),
                        }
                    )
                else:
                    expr = template.format(L=operand_sql(left), R=operand_sql(right))
                    rows.append(
                        {
                            "id": rid,
                            "form": form,
                            "left": left,
                            "right": right,
                            "mode": mode,
                            "expr": expr,
                            "sql": f"SELECT typeof({expr})",
                            **analyze_expr(spark, expr),
                        }
                    )
        return rows
    finally:
        spark.stop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://127.0.0.1:15012")
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/coercion-matrix.json")
    args = ap.parse_args()

    rows: list[dict] = []
    for form, template in FORMS:
        for mode, ansi in MODES:
            unit = run_unit(args.remote, form, template, mode, ansi)
            print(f"wrote unit {form}/{mode}: {len(unit)} rows")
            rows.extend(unit)
    # The literal spot checks run in throwaway per-mode sessions so the
    # fixture pins them.
    for i, (mode, ansi) in enumerate(MODES):
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.remote(args.remote).create()
        try:
            spark.conf.set(ANSI, ansi)
            rows.extend(check_spots(spark, mode, i))
        finally:
            spark.stop()

    doc = {
        "format_version": FORMAT_VERSION,
        "spark_version": "4.2.0",
        "generator": GENERATOR,
        "notes": "Implicit-coercion ground truth: for each form, every pair of operand types "
        "(all scalar types krishiv-types has, plus one array, one map and one struct type) as "
        "`CAST(NULL AS <type>)` operands in both ANSI modes. `ok` rows carry Spark's `typeof` "
        "(the `UNION ALL` query's output column type for `union`); error rows carry the error "
        "class, SQLSTATE and message parameters Spark raised. `CASE`/`IF` vary the value "
        "branches (the condition is `TRUE`); `elt` varies the values (the position is `1`).",
        "operands": [ddl for ddl, _ in OPERANDS],
        "modes": {mode: {ANSI: ansi} for mode, ansi in MODES},
        "rows": rows,
    }
    text = json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
    if LOCAL_PATH.search(text):
        raise SystemExit("refusing to write a fixture that contains a local path")
    Path(args.out).write_text(text, encoding="utf-8")
    per_mode = {mode: sum(1 for r in rows if r["mode"] == mode) for mode, _ in MODES}
    print(f"wrote {len(rows)} rows {per_mode} -> {args.out}")


if __name__ == "__main__":
    main()
