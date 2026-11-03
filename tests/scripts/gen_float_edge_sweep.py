#!/usr/bin/env python3
"""Floating-point edge sweep against Spark 4.2 (Comet 1.2.0 correctness check 1, after P2-T101).

Crosses `-0.0`, `NaN`, `+/-Infinity`, subnormals, the smallest normal, the largest and smallest
finite values and NULL with every arithmetic, comparison, aggregate, ordering, cast, math, hashing
and collection function that takes a floating-point value, for DOUBLE and FLOAT, and runs the lot on
a REAL Spark in local mode (never from memory). The fixture has the shape of `expr-differential.json`
(replayed by `crates/krishiv-testkit/tests/float_edge_sweep.rs` through `expr_diff`), without
sub-expression cases: the sweep is deterministic and exhaustive over the edge values, so there is
nothing to bisect. Usage (the JDK path is the machine's; Spark needs Java 17 or 21):

    env -u SPARK_HOME uv run --no-project --python 3.12 --with 'pyspark==4.2.0' \
        python -I tests/scripts/gen_float_edge_sweep.py > tests/fixtures/spark-4.2.0/float-edge-sweep.json

The table `t(id, p, x, y, d, fx, fy, fd)` has two rows per ordered pair `(x, y)` of edge values
(`p` is the pair): `x`, `y` are the pair in both rows, `d` is `x` in the first row and `y` in the
second, so a window aggregate `... OVER (PARTITION BY p)` aggregates exactly the pair. `fx`, `fy`
and `fd` are the FLOAT versions.
"""

import importlib.util
import json
import pathlib
import sys

DOUBLES = [
    "-0.0",
    "0.0",
    "NaN",
    "Infinity",
    "-Infinity",
    "4.9E-324",  # the smallest subnormal
    "2.225073858507201E-308",  # the largest subnormal
    "2.2250738585072014E-308",  # the smallest normal
    "1.7976931348623157E308",  # the largest finite value
    "-1.7976931348623157E308",
    "1.0",
    "-1.0",
    "0.5",
    "2.5",
    "-2.5",
    None,
]
FLOATS = [
    "-0.0",
    "0.0",
    "NaN",
    "Infinity",
    "-Infinity",
    "1.4E-45",
    "1.1754942E-38",
    "1.17549435E-38",
    "3.4028235E38",
    "-3.4028235E38",
    "1.0",
    "-1.0",
    "0.5",
    "2.5",
    "-2.5",
    None,
]


def num(v, sql_type):
    return f"CAST(NULL AS {sql_type})" if v is None else f"CAST('{v}' AS {sql_type})"


def setup_sql():
    rows = []
    n = len(DOUBLES)
    for xi in range(n):
        for yi in range(n):
            pair = xi * n + yi
            for which in (0, 1):
                dv, fv = (DOUBLES[xi], FLOATS[xi]) if which == 0 else (DOUBLES[yi], FLOATS[yi])
                rows.append(
                    "("
                    + ", ".join(
                        [
                            str(pair * 2 + which),
                            str(pair),
                            num(DOUBLES[xi], "DOUBLE"),
                            num(DOUBLES[yi], "DOUBLE"),
                            num(dv, "DOUBLE"),
                            num(FLOATS[xi], "FLOAT"),
                            num(FLOATS[yi], "FLOAT"),
                            num(fv, "FLOAT"),
                        ]
                    )
                    + ")"
                )
    return (
        "CREATE OR REPLACE TEMPORARY VIEW t AS SELECT * FROM VALUES "
        + ", ".join(rows)
        + " AS t(id, p, x, y, d, fx, fy, fd)"
    )


UNARY = [
    "abs({a})",
    "(- {a})",
    "(+ {a})",
    "floor({a})",
    "ceil({a})",
    "round({a}, 0)",
    "round({a}, 1)",
    "round({a}, -1)",
    "bround({a}, 0)",
    "rint({a})",
    "sqrt({a})",
    "cbrt({a})",
    "exp({a})",
    "expm1({a})",
    "ln({a})",
    "log10({a})",
    "log2({a})",
    "log1p({a})",
    "sin({a})",
    "cos({a})",
    "tan({a})",
    "asin({a})",
    "acos({a})",
    "atan({a})",
    "sinh({a})",
    "cosh({a})",
    "tanh({a})",
    "asinh({a})",
    "acosh({a})",
    "atanh({a})",
    "degrees({a})",
    "radians({a})",
    "signum({a})",
    "isnan({a})",
    "({a} IS NULL)",
    "({a} = {a})",
    "({a} <=> {a})",
    "({a} < {a})",
    "CAST({a} AS TINYINT)",
    "CAST({a} AS SMALLINT)",
    "CAST({a} AS INT)",
    "CAST({a} AS BIGINT)",
    "CAST({a} AS STRING)",
    "CAST({a} AS BOOLEAN)",
    "CAST({a} AS DECIMAL(10,2))",
    "CAST({a} AS DECIMAL(38,10))",
    "CAST({a} AS FLOAT)",
    "CAST({a} AS DOUBLE)",
    "try_cast({a} AS INT)",
    "try_cast({a} AS BIGINT)",
    "try_cast({a} AS DECIMAL(10,2))",
    "hash({a})",
    "xxhash64({a})",
    "array_distinct(array({a}, {a}))",
    "array_max(array({a}, 1))",
    "array_min(array({a}, 1))",
    "sort_array(array({a}, 1, 0))",
    "size(collect_set({a}) OVER (PARTITION BY p))",
]
BINARY = [
    "({a} + {b})",
    "({a} - {b})",
    "({a} * {b})",
    "({a} / {b})",
    "({a} % {b})",
    "pmod({a}, {b})",
    "power({a}, {b})",
    "atan2({a}, {b})",
    "hypot({a}, {b})",
    "log({a}, {b})",
    "({a} = {b})",
    "({a} <> {b})",
    "({a} < {b})",
    "({a} <= {b})",
    "({a} > {b})",
    "({a} >= {b})",
    "({a} <=> {b})",
    "greatest({a}, {b})",
    "least({a}, {b})",
    "nullif({a}, {b})",
    "nanvl({a}, {b})",
    "coalesce({a}, {b})",
    "({a} BETWEEN {b} AND {b})",
    "({a} IN ({b}))",
    "({a} IN ({b}, 0.0))",
    "if({a} > {b}, {a}, {b})",
    "try_add({a}, {b})",
    "try_subtract({a}, {b})",
    "try_multiply({a}, {b})",
    "try_divide({a}, {b})",
    "array_contains(array({a}), {b})",
    "array_position(array({a}), {b})",
    "array_union(array({a}), array({b}))",
    "array_intersect(array({a}), array({b}))",
    "array_except(array({a}), array({b}))",
    "element_at(map({a}, 1), {b})",
    "array_sort(array({a}, {b}))",
    "array_max(array({a}, {b}))",
    "array_min(array({a}, {b}))",
]
WINDOW = [
    "sum({v}) OVER (PARTITION BY p)",
    "avg({v}) OVER (PARTITION BY p)",
    "min({v}) OVER (PARTITION BY p)",
    "max({v}) OVER (PARTITION BY p)",
    "count({v}) OVER (PARTITION BY p)",
    "var_pop({v}) OVER (PARTITION BY p)",
    "var_samp({v}) OVER (PARTITION BY p)",
    "stddev_pop({v}) OVER (PARTITION BY p)",
    "stddev_samp({v}) OVER (PARTITION BY p)",
    "rank() OVER (PARTITION BY p ORDER BY {v})",
    "dense_rank() OVER (PARTITION BY p ORDER BY {v})",
    "row_number() OVER (PARTITION BY p ORDER BY {v}, id)",
    "count(*) OVER (PARTITION BY {v})",
    "sum(id) OVER (PARTITION BY {v})",
    "count(DISTINCT {v}) OVER (PARTITION BY p)",
    "first_value({v}) OVER (PARTITION BY p ORDER BY {v})",
    "last_value({v}) OVER (PARTITION BY p ORDER BY {v} ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)",
    "max_by(id, {v}) OVER (PARTITION BY p)",
    "min_by(id, {v}) OVER (PARTITION BY p)",
]


def expressions():
    """Every SQL expression of the sweep, in a fixed order, without duplicates."""
    out = []
    for a in ("x", "fx"):
        out += [t.format(a=a) for t in UNARY]
    for a, b in (("x", "y"), ("fx", "fy"), ("x", "fy"), ("fx", "y")):
        out += [t.format(a=a, b=b) for t in BINARY]
    for v in ("d", "fd"):
        out += [t.format(v=v) for t in WINDOW]
    return list(dict.fromkeys(out))


def main():
    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("gen_expr_differential", here / "gen_expr_differential.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("float-edge-sweep")
        .config("spark.sql.catalogImplementation", "in-memory")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.datetime.java8API.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("OFF")
    hive = spark._jvm.org.apache.spark.sql.execution.HiveResult.hiveResultString
    setup = setup_sql()
    spark.sql(setup)
    exprs = expressions()
    cases = []
    for ansi in (True, False):
        spark.conf.set("spark.sql.ansi.enabled", "true" if ansi else "false")
        results = {}
        for lo in range(0, len(exprs), 16):
            results.update(base.run_batch(spark, hive, exprs[lo : lo + 16]))
        for e in exprs:
            case = {"sql": e, "ansi": ansi, "kids": [], "root": True}
            case.update(results[e])
            cases.append(case)
    json.dump({"spark": spark.version, "setup": setup, "cases": cases}, sys.stdout, indent=0)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
