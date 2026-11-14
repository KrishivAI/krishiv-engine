#!/usr/bin/env python3
"""Generate Spark 4.2 goldens for the moment / covariance aggregates (acceptance issue SQL-7).

Runs a REAL Spark 4.2 (never from memory) over Spark Connect. Every case is a pair of columns (y, x)
read from a literal array by `range(0, n, 1, P)` so the partitioning is stated and reproducible:
partition p holds the rows [floor(p*n/P), floor((p+1)*n/P)) in order (Spark's `range` slicing), each
partition folds its rows with the aggregate's update expressions, then the P partial buffers are merged
in shuffle-fetch order into an empty final buffer (P = 1 still does one merge into the empty buffer).
The result of a function may depend on P (and for P > 1 on the merge order, which for a local[N]
run is the order the reduce task reads the map outputs: map id ascending in practice).

For every (dataset, P, function) the fixture records the double's bits (`hex` of Double.doubleToLongBits),
the shortest decimal (`text`, Python repr, which round-trips), or `null` / an `error` condition.
Functions: kurtosis, skewness, var_*, stddev_* (over x) and covar_*, corr, regr_* (over (y, x)).
`NaN`/`Infinity` inputs are written as the strings "NaN", "Infinity", "-Infinity".

The big dataset is not stored: it is the splitmix64 stream described in `big_dataset` below (an
independent implementation reproduces it exactly); the fixture keeps the first rows and a checksum.

Usage (from repo root, with a Spark 4.2 Connect server at `$SPARK_REMOTE`, default sc://127.0.0.1:15002;
pass `--local1` is NOT needed: the partitioning is explicit):

    uv run --no-project --python 3.12 --with 'pyspark-client==4.2.0' \
        python tests/scripts/gen_moment_goldens.py > tests/fixtures/spark-4.2.0/moment-goldens.json
"""

import json
import math
import os
import struct
from decimal import Decimal

from pyspark.sql import SparkSession

MASK = (1 << 64) - 1
NAN, INF, NINF = "NaN", "Infinity", "-Infinity"

SINGLE_FNS = ["kurtosis", "skewness", "var_samp", "var_pop", "stddev_samp", "stddev_pop"]
PAIR_FNS = [
    "covar_samp", "covar_pop", "corr", "regr_count", "regr_avgx", "regr_avgy", "regr_sxx", "regr_syy",
    "regr_sxy", "regr_slope", "regr_intercept", "regr_r2",
]  # fmt: skip


def splitmix64(state):
    """One step: returns (new_state, output). Reproduce exactly in any language (wrapping u64)."""
    state = (state + 0x9E3779B97F4A7C15) & MASK
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return state, z ^ (z >> 31)


def unit(out):
    return (out >> 11) / float(1 << 53)  # [0, 1), exact


def big_dataset(n=10000, seed=7012):
    """x = 1000 + 200 * (u1 - 0.5), y = 0.5 * x + 30 * (u2 - 0.5); ~1% NULL in each column.

    Per row, in this order: u1, u2, u3, u4 from one splitmix64 stream (state starts at `seed`);
    x is NULL when u3 < 0.01, y is NULL when u4 < 0.01. Doubles are computed left to right in f64.
    """
    st, rows = seed, []
    for _ in range(n):
        st, a = splitmix64(st)
        st, b = splitmix64(st)
        st, c = splitmix64(st)
        st, d = splitmix64(st)
        x = 1000.0 + 200.0 * (unit(a) - 0.5)
        y = 0.5 * x + 30.0 * (unit(b) - 0.5)
        rows.append((None if unit(d) < 0.01 else y, None if unit(c) < 0.01 else x))
    return rows


# Small datasets: (name, sql type of the columns (y, x), rows (y, x)). Values are Python numbers,
# None, or the strings NaN/Infinity/-Infinity; decimals are strings.
SMALL = [
    ("issue_kurtosis_skewness", "double", [(0.0, -10.0), (0.0, -20.0), (0.0, 100.0), (0.0, 1000.0)]),
    ("issue_regr", "double", [(1.0, 2.0), (2.0, 2.0), (2.0, 3.0), (2.0, 4.0)]),
    ("doc_skew_neg", "double", [(0.0, -1000.0), (0.0, -100.0), (0.0, 10.0), (0.0, 20.0)]),
    ("doc_kurtosis_5", "double", [(0.0, 1.0), (0.0, 10.0), (0.0, 100.0), (0.0, 10.0), (0.0, 1.0)]),
    ("negatives", "double", [(-1.5, -2.5), (-3.25, -0.5), (-7.0, -9.75), (-0.125, -4.0), (-8.5, -1.0)]),
    (
        "big_offset",
        "double",
        [(1e9 + a, 1e9 + b) for a, b in [(0.1, 0.3), (0.2, 0.1), (0.4, 0.6), (0.3, 0.2), (0.5, 0.9)]],
    ),
    ("big_offset_ints", "double", [(1e9 + i, 1e9 + (i * 7 % 5)) for i in range(8)]),
    ("constant_both", "double", [(4.0, 4.0)] * 5),
    ("constant_x", "double", [(1.0, 3.0), (2.0, 3.0), (5.0, 3.0), (4.0, 3.0)]),
    ("constant_y", "double", [(3.0, 1.0), (3.0, 2.0), (3.0, 5.0), (3.0, 4.0)]),
    ("one_row", "double", [(2.5, 7.5)]),
    ("two_rows", "double", [(1.0, 2.0), (4.0, 8.0)]),
    ("two_rows_equal", "double", [(3.0, 3.0), (3.0, 3.0)]),
    ("with_nan", "double", [(1.0, 2.0), (2.0, NAN), (3.0, 4.0), (4.0, 7.0)]),
    ("with_infinity", "double", [(1.0, 2.0), (2.0, INF), (3.0, 4.0)]),
    ("with_nulls", "double", [(1.0, 2.0), (None, 3.0), (3.0, None), (4.0, 9.0), (None, None), (6.0, 1.0)]),
    ("all_null", "double", [(None, None), (None, None)]),
    ("tiny_values", "double", [(1e-9 * i, 1e-9 * (i * i % 7 + 1)) for i in range(1, 9)]),
    ("int_columns", "int", [(1, 2), (5, 3), (9, 4), (2, 10), (7, 7), (-3, 1)]),
    (
        "decimal_columns",
        "decimal(10,2)",
        [("1.10", "2.20"), ("3.35", "0.05"), ("-4.40", "7.75"), ("9.99", "1.01"), ("0.01", "5.55")],
    ),
]  # fmt: skip


def lit(v, sqltype):
    if v is None:
        return f"cast(null as {sqltype})"
    if isinstance(v, str):
        if v in (NAN, INF, NINF):
            return {NAN: "double('NaN')", INF: "double('Infinity')", NINF: "double('-Infinity')"}[v]
        return f"cast('{v}' as {sqltype})"
    if sqltype == "double":
        return f"cast('{v!r}' as double)"  # the text round-trips the double exactly
    return f"cast({v} as {sqltype})"


def arr(rows, idx, sqltype):
    return "array(" + ",".join(lit(r[idx], sqltype) for r in rows) + ")"


def view(spark, rows, sqltype, parts):
    n = len(rows)
    spark.sql(
        f"create or replace temp view t as select element_at({arr(rows, 0, sqltype)}, cast(id + 1 as int)) y, "
        f"element_at({arr(rows, 1, sqltype)}, cast(id + 1 as int)) x from range(0, {n}, 1, {parts})"
    )


def bits(v):
    return struct.pack(">d", v).hex()


def cell(spark, fn):
    args = "y, x" if fn in PAIR_FNS else "x"
    try:
        v = spark.sql(f"select {fn}({args}) from t").collect()[0][0]
    except Exception as e:
        cond = getattr(e, "getErrorClass", lambda: None)() or str(e).split("\n")[0][:200]
        return {"error": cond}
    if v is None:
        return {"null": True}
    if isinstance(v, Decimal):  # regr_avgx/avgy over decimal columns: Average gives a decimal
        return {"decimal": str(v)}
    if isinstance(v, int):  # regr_count
        return {"long": v}
    return {"hex": bits(v), "text": repr(v)}


def slices(n, p):
    return [[(i * n) // p, ((i + 1) * n) // p] for i in range(p)]


def main():
    remote = os.environ.get("SPARK_REMOTE", "sc://127.0.0.1:15002")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    assert spark.version.startswith("4.2"), spark.version
    fns = SINGLE_FNS + PAIR_FNS
    cases = []
    for name, sqltype, rows in SMALL:
        for parts in (1, 2):
            if parts > len(rows):
                continue
            view(spark, rows, sqltype, parts)
            cases.append({
                "dataset": name, "type": sqltype, "partitions": parts, "slices": slices(len(rows), parts),
                "rows": [[a, b] for a, b in rows],
                "results": {fn: cell(spark, fn) for fn in fns},
            })  # fmt: skip
    big = big_dataset()
    for parts in (1, 4):
        view(spark, big, "double", parts)
        cases.append({
            "dataset": "big_splitmix64", "type": "double", "partitions": parts, "slices": slices(len(big), parts),
            "first_rows": [[a, b] for a, b in big[:5]],
            "null_counts": [sum(r[0] is None for r in big), sum(r[1] is None for r in big)],
            "sum_x_nonnull": math.fsum(r[1] for r in big if r[1] is not None),
            "results": {fn: cell(spark, fn) for fn in fns},
        })  # fmt: skip
    out = {
        "spark": spark.version,
        "ansi": spark.conf.get("spark.sql.ansi.enabled"),
        "generator": "tests/scripts/gen_moment_goldens.py",
        "big_dataset": {"n": 10000, "seed": 7012, "rule": big_dataset.__doc__.strip()},
        "cases": cases,
    }
    print(json.dumps(out, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
