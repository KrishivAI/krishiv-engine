#!/usr/bin/env python3
"""Generate Spark 4.2 goldens for `percentile_approx` / `approx_percentile` (acceptance issue SQL-12).

Runs a REAL Spark 4.2 (never from memory) over Spark Connect. Data is generated inside Spark from
`range(0, n, 1, P)` (`id` = 0..n-1) with integer arithmetic only, so any language reproduces it:

    u(id)   = (id * 2654435761 + 12345) % 1000003            (64-bit integers, no overflow for id < 2^31)
    w(id)   = (u(id) * 7919) % 10000019
    uniform_int : u              (INT)        repeated : u % 17           (INT)
    asc : id (BIGINT)            desc : n - 1 - id (BIGINT)
    double : u / 997.0 (DOUBLE, one IEEE division)
    decimal : w / 100 as DECIMAL(12,2)  (unscaled value w, scale 2: exact)
    with_nulls : u, NULL when id % 10 = 3 (INT)

Partitions: `range(0, n, 1, P)` gives partition p the ids [floor(p*n/P), floor((p+1)*n/P)) in order (Spark's
range slicing); every partition folds its values into its own summary (insert, head buffer 50000), the
partials are serialized compressed and merged in reduce-read order into an empty buffer, P = 1 included
(a single merge into an empty summary). Every (dataset, n, P, accuracy) query asks all percentiles at
once; each is its own aggregate expression (own summary). `scalars` are `percentile_approx(v, p, acc)`
for p in PCTS; `array_sorted` is the array form over PCTS, `array_unsorted` over [0.9, 0.1, 0.5].
Doubles are recorded as `{hex, text}` (Double.doubleToLongBits and the shortest decimal); ints as numbers; decimals
as strings. NULL (empty input) is `null`.

Each (n, P) with P > 1 is run `REPEATS` times; `stable` says whether all runs agree (merge order is the
order a reduce task reads the map outputs, not fixed by Spark's contract).

Usage (repo root, Spark 4.2 Connect server at `$SPARK_REMOTE`, default sc://127.0.0.1:15002):

    uv run --no-project --python 3.12 --with 'pyspark-client==4.2.0' \
        python tests/scripts/gen_quantile_goldens.py > tests/fixtures/spark-4.2.0/quantile-goldens.json
"""

import json
import os
import struct
import sys
from decimal import Decimal

from pyspark.sql import SparkSession

SIZES = [1, 2, 10, 100, 1000, 60000, 1500000]
PCTS = [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]
UNSORTED = [0.9, 0.1, 0.5]
ACCS = [1, 10, 100, 10000, 1000000]
PARTS = [1, 4]
REPEATS = 2

U = "((id * 2654435761 + 12345) % 1000003)"
W = f"({U} * 7919 % 10000019)"
DISTS = {
    "uniform_int": (U, "int"),
    "repeated": (f"({U} % 17)", "int"),
    "asc": ("id", "bigint"),
    "desc": ("(N - 1 - id)", "bigint"),
    "double": (f"({U} / 997.0D)", "double"),
    "decimal": (f"cast(cast({W} as decimal(14,0)) / 100 as decimal(12,2))", "decimal(12,2)"),
    "with_nulls": (f"(case when id % 10 = 3 then null else {U} end)", "int"),
}


def conv(v):
    if v is None:
        return None
    if isinstance(v, list):
        return [conv(x) for x in v]
    if isinstance(v, float):
        return {"hex": struct.pack(">d", v).hex(), "text": repr(v)}
    if isinstance(v, Decimal):
        return str(v)
    return v


def query(spark, expr, n, parts, acc):
    e = expr.replace("N", str(n))
    sc = ", ".join(f"percentile_approx(v, {p!r}, {acc})" for p in PCTS)
    ar = f"percentile_approx(v, array({', '.join(repr(p) for p in PCTS)}), {acc})"
    au = f"percentile_approx(v, array({', '.join(repr(p) for p in UNSORTED)}), {acc})"
    row = spark.sql(f"select {sc}, {ar}, {au} from (select {e} as v from range(0, {n}, 1, {parts}))").collect()[0]
    vals = [conv(x) for x in row]
    return {
        "scalars": dict(zip(map(repr, PCTS), vals[: len(PCTS)], strict=True)),
        "array_sorted": vals[-2],
        "array_unsorted": vals[-1],
    }


def main():
    remote = os.environ.get("SPARK_REMOTE", "sc://127.0.0.1:15002")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    assert spark.version.startswith("4.2"), spark.version
    cases = []
    for dist, (expr, typ) in DISTS.items():
        for n in SIZES:
            for parts in PARTS:
                for acc in ACCS:
                    runs = [query(spark, expr, n, parts, acc) for _ in range(REPEATS if parts > 1 else 1)]
                    case = {
                        "dist": dist, "type": typ, "n": n, "partitions": parts, "accuracy": acc,
                        "slices": [[(i * n) // parts, ((i + 1) * n) // parts] for i in range(parts)],
                        "stable": all(r == runs[0] for r in runs),
                        **runs[0],
                    }  # fmt: skip
                    cases.append(case)
                    print(dist, n, parts, acc, case["stable"], file=sys.stderr, flush=True)
    out = {
        "spark": spark.version,
        "generator": "tests/scripts/gen_quantile_goldens.py",
        "pcts": PCTS,
        "unsorted_pcts": UNSORTED,
        "rules": __doc__.split("Usage")[0].strip(),
        "cases": cases,
    }
    print(json.dumps(out, indent=None, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
