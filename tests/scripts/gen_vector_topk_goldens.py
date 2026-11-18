#!/usr/bin/env python3
"""Generate Spark 4.2 goldens for exact top-k vector search (P3-T077, PR-33).

Runs a REAL Spark 4.2 (never from memory) over Spark Connect. The table is rebuilt from a recipe, not stored:
row `i` of `n` has `id = i` and the vector `random_vector(dim, 100 + i, role 0, 1.0)` of gen_vector_goldens.py
(`float32((unit - 0.5) * 2 * scale)`, splitmix64 seeded `seed * 2 + role`), except these rows:
  i % 97 == 5: a NULL vector;  i == 13: element 3 is NaN;  i == 29: all zeros;
  i == 41: element 2 is NULL;  i == 57: element 0 is +Infinity.
The query vector is `random_vector(dim, 999, 1, 1.0)`.

Queries (k in 1, 10, 50): `ORDER BY vector_l2_distance(v, q)` (ASC, NULLS FIRST by default), the same with
NULLS LAST, `ORDER BY vector_cosine_similarity(v, q) DESC` and `ORDER BY vector_inner_product(v, q) DESC`
(DESC, NULLS LAST by default), each with LIMIT k. A cell is the list of `[id, bits]` (the distance's float bits,
or null). The `mismatch` case is a table where row 7 has one element too few: the query raises.

Usage (repo root, a Spark 4.2 Connect server at `$SPARK_REMOTE`):

    uv run --no-project --python 3.12 --with 'pyspark-client==4.2.0' \
        python tests/scripts/gen_vector_topk_goldens.py > tests/fixtures/spark-4.2.0/vector-topk-goldens.json
"""

import json
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StructField, StructType

sys.path.insert(0, os.path.dirname(__file__))
from gen_vector_goldens import bits, error_of, random_vector

NAN, INF = float("nan"), float("inf")
SCHEMA = StructType([StructField("id", IntegerType(), False), StructField("v", ArrayType(FloatType(), True), True)])
ORDERS = {
    "l2": "vector_l2_distance(v, {q}) ASC",
    "l2_nulls_last": "vector_l2_distance(v, {q}) ASC NULLS LAST",
    "cos": "vector_cosine_similarity(v, {q}) DESC",
    "ip": "vector_inner_product(v, {q}) DESC",
}
FN = {"l2": "vector_l2_distance", "l2_nulls_last": "vector_l2_distance",
      "cos": "vector_cosine_similarity", "ip": "vector_inner_product"}  # fmt: skip


def table_rows(n, dim, short_row=None):
    rows = []
    for i in range(n):
        v = random_vector(dim, 100 + i, 0, 1.0)
        if i % 97 == 5:
            v = None
        elif i == 13:
            v[3] = NAN
        elif i == 29:
            v = [0.0] * dim
        elif i == 41:
            v[2] = None
        elif i == 57:
            v[0] = INF
        if short_row == i and v is not None:
            v = v[:-1]
        rows.append((i, v))
    return rows


def literal(q):
    return "array(" + ", ".join(f"CAST({x!r} AS FLOAT)" for x in q) + ")"


def case(spark, n, dim, short_row=None):
    spark.createDataFrame(table_rows(n, dim, short_row), SCHEMA).createOrReplaceTempView("vt")
    q = random_vector(dim, 999, 1, 1.0)
    out = {"n": n, "dim": dim, "short_row": short_row, "results": {}}
    for name, order in ORDERS.items():
        for k in (1, 10, 50):
            sql = f"SELECT id, {FN[name]}(v, {literal(q)}) AS d FROM vt ORDER BY {order.format(q=literal(q))} LIMIT {k}"
            try:
                rows = [[r[0], None if r[1] is None else bits(r[1])] for r in spark.sql(sql).collect()]
            except Exception as e:  # every Spark failure is recorded as a cell
                rows = error_of(e)
            out["results"][f"{name}/{k}"] = rows
    return out


def main():
    spark = SparkSession.builder.remote(os.environ.get("SPARK_REMOTE", "sc://127.0.0.1:15002")).getOrCreate()
    cases = [case(spark, 400, 768), case(spark, 600, 24), case(spark, 60, 16, short_row=7)]
    print(f'{{"spark": {json.dumps(spark.version)},\n"cases": [')
    print(",\n".join(json.dumps(c, separators=(",", ":")) for c in cases))
    print("]}")


if __name__ == "__main__":
    main()
