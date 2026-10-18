#!/usr/bin/env python3
"""Generate Spark `ORDER BY` golden fixtures for the sort operator (P2-T054).

Runs a REAL Spark (never from memory) in a throw-away container, single partition, and records the order
of row ids for each ORDER BY case: NULLS FIRST/LAST x ASC/DESC x NaN/-0.0 x multi-key. Every case ends with
`id ASC`, so ties (-0.0 vs 0.0, NaN vs -NaN) are broken by id and the golden is fully determined by Spark's
comparison semantics, not by the stability of its sort. Usage (from repo root):

    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:4.2.0 \
        /opt/spark/bin/spark-submit --master 'local[1]' /work/gen_sort_goldens.py \
        > tests/fixtures/spark-4.2.0/sort-ordering.json

The fixture goes to stdout (the container user cannot write the mount); Spark logs go to stderr.
"""

import datetime
import json
import sys

# (id, d DOUBLE, f FLOAT, s STRING, i INT); values are SQL literals. `-NaN` has the sign bit set.
ROWS = [
    (1, "1.0", "1.0", "'b'", "1"),
    (2, "-0.0", "-0.0", "'a'", "2"),
    (3, "0.0", "0.0", "'B'", "NULL"),
    (4, "NaN", "NaN", "NULL", "1"),
    (5, "NULL", "NULL", "'aa'", "3"),
    (6, "-Infinity", "-Infinity", "''", "2"),
    (7, "Infinity", "Infinity", "'é'", "NULL"),
    (8, "-NaN", "-NaN", "'a'", "1"),
    (9, "-1.5", "-1.5", "'Z'", "-2147483648"),
    (10, "NULL", "NULL", "'b'", "NULL"),
    (11, "0.0", "-0.0", "'ab'", "2"),
    (12, "-0.0", "0.0", "'éa'", "2147483647"),
    (13, "1.7976931348623157E308", "3.4028235E38", "'\U0001f600'", "0"),
    (14, "4.9E-324", "1.4E-45", "'a\u0000'", "-1"),
    (15, "NaN", "-NaN", "'A'", "3"),
]

# Each case: list of (column, direction, nulls) with nulls None for Spark's default; `id ASC` is appended.
CASES = [[(c, d, n)] for c in ["d", "f", "s", "i"] for d in ["ASC", "DESC"] for n in [None, "FIRST", "LAST"]] + [
    [("i", "ASC", None), ("d", "DESC", None)],
    [("i", "DESC", "FIRST"), ("f", "ASC", "LAST")],
    [("s", "DESC", None), ("i", "ASC", "LAST")],
    [("d", "ASC", "LAST"), ("s", "ASC", "FIRST")],
    [("f", "DESC", "LAST"), ("s", "DESC", "FIRST"), ("i", "ASC", None)],
]


def literal(column, text):
    if text == "NULL":
        return "NULL"
    if column in ("d", "f"):
        ty = "DOUBLE" if column == "d" else "FLOAT"
        if text == "-NaN":
            return f"-CAST('NaN' AS {ty})"
        return f"CAST('{text}' AS {ty})"
    return text


def order_by(case):
    keys = [f"{c} {d}" + (f" NULLS {n}" if n else "") for c, d, n in case]
    return ", ".join([*keys, "id ASC"])


def main():

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[1]").getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions", "1")
    values = ", ".join(
        "(" + ", ".join([str(r[0])] + [literal(c, v) for c, v in zip("dfsi", r[1:], strict=True)]) + ")" for r in ROWS
    )
    spark.sql(
        "SELECT CAST(id AS INT) id, CAST(d AS DOUBLE) d, CAST(f AS FLOAT) f, CAST(s AS STRING) s, CAST(i AS INT) i "
        f"FROM VALUES {values} AS t(id, d, f, s, i)"
    ).createOrReplaceTempView("t")
    cases = []
    for case in CASES:
        sql = f"SELECT id FROM t ORDER BY {order_by(case)}"
        ids = [row["id"] for row in spark.sql(sql).collect()]
        keys = [
            {"column": c, "descending": d == "DESC", "nulls_first": None if n is None else n == "FIRST"}
            for c, d, n in case
        ]
        cases.append({"sql": sql, "keys": keys, "ids": ids})
    doc = {
        "spark_version": spark.version,
        # The Spark image ships Python 3.10, which has no `datetime.UTC`.
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
        "generator": "tests/scripts/gen_sort_goldens.py",
        "notes": "Row ids in Spark's output order; every ORDER BY ends with `id ASC`; nulls_first null: default.",
        "columns": ["id", "d", "f", "s", "i"],
        "rows": [list(r) for r in ROWS],
        "cases": cases,
    }
    sys.stdout.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
