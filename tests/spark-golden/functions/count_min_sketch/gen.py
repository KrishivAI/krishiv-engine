# SPDX-License-Identifier: Apache-2.0
"""Records what Spark answers for count_min_sketch: `python gen.py fixture.json` (PySpark 4.3.0, JDK 21)."""

import json
import sys

from pyspark.sql import SparkSession

INTS = "(1),(2),(2),(-2147483648),(2147483647),(CAST(NULL AS INT))"
VALUES = [
    "(CAST(1 AS TINYINT)),(CAST(-128 AS TINYINT)),(CAST(NULL AS TINYINT))",
    "(CAST(300 AS SMALLINT)),(CAST(-300 AS SMALLINT))",
    INTS,
    "(-9223372036854775808L),(9223372036854775807L),(5L),(-5L)",
    "('a'),('b'),('b'),(''),('héllo wörld 12345'),(CAST(NULL AS STRING))",
    "(X''),(X'00FF80'),(X'0102030405060708090A'),(X'FFFEFD')",
    "(CAST(NULL AS INT))",
    # An untyped NULL value is the statement `count_min_sketch(NULL, ...)` below.
]
QUERIES = [f"SELECT hex(count_min_sketch(c, 0.2d, 0.9d, 42)) FROM VALUES {v} t(c)" for v in VALUES]
ONE = "FROM VALUES (1) t(c)"
QUERIES += [
    f"SELECT hex(count_min_sketch(NULL, 0.2d, 0.9d, 42)) {ONE}",
]
QUERIES += [
    f"SELECT hex(count_min_sketch(c, {a})) {ONE}"
    for a in [
        "0.5d, 0.5d, 1",
        "0.5d, 0.5d, 1L",
        "0.5d, 0.5d, 4294967297L",
        "0.5d, 0.5d, -7",
        "0.5d, 0.5d, -2147483648",
        "2.0d, 0.5d, 1",
        "0.25d+0.25d, 0.5d, 1+1",
        "0.5d, CAST('NaN' AS DOUBLE), 1",
        "0.2d, 0.99d, 2147483647",
        "0.5, 0.5d, 1",
        "0.5d, 0.5f, 1",
        "'0.5', 0.5d, 1",
        "1, 0.5d, 1",
        "0.0d, 0.5d, 1",
        "-1.0d, 0.5d, 1",
        "0.5d, 0.0d, 1",
        "0.5d, 1.0d, 1",
        "0.5d, 1.5d, 1",
        "NULL, 0.5d, 1",
        "CAST(NULL AS DOUBLE), 0.5d, 1",
        "0.5d, NULL, 1",
        "0.5d, 0.5d, NULL",
        "c, 0.5d, 1",
        "0.5d, 0.5d, c",
        "rand(), 0.5d, 1",
        "0.5d, 0.5d, 1.5d",
        "0.5d, 0.5d, '1'",
        "0.5d, 0.5d",
        "0.5d, 0.5d, 1, 2",
    ]
]
QUERIES += [
    f"SELECT hex(count_min_sketch(c, 0.5d, 0.5d, 1)) FROM VALUES ({v}) t(c)"
    for v in [
        "1.5d",
        "1.5f",
        "CAST(1 AS DECIMAL(10,2))",
        "true",
        "DATE'2020-01-01'",
        "TIMESTAMP'2020-01-01 00:00:00'",
        "array(1)",
        "INTERVAL '1' DAY",
    ]
]
QUERIES += [
    f"SELECT hex(count_min_sketch(c, 0.5d, 0.5d, 1)) {ONE} WHERE false",
    "SELECT g, hex(count_min_sketch(c, 0.5d, 0.5d, -7)) FROM VALUES (1,1),(1,2),(2,CAST(NULL AS INT)) t(g,c) "
    "GROUP BY g ORDER BY g",
    "SELECT hex(count_min_sketch(DISTINCT c, 0.5d, 0.5d, 1)) FROM VALUES (1),(1),(2) t(c)",
    "SELECT hex(count_min_sketch(c, 0.5d, 0.5d, 1) FILTER (WHERE c > 1)) FROM VALUES (1),(1),(2) t(c)",
    "SELECT hex(count_min_sketch(confidence => 0.9d, seed => 3, column => c, epsilon => 0.2d)) FROM VALUES (7) t(c)",
    # Enough rows to cross several batches and partitions.
    "SELECT sha2(count_min_sketch(id % 1000, 0.01d, 0.95d, 11), 256) FROM range(0, 100000, 1, 8)",
    "SELECT id % 3 AS g, sha2(count_min_sketch(CAST(id AS STRING), 0.05d, 0.9d, 5), 256) "
    "FROM range(0, 30000, 1, 8) GROUP BY id % 3 ORDER BY g",
]


def main() -> None:
    spark = SparkSession.builder.master("local[2]").config("spark.sql.catalogImplementation", "in-memory").getOrCreate()
    cases = []
    for sql in QUERIES:
        try:
            df = spark.sql(sql)
            rows = ["|".join(str(v) for v in r) for r in df.collect()]
            cases.append({"sql": sql, "columns": df.columns, "rows": rows})
        except Exception as e:  # noqa: BLE001 - the error is the answer
            cases.append({"sql": sql, "error": e.getCondition(), "sqlstate": e.getSqlState()})
    with open(sys.argv[1], "w", encoding="utf-8") as out:
        json.dump({"spark": spark.version, "cases": cases}, out, indent=1, ensure_ascii=False)
        out.write("\n")


if __name__ == "__main__":
    main()
