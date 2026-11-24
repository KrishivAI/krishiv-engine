# SPDX-License-Identifier: Apache-2.0
"""Runs the common calls of a stock pyspark-client (4.3.0 in the 0.3 release check) against a Krishiv server.

Usage: connect_client_probe.py sc://localhost:15002. Each call must work: a SQL_CONF_NOT_FOUND here means the
server lacks a conf the client reads. `show` needs SHOW_STRING (unimplemented): it is the one known failure.
"""

import sys

from pyspark.sql import SparkSession

spark = SparkSession.builder.remote(sys.argv[1]).getOrCreate()


def rows(df):
    return [tuple(r) for r in df.collect()]


def expect(actual, expected):
    if actual != expected:
        raise AssertionError(f"expected {expected!r}, got {actual!r}")


def pandas_rows(pdf):
    return [tuple(r) for r in pdf.itertuples(index=False, name=None)]


def missing_table():
    try:
        spark.sql("SELECT * FROM no_such_table").collect()
    except Exception as e:
        if "TABLE_OR_VIEW_NOT_FOUND" not in str(e):
            raise AssertionError(f"expected TABLE_OR_VIEW_NOT_FOUND, got {str(e)[:200]}") from e
        return
    raise AssertionError("expected TABLE_OR_VIEW_NOT_FOUND, the query returned")


checks = {
    "collect": lambda: expect(rows(spark.sql("SELECT 1 AS a")), [(1,)]),
    "toPandas": lambda: expect(pandas_rows(spark.sql("SELECT 1 AS n, 'x' AS s").toPandas()), [(1, "x")]),
    "toPandas version": lambda: expect(len(spark.sql("SELECT version() AS krishiv").toPandas()), 1),
    "toPandas struct/array": lambda: expect(
        len(spark.sql("SELECT named_struct('a',1) AS st, array(1,2) AS ar, map('k',1) AS m").toPandas()), 1
    ),
    "createDataFrame ddl": lambda: expect(rows(spark.createDataFrame([(1, "a")], "id INT, s STRING")), [(1, "a")]),
    "createDataFrame names": lambda: expect(rows(spark.createDataFrame([(1, "a")], ["id", "s"])), [(1, "a")]),
    "createDataFrame pandas": lambda: expect(
        rows(spark.createDataFrame(__import__("pandas").DataFrame({"a": [1, 2]}))), [(1,), (2,)]
    ),
    "show (known: SHOW_STRING)": lambda: spark.range(3).show(),
    "count": lambda: expect(spark.range(5).count(), 5),
    "schema": lambda: expect(spark.range(1).schema.fieldNames(), ["id"]),
    "conf.get tz": lambda: expect(type(spark.conf.get("spark.sql.session.timeZone")), str),
    "eagerEval": lambda: expect(spark.conf.get("spark.sql.repl.eagerEval.enabled"), "false"),
    "error path": missing_table,
}
bad = 0
for name, fn in checks.items():
    try:
        fn()
        print("ok  ", name)
    except Exception as e:
        msg = str(e).splitlines()[0][:200]
        conf = "SQL_CONF_NOT_FOUND" in str(e)
        print("FAIL", name, "-", msg)
        # A known gap is reported, not counted, unless it is the conf failure this script exists to catch.
        known = "known" in name and not conf
        bad += 0 if known else 1
sys.exit(1 if bad else 0)
