# SPDX-License-Identifier: Apache-2.0
"""Runs the common calls of a stock pyspark-client (4.3.0 in the 0.3 release check) against a Krishiv server.

Usage: connect_client_probe.py sc://localhost:15002. Each call must work: a SQL_CONF_NOT_FOUND here means the
server lacks a conf the client reads. `show` needs SHOW_STRING (unimplemented): it is the one known failure.
"""

import sys

from pyspark.sql import SparkSession

spark = SparkSession.builder.remote(sys.argv[1]).getOrCreate()
checks = {
    "collect": lambda: spark.sql("SELECT 1 AS a").collect(),
    "toPandas": lambda: spark.sql("SELECT version() AS krishiv, 1 AS n, 'x' AS s").toPandas(),
    "toPandas struct/array": lambda: spark.sql(
        "SELECT named_struct('a',1) AS st, array(1,2) AS ar, map('k',1) AS m"
    ).toPandas(),
    "createDataFrame ddl": lambda: spark.createDataFrame([(1, "a")], "id INT, s STRING").collect(),
    "createDataFrame names": lambda: spark.createDataFrame([(1, "a")], ["id", "s"]).collect(),
    "createDataFrame pandas": lambda: spark.createDataFrame(__import__("pandas").DataFrame({"a": [1, 2]})).collect(),
    "show (known: SHOW_STRING)": lambda: spark.range(3).show(),
    "count": lambda: spark.range(5).count(),
    "schema": lambda: spark.range(1).schema,
    "conf.get tz": lambda: spark.conf.get("spark.sql.session.timeZone"),
    "eagerEval": lambda: spark.conf.get("spark.sql.repl.eagerEval.enabled"),
    "error path": lambda: spark.sql("SELECT * FROM no_such_table").collect(),
}
bad = 0
for name, fn in checks.items():
    try:
        fn()
        print("ok  ", name)
    except Exception as e:
        msg = str(e).splitlines()[0][:200]
        conf = "SQL_CONF_NOT_FOUND" in str(e)
        expected = name == "error path" and not conf
        print("ok  " if expected else "FAIL", name, "-", msg)
        # A known gap is reported, not counted, unless it is the conf failure this script exists to catch.
        known = "known" in name and not conf
        bad += 0 if expected or known else 1
sys.exit(1 if bad else 0)
