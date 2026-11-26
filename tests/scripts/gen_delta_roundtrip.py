# SPDX-License-Identifier: Apache-2.0
"""Spark 4.2.0 + delta-spark 4.4.1 and Krishiv on the same Delta tables (P3-T028b, PR-53b, round trip).

Spark starts each table and has the last word; Krishiv commits in between:

1. ``create``: Spark writes ``plain`` (protocol (1, 2)), ``part`` (partitioned by ``x``), ``dv`` (deletion vectors,
   one ``DELETE`` leaving a DV) and ``cdf`` (change data feed and deletion vectors) under ``<out>/delta`` and records
   what it reads, each table's version and protocol (``SPARK-CREATE.json``).
2. Krishiv (``roundtrip_tests::krishiv_commits_on_spark_tables`` in krishiv-connectors-delta) reads each table,
   commits an append (data files written by its Parquet handler, one per partition value), a ``DELETE`` that removes
   whole files (a partition of ``part``; on ``dv`` Spark's files, one with a deletion vector, and its own first file;
   its own first file elsewhere) and a second append, and records the version
   and its read after each commit (``KRISHIV-RT.json``).
3. ``readback``: Spark reads Krishiv's result, its history entries of Krishiv's commits, each Krishiv version by time
   travel and, on ``cdf``, the change feed of Krishiv's versions; then runs ``DELETE``, ``UPDATE`` and ``MERGE`` on
   top (deletion vectors against Krishiv's files on ``dv`` and ``cdf``) and an append, and records the metrics of
   those commits (``SPARK-READBACK.json``).
4. Krishiv (``roundtrip_tests::krishiv_reads_spark_on_top``) reads Spark's last version and its own versions again.

The ``sql_*`` tables (``sql_plain``, ``sql_part``, ``sql_dv``, made like ``plain``, ``part`` and ``dv``) take Krishiv's
SQL instead of direct commits (krishiv-session ``tests/delta_roundtrip.rs``, between steps 2 and 3, and after step 4):
an ``INSERT`` through Krishiv's Delta writer (its data files and statistics), a copy-on-write ``DELETE`` and
``UPDATE`` of rows in Spark's and Krishiv's files (on ``sql_dv`` a Spark file with a deletion vector), on
``sql_part`` a whole-file ``DELETE`` of a partition, and a second ``INSERT``, recorded in ``KRISHIV-SQL.json``. Step
3 reads them as it reads the others, with each Krishiv commit's ``operationMetrics``.

``race_k`` and ``race_s`` (partitioned by ``x``) are the concurrent-writer cases: on ``race_k`` two Krishiv commits
staged from one snapshot (appends both land; ``DELETE`` of the same files: one lands, the other conflicts; ``DELETE``
of different partitions: both land); ``race_s`` ends ``create`` with Spark's ``DELETE`` of partition ``x = 0``, and
Krishiv commits from the version before it (an append and a ``DELETE`` of ``x = 1`` land, a ``DELETE`` of ``x = 0``
conflicts). Step 3 reads them as the others.

No expected value is written by hand: every comparison is between the two engines' records. Run through
``tests/scripts/delta_roundtrip.sh`` (it mounts ``<out>`` at the same path).
"""

import json
import os
import sys

from pyspark.sql import SparkSession

# name -> (PARTITIONED BY clause, table properties)
DV = "'delta.enableDeletionVectors' = 'true'"
TABLES = {
    "plain": ("", ""),
    "part": ("PARTITIONED BY (x)", ""),
    "dv": ("", DV),
    "cdf": ("", f"{DV}, 'delta.enableChangeDataFeed' = 'true'"),
    # Krishiv's SQL on these (krishiv-session ``tests/delta_roundtrip.rs``).
    "sql_plain": ("", ""),
    "sql_part": ("PARTITIONED BY (x)", ""),
    "sql_dv": ("", DV),
    # Concurrent writers (``roundtrip_tests::race_tables``): two Krishiv commits from one snapshot on ``race_k``;
    # on ``race_s`` Krishiv commits from the version before Spark's ``DELETE`` of partition ``x = 0``.
    "race_k": ("PARTITIONED BY (x)", ""),
    "race_s": ("PARTITIONED BY (x)", ""),
}


def typed(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def rows_of(df):
    return [{k: typed(v) for k, v in r.asDict().items()} for r in df.collect()]


def path_of(out_dir, name):
    return os.path.join(out_dir, "delta", name)


def ref(out_dir, name):
    return f"delta.`{path_of(out_dir, name)}`"


def read(spark, t, version=None):
    at = "" if version is None else f" VERSION AS OF {version}"
    return rows_of(spark.sql(f"SELECT id, x, name FROM {t}{at} ORDER BY id NULLS FIRST, name"))


def state(spark, t):
    detail = spark.sql(f"DESCRIBE DETAIL {t}").collect()[0]
    version = spark.sql(f"DESCRIBE HISTORY {t}").agg({"version": "max"}).collect()[0][0]
    return {
        "rows": read(spark, t),
        "version": version,
        "protocol": [detail["minReaderVersion"], detail["minWriterVersion"], sorted(detail["tableFeatures"] or [])],
        "num_files": detail["numFiles"],
    }


def session():
    return (
        SparkSession.builder.master("local[2]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def values(ids):
    return ", ".join(f"({i}, {i % 3}, 'n{i}')" for i in ids)


def create(spark, out_dir):
    result = {}
    for name, (partitioned, props) in TABLES.items():
        t = ref(out_dir, name)
        tblprops = f" TBLPROPERTIES ({props})" if props else ""
        spark.sql(f"CREATE TABLE {t} (id BIGINT, x INT, name STRING) USING delta {partitioned}{tblprops}")
        spark.sql(f"INSERT INTO {t} VALUES {values(range(0, 12))}")
        if props:
            # A deletion vector on a Spark file.
            spark.sql(f"DELETE FROM {t} WHERE id IN (1, 5)")
        if name == "race_s":
            # Spark's winner: Krishiv commits from the version before it.
            spark.sql(f"DELETE FROM {t} WHERE x = 0")
        result[name] = state(spark, t)
    return result


def readback(spark, out_dir):
    krishiv = {}
    for record in ("KRISHIV-RT.json", "KRISHIV-SQL.json"):
        with open(os.path.join(out_dir, record)) as f:
            krishiv.update(json.load(f))
    result = {}
    for name in TABLES:
        t = ref(out_dir, name)
        versions = [s["version"] for s in krishiv[name]["steps"]]
        entry = {"before": state(spark, t)}
        history = spark.sql(f"DESCRIBE HISTORY {t}").collect()
        entry["history"] = {
            str(h["version"]): {
                "operation": h["operation"],
                "readVersion": h["readVersion"],
                "isolationLevel": h["isolationLevel"],
                "isBlindAppend": h["isBlindAppend"],
                "operationParameters": dict(h["operationParameters"] or {}),
                "operationMetrics": dict(h["operationMetrics"] or {}),
            }
            for h in history
            if h["version"] in versions
        }
        entry["at"] = {str(v): read(spark, t, v) for v in versions}
        if name == "cdf":
            entry["changes"] = rows_of(
                spark.read.format("delta")
                .option("readChangeFeed", "true")
                .option("startingVersion", min(versions))
                .option("endingVersion", max(versions))
                .load(path_of(out_dir, name))
                .selectExpr("id", "x", "name", "_change_type", "_commit_version")
                .orderBy("_commit_version", "_change_type", "id")
            )
        # Id 110 is live in Krishiv's last file on every table (a DV on it on ``dv`` and ``cdf``).
        spark.sql(f"DELETE FROM {t} WHERE id = 110")
        spark.sql(f"UPDATE {t} SET name = 's' WHERE id IN (0, 104)")
        spark.sql(
            f"MERGE INTO {t} d USING (SELECT * FROM VALUES (105L, 0, 'm'), (300L, 0, 'n300') AS s(id, x, name)) s "
            "ON d.id = s.id WHEN MATCHED THEN DELETE WHEN NOT MATCHED THEN INSERT *"
        )
        spark.sql(f"INSERT INTO {t} VALUES {values([400])}")
        entry["after"] = state(spark, t)
        # Spark's own DML commits and their metrics, so step 4 can see that each did work on Krishiv's files.
        entry["dml"] = {
            str(h["version"]): {
                "operation": h["operation"],
                "operationMetrics": dict(h["operationMetrics"] or {}),
            }
            for h in spark.sql(f"DESCRIBE HISTORY {t}").collect()
            if h["version"] > entry["before"]["version"]
        }
        result[name] = entry
    return result


def main(step, out_dir):
    spark = session()
    if step == "create":
        result, target = create(spark, out_dir), "SPARK-CREATE.json"
    else:
        result, target = readback(spark, out_dir), "SPARK-READBACK.json"
    result["_oracle"] = {"spark": spark.version, "delta": "4.4.1"}
    with open(os.path.join(out_dir, target), "w") as f:
        json.dump(result, f, indent=1, sort_keys=True, default=typed)
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("create", "readback"):
        sys.exit("usage: gen_delta_roundtrip.py create|readback <out dir>")
    main(sys.argv[1], sys.argv[2])
