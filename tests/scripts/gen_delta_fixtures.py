# SPDX-License-Identifier: Apache-2.0
"""Oracle-written Delta tables for the Delta connector's checkpoint tests (P3-T021 PR-08a).

Writes small tables with Spark 4.2.0 + delta-spark 4.4.1, each with a checkpoint of a
different shape:

- ``v1-checkpoint``: a classic (V1) single-file checkpoint at version 3, partitioned, with
  ``stats_parsed`` (``delta.checkpoint.writeStatsAsStruct``) and a later JSON commit;
- ``v2-checkpoint``: ``delta.checkpointPolicy = v2``: delta-spark writes a JSON top-level
  checkpoint and one Parquet sidecar under ``_delta_log/_sidecars/``;
- ``multipart-checkpoint``: a classic checkpoint split into parts
  (``spark.databricks.delta.checkpoint.partSize = 1``).

Run it inside the Spark image (it fetches delta-spark from Maven Central):

    docker run --rm --user "$(id -u)" -e HOME=/tmp -v "$OUT:/out" \
        -v "$PWD/tests/scripts:/scripts:ro" --entrypoint /opt/spark/bin/spark-submit \
        apache/spark:4.2.0 \
        --packages io.delta:delta-spark_2.13:4.4.1 --conf spark.jars.ivy=/tmp/ivy \
        /scripts/gen_delta_fixtures.py /out

then copy ``$OUT/*`` to ``tests/fixtures/delta-4.4.1/``. Each table's ``EXPECTED.json``
records the rows Spark reads back (ordered by ``id``) and the files of ``_delta_log``, so the
consuming test checks what the oracle wrote, never a hand-written value.
"""

import json
import os
import sys

from pyspark.sql import SparkSession

FORMAT_VERSION = 1


def session() -> SparkSession:
    return (
        SparkSession.builder.appName("krishiv-delta-fixtures")
        .master("local[1]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )


def write_expected(spark: SparkSession, path: str, notes: dict) -> None:
    rows = [r.asDict() for r in spark.read.format("delta").load(path).orderBy("id").collect()]
    log = sorted(
        os.path.relpath(os.path.join(d, f), os.path.join(path, "_delta_log"))
        for d, _, files in os.walk(os.path.join(path, "_delta_log"))
        for f in files
        if not f.startswith(".")
    )
    expected = {
        "format_version": FORMAT_VERSION,
        "generator": "tests/scripts/gen_delta_fixtures.py",
        "spark": spark.version,
        "delta_spark": "4.4.1",
        "rows": rows,
        "delta_log": log,
        **notes,
    }
    with open(os.path.join(path, "EXPECTED.json"), "w") as f:
        json.dump(expected, f, indent=1, sort_keys=True, default=str)
        f.write("\n")


def table(spark: SparkSession, path: str, props: str) -> None:
    spark.sql(
        f"CREATE TABLE delta.`{path}` (id BIGINT, p STRING, t TIMESTAMP) "
        f"USING delta PARTITIONED BY (p) TBLPROPERTIES ({props})"
    )
    for v in range(3):
        spark.sql(
            f"INSERT INTO delta.`{path}` VALUES "
            f"({v * 2}, 'a', TIMESTAMP'2024-01-0{v + 1} 00:00:00'), "
            f"({v * 2 + 1}, 'b', NULL)"
        )


def main(out: str) -> None:
    spark = session()

    v1 = os.path.join(out, "v1-checkpoint")
    table(spark, v1, "'delta.checkpointInterval' = '3'")
    # The checkpoint is at version 3 (interval 3); one more commit after it.
    spark.sql(f"DELETE FROM delta.`{v1}` WHERE id = 0")
    write_expected(spark, v1, {"checkpoint": "classic"})

    v2 = os.path.join(out, "v2-checkpoint")
    table(
        spark,
        v2,
        "'delta.checkpointPolicy' = 'v2', 'delta.checkpointInterval' = '3'",
    )
    spark.sql(f"DELETE FROM delta.`{v2}` WHERE id = 0")
    write_expected(spark, v2, {"checkpoint": "v2"})

    spark.conf.set("spark.databricks.delta.checkpoint.partSize", "1")
    multi = os.path.join(out, "multipart-checkpoint")
    table(spark, multi, "'delta.checkpointInterval' = '3'")
    write_expected(spark, multi, {"checkpoint": "multipart"})
    spark.stop()


if __name__ == "__main__":
    main(sys.argv[1])
