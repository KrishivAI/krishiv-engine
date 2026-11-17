# SPDX-License-Identifier: Apache-2.0
"""Oracle-written Iceberg tables for the Iceberg connector's read-direction conformance (P3-T028a, PR-53a).

Writes small tables with Spark 4.1.0 + iceberg-spark-runtime-4.1_2.13 1.12.0 (the oracle pair the owner chose on
2026-11-06: Iceberg has no runtime for Spark 4.2 or 4.3). A Hadoop catalog with the warehouse at ``/wh`` in the
container, so every path in the metadata and manifests is ``file:/wh/db/<table>/...``; the consuming test serves the
output directory as the root of its file system and the paths resolve without rewriting any Avro.

Run it from the repository root:

    OUT=$(mktemp -d)
    docker run --rm --user "$(id -u)" -e HOME=/tmp -v "$OUT:/wh" \
        -v "$PWD/tests/scripts:/scripts:ro" --entrypoint /opt/spark/bin/spark-submit \
        apache/spark:4.1.0 \
        --packages org.apache.iceberg:iceberg-spark-runtime-4.1_2.13:1.12.0 --conf spark.jars.ivy=/tmp/ivy \
        /scripts/gen_iceberg_fixtures.py /wh

then replace ``tests/fixtures/iceberg-1.12.0/`` with ``$OUT`` (the ``wh`` directory the test serves is the output
directory itself). ``EXPECTED.json`` (next to each table's directory, ``wh/db/<table>.expected.json`` is NOT used:
one ``EXPECTED.json`` at the root) records for every table the rows Spark reads back, the current metadata file
and the snapshot ids oldest first, so the test checks what the oracle wrote, never a hand-written value.

Tables: ``basic`` (primitive types, two appends), ``partitioned`` (identity, bucket and day transforms),
``pos_deletes`` (merge-on-read DELETE and UPDATE: position delete files), ``dvs`` (format version 3: deletion
vectors), ``evolved`` (add, rename, drop, promote a column; rows before and after), ``nested`` (struct, array, map).
Equality deletes are not here: Spark cannot write them (Flink does); PR-25c's own tests cover them.
"""

import glob
import json
import os
import sys

from pyspark.sql import SparkSession

WAREHOUSE = "/wh"


def spark_session() -> SparkSession:
    return (
        SparkSession.builder.master("local[2]")
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.catalog.o", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.o.type", "hadoop")
        .config("spark.sql.catalog.o.warehouse", WAREHOUSE)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def create_tables(spark: SparkSession) -> None:
    sql = spark.sql
    sql("CREATE NAMESPACE IF NOT EXISTS o.db")

    sql(
        "CREATE TABLE o.db.basic (id BIGINT, i INT, s STRING, d DOUBLE, dec DECIMAL(10,2), dt DATE, "
        "ts TIMESTAMP, b BOOLEAN, bin BINARY) USING iceberg TBLPROPERTIES ('format-version'='2')"
    )
    sql(
        "INSERT INTO o.db.basic VALUES "
        "(1, 10, 'one', 1.5, 12.34, DATE'2024-01-02', TIMESTAMP'2024-01-02 03:04:05', true, X'01'), "
        "(2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL), "
        "(3, -30, 'three, \"quoted\"', -3.25, -0.01, DATE'1969-12-31', TIMESTAMP'1969-12-31 23:59:59', false, X'')"
    )
    sql(
        "INSERT INTO o.db.basic VALUES "
        "(4, 40, 'ünï', 4.0, 99999999.99, DATE'2100-12-31', TIMESTAMP'2100-12-31 00:00:00', true, X'ff00')"
    )

    sql(
        "CREATE TABLE o.db.partitioned (id BIGINT, region STRING, ts TIMESTAMP, v INT) USING iceberg "
        "PARTITIONED BY (region, bucket(4, id), days(ts)) TBLPROPERTIES ('format-version'='2')"
    )
    sql(
        "INSERT INTO o.db.partitioned VALUES "
        "(1, 'eu', TIMESTAMP'2024-03-01 10:00:00', 1), (2, 'eu', TIMESTAMP'2024-03-02 10:00:00', 2), "
        "(3, 'us', TIMESTAMP'2024-03-01 11:00:00', 3), (4, NULL, TIMESTAMP'2024-03-03 12:00:00', 4), "
        "(5, 'us', TIMESTAMP'2024-03-02 13:00:00', 5), (6, 'eu', TIMESTAMP'2024-03-01 14:00:00', 6)"
    )

    sql(
        "CREATE TABLE o.db.pos_deletes (id BIGINT, name STRING, v INT) USING iceberg TBLPROPERTIES ("
        "'format-version'='2', 'write.delete.mode'='merge-on-read', 'write.update.mode'='merge-on-read', "
        "'write.merge.mode'='merge-on-read')"
    )
    sql("INSERT INTO o.db.pos_deletes SELECT id, concat('n', id), CAST(id AS INT) * 10 FROM range(1, 21)")
    sql("DELETE FROM o.db.pos_deletes WHERE id IN (3, 7, 11)")
    sql("UPDATE o.db.pos_deletes SET v = -1 WHERE id % 5 = 0")

    sql(
        "CREATE TABLE o.db.dvs (id BIGINT, name STRING) USING iceberg TBLPROPERTIES ("
        "'format-version'='3', 'write.delete.mode'='merge-on-read', 'write.update.mode'='merge-on-read')"
    )
    sql("INSERT INTO o.db.dvs SELECT id, concat('n', id) FROM range(1, 31)")
    sql("DELETE FROM o.db.dvs WHERE id % 4 = 0")
    sql("DELETE FROM o.db.dvs WHERE id IN (1, 2)")

    sql("CREATE TABLE o.db.evolved (id BIGINT, a INT, gone STRING) USING iceberg TBLPROPERTIES ('format-version'='2')")
    sql("INSERT INTO o.db.evolved VALUES (1, 10, 'x'), (2, 20, 'y')")
    sql("ALTER TABLE o.db.evolved ADD COLUMN extra STRING")
    sql("INSERT INTO o.db.evolved VALUES (3, 30, 'z', 'e3')")
    sql("ALTER TABLE o.db.evolved RENAME COLUMN a TO renamed")
    sql("ALTER TABLE o.db.evolved ALTER COLUMN renamed TYPE BIGINT")
    sql("ALTER TABLE o.db.evolved DROP COLUMN gone")
    sql("INSERT INTO o.db.evolved VALUES (4, 9999999999, 'e4')")

    sql(
        "CREATE TABLE o.db.nested (id BIGINT, st STRUCT<a: INT, b: STRING>, arr ARRAY<INT>, m MAP<STRING, INT>) "
        "USING iceberg TBLPROPERTIES ('format-version'='2')"
    )
    sql(
        "INSERT INTO o.db.nested VALUES "
        "(1, named_struct('a', 1, 'b', 'x'), array(1, 2, 3), map('k', 1)), "
        "(2, NULL, array(), map()), "
        "(3, named_struct('a', NULL, 'b', 'z'), NULL, NULL)"
    )


def typed(value):
    """A JSON-safe, type-tagged rendering (so the Rust test compares values, not Python's ``str``)."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return {"binary": bytes(value).hex()}
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {"map": [[typed(k), typed(v)] for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))]}
    if isinstance(value, (list, tuple)) and not hasattr(value, "asDict"):
        return [typed(v) for v in value]
    if hasattr(value, "asDict"):
        return {k: typed(v) for k, v in value.asDict().items()}
    return str(value)  # decimal, date, timestamp: ISO text, rendered the same way by the Rust side


def table_expected(spark: SparkSession, name: str) -> dict:
    df = spark.sql(f"SELECT * FROM o.db.{name} ORDER BY id")
    rows = [{k: typed(v) for k, v in r.asDict().items()} for r in df.collect()]
    snaps = [
        r["snapshot_id"]
        for r in spark.sql(
            f"SELECT snapshot_id FROM o.db.{name}.snapshots ORDER BY committed_at, snapshot_id"
        ).collect()
    ]
    meta_dir = os.path.join(WAREHOUSE, "db", name, "metadata")
    versions = sorted(
        glob.glob(os.path.join(meta_dir, "v*.metadata.json")),
        key=lambda p: int(os.path.basename(p)[1:].split(".")[0]),
    )
    return {
        "columns": df.columns,
        "rows": rows,
        "metadata": f"file:{versions[-1]}",
        "snapshots": snaps,
        "delete_files": [
            r.asDict()
            for r in spark.sql(f"SELECT content, file_format, record_count FROM o.db.{name}.delete_files").collect()
        ],
    }


def main() -> None:
    spark = spark_session()
    create_tables(spark)
    out = {n: table_expected(spark, n) for n in ["basic", "partitioned", "pos_deletes", "dvs", "evolved", "nested"]}
    # Spark's own answer for the time-travel reads of `evolved`: every snapshot, oldest first.
    out["evolved"]["as_of"] = {
        str(s): [r.asDict() for r in spark.sql(f"SELECT * FROM o.db.evolved VERSION AS OF {s} ORDER BY id").collect()]
        for s in out["evolved"]["snapshots"]
    }
    out["evolved"]["as_of"] = json.loads(json.dumps(out["evolved"]["as_of"], default=typed))
    out["_oracle"] = {"spark": spark.version, "iceberg": "1.12.0"}
    with open(os.path.join(WAREHOUSE, "EXPECTED.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=typed)
    spark.stop()


if __name__ == "__main__":
    if sys.argv[1:] != [WAREHOUSE]:
        sys.exit(f"usage: gen_iceberg_fixtures.py {WAREHOUSE}  (mount the output directory at {WAREHOUSE})")
    main()
