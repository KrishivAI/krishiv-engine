# SPDX-License-Identifier: Apache-2.0
"""Spark 4.1.0 + Iceberg 1.12.0 and Krishiv on the same Iceberg tables (P3-T028b, PR-53b, round trip).

Spark starts each table and has the last word; Krishiv works in between:

1. ``create``: Spark writes tables Krishiv did not make, with what Spark's writers leave behind: merge-on-read
   position deletes (v2), deletion vectors (v3), and a table whose schema and partition spec evolved (a column
   promoted from ``int`` to ``bigint``, one renamed, unpartitioned to ``identity(x)``). It records what Spark reads
   (``SPARK-CREATE.json``).
2. Krishiv (``roundtrip_tests::krishiv_meets_spark_tables``) reads each table, appends, runs a copy-on-write
   ``DELETE`` and ``UPDATE`` that touch Spark's files and its own, and records its reads and snapshots
   (``KRISHIV-RT.json``); its last metadata becomes the Hadoop catalog's next ``v<N>.metadata.json``.
3. ``readback``: Spark reads Krishiv's result and its snapshots, time travels to each Krishiv snapshot, then runs
   merge-on-read ``DELETE``, ``UPDATE`` and ``MERGE`` on top (position deletes or DVs against Krishiv's data files)
   and an append (``SPARK-READBACK.json``).
4. Krishiv (``roundtrip_tests::krishiv_reads_spark_on_top``) reads Spark's last metadata and compares.

No expected value is written by hand: every comparison is between the two engines' records. Run through
``tests/scripts/iceberg_roundtrip.sh`` (it mounts ``<out>`` at the same path, since Iceberg metadata holds absolute
paths).
"""

import glob
import json
import os
import re
import sys

from pyspark.sql import SparkSession

# name -> (format version, evolved)
TABLES = {"mor2": (2, False), "mor3": (3, False), "evolved": (2, True)}
MOR = "'write.delete.mode'='merge-on-read', 'write.update.mode'='merge-on-read', 'write.merge.mode'='merge-on-read'"


def typed(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def rows_of(df):
    return [{k: typed(v) for k, v in r.asDict().items()} for r in df.collect()]


def latest_metadata(wh, name):
    meta = os.path.join(wh, "db", name, "metadata")
    versions = [
        p
        for p in glob.glob(os.path.join(meta, "v*.metadata.json"))
        if re.fullmatch(r"v\d+\.metadata\.json", os.path.basename(p))
    ]
    return max(versions, key=lambda p: int(os.path.basename(p)[1:].split(".")[0]))


def state(spark, wh, name, v3):
    t = f"o.db.{name}"
    out = {
        "rows": rows_of(spark.sql(f"SELECT id, x, name FROM {t} ORDER BY id NULLS FIRST, name")),
        "snapshots": rows_of(
            spark.sql(
                "SELECT snapshot_id, parent_id, operation, summary['added-records'] AS added_records, "
                "summary['deleted-records'] AS deleted_records, summary['total-records'] AS total_records, "
                "summary['total-data-files'] AS total_data_files, "
                "summary['total-position-deletes'] AS total_position_deletes "
                f"FROM {t}.snapshots ORDER BY committed_at, snapshot_id"
            )
        ),
        "delete_files": rows_of(
            spark.sql(f"SELECT content, file_format, record_count FROM {t}.delete_files ORDER BY file_path")
        ),
        "specs": [r["spec_id"] for r in spark.sql(f"SELECT DISTINCT spec_id FROM {t}.files ORDER BY 1").collect()],
        "metadata": "file:" + latest_metadata(wh, name),
    }
    if v3:
        out["row_ids"] = rows_of(spark.sql(f"SELECT id, _row_id FROM {t} ORDER BY id NULLS FIRST, name"))
    return out


def session(wh):
    return (
        SparkSession.builder.master("local[2]")
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.catalog.o", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.o.type", "hadoop")
        .config("spark.sql.catalog.o.warehouse", wh)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def values(ids):
    return ", ".join(f"({i}, {i % 3}, 'n{i}')" for i in ids)


def create(spark, wh):
    result = {}
    for name, (version, evolved) in TABLES.items():
        t = f"o.db.{name}"
        if evolved:
            spark.sql(f"CREATE TABLE {t} (id INT, x INT, nm STRING) USING iceberg TBLPROPERTIES ('format-version'='2')")
            spark.sql(f"INSERT INTO {t} VALUES {values(range(0, 6))}")
            spark.sql(f"ALTER TABLE {t} ALTER COLUMN id TYPE BIGINT")
            spark.sql(f"ALTER TABLE {t} RENAME COLUMN nm TO name")
            spark.sql(f"ALTER TABLE {t} ADD PARTITION FIELD x")
            spark.sql(f"INSERT INTO {t} VALUES {values(range(6, 12))}")
        else:
            spark.sql(
                f"CREATE TABLE {t} (id BIGINT, x INT, name STRING) USING iceberg PARTITIONED BY (x) "
                f"TBLPROPERTIES ('format-version'='{version}', {MOR})"
            )
            spark.sql(f"INSERT INTO {t} VALUES {values(range(0, 12))}")
            # Merge-on-read: position deletes on v2, a deletion vector on v3.
            spark.sql(f"DELETE FROM {t} WHERE id IN (1, 5)")
        result[name] = {"format_version": version, "evolved": evolved, "spark": state(spark, wh, name, version == 3)}
    return result


def readback(spark, wh, out_dir):
    with open(os.path.join(out_dir, "KRISHIV-RT.json")) as f:
        krishiv = json.load(f)
    result = {}
    for name, (version, _) in TABLES.items():
        t = f"o.db.{name}"
        v3 = version == 3
        entry = {"before": state(spark, wh, name, v3)}
        # Spark's time travel to each snapshot Krishiv committed.
        entry["at"] = {
            str(s["snapshot_id"]): rows_of(
                spark.sql(f"SELECT id, x, name FROM {t} VERSION AS OF {s['snapshot_id']} ORDER BY id NULLS FIRST, name")
            )
            for s in krishiv[name]["steps"]
        }
        # Merge-on-read on top of Krishiv's files (the evolved table stays copy-on-write, Spark's default).
        spark.sql(f"DELETE FROM {t} WHERE id = 102")
        spark.sql(f"UPDATE {t} SET name = 's' WHERE id IN (0, 103)")
        spark.sql(
            f"MERGE INTO {t} d USING (SELECT * FROM VALUES (104L, 1, 'm'), (300L, 0, 'n300') AS s(id, x, name)) s "
            "ON d.id = s.id WHEN MATCHED THEN DELETE WHEN NOT MATCHED THEN INSERT *"
        )
        spark.sql(f"INSERT INTO {t} VALUES {values([400])}")
        entry["after"] = state(spark, wh, name, v3)
        result[name] = entry
    return result


def main(step, out_dir):
    wh = os.path.join(out_dir, "wh")
    spark = session(wh)
    if step == "create":
        result, target = create(spark, wh), "SPARK-CREATE.json"
    else:
        result, target = readback(spark, wh, out_dir), "SPARK-READBACK.json"
    result["_oracle"] = {"spark": spark.version, "iceberg": "1.12.0"}
    with open(os.path.join(out_dir, target), "w") as f:
        json.dump(result, f, indent=1, sort_keys=True, default=typed)
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("create", "readback"):
        sys.exit("usage: gen_iceberg_roundtrip.py create|readback <out dir>")
    main(sys.argv[1], sys.argv[2])
