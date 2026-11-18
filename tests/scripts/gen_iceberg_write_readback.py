# SPDX-License-Identifier: Apache-2.0
"""Spark 4.1.0 + Iceberg 1.12.0 reads tables Krishiv wrote (P3-T028b, PR-53b, write conformance).

Input: ``<out>/wh``, a Hadoop-catalog warehouse the Rust test ``write_conformance_tests::krishiv_writes_the_warehouse``
wrote through Krishiv's append sink (paths in the metadata are absolute, so ``<out>`` is mounted at the same path).
For every table this records what Spark reads (rows, snapshots with their summaries, history, schema, data files,
partitions, and for format version 3 the ``_row_id`` values), appends two rows through Spark, records the same again
and the metadata file Spark committed, into ``<out>/READBACK.json``. Nothing here is a hand-written expectation: the
Rust test ``krishiv_agrees_with_spark_readback`` compares ``KRISHIV.json`` with this file.

Run through ``tests/scripts/iceberg_write_conformance.sh`` (it mounts ``<out>`` and calls spark-submit).
"""

import glob
import json
import os
import re
import sys

from pyspark.sql import SparkSession

TABLES = ["unpart", "part", "v3", "v3part"]


def typed(value):
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def rows_of(df):
    return [{k: typed(v) for k, v in r.asDict().items()} for r in df.collect()]


def state(spark, name, v3):
    t = f"o.db.{name}"
    out = {
        "rows": rows_of(spark.sql(f"SELECT id, x, name FROM {t} ORDER BY id, name")),
        "snapshots": rows_of(
            spark.sql(
                "SELECT snapshot_id, parent_id, operation, "
                "summary['added-records'] AS added_records, summary['total-records'] AS total_records, "
                f"summary['total-data-files'] AS total_data_files FROM {t}.snapshots ORDER BY committed_at, snapshot_id"
            )
        ),
        "history": [
            r["snapshot_id"]
            for r in spark.sql(f"SELECT snapshot_id FROM {t}.history ORDER BY made_current_at, snapshot_id").collect()
        ],
        "schema": rows_of(spark.sql(f"DESCRIBE TABLE {t}")),
        "files": rows_of(spark.sql(f"SELECT file_path, record_count FROM {t}.files ORDER BY file_path")),
        "partitions": rows_of(spark.sql(f"SELECT record_count, file_count FROM {t}.partitions")),
        "properties": rows_of(spark.sql(f"SHOW TBLPROPERTIES {t}")),
    }
    if v3:
        # No try/except: if Spark cannot project the lineage columns on a Krishiv-written v3 table, the job must fail.
        out["row_ids"] = [r["_row_id"] for r in spark.sql(f"SELECT _row_id FROM {t} ORDER BY _row_id").collect()]
        out["last_updated"] = [
            r["_last_updated_sequence_number"]
            for r in spark.sql(f"SELECT _last_updated_sequence_number FROM {t} ORDER BY id, name").collect()
        ]
    return out


def main(out_dir):
    wh = os.path.join(out_dir, "wh")
    spark = (
        SparkSession.builder.master("local[2]")
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.catalog.o", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.o.type", "hadoop")
        .config("spark.sql.catalog.o.warehouse", wh)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    result = {}
    for name in TABLES:
        v3 = name.startswith("v3")
        t = f"o.db.{name}"
        entry = {"before": state(spark, name, v3)}
        spark.sql(f"INSERT INTO {t} VALUES (200, 2, 'spark200'), (201, NULL, NULL)")
        entry["after"] = state(spark, name, v3)
        meta = os.path.join(wh, "db", name, "metadata")
        # The Hadoop catalog's own names (``v<N>.metadata.json``); Krishiv's test catalog also left ``v<N>-sink`` files.
        versions = [
            p
            for p in glob.glob(os.path.join(meta, "v*.metadata.json"))
            if re.fullmatch(r"v\d+\.metadata\.json", os.path.basename(p))
        ]
        latest = max(versions, key=lambda p: int(os.path.basename(p)[1:].split(".")[0]))
        entry["after"]["metadata"] = f"file:{latest}"
        result[name] = entry
    result["_oracle"] = {"spark": spark.version, "iceberg": "1.12.0"}
    with open(os.path.join(out_dir, "READBACK.json"), "w") as f:
        json.dump(result, f, indent=1, sort_keys=True, default=typed)
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: gen_iceberg_write_readback.py <out dir>")
    main(sys.argv[1])
