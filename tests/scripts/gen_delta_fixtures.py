# SPDX-License-Identifier: Apache-2.0
"""Oracle-written Delta tables for the Delta connector's tests (P3-T021 PR-08a and PR-08b).

Writes small tables with Spark 4.2.0 + delta-spark 4.4.1. PR-08a's three have a checkpoint of a
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

PR-08b's read-path tables (``scan_tables``) are written by ``... /out scan`` (only those, so the
checkpoint tables keep their bytes): deletion vectors, column mapping ``name`` and ``id`` (a
rename, a nested rename and a drop), type widening (``int`` to ``bigint`` and ``float`` to
``double`` after files were written), partition values with NULL and special characters,
``timestamp_ntz``, an unshredded Variant and ``vacuumProtocolCheck``. Their ``EXPECTED.json``
also has ``typed_rows``: every value in a typed form (``typed``) so the Rust test compares
values, not Python's ``str``. One table per refused reader feature that delta-spark can create
is written without rows (``refused_tables``); what it cannot create is recorded in
``REFUSED.json`` with the error, never replaced by a hand-written table.
"""

import json
import os
import shutil
import sys
from datetime import date, datetime
from decimal import Decimal

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


def typed(v: object) -> object:
    """A collected value as JSON with its kind kept: floats by bits-safe ``repr``, bytes as hex,
    timestamps and dates in ISO form, a Variant as its two byte strings, structs as objects."""
    if v is None or isinstance(v, bool | int | str):
        return v
    if isinstance(v, float):
        return {"float": repr(v)}
    if isinstance(v, Decimal):
        return {"decimal": str(v)}
    if isinstance(v, bytes | bytearray):
        return {"binary": bytes(v).hex()}
    if isinstance(v, datetime):
        return {"timestamp": v.isoformat()}
    if isinstance(v, date):
        return {"date": v.isoformat()}
    if hasattr(v, "asDict"):
        return {"struct": {k: typed(x) for k, x in v.asDict(recursive=False).items()}}
    if hasattr(v, "metadata") and hasattr(v, "value"):
        return {"variant": {"metadata": bytes(v.metadata).hex(), "value": bytes(v.value).hex()}}
    if isinstance(v, list):
        return [typed(x) for x in v]
    raise TypeError(f"no typed form for {type(v)}")


def oracle_error(e: Exception, path: str) -> dict:
    """The oracle's error: a captured Spark error, or the JVM throwable behind a Py4J error
    (its class, condition, parameters and first message line), with the table path elided."""
    record: dict = {}
    java = getattr(e, "java_exception", None)
    while java is not None:
        record["jvm_class"] = java.getClass().getName()
        try:
            record["condition"] = java.getErrorClass()
            record["sqlstate"] = java.getSqlState()
            record["parameters"] = {
                k: str(v).replace(path, "<path>") for k, v in dict(java.getMessageParameters()).items()
            }
            break
        except Exception:
            java = java.getCause()
    if java is None and hasattr(e, "getCondition"):
        record["condition"] = e.getCondition()
        record["sqlstate"] = e.getSqlState()
        params = e.getMessageParameters() or {}
        record["parameters"] = {k: v.replace(path, "<path>") for k, v in params.items()}
    message = java.getMessage() if java is not None else str(e)
    record["message"] = (message or "").splitlines()[0].replace(path, "<path>")[:600]
    return record


def protocol(path: str) -> dict:
    """The newest protocol action of the table's JSON commits."""
    found: dict = {}
    log = os.path.join(path, "_delta_log")
    for name in sorted(n for n in os.listdir(log) if n.endswith(".json") and n[0].isdigit()):
        with open(os.path.join(log, name)) as f:
            for line in f:
                action = json.loads(line)
                if "protocol" in action:
                    found = action["protocol"]
    return found


def write_scan_expected(spark: SparkSession, path: str, notes: dict) -> None:
    df = spark.read.format("delta").load(path)
    rows = df.orderBy("id").collect()
    dv_storage: set[str] = set()
    log = os.path.join(path, "_delta_log")
    for name in os.listdir(log):
        if not (name.endswith(".json") and name[0].isdigit()):
            continue
        with open(os.path.join(log, name)) as f:
            for line in f:
                add = json.loads(line).get("add") or {}
                if add.get("deletionVector"):
                    dv_storage.add(add["deletionVector"]["storageType"])
    notes = {
        **notes,
        "schema": df.schema.simpleString(),
        "fields": [f.name for f in df.schema.fields],
        "typed_rows": [[typed(r[i]) for i in range(len(r))] for r in rows],
        "protocol": protocol(path),
        "dv_storage_types": sorted(dv_storage),
        "timezone": spark.conf.get("spark.sql.session.timeZone"),
    }
    write_expected(spark, path, notes)


def drop_hadoop_crc(out: str) -> None:
    for d, _, files in os.walk(out):
        for f in files:
            if f.startswith(".") and f.endswith(".crc"):
                os.remove(os.path.join(d, f))


#: Rows the deletion-vector table holds, and the Parquet row-group size it is written with: a few
#: row groups, each of more rows than one decoded batch (8,192).
DV_ROWS = 40_000
DV_BLOCK_BYTES = 100_000
#: Krishiv's Parquet batch (`krishiv.exec.batchSize`'s default), for deletes at batch edges.
BATCH_ROWS = 8_192


def row_groups(spark: SparkSession, path: str) -> list[int]:
    """The row counts of the row groups of the one data file under ``path`` (its footer, read by
    parquet-mr in the oracle's JVM)."""
    files = [f for f in os.listdir(path) if f.endswith(".parquet")]
    assert len(files) == 1, files
    jvm = spark._jvm
    footer = jvm.org.apache.parquet.hadoop.ParquetFileReader.readFooter(
        spark._jsc.hadoopConfiguration(), jvm.org.apache.hadoop.fs.Path(os.path.join(path, files[0]))
    )
    return [b.getRowCount() for b in footer.getBlocks()]


def deletion_vectors(spark: SparkSession, dv: str) -> None:
    """One file of ``DV_ROWS`` rows in several row groups (ids in file order), then two DELETE
    statements: the first keeps every 100th row and the rows within three of each row-group
    and batch edge; the second (rewriting the DV) deletes the rows right at those edges."""
    spark.sql(
        f"CREATE TABLE delta.`{dv}` (id BIGINT, v INT) USING delta "
        "TBLPROPERTIES ('delta.enableDeletionVectors' = 'true')"
    )
    conf = spark._jsc.hadoopConfiguration()
    conf.setInt("parquet.block.size", DV_BLOCK_BYTES)
    spark.range(DV_ROWS).selectExpr("id", "CAST(id % 7 AS INT) AS v").coalesce(1).write.format("delta").mode(
        "append"
    ).save(dv)
    conf.unset("parquet.block.size")
    groups = row_groups(spark, dv)
    edges: list[int] = []
    start = 0
    for rows in groups:
        edges.extend(range(start, start + rows, BATCH_ROWS))
        start += rows
    edges.append(start)
    near = sorted({e + d for e in edges for d in range(-3, 4) if 0 <= e + d < DV_ROWS})
    exact = sorted({e + d for e in edges for d in (-1, 0) if 0 <= e + d < DV_ROWS})
    keep = ", ".join(map(str, near))
    spark.sql(f"DELETE FROM delta.`{dv}` WHERE NOT (id % 100 = 0 OR id IN ({keep}))")
    spark.sql(f"DELETE FROM delta.`{dv}` WHERE id IN ({', '.join(map(str, exact))})")
    write_scan_expected(
        spark,
        dv,
        {"feature": "deletionVectors", "row_groups": groups, "edges": edges, "deleted_at_edges": exact},
    )


def scan_tables(spark: SparkSession, out: str) -> None:
    def t(name: str) -> str:
        return os.path.join(out, name)

    deletion_vectors(spark, t("deletion-vectors"))

    for mode in ("name", "id"):
        cm = t(f"column-mapping-{mode}")
        spark.sql(
            f"CREATE TABLE delta.`{cm}` (id BIGINT, a STRING, b INT, s STRUCT<x: INT, y: STRING>, p STRING) "
            f"USING delta PARTITIONED BY (p) TBLPROPERTIES ('delta.columnMapping.mode' = '{mode}')"
        )
        spark.sql(
            f"INSERT INTO delta.`{cm}` VALUES (1, 'a1', 10, named_struct('x', 100, 'y', 'y1'), 'p1'), "
            "(2, NULL, 20, NULL, NULL)"
        )
        spark.sql(f"ALTER TABLE delta.`{cm}` RENAME COLUMN a TO a2")
        spark.sql(f"ALTER TABLE delta.`{cm}` RENAME COLUMN s.x TO x2")
        spark.sql(f"ALTER TABLE delta.`{cm}` DROP COLUMN b")
        spark.sql(f"ALTER TABLE delta.`{cm}` ADD COLUMNS (b STRING)")
        spark.sql(
            f"INSERT INTO delta.`{cm}` (id, a2, s, p, b) VALUES "
            "(3, 'a3', named_struct('x2', 300, 'y', 'y3'), 'p1', 'new-b')"
        )
        write_scan_expected(spark, cm, {"feature": "columnMapping", "mode": mode})

    tw = t("type-widening")
    spark.sql(
        f"CREATE TABLE delta.`{tw}` (id BIGINT, i INT, f FLOAT) USING delta "
        "TBLPROPERTIES ('delta.enableTypeWidening' = 'true')"
    )
    spark.sql(f"INSERT INTO delta.`{tw}` VALUES (1, 2147483647, 1.5), (2, NULL, CAST(0.1 AS FLOAT))")
    spark.sql(f"ALTER TABLE delta.`{tw}` ALTER COLUMN i TYPE BIGINT")
    spark.sql(f"ALTER TABLE delta.`{tw}` ALTER COLUMN f TYPE DOUBLE")
    spark.sql(f"INSERT INTO delta.`{tw}` VALUES (3, 9223372036854775807, 0.1)")
    write_scan_expected(spark, tw, {"feature": "typeWidening"})

    pt = t("partitioned")
    spark.sql(
        f"CREATE TABLE delta.`{pt}` (id BIGINT, p STRING, n INT, d DATE, ts TIMESTAMP) USING delta "
        "PARTITIONED BY (p, n, d, ts)"
    )
    spark.sql(
        f"INSERT INTO delta.`{pt}` VALUES "
        "(1, 'a b', 1, DATE'2024-01-01', TIMESTAMP'2024-01-01 12:30:45.123456'), "
        "(2, 'a/b', NULL, NULL, NULL), "
        "(3, 'a=b', -1, DATE'1970-01-01', TIMESTAMP'1970-01-01 00:00:00'), "
        "(4, '50%', 2, DATE'2024-02-29', TIMESTAMP'2024-02-29 23:59:59'), "
        "(5, 'ü:é#?', 3, NULL, NULL), "
        "(6, NULL, NULL, NULL, NULL), "
        "(7, '', 4, NULL, NULL)"
    )
    write_scan_expected(spark, pt, {"feature": "partitions"})

    ntz = t("timestamp-ntz")
    spark.sql(
        f"CREATE TABLE delta.`{ntz}` (id BIGINT, t TIMESTAMP_NTZ, tp TIMESTAMP_NTZ) USING delta PARTITIONED BY (tp)"
    )
    spark.sql(
        f"INSERT INTO delta.`{ntz}` VALUES "
        "(1, TIMESTAMP_NTZ'2024-01-01 12:30:45.5', TIMESTAMP_NTZ'2024-01-01 00:00:00'), "
        "(2, NULL, NULL), "
        "(3, TIMESTAMP_NTZ'1900-01-01 00:00:00', TIMESTAMP_NTZ'2024-06-30 23:59:59.999999')"
    )
    write_scan_expected(spark, ntz, {"feature": "timestampNtz"})

    var = t("variant")
    spark.sql(f"CREATE TABLE delta.`{var}` (id BIGINT, v VARIANT) USING delta")
    spark.sql(
        f"INSERT INTO delta.`{var}` VALUES "
        """(1, parse_json('{"a": 1, "b": [true, null, "x"]}')), (2, NULL), (3, parse_json('12.5'))"""
    )
    write_scan_expected(spark, var, {"feature": "variantType"})

    vac = t("vacuum-protocol-check")
    spark.sql(
        f"CREATE TABLE delta.`{vac}` (id BIGINT) USING delta "
        "TBLPROPERTIES ('delta.feature.vacuumProtocolCheck' = 'supported')"
    )
    spark.sql(f"INSERT INTO delta.`{vac}` VALUES (1), (2)")
    write_scan_expected(spark, vac, {"feature": "vacuumProtocolCheck"})


def refused_tables(spark: SparkSession, out: str) -> None:
    """One table per refused reader feature delta-spark 4.4.1 can create; the attempts that
    fail are recorded in ``REFUSED.json`` with the oracle's error."""
    attempts = {
        "variant-shredding": [
            (
                "CREATE TABLE delta.`{p}` (id BIGINT, v VARIANT) USING delta "
                "TBLPROPERTIES ('delta.enableVariantShredding' = 'true')"
            ),
            (
                "CREATE TABLE delta.`{p}` (id BIGINT, v VARIANT) USING delta "
                "TBLPROPERTIES ('delta.feature.variantShredding-preview' = 'supported')"
            ),
        ],
        "catalog-managed": [
            (
                "CREATE TABLE delta.`{p}` (id BIGINT) USING delta "
                "TBLPROPERTIES ('delta.feature.catalogManaged' = 'supported')"
            ),
        ],
        "coordinated-commits": [
            (
                "CREATE TABLE delta.`{p}` (id BIGINT) USING delta "
                "TBLPROPERTIES ('delta.coordinatedCommits.commitCoordinator-preview' = 'in-memory')"
            ),
            (
                "CREATE TABLE delta.`{p}` (id BIGINT) USING delta "
                "TBLPROPERTIES ('delta.coordinatedCommits.commitCoordinator-preview' = 'in-memory', "
                "'delta.coordinatedCommits.commitCoordinatorConf-preview' = '{{}}')"
            ),
        ],
        "geospatial": [
            "CREATE TABLE delta.`{p}` (id BIGINT, g GEOMETRY(4326)) USING delta",
        ],
    }
    report: dict = {}
    for name, sqls in attempts.items():
        path = os.path.join(out, name)
        errors = []
        for sql in sqls:
            try:
                spark.sql(sql.format(p=path))
            except Exception as e:  # the oracle's refusal is the record
                errors.append({"sql": sql, **oracle_error(e, path)})
                shutil.rmtree(path, ignore_errors=True)
                continue
            report[name] = {"created_by": sql, "protocol": protocol(path)}
            with open(os.path.join(path, "EXPECTED.json"), "w") as f:
                json.dump(
                    {
                        "format_version": FORMAT_VERSION,
                        "generator": "tests/scripts/gen_delta_fixtures.py",
                        "spark": spark.version,
                        "delta_spark": "4.4.1",
                        "refused": True,
                        "created_by": sql,
                        "protocol": protocol(path),
                    },
                    f,
                    indent=1,
                    sort_keys=True,
                )
                f.write("\n")
            break
        else:
            report[name] = {"not_created": errors}
    with open(os.path.join(out, "REFUSED.json"), "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
        f.write("\n")


def probe_unknown_reader_feature(spark: SparkSession, out: str) -> None:
    for name, features in (
        ("PROBE-unknown-reader-feature.json", ["krishivProbeFeature"]),
        ("PROBE-two-unknown-reader-features.json", ["krishivProbeZeta", "krishivProbeAlpha"]),
    ):
        probe_unknown_reader_features(spark, out, name, features)


def probe_unknown_reader_features(spark: SparkSession, out: str, record_name: str, features: list[str]) -> None:
    """Delta's own refusal of a reader feature it does not know, for the shape of
    ``DELTA_UNSUPPORTED_FEATURES_FOR_READ``'s parameters: a table whose last commit adds an
    unknown name to ``readerFeatures``, read by delta-spark. Only the error is kept."""
    path = os.path.join(out, "probe-unknown-feature")
    spark.sql(
        f"CREATE TABLE delta.`{path}` (id BIGINT) USING delta TBLPROPERTIES ('delta.enableDeletionVectors' = 'true')"
    )
    spark.sql(f"INSERT INTO delta.`{path}` VALUES (1)")
    proto = protocol(path)
    proto["readerFeatures"] = [*proto["readerFeatures"], *features]
    proto["writerFeatures"] = [*proto["writerFeatures"], *features]
    with open(os.path.join(path, "_delta_log", f"{2:020}.json"), "w") as f:
        f.write(json.dumps({"protocol": proto}) + "\n")
    record: dict = {"protocol": proto}
    try:
        spark.read.format("delta").load(path).collect()
        record["error"] = None
    except Exception as e:
        record.update(oracle_error(e, path))
    shutil.rmtree(path, ignore_errors=True)
    with open(os.path.join(out, record_name), "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.write("\n")


def main(out: str, which: str = "checkpoint") -> None:
    spark = session()
    if which == "review":
        deletion_vectors(spark, os.path.join(out, "deletion-vectors"))
        probe_unknown_reader_feature(spark, out)
        drop_hadoop_crc(out)
        spark.stop()
        return
    if which == "probe":
        refused_tables(spark, out)
        probe_unknown_reader_feature(spark, out)
        drop_hadoop_crc(out)
        spark.stop()
        return
    if which == "scan":
        scan_tables(spark, out)
        refused_tables(spark, out)
        probe_unknown_reader_feature(spark, out)
        drop_hadoop_crc(out)
        spark.stop()
        return

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
    main(*sys.argv[1:3])
