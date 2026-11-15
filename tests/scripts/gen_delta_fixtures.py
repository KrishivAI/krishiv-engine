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
from typing import Any

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


def write_expected(spark: SparkSession, path: str, notes: dict, rows_as_dicts: bool = True) -> None:
    rows = [r.asDict() for r in spark.read.format("delta").load(path).orderBy("id").collect()] if rows_as_dicts else []
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
    row: Any = v  # a PySpark Row or VariantVal: duck-typed below
    if hasattr(row, "asDict"):
        return {"struct": {k: typed(x) for k, x in row.asDict(recursive=False).items()}}
    if hasattr(row, "metadata") and hasattr(row, "value"):
        return {"variant": {"metadata": bytes(row.metadata).hex(), "value": bytes(row.value).hex()}}
    if isinstance(v, list):
        return [typed(x) for x in v]
    if isinstance(v, dict):
        return {"map": [[typed(k), typed(x)] for k, x in v.items()]}
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
    spark: Any = e  # a PySparkException: duck-typed
    if java is None and hasattr(spark, "getCondition"):
        record["condition"] = spark.getCondition()
        record["sqlstate"] = spark.getSqlState()
        params = spark.getMessageParameters() or {}
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


def write_scan_expected(spark: SparkSession, path: str, notes: dict, rows_as_dicts: bool = True) -> None:
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
    write_expected(spark, path, notes, rows_as_dicts)


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


def outcome(spark: SparkSession, sql: str, path: str, rows: bool = False) -> dict:
    """What the oracle does with ``sql``: its error, or success (with the rows when asked), the
    table path elided as ``<path>``."""
    record: dict = {"sql": sql.replace(path, "<path>")}
    try:
        df = spark.sql(sql)
        got = df.collect()
        record["error"] = None
        if rows:
            record["columns"] = [f.name for f in df.schema.fields]
            record["rows"] = [[typed(r[i]) for i in range(len(r))] for r in got]
    except Exception as e:  # the oracle's error is the record
        record["error"] = oracle_error(e, path)
    return record


def session_probes(spark: SparkSession, out: str) -> None:
    """Delta tables through the session (PR-08b part 2): ``SESSION.json`` records, for one
    oracle table, ``DESCRIBE`` of a ``USING delta LOCATION`` table without a column list, the
    schema-mismatch error, the outcome of each write and DDL Krishiv refuses until PR-37a, the
    unknown-reader-feature refusal through a catalog table and through a path (the two
    ``tableNameOrPath`` forms) and a corrupt commit file. ``session-appends`` is a table of three
    commits whose last one the snapshot-cache tests append while a query runs."""
    base = os.path.join(out, "session-base")
    spark.sql(
        f"CREATE TABLE delta.`{base}` (id BIGINT, name STRING, amount DECIMAL(10, 2), "
        "ts TIMESTAMP, p STRING) USING delta PARTITIONED BY (p)"
    )
    spark.sql(
        f"INSERT INTO delta.`{base}` VALUES (1, 'a', 1.50, TIMESTAMP'2024-01-01 00:00:00', 'x'), "
        "(2, NULL, NULL, NULL, 'y')"
    )
    # The table as the probes find it, kept for the session's tests (the probes write to it).
    write_scan_expected(spark, base, {"feature": "session"})
    shutil.copytree(base, os.path.join(out, "session-table"))
    record: dict = {
        "format_version": FORMAT_VERSION,
        "generator": "tests/scripts/gen_delta_fixtures.py",
        "spark": spark.version,
        "delta_spark": "4.4.1",
    }
    record["create_without_columns"] = outcome(spark, f"CREATE TABLE d USING delta LOCATION '{base}'", base)
    record["describe"] = outcome(spark, "DESCRIBE d", base, rows=True)
    record["select"] = outcome(spark, "SELECT * FROM d ORDER BY id", base, rows=True)
    record["create_matching_columns"] = outcome(
        spark,
        f"CREATE TABLE same (id BIGINT, name STRING, amount DECIMAL(10, 2), ts TIMESTAMP, p STRING) "
        f"USING delta LOCATION '{base}'",
        base,
    )
    record["create_matching_columns_partitioned"] = outcome(
        spark,
        f"CREATE TABLE same2 (id BIGINT, name STRING, amount DECIMAL(10, 2), ts TIMESTAMP, p STRING) "
        f"USING delta PARTITIONED BY (p) LOCATION '{base}'",
        base,
    )
    record["describe_matching_columns"] = outcome(spark, "DESCRIBE same2", base, rows=True)
    record["schema_mismatch"] = outcome(
        spark, f"CREATE TABLE m (id BIGINT, other STRING) USING delta LOCATION '{base}'", base
    )
    record["schema_mismatch_type"] = outcome(
        spark,
        f"CREATE TABLE m2 (id INT, name STRING, amount DECIMAL(10, 2), ts TIMESTAMP, p STRING) "
        f"USING delta LOCATION '{base}'",
        base,
    )
    missing = os.path.join(out, "session-missing")
    empty = os.path.join(out, "session-empty")
    os.makedirs(empty, exist_ok=True)
    record["create_missing_location"] = outcome(spark, f"CREATE TABLE gone USING delta LOCATION '{missing}'", missing)
    record["create_empty_location"] = outcome(spark, f"CREATE TABLE hollow USING delta LOCATION '{empty}'", empty)
    shutil.rmtree(missing, ignore_errors=True)
    shutil.rmtree(empty, ignore_errors=True)
    record["create_missing_location_with_columns"] = outcome(
        spark, f"CREATE TABLE gone2 (id BIGINT) USING delta LOCATION '{missing}'", missing
    )
    shutil.rmtree(missing, ignore_errors=True)
    record["create_without_location"] = outcome(spark, "CREATE TABLE managed (id BIGINT) USING delta", base)
    record["ctas"] = outcome(spark, "CREATE TABLE c USING delta AS SELECT 1 AS id", base)
    record["insert"] = outcome(spark, "INSERT INTO d VALUES (3, 'c', 3.00, NULL, 'x')", base)
    record["alter_add_columns"] = outcome(spark, "ALTER TABLE d ADD COLUMNS (extra INT)", base)
    record["alter_set_properties"] = outcome(spark, "ALTER TABLE d SET TBLPROPERTIES ('k' = 'v')", base)

    refused = os.path.join(out, "session-refused")
    spark.sql(
        f"CREATE TABLE delta.`{refused}` (id BIGINT) USING delta TBLPROPERTIES ('delta.enableDeletionVectors' = 'true')"
    )
    spark.sql(f"INSERT INTO delta.`{refused}` VALUES (1)")
    spark.sql(f"CREATE TABLE refused USING delta LOCATION '{refused}'")
    proto = protocol(refused)
    proto["readerFeatures"] = [*proto["readerFeatures"], "krishivProbeFeature"]
    proto["writerFeatures"] = [*proto["writerFeatures"], "krishivProbeFeature"]
    with open(os.path.join(refused, "_delta_log", f"{2:020}.json"), "w") as f:
        f.write(json.dumps({"protocol": proto}) + "\n")
    record["refused_catalog_table"] = outcome(spark, "SELECT * FROM refused", refused)
    record["refused_qualified_catalog_table"] = outcome(spark, "SELECT * FROM default.refused", refused)
    record["refused_path_sql"] = outcome(spark, f"SELECT * FROM delta.`{refused}`", refused)
    record["refused_path_trailing_slash"] = outcome(spark, f"SELECT * FROM delta.`{refused}/`", refused)
    try:
        spark.read.format("delta").load(refused).collect()
        record["refused_path_reader"] = {"error": None}
    except Exception as e:
        record["refused_path_reader"] = {"error": oracle_error(e, refused)}
    shutil.rmtree(refused, ignore_errors=True)

    corrupt = os.path.join(out, "session-corrupt")
    spark.sql(f"CREATE TABLE delta.`{corrupt}` (id BIGINT) USING delta")
    spark.sql(f"INSERT INTO delta.`{corrupt}` VALUES (1)")
    with open(os.path.join(corrupt, "_delta_log", f"{1:020}.json"), "w") as f:
        f.write("{not json\n")
    for crc in [n for n in os.listdir(os.path.join(corrupt, "_delta_log")) if n.endswith(".crc")]:
        os.remove(os.path.join(corrupt, "_delta_log", crc))
    record["corrupt_commit"] = outcome(spark, f"SELECT * FROM delta.`{corrupt}`", corrupt)
    shutil.rmtree(corrupt, ignore_errors=True)
    for name in ("d", "same2", "managed", "c"):
        spark.sql(f"DROP TABLE IF EXISTS {name}")
    shutil.rmtree(base, ignore_errors=True)

    appends = os.path.join(out, "session-appends")
    spark.sql(f"CREATE TABLE delta.`{appends}` (id BIGINT, v STRING) USING delta")
    spark.sql(f"INSERT INTO delta.`{appends}` VALUES (1, 'a'), (2, 'b')")
    spark.sql(f"INSERT INTO delta.`{appends}` VALUES (3, 'c')")
    write_scan_expected(spark, appends, {"feature": "appends"})
    with open(os.path.join(out, "SESSION.json"), "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.write("\n")


def commits(path: str) -> list[dict]:
    """Every JSON commit of the table's log, in version order: its version, ``commitInfo``
    (``timestamp``, ``txnId`` and ``engineInfo`` dropped), the protocol and metadata actions, and
    its ``add`` and ``remove`` actions (``stats`` parsed; times and sizes dropped)."""
    log = os.path.join(path, "_delta_log")
    out = []
    for name in sorted(n for n in os.listdir(log) if n.endswith(".json") and n[0].isdigit()):
        record: dict = {"version": int(name.split(".")[0]), "adds": [], "removes": []}
        with open(os.path.join(log, name)) as f:
            for line in f:
                action = json.loads(line)
                if "commitInfo" in action:
                    info = dict(action["commitInfo"])
                    for k in ("timestamp", "txnId", "engineInfo", "inCommitTimestamp"):
                        info.pop(k, None)
                    record["commitInfo"] = info
                elif "add" in action:
                    add = action["add"]
                    record["adds"].append(
                        {
                            "path": add["path"],
                            "partitionValues": add["partitionValues"],
                            "stats": json.loads(add["stats"]) if add.get("stats") else None,
                        }
                    )
                elif "remove" in action:
                    rm = action["remove"]
                    record["removes"].append({"path": rm["path"], "partitionValues": rm.get("partitionValues")})
                elif "protocol" in action:
                    record["protocol"] = action["protocol"]
                elif "metaData" in action:
                    record["partitionColumns"] = action["metaData"]["partitionColumns"]
        out.append(record)
    return out


#: The ``stats`` table's columns: every type delta-spark keeps min/max for, the types it keeps
#: only null counts for, nested columns and a partition column (no statistics).
STATS_SCHEMA = (
    "id INT, k INT, b TINYINT, s SMALLINT, i INT, l BIGINT, f FLOAT, d DOUBLE, dec DECIMAL(10, 2), "
    "dec38 DECIMAL(38, 4), str STRING, dt DATE, ts TIMESTAMP, ntz TIMESTAMP_NTZ, bool BOOLEAN, "
    "bin BINARY, st STRUCT<x: INT, y: STRING>, arr ARRAY<INT>, m MAP<STRING, INT>, p STRING"
)

#: One INSERT per case (one file each): ``k`` names the case. The string cases are delta-spark's
#: truncation rules (`StatisticsCollection.truncateMinStringAgg`/`truncateMaxStringAgg`, prefix
#: 32): exactly 32 characters, 33, a surrogate pair across the cut, a non-ASCII character after
#: it (the U+10FFFF tie-breaker), U+10FFFF characters after it (skipped), and so many of them
#: that the max is dropped.
STATS_CASES = [
    "(1, 1, 1, 10, 100, 1000, 1.5, 2.5, 12.34, 123456789012345678901234567890.1234, 'abc', "
    "DATE'2024-01-01', TIMESTAMP'2024-01-01 12:30:45.123456', TIMESTAMP_NTZ'2024-01-01 12:30:45.5', "
    "true, X'0102', named_struct('x', 1, 'y', 'yy'), array(1, 2), map('a', 1), 'x'), "
    "(3, 2, -5, -10, -100, -1000, -0.5, -2.5, -12.34, -1.0001, 'abd', DATE'1969-12-31', "
    "TIMESTAMP'1969-12-31 23:59:59.999999', TIMESTAMP_NTZ'1900-01-01 00:00:00', false, NULL, "
    "named_struct('x', NULL, 'y', 'aa'), NULL, NULL, 'x'), "
    "(4, 3, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, "
    "NULL, NULL, 'x')",
    "(5, 4, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'y')",
    "(6, 5, NULL, NULL, NULL, NULL, CAST('NaN' AS FLOAT), CAST('NaN' AS DOUBLE), NULL, NULL, NULL, NULL, "
    "NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'x'), (7, 5, NULL, NULL, NULL, NULL, CAST(-0.0 AS FLOAT), "
    "1.0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'x')",
]
STATS_STRINGS = [
    "a" * 32,
    "a" * 33,
    "z" * 40,
    "a" * 31 + "\U0001f600" + "bc",
    "a" * 32 + "é" * 3,
    "a" * 32 + "\U0010ffff" * 2 + "b",
    "a" * 32 + "\U0010ffff" * 20 + "b",
    "é" * 40,
]


def sql_string(v: str) -> str:
    return "'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'"


def stats_table(spark: SparkSession, out: str) -> None:
    """``write-stats``: one file per case of ``STATS_CASES`` and ``STATS_STRINGS`` (``k`` from
    10 on), so ``delta_stats_match_spark`` writes each case's rows with Krishiv and compares its
    ``stats`` with the oracle's."""
    path = os.path.join(out, "write-stats")
    spark.sql(f"CREATE TABLE delta.`{path}` ({STATS_SCHEMA}) USING delta PARTITIONED BY (p)")
    for case in STATS_CASES:
        spark.sql(f"INSERT INTO delta.`{path}` VALUES {case}")
    nulls = ", ".join(["NULL"] * 8)
    for n, v in enumerate(STATS_STRINGS):
        spark.sql(
            f"INSERT INTO delta.`{path}` VALUES ({100 + n}, {10 + n}, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, "
            f"{sql_string(v)}, {nulls}, 'x')"
        )
    # The rows by case, and every commit's adds with their parsed stats.
    # `rows` stays empty: its NaN cells are not JSON (``typed_rows`` has them as strings).
    write_scan_expected(
        spark, path, {"feature": "stats", "commits": commits(path), "strings": STATS_STRINGS}, rows_as_dicts=False
    )


def run_steps(spark: SparkSession, path: str, steps: list[tuple[str, Any]]) -> list[dict]:
    """Runs each named step (SQL text, or a callable of the session) on the table at ``path``
    and records its outcome: the error, or the rows read back (by ``id``) and the commit it
    wrote."""
    out = []
    for name, step in steps:
        before = len(commits(path)) if os.path.isdir(os.path.join(path, "_delta_log")) else 0
        record: dict = {"step": name}
        if isinstance(step, str):
            record["sql"] = step.replace(path, "<path>")
        try:
            spark.sql(step).collect() if isinstance(step, str) else step(spark)
            record["error"] = None
        except Exception as e:  # the oracle's error is the record
            record["error"] = oracle_error(e, path)
        if os.path.isdir(os.path.join(path, "_delta_log")):
            log = commits(path)
            record["commits"] = log[before:]
            df = spark.read.format("delta").load(path)
            record["rows"] = [[typed(r[i]) for i in range(len(r))] for r in df.orderBy("id").collect()]
        out.append(record)
    return out


def write_modes(spark: SparkSession, out: str) -> None:
    """``WRITES.json``: the write modes against the oracle (``write_modes_match_oracle``), each
    step on a table partitioned by ``p``: CTAS, append, overwrite (truncate), dynamic partition
    overwrite, ``replaceWhere`` (matching and not), a static partition overwrite, v1
    ``saveAsTable`` overwrite (same schema, another schema) and an overwrite of an append-only
    table. Each step records the oracle's error or its commit and rows."""
    path = os.path.join(out, "write-modes")
    t = f"delta.`{path}`"

    def replace_where(where: str, rows: str):
        def step(s: SparkSession) -> None:
            s.sql(f"SELECT * FROM VALUES {rows} AS v(id, v, p)").selectExpr(
                "CAST(id AS BIGINT) AS id", "v", "p"
            ).write.format("delta").mode("overwrite").option("replaceWhere", where).save(path)

        return step

    def dynamic(s: SparkSession) -> None:
        s.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
        try:
            s.sql(f"INSERT OVERWRITE {t} VALUES (20, 'dyn', 'x'), (21, 'dyn', 'z')").collect()
        finally:
            s.conf.unset("spark.sql.sources.partitionOverwriteMode")

    steps: list[tuple[str, Any]] = [
        (
            "ctas",
            f"CREATE TABLE {t} USING delta PARTITIONED BY (p) AS "
            "SELECT CAST(id AS BIGINT) AS id, v, p FROM VALUES (1, 'a', 'x'), (2, 'b', 'y') AS s(id, v, p)",
        ),
        ("append", f"INSERT INTO {t} VALUES (3, 'c', 'x'), (4, 'd', NULL)"),
        ("overwrite", f"INSERT OVERWRITE {t} VALUES (5, 'e', 'x'), (6, 'f', 'y')"),
        ("append_again", f"INSERT INTO {t} VALUES (7, 'g', 'y'), (8, 'h', 'z')"),
        ("dynamic_overwrite", dynamic),
        ("replace_where", replace_where("p = 'y'", "(30, 'rw', 'y')")),
        ("replace_where_mismatch", replace_where("p = 'y'", "(31, 'rw', 'x')")),
        ("static_partition_overwrite", f"INSERT OVERWRITE {t} PARTITION (p = 'z') VALUES (40, 'st')"),
        (
            "insert_overwrite_dataframe_mode",
            lambda s: (
                s.sql("SELECT CAST(50 AS BIGINT) AS id, 'df' AS v, 'x' AS p")
                .write.format("delta")
                .mode("overwrite")
                .save(path)
            ),
        ),
        (
            "append_dataframe",
            lambda s: (
                s.sql("SELECT CAST(51 AS BIGINT) AS id, 'df' AS v, 'w' AS p")
                .write.format("delta")
                .mode("append")
                .save(path)
            ),
        ),
        (
            "save_errorifexists",
            lambda s: s.sql("SELECT CAST(52 AS BIGINT) AS id, 'df' AS v, 'w' AS p").write.format("delta").save(path),
        ),
        (
            "save_ignore",
            lambda s: (
                s.sql("SELECT CAST(53 AS BIGINT) AS id, 'df' AS v, 'w' AS p")
                .write.format("delta")
                .mode("ignore")
                .save(path)
            ),
        ),
        ("insert_reading_itself", f"INSERT INTO {t} SELECT id + 100, v, p FROM {t}"),
    ]
    record: dict = {
        "format_version": FORMAT_VERSION,
        "generator": "tests/scripts/gen_delta_fixtures.py",
        "spark": spark.version,
        "delta_spark": "4.4.1",
        "steps": run_steps(spark, path, steps),
    }

    # v1 saveAsTable(mode=overwrite) on an existing catalog table.
    managed = os.path.join(out, "write-save-as-table")
    spark.sql(
        f"CREATE TABLE sat USING delta PARTITIONED BY (p) LOCATION '{managed}' AS "
        "SELECT CAST(1 AS BIGINT) AS id, 'a' AS v, 'x' AS p"
    )
    sat_steps: list[tuple[str, Any]] = [
        (
            "save_as_table_overwrite_same_schema",
            lambda s: (
                s.sql("SELECT CAST(2 AS BIGINT) AS id, 'b' AS v, 'y' AS p")
                .write.format("delta")
                .mode("overwrite")
                .partitionBy("p")
                .saveAsTable("sat")
            ),
        ),
        (
            "save_as_table_overwrite_other_schema",
            lambda s: (
                s.sql("SELECT CAST(3 AS BIGINT) AS id, 3 AS other, 'y' AS p")
                .write.format("delta")
                .mode("overwrite")
                .partitionBy("p")
                .saveAsTable("sat")
            ),
        ),
        (
            "save_as_table_overwrite_other_partitioning",
            lambda s: (
                s.sql("SELECT CAST(4 AS BIGINT) AS id, 'd' AS v, 'y' AS p")
                .write.format("delta")
                .mode("overwrite")
                .saveAsTable("sat")
            ),
        ),
        (
            "save_as_table_append",
            lambda s: (
                s.sql("SELECT CAST(5 AS BIGINT) AS id, 'e' AS v, 'x' AS p")
                .write.format("delta")
                .mode("append")
                .saveAsTable("sat")
            ),
        ),
    ]
    record["save_as_table"] = run_steps(spark, managed, sat_steps)
    spark.sql("DROP TABLE IF EXISTS sat")

    ao = os.path.join(out, "write-append-only")
    spark.sql(
        f"CREATE TABLE delta.`{ao}` (id BIGINT, p STRING) USING delta PARTITIONED BY (p) "
        "TBLPROPERTIES ('delta.appendOnly' = 'true')"
    )
    spark.sql(f"INSERT INTO delta.`{ao}` VALUES (1, 'x')")
    record["append_only"] = run_steps(
        spark,
        ao,
        [
            ("append", f"INSERT INTO delta.`{ao}` VALUES (2, 'y')"),
            ("overwrite", f"INSERT OVERWRITE delta.`{ao}` VALUES (3, 'x')"),
        ],
    )
    created = os.path.join(out, "write-created")
    record["create_new_location"] = run_steps(
        spark,
        created,
        [("create", f"CREATE TABLE delta.`{created}` (id BIGINT, p STRING) USING delta PARTITIONED BY (p)")],
    )
    record["ctas_existing_location"] = run_steps(
        spark,
        created,
        [("ctas", f"CREATE TABLE ctas_x USING delta LOCATION '{created}' AS SELECT CAST(1 AS BIGINT) AS id, 'x' AS p")],
    )
    spark.sql("DROP TABLE IF EXISTS ctas_x")
    for d in (path, managed, ao, created):
        shutil.rmtree(d, ignore_errors=True)
    with open(os.path.join(out, "WRITES.json"), "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.write("\n")


#: Tables for the writer allow-list (design section 4.2): each created by delta-spark with one
#: row. ``refused`` rows are features Krishiv does not write; the others it appends to.
WRITE_FEATURE_TABLES: list[tuple[str, bool, list[str]]] = [
    ("append-only", False, ["CREATE TABLE {t} (id BIGINT) USING delta TBLPROPERTIES ('delta.appendOnly' = 'true')"]),
    (
        "change-data-feed",
        False,
        ["CREATE TABLE {t} (id BIGINT) USING delta TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"],
    ),
    (
        "column-mapping",
        False,
        [
            "CREATE TABLE {t} (id BIGINT, s STRUCT<x: INT>) USING delta "
            "TBLPROPERTIES ('delta.columnMapping.mode' = 'name')"
        ],
    ),
    (
        "deletion-vectors",
        False,
        ["CREATE TABLE {t} (id BIGINT) USING delta TBLPROPERTIES ('delta.enableDeletionVectors' = 'true')"],
    ),
    (
        "domain-metadata",
        False,
        ["CREATE TABLE {t} (id BIGINT) USING delta TBLPROPERTIES ('delta.feature.domainMetadata' = 'supported')"],
    ),
    (
        "in-commit-timestamp",
        False,
        ["CREATE TABLE {t} (id BIGINT) USING delta TBLPROPERTIES ('delta.enableInCommitTimestamps' = 'true')"],
    ),
    (
        "v2-checkpoint",
        False,
        ["CREATE TABLE {t} (id BIGINT) USING delta TBLPROPERTIES ('delta.checkpointPolicy' = 'v2')"],
    ),
    (
        "vacuum-protocol-check",
        False,
        ["CREATE TABLE {t} (id BIGINT) USING delta TBLPROPERTIES ('delta.feature.vacuumProtocolCheck' = 'supported')"],
    ),
    ("timestamp-ntz", False, ["CREATE TABLE {t} (id BIGINT, n TIMESTAMP_NTZ) USING delta"]),
    (
        "type-widening",
        False,
        ["CREATE TABLE {t} (id BIGINT, i INT) USING delta TBLPROPERTIES ('delta.enableTypeWidening' = 'true')"],
    ),
    (
        "row-tracking",
        True,
        ["CREATE TABLE {t} (id BIGINT) USING delta TBLPROPERTIES ('delta.enableRowTracking' = 'true')"],
    ),
    (
        "iceberg-compat-v2",
        True,
        [
            "CREATE TABLE {t} (id BIGINT) USING delta TBLPROPERTIES ('delta.columnMapping.mode' = 'name', "
            "'delta.enableIcebergCompatV2' = 'true')"
        ],
    ),
    ("clustering", True, ["CREATE TABLE {t} (id BIGINT) USING delta CLUSTER BY (id)"]),
    (
        "check-constraints",
        True,
        ["CREATE TABLE {t} (id BIGINT) USING delta TBLPROPERTIES ('delta.feature.checkConstraints' = 'supported')"],
    ),
    ("generated-columns", True, ["CREATE TABLE {t} (id BIGINT, g BIGINT GENERATED ALWAYS AS (id * 2)) USING delta"]),
    ("identity-columns", True, ["CREATE TABLE {t} (id BIGINT, g BIGINT GENERATED ALWAYS AS IDENTITY) USING delta"]),
    (
        "column-defaults",
        True,
        ["CREATE TABLE {t} (id BIGINT) USING delta TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported')"],
    ),
    ("legacy-1-3", True, ["CREATE TABLE {t} (id BIGINT) USING delta TBLPROPERTIES ('delta.minWriterVersion' = '3')"]),
    ("legacy-1-4", True, ["CREATE TABLE {t} (id BIGINT) USING delta TBLPROPERTIES ('delta.minWriterVersion' = '4')"]),
    (
        "legacy-2-5",
        True,
        [
            "CREATE TABLE {t} (id BIGINT) USING delta TBLPROPERTIES ('delta.minReaderVersion' = '2', "
            "'delta.minWriterVersion' = '5')"
        ],
    ),
    (
        "legacy-2-6",
        True,
        [
            "CREATE TABLE {t} (id BIGINT) USING delta TBLPROPERTIES ('delta.minReaderVersion' = '2', "
            "'delta.minWriterVersion' = '6')"
        ],
    ),
    (
        "check-on-1-2",
        True,
        [
            "CREATE TABLE {t} (id BIGINT) USING delta",
            "ALTER TABLE {t} ADD CONSTRAINT positive CHECK (id > 0)",
        ],
    ),
]


def write_feature_tables(spark: SparkSession, out: str) -> None:
    """``write-features/<name>``: one table per row of ``WRITE_FEATURE_TABLES`` with one row
    (``id`` = 1, other columns NULL), and ``write-features/FEATURES.json`` with each table's
    protocol, what created it and whether Krishiv refuses writes to it."""
    root = os.path.join(out, "write-features")
    os.makedirs(root, exist_ok=True)
    report: dict = {
        "format_version": FORMAT_VERSION,
        "generator": "tests/scripts/gen_delta_fixtures.py",
        "spark": spark.version,
        "delta_spark": "4.4.1",
        "tables": {},
    }
    for name, refused, sqls in WRITE_FEATURE_TABLES:
        path = os.path.join(root, name)
        t = f"delta.`{path}`"
        entry: dict = {"refused": refused, "created_by": [q.replace("{t}", "<t>") for q in sqls]}
        try:
            for q in sqls:
                spark.sql(q.format(t=t)).collect()
            spark.sql(f"INSERT INTO {t} (id) VALUES (1)").collect()
        except Exception as e:  # the oracle's error is the record
            entry["error"] = oracle_error(e, path)
            shutil.rmtree(path, ignore_errors=True)
            report["tables"][name] = entry
            continue
        entry["protocol"] = protocol(path)
        df = spark.read.format("delta").load(path)
        entry["fields"] = [f.name for f in df.schema.fields]
        entry["rows"] = [[typed(r[i]) for i in range(len(r))] for r in df.orderBy("id").collect()]
        report["tables"][name] = entry
    with open(os.path.join(root, "FEATURES.json"), "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
        f.write("\n")


def main(out: str, which: str = "checkpoint") -> None:
    spark = session()
    if which == "writes":
        stats_table(spark, out)
        write_modes(spark, out)
        write_feature_tables(spark, out)
        drop_hadoop_crc(out)
        spark.stop()
        return
    if which == "session":
        session_probes(spark, out)
        drop_hadoop_crc(out)
        spark.stop()
        return
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
