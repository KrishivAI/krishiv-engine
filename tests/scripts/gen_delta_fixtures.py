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

``... /out dml`` writes the DML oracle (``dml_cases``): per scenario a ``before`` table and
``dml/DML.json`` with each ``DELETE``/``UPDATE`` step's output, commits, rows and history.

``... /out insert-path`` writes ``INSERT-PATH.json`` (``insert_by_path``): ``INSERT INTO
delta.`<path>``` in each form, its arity refusals and its refused targets.

``... /out merge`` writes the MERGE oracle (``merge_cases``): the same for ``MERGE`` steps, in
``merge/MERGE.json``, with the file-backed sources the steps read. Use a fresh ``$OUT`` for each
attempt: the run refuses an ``$OUT/merge`` left by an earlier or interrupted one.

``... /out create-path`` writes ``CREATE-PATH.json`` (``create_by_path``): ``CREATE TABLE
delta.`<path>``` in each form, the log it writes and its refusals.

``... /out dvs`` writes the deletion-vector oracle (``dv_cases``, PR-37d): ``DELETE``, ``UPDATE``
and ``MERGE`` steps on tables with ``delta.enableDeletionVectors = true``, in ``dvs/DVS.json``,
with every commit's deletion vector descriptors and the change-data-feed rows Spark derives.

``... /out dvs-no-stats`` writes the same for a table whose files carry no statistics, in
``dvs-no-stats/NOSTATS.json``.

``... /out options`` writes ``OPTIONS.json`` (``write_options``, PR-37e): delta-spark's handling
of each Delta write option, and Spark 4.2.0's parse of the ``WITH (...)`` statement options.

``... /out catalog-options`` writes ``CATALOG-OPTIONS.json`` (``catalog_write_options``): which
Delta write options delta-spark reads from the DataFrame writers of a catalog table
(``insertInto``, ``saveAsTable`` append, ``writeTo(..).append()``), and the session's
``spark.databricks.delta.commitInfo.userMetadata`` against the option.

``... /out dml-user-metadata`` writes ``DML-USER-METADATA.json`` (``dml_user_metadata``): the
``commitInfo.userMetadata`` delta-spark gives ``DELETE``, ``UPDATE`` and ``MERGE`` commits from that
session conf, and ``insertInto``'s error for a frame with an extra column.

``... /out readback /in`` has Spark read the Krishiv-written tables under ``/in``
(``tests/fixtures/delta-4.4.1/krishiv-written/``) and records ``READBACK.json`` there.

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
    # The classified causes below it, nearest first (Spark's `TASK_WRITE_FAILED` wraps the
    # condition the write raised).
    causes = []
    cause = java.getCause() if java is not None else None
    while cause is not None and len(causes) < 8:
        try:
            condition = cause.getErrorClass()
        except Exception:
            condition = None
        if condition is not None:
            causes.append(
                {
                    "jvm_class": cause.getClass().getName(),
                    "condition": condition,
                    "sqlstate": cause.getSqlState(),
                    "parameters": {
                        k: str(v).replace(path, "<path>") for k, v in dict(cause.getMessageParameters()).items()
                    },
                    "message": (cause.getMessage() or "").splitlines()[0].replace(path, "<path>")[:600],
                }
            )
        cause = cause.getCause()
    if causes:
        record["causes"] = causes
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
                    removed = {"path": rm["path"], "partitionValues": rm.get("partitionValues")}
                    if rm.get("deletionVector"):
                        removed["deletionVector"] = rm["deletionVector"]
                    record["removes"].append(removed)
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


def insert_by_path(spark: SparkSession, out: str) -> None:
    """``INSERT-PATH.json``: ``INSERT INTO delta.`<path>``` beside ``WRITES.json`` (which
    replays its path steps through a catalog table): the forms of the statement on an existing
    table (column list, ``BY NAME``, ``TABLE``, a static partition, a trailing slash, a ``file:``
    URI), then each refusal: a missing path, an empty directory, a Parquet directory, a name
    without a ``/``, a relative path, an arity mismatch (also under a static partition). Each step
    records the oracle's error or its commit and rows; a refused target's path is elided as
    ``<missing>``."""
    path = os.path.join(out, "insert-path")
    t = f"delta.`{path}`"
    spark.sql(f"CREATE TABLE {t} (id BIGINT, v STRING, p STRING) USING delta PARTITIONED BY (p)")
    spark.sql(f"INSERT INTO {t} VALUES (1, 'a', 'x')")
    steps: list[tuple[str, Any]] = [
        ("values", f"INSERT INTO {t} VALUES (2, 'b', 'y')"),
        ("table_keyword", f"INSERT INTO TABLE {t} VALUES (3, 'c', 'x')"),
        ("column_list", f"INSERT INTO {t} (id, p, v) VALUES (4, 'y', 'd')"),
        ("column_list_partial", f"INSERT INTO {t} (id, p) VALUES (5, 'x')"),
        ("by_name", f"INSERT INTO {t} BY NAME SELECT 'e' AS v, CAST(6 AS BIGINT) AS id, 'y' AS p"),
        ("static_partition", f"INSERT INTO {t} PARTITION (p = 'z') VALUES (7, 'f')"),
        ("trailing_slash", f"INSERT INTO delta.`{path}/` VALUES (8, 'g', 'x')"),
        ("file_uri", f"INSERT INTO delta.`file:{path}` VALUES (9, 'h', 'y')"),
        ("reads_itself", f"INSERT INTO {t} SELECT id + 100, v, p FROM {t} WHERE id < 3"),
        ("arity", f"INSERT INTO {t} VALUES (10, 'i')"),
        ("arity_too_many", f"INSERT INTO {t} VALUES (10, 'i', 'x', 1)"),
        ("arity_column_list", f"INSERT INTO {t} (id, v) VALUES (10, 'i', 'x')"),
        ("catalog_create", f"CREATE TABLE insert_path_cat USING delta LOCATION '{path}'"),
        ("catalog_arity", "INSERT INTO insert_path_cat VALUES (10, 'i')"),
        ("catalog_arity_too_many", "INSERT INTO insert_path_cat VALUES (10, 'i', 'x', 1)"),
        ("catalog_drop", "DROP TABLE insert_path_cat"),
        ("static_partition_arity", f"INSERT INTO {t} PARTITION (p = 'z') VALUES (7)"),
        ("static_partition_arity_too_many", f"INSERT INTO {t} PARTITION (p = 'z') VALUES (7, 'f', 'x')"),
        ("overwrite", f"INSERT OVERWRITE {t} VALUES (11, 'j', 'x')"),
    ]
    record: dict = {
        "format_version": FORMAT_VERSION,
        "generator": "tests/scripts/gen_delta_fixtures.py",
        "spark": spark.version,
        "delta_spark": "4.4.1",
        "steps": run_steps(spark, path, steps),
    }
    missing = os.path.join(out, "insert-missing")
    empty = os.path.join(out, "insert-empty")
    os.makedirs(empty)
    parquet = os.path.join(out, "insert-parquet")
    spark.sql("SELECT CAST(1 AS BIGINT) AS id").write.parquet(parquet)
    refused = []
    for name, target in (("missing", missing), ("empty_directory", empty), ("parquet_directory", parquet)):
        step = outcome(spark, f"INSERT INTO delta.`{target}` VALUES (1)", target)
        step["sql"] = step["sql"].replace("<path>", "<missing>")
        step["step"] = name
        step["created_log"] = os.path.isdir(os.path.join(target, "_delta_log"))
        refused.append(step)
    no_slash = outcome(spark, "INSERT INTO delta.`krishiv_no_slash` VALUES (1)", "krishiv_no_slash")
    no_slash["step"] = "no_slash"
    refused.append(no_slash)
    relative = outcome(spark, "INSERT INTO delta.`krishiv_rel/dir` VALUES (1)", "krishiv_rel/dir")
    relative["step"] = "relative_path"
    relative["created_log"] = os.path.isdir(os.path.join("krishiv_rel", "dir", "_delta_log"))
    refused.append(relative)
    record["refused"] = refused
    for d in (path, missing, empty, parquet):
        shutil.rmtree(d, ignore_errors=True)
    with open(os.path.join(out, "INSERT-PATH.json"), "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.write("\n")


def write_edges(spark: SparkSession, out: str) -> None:
    """``WRITE-EDGES.json``: write edge cases next to ``WRITES.json`` (review of PR-37a part 2):
    an empty-string partition value (stored as null) under a dynamic overwrite, an append and
    ``replaceWhere``; a NULL written into a ``NOT NULL`` column (top level and nested) through
    the DataFrame writer and SQL; a ``''`` into a ``NOT NULL`` partition column;
    ``maxRecordsPerFile`` (a number, ``0``, ``-1`` and a value that is not one; also through
    ``insertInto`` and ``saveAsTable`` on a catalog table); whether ``insertInto`` overwrite reads a
    ``partitionOverwriteMode`` writer option (it does not: the session configuration decides)."""
    record: dict = {
        "format_version": FORMAT_VERSION,
        "generator": "tests/scripts/gen_delta_fixtures.py",
        "spark": spark.version,
        "delta_spark": "4.4.1",
    }
    empty = os.path.join(out, "write-empty-partition")
    t = f"delta.`{empty}`"
    spark.sql(f"CREATE TABLE {t} (id BIGINT, v STRING, p STRING) USING delta PARTITIONED BY (p)")
    spark.sql(f"INSERT INTO {t} VALUES (1, 'a', NULL), (2, 'b', 'x')")

    def dynamic(s: SparkSession) -> None:
        s.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
        try:
            s.sql(f"INSERT OVERWRITE {t} VALUES (3, 'c', '')").collect()
        finally:
            s.conf.unset("spark.sql.sources.partitionOverwriteMode")

    def replace_where(where: str, rows: str):
        def step(s: SparkSession) -> None:
            s.sql(f"SELECT * FROM VALUES {rows} AS v(id, v, p)").selectExpr(
                "CAST(id AS BIGINT) AS id", "v", "p"
            ).write.format("delta").mode("overwrite").option("replaceWhere", where).save(empty)

        return step

    record["empty_partition"] = run_steps(
        spark,
        empty,
        [
            ("dynamic_overwrite_empty", dynamic),
            ("append_empty", f"INSERT INTO {t} VALUES (4, 'd', ''), (5, 'e', NULL)"),
            ("replace_where_empty", replace_where("p = ''", "(6, 'f', '')")),
            ("replace_where_null_with_empty", replace_where("p IS NULL", "(7, 'g', '')")),
        ],
    )
    nn = os.path.join(out, "write-not-null")
    n = f"delta.`{nn}`"
    spark.sql(f"CREATE TABLE {n} (id BIGINT, a INT NOT NULL, s STRUCT<x: INT NOT NULL, y: INT>) USING delta")
    spark.sql(f"INSERT INTO {n} VALUES (1, 1, named_struct('x', 1, 'y', 1))")

    def append_df(select: str):
        def step(s: SparkSession) -> None:
            s.sql(select).write.format("delta").mode("append").save(nn)

        return step

    record["not_null"] = run_steps(
        spark,
        nn,
        [
            (
                "dataframe_null_top_level",
                append_df("SELECT CAST(2 AS BIGINT) AS id, CAST(NULL AS INT) AS a, named_struct('x', 2, 'y', 2) AS s"),
            ),
            (
                "dataframe_null_nested",
                append_df("SELECT CAST(3 AS BIGINT) AS id, 3 AS a, named_struct('x', CAST(NULL AS INT), 'y', 3) AS s"),
            ),
            (
                "dataframe_null_struct",
                append_df("SELECT CAST(4 AS BIGINT) AS id, 4 AS a, CAST(NULL AS STRUCT<x: INT, y: INT>) AS s"),
            ),
            ("sql_null_top_level", f"INSERT INTO {n} VALUES (5, NULL, named_struct('x', 5, 'y', 5))"),
        ],
    )
    # An empty string into a NOT NULL partition column: delta-spark stores it as NULL first, so
    # the NOT NULL check sees the NULL (review of 48ad1876d, finding 1).
    np_ = os.path.join(out, "write-not-null-partition")
    q = f"delta.`{np_}`"
    spark.sql(f"CREATE TABLE {q} (id BIGINT, p STRING NOT NULL) USING delta PARTITIONED BY (p)")
    spark.sql(f"INSERT INTO {q} VALUES (1, 'x')")

    def append_empty_df(s: SparkSession) -> None:
        s.sql("SELECT CAST(3 AS BIGINT) AS id, '' AS p").write.format("delta").mode("append").save(np_)

    record["not_null_partition"] = run_steps(
        spark,
        np_,
        [
            ("sql_empty_partition", f"INSERT INTO {q} VALUES (2, '')"),
            ("dataframe_empty_partition", append_empty_df),
        ],
    )
    mr = os.path.join(out, "write-max-records")

    def max_records(value: str, mode: str):
        def step(s: SparkSession) -> None:
            s.sql("SELECT CAST(id AS BIGINT) AS id FROM range(5)").repartition(1).write.format("delta").mode(
                mode
            ).option("maxRecordsPerFile", value).save(mr)

        return step

    record["max_records"] = run_steps(
        spark,
        mr,
        [
            ("create_two", max_records("2", "errorifexists")),
            ("append_one", max_records("1", "append")),
            ("append_not_a_number", max_records("abc", "append")),
            ("append_zero", max_records("0", "append")),
            ("append_negative", max_records("-1", "append")),
        ],
    )
    # The same option on the catalog-table writers: ``insertInto`` (append and overwrite) and
    # ``saveAsTable`` append (review of 1254d3d76, finding 1).
    mt = os.path.join(out, "write-max-records-table")
    spark.sql(f"CREATE TABLE mrt (id BIGINT) USING delta LOCATION '{mt}'")

    def table_write(value: str, how: str):
        def step(s: SparkSession) -> None:
            w = (
                s.sql("SELECT CAST(id AS BIGINT) AS id FROM range(5)")
                .repartition(1)
                .write.format("delta")
                .option("maxRecordsPerFile", value)
            )
            if how == "insert_into":
                w.mode("append").insertInto("mrt")
            elif how == "insert_into_overwrite":
                w.mode("overwrite").insertInto("mrt")
            else:
                w.mode("append").saveAsTable("mrt")

        return step

    record["max_records_table"] = run_steps(
        spark,
        mt,
        [
            ("insert_into_one", table_write("1", "insert_into")),
            ("insert_into_overwrite_one", table_write("1", "insert_into_overwrite")),
            ("insert_into_not_a_number", table_write("abc", "insert_into")),
            ("save_as_table_append_one", table_write("1", "save_as_table")),
        ],
    )
    spark.sql("DROP TABLE mrt")
    # ``insertInto`` overwrite of a partitioned catalog table: is the overwrite dynamic by a
    # ``partitionOverwriteMode`` writer option, or by the session configuration only? (review
    # of 7ba0c47db, nit 2.)
    om = os.path.join(out, "write-insert-into-overwrite-mode")
    spark.sql(f"CREATE TABLE iom (id BIGINT, p STRING) USING delta PARTITIONED BY (p) LOCATION '{om}'")
    spark.sql("INSERT INTO iom VALUES (1, 'a'), (2, 'b')")

    def overwrite_with_option(row_id: int, option: str, conf: str | None):
        def step(s: SparkSession) -> None:
            if conf is not None:
                s.conf.set("spark.sql.sources.partitionOverwriteMode", conf)
            try:
                s.sql(f"SELECT CAST({row_id} AS BIGINT) AS id, 'a' AS p").write.mode("overwrite").option(
                    "partitionOverwriteMode", option
                ).insertInto("iom")
            finally:
                s.conf.unset("spark.sql.sources.partitionOverwriteMode")

        return step

    record["insert_into_overwrite_mode"] = run_steps(
        spark,
        om,
        [
            ("option_dynamic_conf_unset", overwrite_with_option(3, "dynamic", None)),
            ("insert_b", "INSERT INTO iom VALUES (4, 'b')"),
            ("option_static_conf_dynamic", overwrite_with_option(5, "static", "dynamic")),
        ],
    )
    spark.sql("DROP TABLE iom")
    for d in (empty, nn, np_, mr, mt, om):
        shutil.rmtree(d, ignore_errors=True)
    with open(os.path.join(out, "WRITE-EDGES.json"), "w") as f:
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


#: The DML scenarios (``gen_delta_fixtures.py /out dml``): a table set up by ``setup`` (``{t}`` is
#: the table), copied to ``dml/<name>/before``, then each step run on its own copy of ``before``.
#: A step is (name, SQL, Spark confs set around it).
DML_COLUMNS = "id BIGINT, v STRING, n INT"
DML_ROWS = [
    "(1, 'a', 1), (2, 'b', NULL), (3, 'c', 3)",
    "(4, 'd', 4), (5, NULL, 50), (6, 'f', NULL)",
]
DML_PART_ROWS = [
    "(1, 'a', 1, 'x'), (2, 'b', NULL, 'x'), (3, 'c', 3, 'y')",
    "(4, 'd', 4, 'y'), (5, NULL, 50, NULL), (6, 'f', NULL, 'z')",
]
NO_DVS = {
    "spark.databricks.delta.delete.deletionVectors.persistent": "false",
    "spark.databricks.delta.update.deletionVectors.persistent": "false",
}
DML_SCENARIOS: list[tuple[str, list[str], list[tuple[str, str, dict]]]] = [
    (
        "unpartitioned",
        [f"CREATE TABLE {{t}} ({DML_COLUMNS}) USING delta"] + [f"INSERT INTO {{t}} VALUES {r}" for r in DML_ROWS],
        [
            ("delete_rows", "DELETE FROM {t} WHERE id = 2", {}),
            ("delete_null_condition", "DELETE FROM {t} WHERE n > 3", {}),
            ("delete_whole_file_by_rows", "DELETE FROM {t} WHERE id <= 3", {}),
            ("delete_nothing", "DELETE FROM {t} WHERE id = 999", {}),
            ("delete_all", "DELETE FROM {t}", {}),
            ("delete_alias", "DELETE FROM {t} AS x WHERE x.v = 'c' OR x.v IS NULL", {}),
            ("update_rows", "UPDATE {t} SET v = 'u', n = n + 1 WHERE id >= 3", {}),
            ("update_null_condition", "UPDATE {t} SET n = 0 WHERE n < 4", {}),
            ("update_is_null", "UPDATE {t} SET n = -1 WHERE n IS NULL", {}),
            ("update_all", "UPDATE {t} SET v = concat(v, '!')", {}),
            ("update_store_assignment", "UPDATE {t} SET n = id * 10, v = 7 WHERE id = 1", {}),
            ("update_overflow", "UPDATE {t} SET n = id * 10000000000 WHERE id = 1", {}),
            ("update_nothing", "UPDATE {t} SET n = 0 WHERE id = 999", {}),
            # Errors raised while the new values are written (`TASK_WRITE_FAILED` around them).
            ("update_divide_by_zero", "UPDATE {t} SET n = n div 0 WHERE id = 1", {}),
            ("update_arithmetic_overflow", "UPDATE {t} SET n = n + 2147483647 WHERE id = 3", {}),
            ("update_explicit_cast_overflow", "UPDATE {t} SET n = CAST(id * 10000000000 AS INT) WHERE id = 1", {}),
            # Subqueries in the condition (delta-spark refuses them).
            ("delete_in_subquery", "DELETE FROM {t} WHERE id IN (SELECT id FROM {t} WHERE n = 3)", {}),
            ("delete_exists_subquery", "DELETE FROM {t} WHERE EXISTS (SELECT 1)", {}),
            ("delete_scalar_subquery", "DELETE FROM {t} WHERE id = (SELECT max(id) FROM {t})", {}),
            ("update_in_subquery", "UPDATE {t} SET n = 0 WHERE id IN (SELECT 2)", {}),
        ],
    ),
    (
        "partitioned",
        [f"CREATE TABLE {{t}} ({DML_COLUMNS}, p STRING) USING delta PARTITIONED BY (p)"]
        + [f"INSERT INTO {{t}} VALUES {r}" for r in DML_PART_ROWS],
        [
            ("delete_partition", "DELETE FROM {t} WHERE p = 'x'", {}),
            ("delete_partition_null", "DELETE FROM {t} WHERE p IS NULL", {}),
            ("delete_partition_in", "DELETE FROM {t} WHERE p IN ('y', 'z')", {}),
            ("delete_partition_and_rows", "DELETE FROM {t} WHERE p = 'y' AND id = 3", {}),
            ("delete_rows_any_partition", "DELETE FROM {t} WHERE n IS NULL", {}),
            ("delete_all", "DELETE FROM {t}", {}),
            ("update_partition_only", "UPDATE {t} SET v = 'all' WHERE p = 'y'", {}),
            ("update_moves_partition", "UPDATE {t} SET p = 'w' WHERE id = 4", {}),
            ("update_partition_to_null", "UPDATE {t} SET p = NULL WHERE p = 'z'", {}),
            ("update_rows", "UPDATE {t} SET n = n * 2 WHERE id > 1", {}),
            ("delete_partition_and_subquery", "DELETE FROM {t} WHERE p = 'y' AND id IN (SELECT 3)", {}),
        ],
    ),
    (
        "column-mapping",
        [
            f"CREATE TABLE {{t}} ({DML_COLUMNS}, p STRING) USING delta PARTITIONED BY (p) "
            "TBLPROPERTIES ('delta.columnMapping.mode' = 'name')"
        ]
        + [f"INSERT INTO {{t}} VALUES {r}" for r in DML_PART_ROWS]
        + ["ALTER TABLE {t} RENAME COLUMN v TO v2"],
        [
            ("delete_rows", "DELETE FROM {t} WHERE v2 = 'c'", {}),
            ("delete_partition", "DELETE FROM {t} WHERE p = 'x'", {}),
            ("update_rows", "UPDATE {t} SET v2 = 'u' WHERE n IS NOT NULL", {}),
        ],
    ),
    (
        "deletion-vectors",
        [
            f"CREATE TABLE {{t}} ({DML_COLUMNS}) USING delta TBLPROPERTIES ('delta.enableDeletionVectors' = 'true')",
            "INSERT INTO {t} SELECT id, CAST(id AS STRING), CAST(id % 7 AS INT) FROM range(1, 101)",
            "DELETE FROM {t} WHERE id % 10 = 0",
        ],
        [
            ("delete_rows_copy_on_write", "DELETE FROM {t} WHERE id < 5", NO_DVS),
            ("update_rows_copy_on_write", "UPDATE {t} SET v = 'u' WHERE n = 3", NO_DVS),
            ("delete_all", "DELETE FROM {t}", NO_DVS),
            ("delete_rows_with_dvs", "DELETE FROM {t} WHERE id < 5", {}),
            ("update_rows_with_dvs", "UPDATE {t} SET v = 'u' WHERE n = 3", {}),
        ],
    ),
    (
        "change-data-feed",
        [
            f"CREATE TABLE {{t}} ({DML_COLUMNS}, p STRING) USING delta PARTITIONED BY (p) "
            "TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"
        ]
        + [f"INSERT INTO {{t}} VALUES {r}" for r in DML_PART_ROWS],
        [
            ("delete_partition", "DELETE FROM {t} WHERE p = 'x'", {}),
            ("delete_all", "DELETE FROM {t}", {}),
            ("delete_rows", "DELETE FROM {t} WHERE id = 3", {}),
            ("update_rows", "UPDATE {t} SET v = 'u' WHERE id = 3", {}),
            ("update_partition_only", "UPDATE {t} SET v = 'u' WHERE p = 'y'", {}),
        ],
    ),
    (
        # A struct column `s` with a field `p` beside the partition column `p`: `s.p` is the
        # struct's field, never the partition column (`x.p` under the alias `x` is the partition).
        "struct-partition",
        [
            "CREATE TABLE {t} (id BIGINT, s STRUCT<p: INT, q: INT>, p INT) USING delta PARTITIONED BY (p)",
            "INSERT INTO {t} VALUES (1, named_struct('p', 1, 'q', 1), 1), (2, named_struct('p', 2, 'q', 2), 1)",
            "INSERT INTO {t} VALUES (3, named_struct('p', 2, 'q', 3), 1), (4, named_struct('p', 1, 'q', 4), 2)",
            "INSERT INTO {t} VALUES (5, named_struct('p', 3, 'q', 5), 1), (6, named_struct('p', 3, 'q', 6), 2)",
        ],
        [
            ("delete_struct_field", "DELETE FROM {t} WHERE s.p = 1", {}),
            ("delete_struct_field_none_match", "DELETE FROM {t} WHERE s.p = 9", {}),
            ("delete_qualified_partition", "DELETE FROM {t} AS x WHERE x.p = 2", {}),
            ("delete_struct_field_and_partition", "DELETE FROM {t} WHERE p = 1 AND s.p = 2", {}),
            ("update_struct_field", "UPDATE {t} SET id = id + 100 WHERE s.p = 2", {}),
        ],
    ),
    (
        # A TIMESTAMP partition column compared with a string literal under a non-UTC session
        # zone: Spark casts the literal in the session zone.
        "timestamp-partition",
        [
            "CREATE TABLE {t} (id BIGINT, ts TIMESTAMP) USING delta PARTITIONED BY (ts)",
            "INSERT INTO {t} VALUES (1, TIMESTAMP '2024-01-01 00:00:00'), (2, TIMESTAMP '2024-01-01 08:00:00')",
            "INSERT INTO {t} VALUES (3, TIMESTAMP '2024-01-01 08:00:00'), (4, TIMESTAMP '2024-01-01 16:00:00')",
        ],
        [
            ("delete_string_literal_utc", "DELETE FROM {t} WHERE ts = '2024-01-01 08:00:00'", {}),
            (
                "delete_string_literal_zone",
                "DELETE FROM {t} WHERE ts = '2024-01-01 00:00:00'",
                {"spark.sql.session.timeZone": "America/Los_Angeles"},
            ),
            (
                "delete_timestamp_literal_zone",
                "DELETE FROM {t} WHERE ts = TIMESTAMP '2024-01-01 00:00:00'",
                {"spark.sql.session.timeZone": "America/Los_Angeles"},
            ),
            (
                "delete_string_range_zone",
                "DELETE FROM {t} WHERE ts < '2024-01-01 00:00:00'",
                {"spark.sql.session.timeZone": "America/Los_Angeles"},
            ),
            (
                "update_string_literal_zone",
                "UPDATE {t} SET id = id + 100 WHERE ts = '2024-01-01 00:00:00'",
                {"spark.sql.session.timeZone": "America/Los_Angeles"},
            ),
        ],
    ),
    (
        "append-only",
        [
            f"CREATE TABLE {{t}} ({DML_COLUMNS}) USING delta TBLPROPERTIES ('delta.appendOnly' = 'true')",
            f"INSERT INTO {{t}} VALUES {DML_ROWS[0]}",
        ],
        [
            ("delete_rows", "DELETE FROM {t} WHERE id = 2", {}),
            ("delete_all", "DELETE FROM {t}", {}),
            ("update_rows", "UPDATE {t} SET n = 0 WHERE id = 2", {}),
        ],
    ),
]


def dml_cases(spark: SparkSession, out: str) -> None:
    """``dml/``: per scenario of ``DML_SCENARIOS`` the table before the steps (``before``) and
    ``DML.json``: per step its SQL, the command's output (columns and rows) or error, the commits
    it wrote, the rows read back (by ``id``) and ``DESCRIBE HISTORY``'s newest row (operation,
    parameters, metrics). Each step runs on its own copy of ``before``."""
    root = os.path.join(out, "dml")
    os.makedirs(root, exist_ok=True)
    record: dict = {
        "format_version": FORMAT_VERSION,
        "generator": "tests/scripts/gen_delta_fixtures.py dml",
        "spark": spark.version,
        "delta_spark": "4.4.1",
        "scenarios": {},
    }
    for name, setup, steps in DML_SCENARIOS:
        before = os.path.join(root, name, "before")
        for q in setup:
            spark.sql(q.format(t=f"delta.`{before}`")).collect()
        df = spark.read.format("delta").load(before)
        scenario: dict = {
            "setup": [q.replace("{t}", "<t>") for q in setup],
            "fields": [f.name for f in df.schema.fields],
            "rows": [[typed(r[i]) for i in range(len(r))] for r in df.orderBy("id").collect()],
            "versions": len(commits(before)),
            "steps": [],
        }
        for step, sql, confs in steps:
            work = os.path.join(out, "dml-work", name, step)
            shutil.copytree(before, work)
            text = sql.format(t=f"delta.`{work}`")
            entry: dict = {"step": step, "sql": sql.replace("{t}", "<t>"), "confs": confs}
            for k, v in confs.items():
                spark.conf.set(k, v)
            try:
                result = spark.sql(text)
                entry["output_fields"] = [f.name for f in result.schema.fields]
                entry["output"] = [[typed(r[i]) for i in range(len(r))] for r in result.collect()]
                entry["error"] = None
            except Exception as e:  # the oracle's error is the record
                entry["error"] = oracle_error(e, work)
            finally:
                for k in confs:
                    spark.conf.unset(k)
            log = commits(work)
            entry["commits"] = log[scenario["versions"] :]
            back = spark.read.format("delta").load(work)
            entry["rows"] = [[typed(r[i]) for i in range(len(r))] for r in back.orderBy("id").collect()]
            newest = spark.sql(f"DESCRIBE HISTORY delta.`{work}` LIMIT 1").collect()[0]
            entry["history"] = {
                "version": newest["version"],
                "operation": newest["operation"],
                "operationParameters": dict(newest["operationParameters"] or {}),
                "operationMetrics": dict(newest["operationMetrics"] or {}),
            }
            scenario["steps"].append(entry)
            shutil.rmtree(work)
        record["scenarios"][name] = scenario
    shutil.rmtree(os.path.join(out, "dml-work"), ignore_errors=True)
    with open(os.path.join(root, "DML.json"), "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.write("\n")


#: MERGE sources: an inline query (ids 2 and 4 match, 7 and NULL do not), one with a target row
#: matched twice (id 2), one with keys of the `n` column (NULL among them) and an empty one.
MERGE_SRC = (
    "(SELECT CAST(id AS BIGINT) AS id, v, n FROM VALUES (2, 'B', 20), (4, 'D', 40), (7, 'G', 70), "
    "(NULL, 'N', 0) AS s(id, v, n)) s"
)
MERGE_DUP_SRC = (
    "(SELECT CAST(id AS BIGINT) AS id, v, n FROM VALUES (2, 'x', 1), (2, 'y', 2), (8, 'z', 3) AS s(id, v, n)) s"
)
MERGE_N_SRC = "(SELECT CAST(id AS BIGINT) AS id, v, n FROM VALUES (10, 'k', 1), (11, 'l', NULL) AS s(id, v, n)) s"
MERGE_EMPTY_SRC = "(SELECT CAST(1 AS BIGINT) AS id, 'e' AS v, 1 AS n WHERE false) s"
MERGE_NONE_SRC = "(SELECT CAST(99 AS BIGINT) AS id, 'z' AS v, 9 AS n) s"
MERGE_WIDE_SRC = "(SELECT CAST(2 AS BIGINT) AS id, 'W' AS v, 2 AS n, 'extra' AS e) s"
#: A source without the target's `n` column.
MERGE_NARROW_SRC = "(SELECT CAST(id AS BIGINT) AS id, v FROM VALUES (2, 'B'), (7, 'G') AS s(id, v)) s"
MERGE_PART_SRC = (
    "(SELECT CAST(id AS BIGINT) AS id, v, n, p FROM VALUES (2, 'B', 20, 'x'), (4, 'D', 40, 'q'), "
    "(7, 'G', 70, 'q') AS s(id, v, n, p)) s"
)
MERGE_PART_NONE_SRC = "(SELECT CAST(99 AS BIGINT) AS id, 'z' AS v, 9 AS n, 'q' AS p) s"
#: The rows of the file-backed sources (`<src_parquet>`, `<src_delta>`), written once by the
#: generator.
MERGE_FILE_SOURCE = "SELECT CAST(id AS BIGINT) AS id, v, n FROM VALUES (3, 'P', 30), (9, 'Q', 90) AS s(id, v, n)"


AUTO_MERGE = {"spark.databricks.delta.schema.autoMerge.enabled": "true"}


def merge_on(src: str, clauses: str, on: str = "t.id = s.id") -> str:
    return f"MERGE INTO {{t}} t USING {src} ON {on} {clauses}"


MERGE_SCENARIOS: list[tuple[str, list[str], list[tuple[str, str, dict]]]] = [
    (
        "unpartitioned",
        DML_SCENARIOS[0][1],
        [
            ("update_matched", merge_on(MERGE_SRC, "WHEN MATCHED THEN UPDATE SET v = s.v"), {}),
            ("update_star", merge_on(MERGE_SRC, "WHEN MATCHED THEN UPDATE SET *"), {}),
            ("update_qualified_key", merge_on(MERGE_SRC, "WHEN MATCHED THEN UPDATE SET t.v = s.v, n = t.n + 1"), {}),
            ("delete_matched", merge_on(MERGE_SRC, "WHEN MATCHED THEN DELETE"), {}),
            ("insert_star", merge_on(MERGE_SRC, "WHEN NOT MATCHED THEN INSERT *"), {}),
            ("insert_by_target", merge_on(MERGE_SRC, "WHEN NOT MATCHED BY TARGET THEN INSERT *"), {}),
            (
                "insert_values_partial",
                merge_on(MERGE_SRC, "WHEN NOT MATCHED THEN INSERT (id, v) VALUES (s.id, s.v)"),
                {},
            ),
            (
                "insert_two_clauses",
                merge_on(
                    MERGE_SRC,
                    "WHEN NOT MATCHED AND s.id > 6 THEN INSERT * "
                    "WHEN NOT MATCHED THEN INSERT (id, v, n) VALUES (s.id, 'x', -1)",
                ),
                {},
            ),
            ("upsert", merge_on(MERGE_SRC, "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *"), {}),
            ("nmbs_delete", merge_on(MERGE_SRC, "WHEN NOT MATCHED BY SOURCE THEN DELETE"), {}),
            (
                "nmbs_update_conditional",
                merge_on(MERGE_SRC, "WHEN NOT MATCHED BY SOURCE AND t.n IS NULL THEN UPDATE SET n = -1"),
                {},
            ),
            (
                "all_kinds",
                merge_on(
                    MERGE_SRC,
                    "WHEN MATCHED THEN UPDATE SET n = s.n WHEN NOT MATCHED THEN INSERT * "
                    "WHEN NOT MATCHED BY SOURCE THEN DELETE",
                ),
                {},
            ),
            (
                "clause_order",
                merge_on(
                    MERGE_SRC,
                    "WHEN MATCHED AND s.n > 30 THEN DELETE WHEN MATCHED AND s.n > 10 THEN UPDATE SET n = s.n "
                    "WHEN NOT MATCHED AND s.id > 6 THEN INSERT * "
                    "WHEN NOT MATCHED THEN INSERT (id, v, n) VALUES (s.id, 'x', -1) "
                    "WHEN NOT MATCHED BY SOURCE AND t.id < 2 THEN UPDATE SET v = 'gone' "
                    "WHEN NOT MATCHED BY SOURCE AND t.id > 4 THEN DELETE",
                ),
                {},
            ),
            ("matched_condition_never_holds", merge_on(MERGE_SRC, "WHEN MATCHED AND s.n > 100 THEN DELETE"), {}),
            (
                "target_only_on_conjunct",
                merge_on(MERGE_SRC, "WHEN MATCHED THEN UPDATE SET v = 'c'", "t.id = s.id AND t.n > 2"),
                {},
            ),
            (
                "null_keys",
                merge_on(
                    MERGE_N_SRC,
                    "WHEN MATCHED THEN UPDATE SET v = s.v WHEN NOT MATCHED THEN INSERT * "
                    "WHEN NOT MATCHED BY SOURCE AND t.n IS NULL THEN DELETE",
                    "t.n = s.n",
                ),
                {},
            ),
            (
                "null_safe_keys",
                merge_on(
                    MERGE_N_SRC, "WHEN MATCHED THEN UPDATE SET v = s.v WHEN NOT MATCHED THEN INSERT *", "t.n <=> s.n"
                ),
                {},
            ),
            ("nothing_matched", merge_on(MERGE_NONE_SRC, "WHEN MATCHED THEN UPDATE SET v = s.v"), {}),
            ("empty_source", merge_on(MERGE_EMPTY_SRC, "WHEN MATCHED THEN DELETE WHEN NOT MATCHED THEN INSERT *"), {}),
            ("empty_source_nmbs", merge_on(MERGE_EMPTY_SRC, "WHEN NOT MATCHED BY SOURCE THEN DELETE"), {}),
            (
                "store_assignment",
                merge_on(
                    MERGE_SRC,
                    "WHEN MATCHED THEN UPDATE SET n = s.id * 10 "
                    "WHEN NOT MATCHED THEN INSERT (id, v, n) VALUES (s.id, s.n, s.n)",
                ),
                {},
            ),
            ("update_overflow", merge_on(MERGE_SRC, "WHEN MATCHED THEN UPDATE SET n = s.id * 10000000000"), {}),
            ("multiple_match_update", merge_on(MERGE_DUP_SRC, "WHEN MATCHED THEN UPDATE SET v = s.v"), {}),
            (
                "multiple_match_conditional_update",
                merge_on(MERGE_DUP_SRC, "WHEN MATCHED AND s.v = 'x' THEN UPDATE SET v = s.v"),
                {},
            ),
            ("multiple_match_delete", merge_on(MERGE_DUP_SRC, "WHEN MATCHED THEN DELETE"), {}),
            (
                "multiple_match_delete_insert",
                merge_on(MERGE_DUP_SRC, "WHEN MATCHED THEN DELETE WHEN NOT MATCHED THEN INSERT *"),
                {},
            ),
            ("multiple_match_conditional_delete", merge_on(MERGE_DUP_SRC, "WHEN MATCHED AND s.n > 0 THEN DELETE"), {}),
            ("multiple_match_insert_only", merge_on(MERGE_DUP_SRC, "WHEN NOT MATCHED THEN INSERT *"), {}),
            (
                "multiple_match_insert_only_unoptimized",
                merge_on(MERGE_DUP_SRC, "WHEN NOT MATCHED THEN INSERT *"),
                {"spark.databricks.delta.merge.optimizeInsertOnlyMerge.enabled": "false"},
            ),
            ("multiple_match_nmbs_only", merge_on(MERGE_DUP_SRC, "WHEN NOT MATCHED BY SOURCE THEN DELETE"), {}),
            (
                "non_last_matched_omits_condition",
                merge_on(MERGE_SRC, "WHEN MATCHED THEN DELETE WHEN MATCHED THEN UPDATE SET v = 'x'"),
                {},
            ),
            (
                "non_last_not_matched_omits_condition",
                merge_on(MERGE_SRC, "WHEN NOT MATCHED THEN INSERT * WHEN NOT MATCHED AND s.id > 5 THEN INSERT *"),
                {},
            ),
            (
                "non_last_nmbs_omits_condition",
                merge_on(
                    MERGE_SRC,
                    "WHEN NOT MATCHED BY SOURCE THEN DELETE WHEN NOT MATCHED BY SOURCE AND t.id > 5 THEN DELETE",
                ),
                {},
            ),
            ("unresolved_source_column", merge_on(MERGE_SRC, "WHEN MATCHED THEN UPDATE SET v = s.zzz"), {}),
            ("unresolved_target_column", merge_on(MERGE_SRC, "WHEN MATCHED THEN UPDATE SET zzz = s.v"), {}),
            (
                "insert_reads_target",
                merge_on(MERGE_SRC, "WHEN NOT MATCHED THEN INSERT (id, v, n) VALUES (s.id, t.v, s.n)"),
                {},
            ),
            ("nmbs_reads_source", merge_on(MERGE_SRC, "WHEN NOT MATCHED BY SOURCE THEN UPDATE SET v = s.v"), {}),
            ("unresolved_on", merge_on(MERGE_SRC, "WHEN MATCHED THEN DELETE", "t.id = s.zzz"), {}),
            (
                "insert_duplicate_columns",
                merge_on(MERGE_SRC, "WHEN NOT MATCHED THEN INSERT (id, id) VALUES (s.id, s.id)"),
                {},
            ),
            ("update_duplicate_columns", merge_on(MERGE_SRC, "WHEN MATCHED THEN UPDATE SET v = s.v, v = 'x'"), {}),
            ("non_deterministic_on", merge_on(MERGE_SRC, "WHEN MATCHED THEN DELETE", "t.id = s.id AND rand() < 2"), {}),
            ("subquery_in_clause", merge_on(MERGE_SRC, "WHEN MATCHED AND s.id IN (SELECT 2) THEN DELETE"), {}),
            (
                "schema_evolution_conf",
                merge_on(MERGE_WIDE_SRC, "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *"),
                {"spark.databricks.delta.schema.autoMerge.enabled": "true"},
            ),
            ("star_with_extra_source_column", merge_on(MERGE_WIDE_SRC, "WHEN MATCHED THEN UPDATE SET *"), {}),
            (
                "source_parquet_files",
                merge_on(
                    "(SELECT * FROM parquet.`<src_parquet>`) s",
                    "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *",
                ),
                {},
            ),
            (
                "source_delta_table",
                merge_on("delta.`<src_delta>` s", "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *"),
                {},
            ),
            # Schema evolution on, but no clause brings a new column: the MERGE runs.
            (
                "schema_evolution_conf_no_new_column",
                merge_on(MERGE_SRC, "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *"),
                AUTO_MERGE,
            ),
            (
                "schema_evolution_conf_explicit_assignments",
                merge_on(
                    MERGE_WIDE_SRC,
                    "WHEN MATCHED THEN UPDATE SET v = s.e WHEN NOT MATCHED THEN INSERT (id) VALUES (s.id)",
                ),
                AUTO_MERGE,
            ),
            (
                "schema_evolution_conf_narrow_source",
                merge_on(MERGE_NARROW_SRC, "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *"),
                AUTO_MERGE,
            ),
            ("star_narrow_source", merge_on(MERGE_NARROW_SRC, "WHEN NOT MATCHED THEN INSERT *"), {}),
            ("update_star_narrow_source", merge_on(MERGE_NARROW_SRC, "WHEN MATCHED THEN UPDATE SET *"), {}),
            (
                "subquery_in_insert_clause",
                merge_on(MERGE_SRC, "WHEN NOT MATCHED AND s.id IN (SELECT 7) THEN INSERT *"),
                {},
            ),
            (
                "subquery_in_nmbs_clause",
                merge_on(MERGE_SRC, "WHEN NOT MATCHED BY SOURCE AND t.id IN (SELECT 1) THEN DELETE"),
                {},
            ),
            (
                "subquery_in_update_clause",
                merge_on(MERGE_SRC, "WHEN MATCHED AND s.id IN (SELECT 2) THEN UPDATE SET v = s.v"),
                {},
            ),
            ("non_deterministic_clause", merge_on(MERGE_SRC, "WHEN MATCHED AND rand() < 2 THEN DELETE"), {}),
        ],
    ),
    (
        "partitioned",
        DML_SCENARIOS[1][1],
        [
            ("update_moves_partition", merge_on(MERGE_SRC, "WHEN MATCHED THEN UPDATE SET p = 'w'"), {}),
            (
                "upsert_new_partition",
                merge_on(MERGE_PART_SRC, "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *"),
                {},
            ),
            ("nmbs_delete_partition", merge_on(MERGE_SRC, "WHEN NOT MATCHED BY SOURCE AND t.p = 'z' THEN DELETE"), {}),
            ("on_partition_conjunct", merge_on(MERGE_SRC, "WHEN MATCHED THEN DELETE", "t.id = s.id AND t.p = 'y'"), {}),
            ("insert_only", merge_on(MERGE_PART_SRC, "WHEN NOT MATCHED THEN INSERT *"), {}),
            (
                "upsert_and_conditional_nmbs",
                merge_on(
                    MERGE_PART_SRC,
                    "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT * "
                    "WHEN NOT MATCHED BY SOURCE AND t.p = 'z' THEN UPDATE SET v = 'nmbs'",
                ),
                {},
            ),
        ],
    ),
    (
        "column-mapping",
        DML_SCENARIOS[2][1],
        [
            (
                "upsert",
                merge_on(
                    MERGE_SRC,
                    "WHEN MATCHED THEN UPDATE SET v2 = s.v "
                    "WHEN NOT MATCHED THEN INSERT (id, v2, n, p) VALUES (s.id, s.v, s.n, 'n')",
                ),
                {},
            ),
            ("nmbs_delete", merge_on(MERGE_SRC, "WHEN NOT MATCHED BY SOURCE AND t.v2 IS NULL THEN DELETE"), {}),
        ],
    ),
    (
        "deletion-vectors",
        DML_SCENARIOS[3][1],
        [
            (
                "update_copy_on_write",
                merge_on(MERGE_SRC, "WHEN MATCHED THEN UPDATE SET v = s.v WHEN NOT MATCHED THEN INSERT *"),
                {"spark.databricks.delta.merge.deletionVectors.persistent": "false"},
            ),
        ],
    ),
    (
        "change-data-feed",
        DML_SCENARIOS[4][1],
        [
            ("insert_only", merge_on(MERGE_PART_SRC, "WHEN NOT MATCHED THEN INSERT *"), {}),
            (
                "insert_no_match",
                merge_on(MERGE_PART_NONE_SRC, "WHEN MATCHED THEN DELETE WHEN NOT MATCHED THEN INSERT *"),
                {},
            ),
            (
                "upsert",
                merge_on(MERGE_PART_SRC, "WHEN MATCHED THEN UPDATE SET v = s.v WHEN NOT MATCHED THEN INSERT *"),
                {},
            ),
            ("delete_matched", merge_on(MERGE_SRC, "WHEN MATCHED THEN DELETE"), {}),
        ],
    ),
    (
        "append-only",
        DML_SCENARIOS[-1][1],
        [
            ("insert_only", merge_on(MERGE_SRC, "WHEN NOT MATCHED THEN INSERT *"), {}),
            ("update_matched", merge_on(MERGE_SRC, "WHEN MATCHED THEN UPDATE SET v = s.v"), {}),
            ("delete_matched", merge_on(MERGE_SRC, "WHEN MATCHED THEN DELETE"), {}),
            (
                "upsert_no_match",
                merge_on(MERGE_NONE_SRC, "WHEN MATCHED THEN UPDATE SET v = s.v WHEN NOT MATCHED THEN INSERT *"),
                {},
            ),
        ],
    ),
]


def cdc_files(path: str, from_version: int) -> int:
    """The ``cdc`` actions of the table's JSON commits from ``from_version`` on."""
    log = os.path.join(path, "_delta_log")
    n = 0
    for name in sorted(x for x in os.listdir(log) if x.endswith(".json") and x[0].isdigit()):
        if int(name.split(".")[0]) < from_version:
            continue
        with open(os.path.join(log, name)) as f:
            n += sum(1 for line in f if "cdc" in json.loads(line))
    return n


def merge_cases(spark: SparkSession, out: str) -> None:
    """``merge/``: as ``dml_cases`` for the ``MERGE`` steps of ``MERGE_SCENARIOS`` (``MERGE.json``),
    plus the file-backed sources the steps read (``source-parquet``, ``source-delta``; ``<src_parquet>``
    and ``<src_delta>`` in the SQL) and each step's ``cdc`` action count."""
    root = os.path.join(out, "merge")
    if os.path.exists(root):
        raise SystemExit(f"{root} exists: write the MERGE oracle into a fresh output directory")
    os.makedirs(root)
    src_parquet = os.path.join(root, "source-parquet")
    src_delta = os.path.join(root, "source-delta")
    spark.sql(MERGE_FILE_SOURCE).coalesce(1).write.parquet(src_parquet)
    spark.sql(MERGE_FILE_SOURCE).coalesce(1).write.format("delta").save(src_delta)
    record: dict = {
        "format_version": FORMAT_VERSION,
        "generator": "tests/scripts/gen_delta_fixtures.py merge",
        "spark": spark.version,
        "delta_spark": "4.4.1",
        "scenarios": {},
    }
    for name, setup, steps in MERGE_SCENARIOS:
        before = os.path.join(root, name, "before")
        for q in setup:
            spark.sql(q.format(t=f"delta.`{before}`")).collect()
        df = spark.read.format("delta").load(before)
        scenario: dict = {
            "setup": [q.replace("{t}", "<t>") for q in setup],
            "fields": [f.name for f in df.schema.fields],
            "rows": [[typed(r[i]) for i in range(len(r))] for r in df.orderBy("id").collect()],
            "versions": len(commits(before)),
            "steps": [],
        }
        for step, sql, confs in steps:
            work = os.path.join(out, "merge-work", name, step)
            shutil.copytree(before, work)
            text = (
                sql.format(t=f"delta.`{work}`").replace("<src_parquet>", src_parquet).replace("<src_delta>", src_delta)
            )
            entry: dict = {"step": step, "sql": sql.replace("{t}", "<t>"), "confs": confs}
            for k, v in confs.items():
                spark.conf.set(k, v)
            try:
                result = spark.sql(text)
                entry["output_fields"] = [f.name for f in result.schema.fields]
                entry["output"] = [[typed(r[i]) for i in range(len(r))] for r in result.collect()]
                entry["error"] = None
            except Exception as e:  # the oracle's error is the record
                entry["error"] = oracle_error(e, work)
            finally:
                for k in confs:
                    spark.conf.unset(k)
            log = commits(work)
            entry["commits"] = log[scenario["versions"] :]
            entry["cdc_files"] = cdc_files(work, scenario["versions"])
            back = spark.read.format("delta").load(work)
            entry["fields"] = [f.name for f in back.schema.fields]
            entry["rows"] = [[typed(r[i]) for i in range(len(r))] for r in back.orderBy("id").collect()]
            newest = spark.sql(f"DESCRIBE HISTORY delta.`{work}` LIMIT 1").collect()[0]
            entry["history"] = {
                "version": newest["version"],
                "operation": newest["operation"],
                "operationParameters": dict(newest["operationParameters"] or {}),
                "operationMetrics": dict(newest["operationMetrics"] or {}),
            }
            scenario["steps"].append(entry)
            shutil.rmtree(work)
        record["scenarios"][name] = scenario
    shutil.rmtree(os.path.join(out, "merge-work"), ignore_errors=True)
    with open(os.path.join(root, "MERGE.json"), "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.write("\n")


#: A deletion-vector table of three files: ids 1-3, 4-6 and 7-100, the third with a DV
#: (`id % 10 = 0` deleted) before the steps.
DV_TABLE = [
    f"CREATE TABLE {{t}} ({DML_COLUMNS}) USING delta TBLPROPERTIES ('delta.enableDeletionVectors' = 'true')",
    f"INSERT INTO {{t}} VALUES {DML_ROWS[0]}",
    f"INSERT INTO {{t}} VALUES {DML_ROWS[1]}",
    "INSERT INTO {t} SELECT id, CAST(id AS STRING), CAST(id % 7 AS INT) FROM range(7, 101)",
    "DELETE FROM {t} WHERE id % 10 = 0",
]
NO_MERGE_DVS = {"spark.databricks.delta.merge.deletionVectors.persistent": "false"}
DV_SCENARIOS: list[tuple[str, list[str], list[tuple[str, str, dict]]]] = [
    (
        "unpartitioned",
        DV_TABLE,
        [
            ("delete_rows", "DELETE FROM {t} WHERE id = 2", {}),
            ("delete_combines_existing_dv", "DELETE FROM {t} WHERE id IN (11, 12, 13)", {}),
            ("delete_whole_file", "DELETE FROM {t} WHERE id <= 3", {}),
            ("delete_whole_file_with_dv", "DELETE FROM {t} WHERE id >= 7", {}),
            ("delete_mixed", "DELETE FROM {t} WHERE id <= 4 OR id = 50", {}),
            ("delete_null_condition", "DELETE FROM {t} WHERE n > 3", {}),
            ("delete_nothing", "DELETE FROM {t} WHERE id = 999", {}),
            ("delete_all", "DELETE FROM {t}", {}),
            ("delete_rows_copy_on_write", "DELETE FROM {t} WHERE id IN (2, 11)", NO_DVS),
            ("update_rows", "UPDATE {t} SET v = 'u' WHERE id >= 3 AND id <= 12", {}),
            ("update_whole_file", "UPDATE {t} SET n = 0 WHERE id <= 3", {}),
            ("update_nothing", "UPDATE {t} SET n = 0 WHERE id = 999", {}),
            ("update_all", "UPDATE {t} SET v = 'w'", {}),
            ("update_rows_copy_on_write", "UPDATE {t} SET v = 'u' WHERE id IN (2, 11)", NO_DVS),
            ("merge_update", merge_on(MERGE_SRC, "WHEN MATCHED THEN UPDATE SET v = s.v"), {}),
            ("merge_delete", merge_on(MERGE_SRC, "WHEN MATCHED THEN DELETE"), {}),
            (
                "merge_upsert",
                merge_on(MERGE_SRC, "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *"),
                {},
            ),
            ("merge_insert_only", merge_on(MERGE_SRC, "WHEN NOT MATCHED THEN INSERT *"), {}),
            (
                "merge_nmbs_delete",
                merge_on(MERGE_SRC, "WHEN NOT MATCHED BY SOURCE AND t.id < 7 THEN DELETE"),
                {},
            ),
            (
                "merge_all_kinds",
                merge_on(
                    MERGE_SRC,
                    "WHEN MATCHED AND t.id = 2 THEN DELETE WHEN MATCHED THEN UPDATE SET n = s.n "
                    "WHEN NOT MATCHED THEN INSERT * WHEN NOT MATCHED BY SOURCE AND t.id = 11 THEN UPDATE SET n = -1",
                ),
                {},
            ),
            (
                "merge_whole_file",
                merge_on(
                    "(SELECT CAST(id AS BIGINT) AS id FROM VALUES (1), (2), (3) AS s(id)) s",
                    "WHEN MATCHED THEN UPDATE SET n = 9",
                ),
                {},
            ),
            (
                "merge_update_copy_on_write",
                merge_on(MERGE_SRC, "WHEN MATCHED THEN UPDATE SET v = s.v"),
                NO_MERGE_DVS,
            ),
        ],
    ),
    (
        "partitioned",
        [
            f"CREATE TABLE {{t}} ({DML_COLUMNS}, p STRING) USING delta PARTITIONED BY (p) "
            "TBLPROPERTIES ('delta.enableDeletionVectors' = 'true')"
        ]
        + [f"INSERT INTO {{t}} VALUES {r}" for r in DML_PART_ROWS],
        [
            ("delete_partition", "DELETE FROM {t} WHERE p = 'x'", {}),
            ("delete_rows", "DELETE FROM {t} WHERE n IS NULL", {}),
            ("delete_partition_and_rows", "DELETE FROM {t} WHERE p = 'x' AND id = 1", {}),
            ("update_partition_only", "UPDATE {t} SET v = 'all' WHERE p = 'y'", {}),
            ("update_rows", "UPDATE {t} SET n = n * 2 WHERE id > 1", {}),
            ("update_moves_partition", "UPDATE {t} SET p = 'w' WHERE id = 1", {}),
            (
                "merge_upsert",
                merge_on(MERGE_PART_SRC, "WHEN MATCHED THEN UPDATE SET v = s.v WHEN NOT MATCHED THEN INSERT *"),
                {},
            ),
        ],
    ),
    (
        "change-data-feed",
        [
            f"CREATE TABLE {{t}} ({DML_COLUMNS}, p STRING) USING delta PARTITIONED BY (p) "
            "TBLPROPERTIES ('delta.enableDeletionVectors' = 'true', 'delta.enableChangeDataFeed' = 'true')"
        ]
        + [f"INSERT INTO {{t}} VALUES {r}" for r in DML_PART_ROWS],
        [
            ("delete_rows", "DELETE FROM {t} WHERE id = 1", {}),
            ("delete_whole_file", "DELETE FROM {t} WHERE id = 3", {}),
            ("delete_rows_and_file", "DELETE FROM {t} WHERE id IN (1, 3)", {}),
            ("delete_partition", "DELETE FROM {t} WHERE p = 'x'", {}),
            ("update_rows", "UPDATE {t} SET v = 'u' WHERE id = 1", {}),
            ("merge_delete", merge_on(MERGE_PART_SRC, "WHEN MATCHED THEN DELETE"), {}),
        ],
    ),
]


#: A deletion-vector table whose files carry no statistics (``stats.collect`` off while they are
#: written): delta-spark computes a new vector's file statistics from the file
#: (``DMLWithDeletionVectorsHelper.getActionsWithStats``) and never removes such a file whole
#: (``TouchedFileWithDV.isFullyReplaced`` needs ``numRecords``).
DV_NO_STATS_SCENARIOS: list[tuple[str, list[str], list[tuple[str, str, dict]]]] = [
    (
        "no-stats",
        [
            f"CREATE TABLE {{t}} ({DML_COLUMNS}) USING delta TBLPROPERTIES ('delta.enableDeletionVectors' = 'true')",
            "SET spark.databricks.delta.stats.collect = false",
            f"INSERT INTO {{t}} VALUES {DML_ROWS[0]}",
            f"INSERT INTO {{t}} VALUES {DML_ROWS[1]}",
            "SET spark.databricks.delta.stats.collect = true",
        ],
        [
            ("delete_rows", "DELETE FROM {t} WHERE id = 2", {}),
            ("delete_every_row_of_a_file", "DELETE FROM {t} WHERE id <= 3", {}),
            ("update_rows", "UPDATE {t} SET v = 'u' WHERE id = 5", {}),
            ("merge_delete", merge_on(MERGE_SRC, "WHEN MATCHED THEN DELETE"), {}),
        ],
    ),
]


def dv_commits(path: str) -> list[dict]:
    """``commits`` with every add's and remove's deletion vector descriptor (``None`` without
    one) and ``dataChange``, and the commit's ``cdc`` action count."""
    log = os.path.join(path, "_delta_log")
    out = commits(path)
    for record in out:
        record["add_dvs"] = []
        record["remove_dvs"] = []
        record["cdc"] = 0
        with open(os.path.join(log, f"{record['version']:020}.json")) as f:
            for line in f:
                action = json.loads(line)
                if "add" in action:
                    a = action["add"]
                    record["add_dvs"].append(
                        {"path": a["path"], "deletionVector": a.get("deletionVector"), "dataChange": a["dataChange"]}
                    )
                elif "remove" in action:
                    r = action["remove"]
                    record["remove_dvs"].append(
                        {"path": r["path"], "deletionVector": r.get("deletionVector"), "dataChange": r["dataChange"]}
                    )
                elif "cdc" in action:
                    record["cdc"] += 1
    return out


def change_rows(spark: SparkSession, path: str, from_version: int) -> list[str]:
    """The change data feed of ``path`` from ``from_version`` on: each change as
    ``to_json(struct(*))`` without ``_commit_timestamp``, by version, id and change type."""
    from pyspark.sql.functions import col, struct, to_json

    df = (
        spark.read.format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", from_version)
        .load(path)
        .drop("_commit_timestamp")
    )
    return [
        r[0]
        for r in df.orderBy("_commit_version", "id", "_change_type")
        .select(to_json(struct(*[col(c) for c in df.columns])))
        .collect()
    ]


def dv_cases(
    spark: SparkSession,
    out: str,
    dirname: str = "dvs",
    record_name: str = "DVS.json",
    scenarios: list[tuple[str, list[str], list[tuple[str, str, dict]]]] | None = None,
) -> None:
    """``dvs/``: as ``merge_cases`` for the deletion-vector steps of ``DV_SCENARIOS``
    (``DVS.json``, P3-T021 PR-37d): every step's commits with their deletion vector descriptors
    (``dv_commits``) and, on the change-data-feed table, the change rows Spark derives
    (``change_rows``). ``dvs-no-stats`` writes ``DV_NO_STATS_SCENARIOS`` the same way into
    ``dvs-no-stats/NOSTATS.json``."""
    if scenarios is None:
        scenarios = DV_SCENARIOS
    root = os.path.join(out, dirname)
    os.makedirs(root, exist_ok=True)
    record: dict = {
        "format_version": FORMAT_VERSION,
        "generator": f"tests/scripts/gen_delta_fixtures.py {dirname}",
        "spark": spark.version,
        "delta_spark": "4.4.1",
        "scenarios": {},
    }
    for name, setup, steps in scenarios:
        before = os.path.join(root, name, "before")
        for q in setup:
            spark.sql(q.format(t=f"delta.`{before}`")).collect()
        df = spark.read.format("delta").load(before)
        scenario: dict = {
            "setup": [q.replace("{t}", "<t>") for q in setup],
            "fields": [f.name for f in df.schema.fields],
            "rows": [[typed(r[i]) for i in range(len(r))] for r in df.orderBy("id").collect()],
            "versions": len(commits(before)),
            "steps": [],
        }
        cdf = name == "change-data-feed"
        for step, sql, confs in steps:
            work = os.path.join(out, f"{dirname}-work", name, step)
            shutil.copytree(before, work)
            text = sql.format(t=f"delta.`{work}`")
            entry: dict = {"step": step, "sql": sql.replace("{t}", "<t>"), "confs": confs}
            for k, v in confs.items():
                spark.conf.set(k, v)
            try:
                result = spark.sql(text)
                entry["output_fields"] = [f.name for f in result.schema.fields]
                entry["output"] = [[typed(r[i]) for i in range(len(r))] for r in result.collect()]
                entry["error"] = None
            except Exception as e:  # the oracle's error is the record
                entry["error"] = oracle_error(e, work)
            finally:
                for k in confs:
                    spark.conf.unset(k)
            log = dv_commits(work)
            entry["commits"] = log[scenario["versions"] :]
            back = spark.read.format("delta").load(work)
            entry["rows"] = [[typed(r[i]) for i in range(len(r))] for r in back.orderBy("id").collect()]
            if cdf and entry["commits"]:
                entry["changes"] = change_rows(spark, work, scenario["versions"])
            newest = spark.sql(f"DESCRIBE HISTORY delta.`{work}` LIMIT 1").collect()[0]
            entry["history"] = {
                "version": newest["version"],
                "operation": newest["operation"],
                "operationParameters": dict(newest["operationParameters"] or {}),
                "operationMetrics": dict(newest["operationMetrics"] or {}),
            }
            scenario["steps"].append(entry)
            shutil.rmtree(work)
        record["scenarios"][name] = scenario
    shutil.rmtree(os.path.join(out, f"{dirname}-work"), ignore_errors=True)
    with open(os.path.join(root, record_name), "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.write("\n")


def readback(spark: SparkSession, out: str, src: str) -> None:
    """Spark reads the Krishiv-written tables under ``src`` (``krishiv-written/``, made by the
    connector's ``write_readback_tables``): per table its rows as ``to_json(struct(*))`` by
    ``id``, ``DESCRIBE HISTORY`` (version and operation), the protocol, a data-skipping count
    (``id >= 2``) and an append of its own, made on a copy under ``out``. Writes
    ``out/READBACK.json``."""
    from pyspark.sql.functions import col, struct, to_json

    tables: dict = {}
    for name in sorted(os.listdir(src)):
        path = os.path.join(src, name)
        if not os.path.isdir(os.path.join(path, "_delta_log")):
            continue
        record: dict = {"error": None}
        try:
            df = spark.read.format("delta").load(path)
            record["rows"] = [
                r[0] for r in df.orderBy("id").select(to_json(struct(*[col(c) for c in df.columns]))).collect()
            ]
            record["count_id_ge_2"] = df.where("id >= 2").count()
            history = spark.sql(f"DESCRIBE HISTORY delta.`{path}`").orderBy("version").collect()
            record["history"] = [{"version": h["version"], "operation": h["operation"]} for h in history]
            record["protocol"] = protocol(path)
            props = spark.sql(f"SHOW TBLPROPERTIES delta.`{path}`").collect()
            if any(r["key"] == "delta.enableChangeDataFeed" and r["value"] == "true" for r in props):
                # The change rows Spark derives from the newest (Krishiv's) commit.
                record["changes"] = change_rows(spark, path, len(commits(path)) - 1)
        except Exception as e:  # the oracle's error is the record
            record["error"] = oracle_error(e, path)
            tables[name] = record
            continue
        copy = os.path.join(out, name)
        shutil.copytree(path, copy)
        try:
            df.orderBy("id").limit(1).write.format("delta").mode("append").save(copy)
            record["spark_append"] = {
                "error": None,
                "rows": spark.read.format("delta").load(copy).count(),
                "protocol": protocol(copy),
            }
        except Exception as e:
            record["spark_append"] = {"error": oracle_error(e, copy)}
        tables[name] = record
    with open(os.path.join(out, "READBACK.json"), "w") as f:
        json.dump(
            {
                "format_version": FORMAT_VERSION,
                "generator": "tests/scripts/gen_delta_fixtures.py readback",
                "spark": spark.version,
                "delta_spark": "4.4.1",
                "tables": tables,
            },
            f,
            indent=1,
            sort_keys=True,
        )
        f.write("\n")


def log_actions(path: str) -> list[dict]:
    """Every JSON commit of the table's log as written, in version order: each action, with the
    values that differ per run dropped (``commitInfo`` ``timestamp``, ``txnId``, ``engineInfo``,
    ``inCommitTimestamp``; ``metaData`` ``id`` and ``createdTime``; ``add`` ``modificationTime``
    and ``size``)."""
    log = os.path.join(path, "_delta_log")
    out = []
    for name in sorted(n for n in os.listdir(log) if n.endswith(".json") and n[0].isdigit()):
        actions = []
        with open(os.path.join(log, name)) as f:
            for line in f:
                action = json.loads(line)
                for kind, drop in (
                    ("commitInfo", ("timestamp", "txnId", "engineInfo", "inCommitTimestamp")),
                    ("metaData", ("id", "createdTime")),
                    ("add", ("modificationTime", "size")),
                ):
                    if kind in action:
                        for k in drop:
                            action[kind].pop(k, None)
                actions.append(action)
        out.append({"version": int(name.split(".")[0]), "actions": actions})
    return out


def create_by_path(spark: SparkSession, out: str) -> None:
    """``CREATE-PATH.json``: ``CREATE TABLE delta.`<path>``` (row 74). Each case runs on a path of
    its own (``<path>`` in the record) and records the oracle's error, or the log's commits as
    written (``log_actions``) and what ``SELECT *`` reads (columns and rows). The cases: plain,
    partitioned, ``NOT NULL`` and comments, ``TBLPROPERTIES``, every Spark type family, a second
    ``CREATE`` over the table (same, other, no columns; ``IF NOT EXISTS``; other partitioning),
    an empty directory, a Parquet directory, a trailing slash, a ``file:`` URI, a ``LOCATION``
    beside the path, ``OR REPLACE``, CTAS, no columns, a relative path and a name without ``/``.
    No catalog entry is made: ``tables_after`` lists ``default`` at the end."""
    root = os.path.join(out, "create-path")
    os.makedirs(root)
    n = [0]

    def fresh() -> str:
        n[0] += 1
        return os.path.join(root, f"t{n[0]}")

    cases: list[dict] = []

    def case(name: str, sql_of, path: str, setup=None) -> None:
        if setup is not None:
            setup(path)
        sql = sql_of(path)
        record: dict = {"case": name, "sql": sql.replace(path, "<path>")}
        before = len(log_actions(path)) if os.path.isdir(os.path.join(path, "_delta_log")) else 0
        try:
            spark.sql(sql).collect()
            record["error"] = None
        except Exception as e:  # the oracle's error is the record
            record["error"] = oracle_error(e, path)
        record["created_log"] = os.path.isdir(os.path.join(path, "_delta_log"))
        if record["created_log"]:
            record["commits"] = log_actions(path)[before:]
            try:
                df = spark.sql(f"SELECT * FROM delta.`{path}`")
                record["columns"] = [[f.name, f.dataType.simpleString(), f.nullable] for f in df.schema.fields]
                record["rows"] = [[typed(r[i]) for i in range(len(r))] for r in df.collect()]
            except Exception as e:  # a table the oracle cannot read back
                record["read_error"] = oracle_error(e, path)
        cases.append(record)

    def t(p: str) -> str:
        return f"delta.`{p}`"

    plain = fresh()
    case("plain", lambda p: f"CREATE TABLE {t(p)} (id BIGINT, v STRING) USING delta", plain)
    case(
        "partitioned",
        lambda p: f"CREATE TABLE {t(p)} (id BIGINT, v STRING, p STRING, n INT) USING delta PARTITIONED BY (p, n)",
        fresh(),
    )
    case(
        "partition_first",
        lambda p: f"CREATE TABLE {t(p)} (p STRING, id BIGINT) USING delta PARTITIONED BY (p)",
        fresh(),
    )
    case(
        "not_null_comments",
        lambda p: f"CREATE TABLE {t(p)} (id BIGINT NOT NULL COMMENT 'the key', v STRING) USING delta COMMENT 'a table'",
        fresh(),
    )
    case(
        "tblproperties",
        lambda p: (
            f"CREATE TABLE {t(p)} (id BIGINT) USING delta "
            "TBLPROPERTIES ('delta.appendOnly' = 'true', 'krishiv.k' = 'v')"
        ),
        fresh(),
    )
    case(
        "types",
        lambda p: (
            f"CREATE TABLE {t(p)} (b TINYINT, s SMALLINT, i INT, l BIGINT, f FLOAT, d DOUBLE, "
            "dec DECIMAL(10, 2), str STRING, c CHAR(3), vc VARCHAR(5), bool BOOLEAN, bin BINARY, dt DATE, "
            "ts TIMESTAMP, ntz TIMESTAMP_NTZ, arr ARRAY<INT>, m MAP<STRING, INT>, st STRUCT<x: INT, y: STRING>) "
            "USING delta"
        ),
        fresh(),
    )
    case("again_same", lambda p: f"CREATE TABLE {t(p)} (id BIGINT, v STRING) USING delta", plain)
    case("again_other", lambda p: f"CREATE TABLE {t(p)} (id INT) USING delta", plain)
    case("again_no_columns", lambda p: f"CREATE TABLE {t(p)} USING delta", plain)
    case("if_not_exists_same", lambda p: f"CREATE TABLE IF NOT EXISTS {t(p)} (id BIGINT, v STRING) USING delta", plain)
    case("if_not_exists_other", lambda p: f"CREATE TABLE IF NOT EXISTS {t(p)} (id INT) USING delta", plain)
    case(
        "again_partitioned",
        lambda p: f"CREATE TABLE {t(p)} (id BIGINT, v STRING) USING delta PARTITIONED BY (v)",
        plain,
    )
    case("empty_directory", lambda p: f"CREATE TABLE {t(p)} (id BIGINT) USING delta", fresh(), os.makedirs)
    case(
        "parquet_directory",
        lambda p: f"CREATE TABLE {t(p)} (id BIGINT) USING delta",
        fresh(),
        lambda p: spark.sql("SELECT CAST(1 AS BIGINT) AS id").write.parquet(p),
    )
    case("trailing_slash", lambda p: f"CREATE TABLE delta.`{p}/` (id BIGINT) USING delta", fresh())
    case("file_uri", lambda p: f"CREATE TABLE delta.`file:{p}` (id BIGINT) USING delta", fresh())
    case(
        "location_too",
        lambda p: f"CREATE TABLE {t(p)} (id BIGINT) USING delta LOCATION '{p}'",
        fresh(),
    )
    case("or_replace_new", lambda p: f"CREATE OR REPLACE TABLE {t(p)} (id BIGINT) USING delta", fresh())
    case("or_replace_existing", lambda p: f"CREATE OR REPLACE TABLE {t(p)} (id INT, w STRING) USING delta", plain)
    case("ctas", lambda p: f"CREATE TABLE {t(p)} USING delta AS SELECT CAST(1 AS BIGINT) AS id, 'a' AS v", fresh())
    case("no_columns", lambda p: f"CREATE TABLE {t(p)} USING delta", fresh())
    case("using_parquet", lambda p: f"CREATE TABLE {t(p)} (id BIGINT) USING parquet", fresh())
    case("no_using", lambda p: f"CREATE TABLE {t(p)} (id BIGINT)", fresh())
    # The relative cases resolve against the JVM's working directory: the run owns these names
    # there, so it stops rather than write into (and later remove) a directory it did not make.
    for owned in ("krishiv_rel", "krishiv_no_slash"):
        if os.path.exists(owned):
            raise SystemExit(f"{owned} already exists in the working directory; move it first")
    case("relative_path", lambda p: "CREATE TABLE delta.`krishiv_rel/dir` (id BIGINT) USING delta", "krishiv_rel/dir")
    case("no_slash", lambda p: "CREATE TABLE delta.`krishiv_no_slash` (id BIGINT) USING delta", "krishiv_no_slash")
    record: dict = {
        "format_version": FORMAT_VERSION,
        "generator": "tests/scripts/gen_delta_fixtures.py",
        "spark": spark.version,
        "delta_spark": "4.4.1",
        "cases": cases,
        "tables_after": sorted(r.tableName for r in spark.sql("SHOW TABLES IN default").collect()),
    }
    shutil.rmtree(root, ignore_errors=True)
    for owned in ("krishiv_rel", "krishiv_no_slash"):
        shutil.rmtree(owned, ignore_errors=True)
    with open(os.path.join(out, "CREATE-PATH.json"), "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.write("\n")


# Delta write options (PR-37e): (case, options, save mode, partitioned). Each runs on a fresh
# two-row table written with the defaults, appending two rows with the options.
OPTION_CASES: list[tuple[str, dict[str, str], str, bool]] = [
    ("merge_schema_bad", {"mergeSchema": "maybe"}, "append", False),
    ("merge_schema_true_same", {"mergeSchema": "true"}, "append", False),
    ("overwrite_schema_bad", {"overwriteSchema": "maybe"}, "overwrite", False),
    ("data_change_bad", {"dataChange": "maybe"}, "append", False),
    ("data_change_false_append", {"dataChange": "false"}, "append", False),
    ("optimize_write_bad", {"optimizeWrite": "maybe"}, "append", False),
    ("optimize_write_true", {"optimizeWrite": "true"}, "append", False),
    ("txn_version_only", {"txnVersion": "1"}, "append", False),
    ("txn_app_only", {"txnAppId": "a"}, "append", False),
    ("txn_version_negative", {"txnVersion": "-1", "txnAppId": "a"}, "append", False),
    ("txn_both", {"txnVersion": "1", "txnAppId": "a"}, "append", False),
    ("pom_bad", {"partitionOverwriteMode": "bogus"}, "overwrite", True),
    ("pom_bad_append", {"partitionOverwriteMode": "bogus"}, "append", True),
    ("pom_bad_unpartitioned", {"partitionOverwriteMode": "bogus"}, "overwrite", False),
    ("pom_bad_append_unpartitioned", {"partitionOverwriteMode": "bogus"}, "append", False),
    ("unknown_option", {"someUnknownOption": "x"}, "append", False),
    ("user_metadata", {"userMetadata": "hello"}, "append", False),
    ("user_metadata_overwrite", {"userMetadata": "hello"}, "overwrite", False),
    ("replace_using", {"replaceUsing": "id"}, "overwrite", False),
    ("write_partition_columns", {"writePartitionColumns": "true"}, "append", True),
    ("max_records_bad", {"maxRecordsPerFile": "abc"}, "append", False),
]

# Statement-level `WITH (...)` options: INSERT's (Spark 4.2.0 parses them), DELETE's and
# UPDATE's (Spark 4.3's syntax). `<pt>` is a table partitioned by `p`, `<t>` one that is not.
OPTION_STATEMENTS: list[tuple[str, str]] = [
    ("insert_with_user_metadata", "INSERT INTO <t> WITH ('userMetadata' = 'm1') VALUES (5, 1)"),
    ("insert_with_unknown", "INSERT INTO <t> WITH ('someUnknownOption' = 'x') VALUES (5, 1)"),
    ("insert_with_merge_schema_bad", "INSERT INTO <t> WITH ('mergeSchema' = 'maybe') VALUES (5, 1)"),
    ("insert_with_txn_version_only", "INSERT INTO <t> WITH ('txnVersion' = '1') VALUES (5, 1)"),
    (
        "insert_overwrite_with_pom_bad",
        "INSERT OVERWRITE <t> WITH ('partitionOverwriteMode' = 'bogus') VALUES (5, 1)",
    ),
    (
        "insert_overwrite_partitioned_with_pom_bad",
        "INSERT OVERWRITE <pt> WITH ('partitionOverwriteMode' = 'bogus') VALUES (5, 1)",
    ),
    (
        "insert_partitioned_with_pom_bad",
        "INSERT INTO <pt> WITH ('partitionOverwriteMode' = 'bogus') VALUES (5, 1)",
    ),
    (
        "insert_overwrite_partitioned_with_pom_dynamic",
        "INSERT OVERWRITE <pt> WITH ('partitionOverwriteMode' = 'dynamic') VALUES (5, 1)",
    ),
    ("delete_with_options", "DELETE FROM <t> WITH ('userMetadata' = 'x') WHERE id = 0"),
    ("update_with_options", "UPDATE <t> WITH ('userMetadata' = 'x') SET id = 3 WHERE id = 0"),
]


# `mergeSchema` against the table's types (`SchemaMergingUtils.mergeDataTypes`): (case, table,
# the appended row's SELECT list, options). `plain` is `(id BIGINT, n INT, d DECIMAL(10,2),
# s STRUCT<a: INT>)` with one row; `widening` is `(id BIGINT, i INT)` with
# `delta.enableTypeWidening`, as `write-features/type-widening`.
MERGE_SCHEMA_BASE = (
    "CAST(5 AS BIGINT) AS id, CAST(5 AS INT) AS n, CAST(1.5 AS DECIMAL(10,2)) AS d, "
    "named_struct('a', CAST(1 AS INT)) AS s"
)
MERGE_SCHEMA_CASES: list[tuple[str, str, str, dict[str, str]]] = [
    ("same_types", "plain", MERGE_SCHEMA_BASE, {"mergeSchema": "true"}),
    (
        "bigint_into_int",
        "plain",
        MERGE_SCHEMA_BASE.replace("CAST(5 AS INT) AS n", "CAST(5 AS BIGINT) AS n"),
        {"mergeSchema": "true"},
    ),
    (
        "bigint_into_int_no_merge",
        "plain",
        MERGE_SCHEMA_BASE.replace("CAST(5 AS INT) AS n", "CAST(5 AS BIGINT) AS n"),
        {},
    ),
    (
        "smallint_into_int",
        "plain",
        MERGE_SCHEMA_BASE.replace("CAST(5 AS INT) AS n", "CAST(5 AS SMALLINT) AS n"),
        {"mergeSchema": "true"},
    ),
    (
        "tinyint_into_int",
        "plain",
        MERGE_SCHEMA_BASE.replace("CAST(5 AS INT) AS n", "CAST(5 AS TINYINT) AS n"),
        {"mergeSchema": "true"},
    ),
    ("null_into_int", "plain", MERGE_SCHEMA_BASE.replace("CAST(5 AS INT) AS n", "NULL AS n"), {"mergeSchema": "true"}),
    ("string_into_int", "plain", MERGE_SCHEMA_BASE.replace("CAST(5 AS INT) AS n", "'5' AS n"), {"mergeSchema": "true"}),
    (
        "int_into_bigint",
        "plain",
        MERGE_SCHEMA_BASE.replace("CAST(5 AS BIGINT) AS id", "CAST(5 AS INT) AS id"),
        {"mergeSchema": "true"},
    ),
    (
        "decimal_precision",
        "plain",
        MERGE_SCHEMA_BASE.replace("DECIMAL(10,2)", "DECIMAL(12,2)"),
        {"mergeSchema": "true"},
    ),
    (
        "nested_bigint_into_int",
        "plain",
        MERGE_SCHEMA_BASE.replace("named_struct('a', CAST(1 AS INT))", "named_struct('a', CAST(1 AS BIGINT))"),
        {"mergeSchema": "true"},
    ),
    (
        "widening_bigint_into_int",
        "widening",
        "CAST(5 AS BIGINT) AS id, CAST(5 AS BIGINT) AS i",
        {"mergeSchema": "true"},
    ),
    ("widening_string_into_int", "widening", "CAST(5 AS BIGINT) AS id, '5' AS i", {"mergeSchema": "true"}),
    (
        "widening_short_into_int",
        "widening",
        "CAST(5 AS BIGINT) AS id, CAST(5 AS SMALLINT) AS i",
        {"mergeSchema": "true"},
    ),
]


# More cases (review 2 of PR-37e): (case, table, SELECT list, options, save mode, writer), the
# writer `save` (a DataFrame `save(path)`) or `saveAsTable` (an append to the catalog table at
# the path). `short` is `(id BIGINT, n SMALLINT)`, `varchar` is `(id BIGINT, v VARCHAR(5))`.
MERGE_SCHEMA_MORE: list[tuple[str, str, str, dict[str, str], str, str]] = [
    (
        "bigint_into_int_overwrite_no_merge",
        "plain",
        MERGE_SCHEMA_BASE.replace("CAST(5 AS INT) AS n", "CAST(5 AS BIGINT) AS n"),
        {},
        "overwrite",
        "save",
    ),
    ("int_into_short_no_merge", "short", "CAST(5 AS BIGINT) AS id, CAST(5 AS INT) AS n", {}, "append", "save"),
    (
        "int_into_short_merge",
        "short",
        "CAST(5 AS BIGINT) AS id, CAST(5 AS INT) AS n",
        {"mergeSchema": "true"},
        "append",
        "save",
    ),
    (
        "widening_bigint_into_int_no_merge",
        "widening",
        "CAST(5 AS BIGINT) AS id, CAST(5 AS BIGINT) AS i",
        {},
        "append",
        "save",
    ),
    (
        "bigint_into_int_save_as_table_append",
        "plain",
        MERGE_SCHEMA_BASE.replace("CAST(5 AS INT) AS n", "CAST(5 AS BIGINT) AS n"),
        {},
        "append",
        "saveAsTable",
    ),
    ("string_into_varchar", "varchar", "CAST(5 AS BIGINT) AS id, 'ab' AS v", {}, "append", "save"),
]


def merge_schema_cases(spark: SparkSession, base: str) -> dict:
    """Each case of ``MERGE_SCHEMA_CASES`` and ``MERGE_SCHEMA_MORE`` on a fresh table: the error,
    or the commit's parameters, the table's column types after it, and its rows."""
    out: dict = {}
    cases = [(n, k, sel, o, "append", "save") for n, k, sel, o in MERGE_SCHEMA_CASES] + MERGE_SCHEMA_MORE
    for name, kind, select, options, mode, writer in cases:
        path = os.path.join(base, "merge-schema-" + name)
        shutil.rmtree(path, ignore_errors=True)
        if kind == "short":
            spark.sql(f"CREATE TABLE delta.`{path}` (id BIGINT, n SMALLINT) USING delta")
            spark.sql(f"INSERT INTO delta.`{path}` VALUES (1, 1)")
        elif kind == "varchar":
            spark.sql(f"CREATE TABLE delta.`{path}` (id BIGINT, v VARCHAR(5)) USING delta")
            spark.sql(f"INSERT INTO delta.`{path}` VALUES (1, 'a')")
        elif kind == "plain":
            spark.sql(f"CREATE TABLE delta.`{path}` (id BIGINT, n INT, d DECIMAL(10,2), s STRUCT<a: INT>) USING delta")
            spark.sql(f"INSERT INTO delta.`{path}` VALUES (1, 1, 1.00, named_struct('a', 1))")
        else:
            spark.sql(
                f"CREATE TABLE delta.`{path}` (id BIGINT, i INT) USING delta "
                "TBLPROPERTIES ('delta.enableTypeWidening' = 'true')"
            )
            spark.sql(f"INSERT INTO delta.`{path}` VALUES (1, 1)")
        w = spark.sql(f"SELECT {select}").write.format("delta").mode(mode)
        for k, v in options.items():
            w = w.option(k, v)
        case: dict = {"table": kind, "select": select, "options": options, "mode": mode, "writer": writer}
        try:
            if writer == "saveAsTable":
                table = "merge_schema_" + name
                spark.sql(f"DROP TABLE IF EXISTS {table}")
                spark.sql(f"CREATE TABLE {table} USING delta LOCATION '{path}'")
                w.saveAsTable(table)
            else:
                w.save(path)
            case["error"] = None
            info = commits(path)[-1]["commitInfo"]
            case["commit"] = {
                "operation": info.get("operation"),
                "operationParameters": info.get("operationParameters"),
            }
            case["types"] = {f.name: f.dataType.simpleString() for f in spark.read.format("delta").load(path).schema}
            case["rows"] = [
                [typed(v) for v in r] for r in spark.read.format("delta").load(path).orderBy("id").collect()
            ]
        except Exception as e:  # the oracle records every failure
            case["error"] = oracle_error(e, path)
        case["versions"] = len(commits(path))
        out[name] = case
    return out


def write_options(spark: SparkSession, out: str) -> None:
    """``OPTIONS.json``: what delta-spark does with each write option of ``OPTION_CASES`` (the
    error, or the commit's operation, parameters and ``userMetadata``), and Spark 4.2.0's parse
    of the statement-level ``WITH (...)`` options."""
    record: dict = {
        "format_version": FORMAT_VERSION,
        "spark": spark.version,
        "delta_spark": "4.4.1",
        "generator": "gen_delta_fixtures.py options",
        "cases": {},
        "statements": {},
    }
    base = "/tmp/krishiv-options"

    def fresh(name: str, partitioned: bool) -> str:
        path = os.path.join(base, name)
        shutil.rmtree(path, ignore_errors=True)
        w = spark.range(2).selectExpr("id", "CAST(id % 2 AS INT) AS p").write.format("delta")
        (w.partitionBy("p") if partitioned else w).save(path)
        return path

    def last_commit_info(path: str) -> dict:
        info = commits(path)[-1]["commitInfo"]
        return {
            "operation": info.get("operation"),
            "operationParameters": info.get("operationParameters"),
            "userMetadata": info.get("userMetadata"),
        }

    for name, options, mode, partitioned in OPTION_CASES:
        path = fresh(name, partitioned)
        w = spark.range(10, 12).selectExpr("id", "CAST(id % 2 AS INT) AS p").write.format("delta").mode(mode)
        for k, v in options.items():
            w = w.option(k, v)
        case: dict = {"options": options, "mode": mode, "partitioned": partitioned}
        try:
            w.save(path)
            case["error"] = None
            case["commit"] = last_commit_info(path)
        except Exception as e:  # the oracle records every failure
            case["error"] = oracle_error(e, path)
        case["versions"] = len(commits(path))
        record["cases"][name] = case
    for name, sql in OPTION_STATEMENTS:
        # `<pt>`: a table partitioned by `p`.
        path = fresh(name, "<pt>" in sql)
        try:
            spark.sql(sql.replace("<t>", f"delta.`{path}`").replace("<pt>", f"delta.`{path}`"))
            record["statements"][name] = {"sql": sql, "error": None, "commit": last_commit_info(path)}
        except Exception as e:  # the oracle records every failure
            record["statements"][name] = {"sql": sql, "error": oracle_error(e, path)}
        record["statements"][name]["versions"] = len(commits(path))
    record["merge_schema"] = merge_schema_cases(spark, base)
    with open(os.path.join(out, "OPTIONS.json"), "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.write("\n")


CATALOG_WRITERS = ["insert_into", "insert_into_overwrite", "save_as_table_append", "write_to_append"]

# (name, writer options, session configuration set around the write)
CATALOG_OPTION_CASES: list[tuple[str, dict[str, str], dict[str, str]]] = [
    ("user_metadata", {"userMetadata": "hello"}, {}),
    ("merge_schema_true", {"mergeSchema": "true"}, {}),
    ("merge_schema_maybe", {"mergeSchema": "maybe"}, {}),
    ("data_change_false", {"dataChange": "false"}, {}),
    ("txn_version_only", {"txnVersion": "1"}, {}),
    ("txn_pair", {"txnAppId": "app", "txnVersion": "1"}, {}),
    ("conf_user_metadata", {}, {"spark.databricks.delta.commitInfo.userMetadata": "from-conf"}),
    (
        "option_and_conf_user_metadata",
        {"userMetadata": "from-option"},
        {"spark.databricks.delta.commitInfo.userMetadata": "from-conf"},
    ),
]


def has_txn(path: str, version: int) -> bool:
    """Whether the table's commit ``version`` carries a ``txn`` action (idempotent writes)."""
    with open(os.path.join(path, "_delta_log", f"{version:020}.json")) as f:
        return any("txn" in json.loads(line) for line in f)


def catalog_write_options(spark: SparkSession, out: str) -> None:
    """``CATALOG-OPTIONS.json``: for each DataFrame writer of an existing Delta catalog table
    (``CATALOG_WRITERS``) and each case of ``CATALOG_OPTION_CASES``, the error, or the commit's
    operation, parameters, ``userMetadata`` and whether it carries a ``txn`` action. The table is
    ``(id BIGINT, p INT)``, unpartitioned, with one row; the write adds ``(2, 2)``."""
    record: dict = {
        "format_version": FORMAT_VERSION,
        "spark": spark.version,
        "delta_spark": "4.4.1",
        "generator": "gen_delta_fixtures.py catalog-options",
        "cases": {},
    }
    base = "/tmp/krishiv-catalog-options"
    for writer in CATALOG_WRITERS:
        for name, options, conf in CATALOG_OPTION_CASES:
            key = f"{writer}.{name}"
            path = os.path.join(base, key)
            shutil.rmtree(path, ignore_errors=True)
            spark.sql("DROP TABLE IF EXISTS cot")
            spark.sql(f"CREATE TABLE cot (id BIGINT, p INT) USING delta LOCATION '{path}'")
            spark.sql("INSERT INTO cot VALUES (1, 1)")
            df = spark.sql("SELECT CAST(2 AS BIGINT) AS id, 2 AS p")
            case: dict = {"writer": writer, "options": options, "conf": conf}
            for k, v in conf.items():
                spark.conf.set(k, v)
            try:
                if writer.startswith("insert_into"):
                    w = df.write.mode("overwrite" if writer == "insert_into_overwrite" else "append")
                    for k, v in options.items():
                        w = w.option(k, v)
                    w.insertInto("cot")
                elif writer == "save_as_table_append":
                    w = df.write.format("delta").mode("append")
                    for k, v in options.items():
                        w = w.option(k, v)
                    w.saveAsTable("cot")
                else:
                    w2 = df.writeTo("cot")
                    for k, v in options.items():
                        w2 = w2.option(k, v)
                    w2.append()
                case["error"] = None
                last = commits(path)[-1]
                info = last["commitInfo"]
                case["commit"] = {
                    "operation": info.get("operation"),
                    "operationParameters": info.get("operationParameters"),
                    "userMetadata": info.get("userMetadata"),
                    "txn": has_txn(path, last["version"]),
                }
            except Exception as e:  # the oracle records every failure
                case["error"] = oracle_error(e, path)
            finally:
                for k in conf:
                    spark.conf.unset(k)
            case["versions"] = len(commits(path))
            record["cases"][key] = case
    spark.sql("DROP TABLE IF EXISTS cot")
    shutil.rmtree(base, ignore_errors=True)
    with open(os.path.join(out, "CATALOG-OPTIONS.json"), "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.write("\n")


#: ``DML-USER-METADATA.json`` statements: (name, SQL on ``dut`` with ``s`` the one-row source).
DML_USER_METADATA_STEPS: list[tuple[str, str]] = [
    ("delete_where", "DELETE FROM dut WHERE id = 1"),
    ("delete_all", "DELETE FROM dut"),
    ("delete_partition", "DELETE FROM dut WHERE p = 1"),
    ("update", "UPDATE dut SET p = 9 WHERE id = 2"),
    ("merge_update", "MERGE INTO dut t USING s ON t.id = s.id WHEN MATCHED THEN UPDATE SET p = s.p"),
    (
        "merge_insert_only",
        "MERGE INTO dut t USING (SELECT CAST(7 AS BIGINT) AS id, 7 AS p) s ON t.id = s.id "
        "WHEN NOT MATCHED THEN INSERT *",
    ),
    (
        "merge_all",
        "MERGE INTO dut t USING s ON t.id = s.id WHEN MATCHED THEN DELETE WHEN NOT MATCHED THEN INSERT *",
    ),
]

#: ``DML-USER-METADATA.json`` session confs: unset, and set.
DML_USER_METADATA_CONFS: list[tuple[str, dict[str, str]]] = [
    ("no_conf", {}),
    ("conf", {"spark.databricks.delta.commitInfo.userMetadata": "from-conf"}),
]


def dml_user_metadata(spark: SparkSession, out: str) -> None:
    """``DML-USER-METADATA.json``: the ``commitInfo.userMetadata`` of each ``DELETE``, ``UPDATE``
    and ``MERGE`` (``DML_USER_METADATA_STEPS``) with and without
    ``spark.databricks.delta.commitInfo.userMetadata`` set (``DML_USER_METADATA_CONFS``), on a
    table partitioned by ``p`` holding ``(1, 1), (2, 2), (3, 3)`` (``dv``: with
    ``delta.enableDeletionVectors``; the source ``s`` is ``(2, 5)``). Also ``insert_into_extra``:
    ``insertInto`` of a frame with one column more than the table, with and without
    ``mergeSchema``; ``create_empty``: ``CREATE TABLE`` without a query, under each conf."""
    record: dict = {
        "format_version": FORMAT_VERSION,
        "spark": spark.version,
        "delta_spark": "4.4.1",
        "generator": "gen_delta_fixtures.py dml-user-metadata",
        "cases": {},
    }
    base = "/tmp/krishiv-dml-user-metadata"
    spark.sql("SELECT CAST(2 AS BIGINT) AS id, 5 AS p").createOrReplaceTempView("s")
    for kind, props in [("plain", ""), ("dv", " TBLPROPERTIES ('delta.enableDeletionVectors' = 'true')")]:
        for step, sql in DML_USER_METADATA_STEPS:
            for conf_name, conf in DML_USER_METADATA_CONFS:
                key = f"{kind}.{step}.{conf_name}"
                path = os.path.join(base, key)
                shutil.rmtree(path, ignore_errors=True)
                spark.sql("DROP TABLE IF EXISTS dut")
                spark.sql(
                    f"CREATE TABLE dut (id BIGINT, p INT) USING delta PARTITIONED BY (p) LOCATION '{path}'{props}"
                )
                spark.sql("INSERT INTO dut VALUES (1, 1), (2, 2), (3, 3)")
                case: dict = {"sql": sql, "conf": conf, "deletion_vectors": kind == "dv"}
                for k, v in conf.items():
                    spark.conf.set(k, v)
                try:
                    spark.sql(sql)
                    case["error"] = None
                    info = commits(path)[-1]["commitInfo"]
                    case["commit"] = {
                        "operation": info.get("operation"),
                        "userMetadata": info.get("userMetadata"),
                    }
                except Exception as e:  # the oracle records every failure
                    case["error"] = oracle_error(e, path)
                finally:
                    for k in conf:
                        spark.conf.unset(k)
                case["versions"] = len(commits(path))
                record["cases"][key] = case
    for conf_name, conf in DML_USER_METADATA_CONFS:
        key = f"create_empty.{conf_name}"
        path = os.path.join(base, key)
        shutil.rmtree(path, ignore_errors=True)
        spark.sql("DROP TABLE IF EXISTS dut")
        for k, v in conf.items():
            spark.conf.set(k, v)
        try:
            spark.sql(f"CREATE TABLE dut (id BIGINT, p INT) USING delta LOCATION '{path}'")
        finally:
            for k in conf:
                spark.conf.unset(k)
        info = commits(path)[-1]["commitInfo"]
        record["cases"][key] = {
            "sql": "CREATE TABLE dut (id BIGINT, p INT) USING delta LOCATION <path>",
            "conf": conf,
            "error": None,
            "commit": {"operation": info.get("operation"), "userMetadata": info.get("userMetadata")},
            "versions": len(commits(path)),
        }
    for name, options in [("insert_into_extra", {}), ("insert_into_extra_merge_schema", {"mergeSchema": "true"})]:
        path = os.path.join(base, name)
        shutil.rmtree(path, ignore_errors=True)
        spark.sql("DROP TABLE IF EXISTS dut")
        spark.sql(f"CREATE TABLE dut (id BIGINT, p INT) USING delta LOCATION '{path}'")
        spark.sql("INSERT INTO dut VALUES (1, 1)")
        case = {"options": options}
        w = spark.sql("SELECT CAST(2 AS BIGINT) AS id, 2 AS p, 'x' AS extra").write.mode("append")
        for k, v in options.items():
            w = w.option(k, v)
        try:
            w.insertInto("dut")
            case["error"] = None
        except Exception as e:  # the oracle records every failure
            case["error"] = oracle_error(e, path)
        case["versions"] = len(commits(path))
        case["schema"] = [f.name for f in spark.table("dut").schema.fields]
        record["cases"][name] = case
    spark.sql("DROP TABLE IF EXISTS dut")
    shutil.rmtree(base, ignore_errors=True)
    with open(os.path.join(out, "DML-USER-METADATA.json"), "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.write("\n")


def main(out: str, which: str = "checkpoint", src: str = "") -> None:
    spark = session()
    if which == "dml-user-metadata":
        dml_user_metadata(spark, out)
        spark.stop()
        return
    if which == "catalog-options":
        catalog_write_options(spark, out)
        spark.stop()
        return
    if which == "options":
        write_options(spark, out)
        spark.stop()
        return
    if which == "readback":
        readback(spark, out, src)
        spark.stop()
        return
    if which == "dvs-no-stats":
        dv_cases(spark, out, "dvs-no-stats", "NOSTATS.json", DV_NO_STATS_SCENARIOS)
        drop_hadoop_crc(out)
        spark.stop()
        return
    if which == "dvs":
        dv_cases(spark, out)
        drop_hadoop_crc(out)
        spark.stop()
        return
    if which == "merge":
        merge_cases(spark, out)
        drop_hadoop_crc(out)
        spark.stop()
        return
    if which == "dml":
        dml_cases(spark, out)
        drop_hadoop_crc(out)
        spark.stop()
        return
    if which == "writes":
        stats_table(spark, out)
        write_modes(spark, out)
        write_feature_tables(spark, out)
        drop_hadoop_crc(out)
        spark.stop()
        return
    if which == "insert-path":
        insert_by_path(spark, out)
        spark.stop()
        return
    if which == "create-path":
        create_by_path(spark, out)
        spark.stop()
        return
    if which == "edges":
        write_edges(spark, out)
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
    main(*sys.argv[1:4])
