# SPDX-License-Identifier: Apache-2.0
"""Capture Spark write-semantics goldens (P3-T023) from a real Spark Connect server.

Input : tests/spark-golden/write-semantics/*.json  (cases; `expect` is source-derived, `status: unverified-by-run`)
Output: tests/fixtures/spark-<ver>/write-semantics/<file>.json with an `observed` block per case
        (rows or error class + SQLSTATE + parameters), `status: oracle`.

Usage: gen_write_semantics_goldens.py --remote sc://127.0.0.1:15013 --version 4.3.0

Needs `pyspark-client` matching the server. Cases whose target is `dsv2 ...` need a V2 catalog named `testcat`
(the InMemoryTableCatalog from the Spark sql/core test jar, or Iceberg 1.12 on Spark 4.1 per the T023 plan);
without one they are recorded as `skipped`. Hive-serde cases need a Hive-enabled server. Each case runs in a fresh
database so cases do not interact. Statements that must run through a classic-only client (SAT invalid-mode) are
recorded as `skipped` when the Connect client rejects them first.
"""

from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
CASES = ROOT / "tests" / "spark-golden" / "write-semantics"
FORMAT_VERSION = 1


def run_sql(spark, stmt, conf):
    saved = {k: spark.conf.get(k, None) for k in conf}
    for k, v in conf.items():
        spark.conf.set(k, v)
    try:
        return spark.sql(stmt)
    finally:
        for k, v in saved.items():
            if v is None:
                spark.conf.unset(k)
            else:
                spark.conf.set(k, v)


def err_record(exc):
    cls = getattr(exc, "getErrorClass", lambda: None)() or getattr(exc, "_errorClass", None)
    return {
        "kind": "error",
        "class": cls,
        "sqlstate": getattr(exc, "getSqlState", lambda: None)(),
        "params": getattr(exc, "getMessageParameters", lambda: None)(),
        "message": str(exc).splitlines()[0],
    }


def rows_of(df):
    return [list(r) for r in df.collect()]


def observe_sql_case(spark, case):
    for stmt in case.get("setup", []):
        spark.sql(stmt)
    conf = case.get("conf", {})
    try:
        run_sql(spark, case["stmt"], conf).collect()
    except Exception as exc:
        return err_record(exc)
    q = case["expect"].get("query", "SELECT * FROM t")
    df = spark.sql(q)
    return {"kind": "rows", "rows": sorted(rows_of(df), key=repr), "schema": df.schema.simpleString()}


def observe_save_as_table(spark, case):
    ex = case.get("existing")
    if ex and "view" in ex:
        spark.sql(ex["view"])
    elif ex:
        w = spark.createDataFrame(ex["rows"], ex["schema"]).write.format(ex["format"])
        if ex["partitionBy"]:
            w = w.partitionBy(*ex["partitionBy"])
        w.saveAsTable("t")
    wr = case["write"]
    conf = case.get("conf", {})
    for k, v in conf.items():
        spark.conf.set(k, v)
    mode = wr["mode"]
    note = None
    if mode.lower() == "errorifexists":
        # The Connect server rejects the string "errorifexists"; classic maps it to the same
        # SaveMode.ErrorIfExists as "error" (DataFrameWriter.mode), so run it as "error".
        mode, note = "error", "run as mode=error: the Connect server rejects 'errorifexists'"
    try:
        w = spark.sql(wr["query"]).write.format(wr["format"]).mode(mode)
        if wr["partitionBy"]:
            w = w.partitionBy(*wr["partitionBy"])
        for k, v in wr.get("options", {}).items():
            w = w.option(k, v)
        w.saveAsTable("t")
    except Exception as exc:
        rec = err_record(exc)
        if note:
            rec["note"] = note
        return rec
    finally:
        for k in conf:
            spark.conf.unset(k)
    try:
        df = spark.table("t")
        rec = {"kind": "rows", "rows": sorted(rows_of(df), key=repr), "schema": df.schema.simpleString()}
        if note:
            rec["note"] = note
        return rec
    except Exception as exc:
        rec = err_record(exc)
        rec["phase"] = "read-back"
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", required=True)
    ap.add_argument("--version", default="4.3.0")
    args = ap.parse_args()
    from pyspark.sql import SparkSession  # imported late: the module is importable without a server

    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    out_dir = ROOT / "tests" / "fixtures" / f"spark-{args.version}" / "write-semantics"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(CASES.glob("*.json")):
        doc = json.loads(path.read_text())
        for i, case in enumerate(doc["cases"]):
            db = f"ws_{path.stem}_{i}"
            spark.sql(f"DROP DATABASE IF EXISTS {db} CASCADE")
            spark.sql(f"CREATE DATABASE {db}")
            spark.sql(f"USE {db}")
            for tv in spark.catalog.listTables():  # temp views are session-wide: drop leftovers of earlier cases
                if tv.isTemporary:
                    spark.catalog.dropTempView(tv.name)
            try:
                if "write" in case:
                    case["observed"] = observe_save_as_table(spark, case)
                elif "stmt" in case:
                    if str(case.get("target", "")).startswith("dsv2"):
                        case["observed"] = {"kind": "skipped", "reason": "needs V2 catalog testcat"}
                    else:
                        case["observed"] = observe_sql_case(spark, case)
                else:
                    case["observed"] = {"kind": "skipped", "reason": "DataFrame-API case: see 'how' (not scripted)"}
            finally:
                spark.sql("USE default")
                spark.sql(f"DROP DATABASE IF EXISTS {db} CASCADE")
        doc["status"] = "oracle"
        doc["format_version"] = FORMAT_VERSION
        doc["spark_version"] = args.version
        (out_dir / path.name).write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
