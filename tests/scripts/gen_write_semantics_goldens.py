# SPDX-License-Identifier: Apache-2.0
"""Capture Spark write-semantics goldens (P3-T023) from a real Spark Connect server.

Input : tests/spark-golden/write-semantics/*.json  (cases; `expect` is source-derived, `status: unverified-by-run`)
Output: tests/fixtures/spark-<ver>/write-semantics/<file>.json with an `observed` block per case
        (rows or error class + SQLSTATE + parameters), `status: oracle`.

Usage: gen_write_semantics_goldens.py --remote sc://127.0.0.1:15013 --version 4.3.0

Needs `pyspark-client` matching the server; the connected server's version must equal `--version`. Cases whose
target is `dsv2 ...` are always recorded as `skipped` (this script has no V2 setup or read-back path: they stay
source-derived until one is written). Hive-serde cases need a Hive-enabled server. Each case runs in a fresh,
uniquely named database (never dropped before it is created) so cases and runs do not interact; the scratch name is
replaced by `default` in the recorded diagnostics. Cases that need a classic-only client (the writer mode strings
`default` and `foo`, which the Connect client rejects while building the command) are recorded as `skipped`.

The output carries only `observed`. Curated `expect` overrides (and `spark_expected_by_source`) live in the input
files and are edited by hand after a run; regeneration never rewrites them.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[2]
CASES = ROOT / "tests" / "spark-golden" / "write-semantics"
FORMAT_VERSION = 1


CLASSIC_ONLY_MODES = {"default", "foo"}


def set_conf(spark, conf):
    """Set `conf`; return what to pass to `restore_conf` (the prior value of each key, None if unset)."""
    saved = {k: spark.conf.get(k, None) for k in conf}
    for k, v in conf.items():
        spark.conf.set(k, v)
    return saved


def restore_conf(spark, saved):
    for k, v in saved.items():
        if v is None:
            spark.conf.unset(k)
        else:
            spark.conf.set(k, v)


def run_sql(spark, stmt, conf):
    saved = set_conf(spark, conf)
    try:
        return spark.sql(stmt)
    finally:
        restore_conf(spark, saved)


def fields_of(df):
    return [{"name": f.name, "type": f.dataType.simpleString(), "nullable": f.nullable} for f in df.schema.fields]


def first_line(exc):
    for line in str(exc).splitlines():
        if line.strip():
            return line.strip()
    return type(exc).__name__


def err_record(exc):
    cls = getattr(exc, "getErrorClass", lambda: None)() or getattr(exc, "_errorClass", None)
    return {
        "kind": "error",
        "class": cls,
        "sqlstate": getattr(exc, "getSqlState", lambda: None)(),
        "params": getattr(exc, "getMessageParameters", lambda: None)(),
        "message": first_line(exc),
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
        rec = err_record(exc)
        if "rows" in case["expect"]:  # the case states what the table holds after the refusal
            try:
                rec["after"] = sorted(rows_of(spark.sql(case["expect"].get("query", "SELECT * FROM t"))), key=repr)
            except Exception as read_exc:
                rec["after_error"] = first_line(read_exc)
        return rec
    q = case["expect"].get("query", "SELECT * FROM t")
    df = spark.sql(q)
    return {
        "kind": "rows",
        "rows": sorted(rows_of(df), key=repr),
        "schema": df.schema.simpleString(),
        "fields": fields_of(df),
    }


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
    if wr["mode"].lower() in CLASSIC_ONLY_MODES:
        return {"kind": "skipped", "reason": "classic-only: the Connect client rejects this writer mode string"}
    conf = case.get("conf", {})
    saved = set_conf(spark, conf)
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
        restore_conf(spark, saved)
    try:
        df = spark.table("t")
        rec = {
            "kind": "rows",
            "rows": sorted(rows_of(df), key=repr),
            "schema": df.schema.simpleString(),
            "fields": fields_of(df),
        }
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
    if not spark.version.startswith(args.version):
        raise SystemExit(f"the server is Spark {spark.version}, not {args.version}: refusing to write its results")
    run_id = uuid.uuid4().hex[:8]
    out_dir = ROOT / "tests" / "fixtures" / f"spark-{args.version}" / "write-semantics"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(CASES.glob("*.json")):
        doc = json.loads(path.read_text())
        for i, case in enumerate(doc["cases"]):
            db = f"ws_{run_id}_{path.stem}_{i}"
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
                        case["observed"] = {"kind": "skipped", "reason": "V2 cases have no capture path in this script"}
                    else:
                        case["observed"] = observe_sql_case(spark, case)
                else:
                    case["observed"] = {"kind": "skipped", "reason": "DataFrame-API case: see 'how' (not scripted)"}
                case["observed"] = json.loads(json.dumps(case["observed"], default=str).replace(db, "default"))
            finally:
                spark.sql("USE default")
                spark.sql(f"DROP DATABASE IF EXISTS {db} CASCADE")  # created above by this run
        doc["status"] = "oracle"
        doc["format_version"] = FORMAT_VERSION
        doc["spark_version"] = args.version
        (out_dir / path.name).write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
