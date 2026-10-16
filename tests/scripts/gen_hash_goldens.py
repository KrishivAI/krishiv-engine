#!/usr/bin/env python3
"""Generate Spark `hash()` / `xxhash64()` golden fixtures (P0-T20, ADR 0010).

Runs against a REAL Spark Connect server (never from memory) and records the Spark
version it ran against. Usage (from repo root):

    docker compose -f bench/baselines/docker-compose.yml up -d spark
    (cd bench && uv run python ../tests/scripts/gen_hash_goldens.py \
        --remote sc://127.0.0.1:15002 --out ../tests/fixtures/spark-4.2.0/hash.json)
    docker compose -f bench/baselines/docker-compose.yml stop spark
"""
import argparse
import datetime
import json

# (category, [expression text, ...]); an expression is the argument list of hash(...).
CASES = [
    ("boolean", ["true", "false"]),
    ("tinyint", ["CAST(0 AS TINYINT)", "CAST(1 AS TINYINT)", "CAST(-128 AS TINYINT)", "CAST(127 AS TINYINT)"]),
    ("smallint", ["CAST(0 AS SMALLINT)", "CAST(-32768 AS SMALLINT)", "CAST(32767 AS SMALLINT)"]),
    ("int", ["0", "1", "-1", "42", "-2147483648", "2147483647"]),
    ("bigint", ["0L", "1L", "-1L", "-9223372036854775808L", "9223372036854775807L", "4294967296L"]),
    ("float", ["CAST(0.0 AS FLOAT)", "-CAST(0.0 AS FLOAT)", "CAST(1.5 AS FLOAT)", "CAST('NaN' AS FLOAT)",
               "CAST('Infinity' AS FLOAT)", "CAST('-Infinity' AS FLOAT)", "CAST(-1.25 AS FLOAT)"]),
    ("double", ["0.0D", "-0.0D", "1.5D", "CAST('NaN' AS DOUBLE)", "CAST('Infinity' AS DOUBLE)",
                "CAST('-Infinity' AS DOUBLE)", "-1.25D", "1.7976931348623157E308D", "4.9E-324D"]),
    ("decimal_le18", ["CAST(0 AS DECIMAL(10,2))", "CAST(123.45 AS DECIMAL(10,2))", "CAST(-123.45 AS DECIMAL(10,2))",
                      "CAST(999999999999999999 AS DECIMAL(18,0))", "CAST(0.000001 AS DECIMAL(18,6))"]),
    ("decimal_gt18", ["CAST(0 AS DECIMAL(38,10))", "CAST(123456789012345678901.1234567890 AS DECIMAL(38,10))",
                      "CAST(-123456789012345678901.1234567890 AS DECIMAL(38,10))",
                      "CAST(12345678901234567890 AS DECIMAL(20,0))"]),
    ("string", ["''", "'a'", "'hello world'", "'Spark SQL'", "'héllo'", "'日本語'",
                "'\U0001f600'", "'a\u0000b'"]),
    ("binary", ["X''", "X'00'", "X'DEADBEEF'", "CAST('abc' AS BINARY)", "X'0001020304050607'"]),
    ("date", ["DATE'1970-01-01'", "DATE'2024-02-29'", "DATE'1969-12-31'", "DATE'0001-01-01'", "DATE'9999-12-31'"]),
    ("timestamp", ["TIMESTAMP'1970-01-01 00:00:00+00:00'", "TIMESTAMP'2024-02-29 12:34:56.789012+00:00'",
                   "TIMESTAMP'1969-12-31 23:59:59.999999+00:00'"]),
    ("timestamp_ntz", ["TIMESTAMP_NTZ'1970-01-01 00:00:00'", "TIMESTAMP_NTZ'2024-02-29 12:34:56.789012'",
                       "TIMESTAMP_NTZ'1969-12-31 23:59:59.999999'"]),
    ("null", ["CAST(NULL AS INT)", "CAST(NULL AS STRING)", "CAST(NULL AS DOUBLE)", "NULL"]),
    ("array", ["array(1, 2, 3)", "CAST(array() AS ARRAY<INT>)", "array(1, NULL, 3)", "CAST(NULL AS ARRAY<INT>)"]),
    ("map", ["map('a', 1, 'b', 2)", "CAST(map() AS MAP<STRING,INT>)", "map('a', CAST(NULL AS INT))"]),
    ("struct", ["named_struct('a', 1, 'b', 'x')", "named_struct('a', CAST(NULL AS INT), 'b', 'x')",
                "named_struct('a', 1, 'b', CAST(NULL AS STRING))"]),
    ("nested_array_struct", ["array(named_struct('a', 1, 'b', 'x'), named_struct('a', 2, 'b', 'y'))",
                             "array(named_struct('a', 1, 'b', CAST(NULL AS STRING)), CAST(NULL AS STRUCT<a:INT,b:STRING>))"]),
    ("multi_arg", ["1, 'a'", "1, 'a', 2.5D", "'a', 1", "CAST(NULL AS INT), 'a'", "1, CAST(NULL AS STRING)",
                   "1, 2, 3, 4", "true, 1L, 'x', X'FF', DATE'2024-01-01'"]),
]


# Spark 4.x rejects hash() on MAP by default (DATATYPE_MISMATCH.HASH_MAP_TYPE). Map goldens are
# therefore produced with this legacy conf, which is recorded per case in the fixture.
LEGACY_MAP_CONF = "spark.sql.legacy.allowHashOnMapType"
MAP_DEFAULT_ERROR_EXPR = "map('a', 1, 'b', 2)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://127.0.0.1:15002")
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/hash.json")
    args = ap.parse_args()

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    version = spark.version
    cases = []
    for category, exprs in CASES:
        for expr in exprs:
            conf = {LEGACY_MAP_CONF: "true"} if category == "map" else {}
            for k, v in conf.items():
                spark.conf.set(k, v)
            row = spark.sql(
                f"SELECT hash({expr}) AS h, xxhash64({expr}) AS x, typeof(struct({expr})) AS t"
            ).collect()[0]
            cases.append({
                "id": f"{category}_{len([c for c in cases if c['category'] == category])}",
                "category": category,
                "expr": expr,
                # Spark type of the hashed argument list: `struct<col1:T1,...>` text from typeof().
                "spark_type": row["t"],
                "hash": row["h"],
                "xxhash64": row["x"],
                "conf": conf,
            })
            for k in conf:
                spark.conf.unset(k)
    # Default-conf behaviour for MAP: an analysis error, recorded as its own fixture section.
    try:
        spark.sql(f"SELECT hash({MAP_DEFAULT_ERROR_EXPR})").collect()
        raise SystemExit("expected hash(map) to fail under default conf")
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        err = getattr(e, "getCondition", lambda: None)() or "UNKNOWN"
    errors = [{"expr": MAP_DEFAULT_ERROR_EXPR, "category": "map", "error_class": err}]
    doc = {
        "spark_version": version,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "generator": "tests/scripts/gen_hash_goldens.py",
        "notes": "hash() = Murmur3 x86_32 seed 42; xxhash64() seed 42. Values are signed ints as Spark returns them.",
        "cases": cases,
        "default_conf_errors": errors,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {len(cases)} cases against Spark {version} -> {args.out}")


if __name__ == "__main__":
    main()
