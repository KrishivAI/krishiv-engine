# SPDX-License-Identifier: Apache-2.0
"""Iceberg type -> Spark type goldens from a real Spark 4.1.0 + iceberg-spark-runtime 1.12.0 run (P3-T022a).

For each case, an Iceberg schema (Iceberg's JSON form) is parsed by Iceberg's own ``SchemaParser`` and converted by
``SparkSchemaUtil.convert``, the conversion Spark's Iceberg integration applies to every table it loads
(``TypeToSparkType``). The output records the Spark schema's ``json()`` or, when the conversion refuses, the exception
class PySpark raises for it (PySpark converts the Py4J error, so this is PySpark's class name, e.g.
``UnsupportedOperationException``) and the message. Iceberg types Spark cannot write (``fixed``, ``uuid``, ``time``,
the nanosecond timestamps, ``unknown``) are covered too, since the schema is built directly, not by Spark DDL. A
column with a format v3 default value is recorded as well: Spark shows it as ``EXISTS_DEFAULT``/``CURRENT_DEFAULT``
metadata, which Krishiv refuses until it fills missing columns from it.

Run it from the repository root:

    docker run --rm --user "$(id -u)" -e HOME=/tmp -e TZ=UTC \
        -v "$PWD/tests/scripts:/scripts:ro" -v "$PWD/tests/fixtures/iceberg-1.12.0:/out" \
        --entrypoint /opt/spark/bin/spark-submit apache/spark:4.1.0 \
        --packages org.apache.iceberg:iceberg-spark-runtime-4.1_2.13:1.12.0 --conf spark.jars.ivy=/tmp/ivy \
        /scripts/gen_iceberg_type_goldens.py /out/types.json

The consuming test is ``crates/krishiv-connectors-iceberg/src/iceberg_v0/type_golden_tests.rs``.
"""

import json
import sys

from pyspark.sql import SparkSession


def field(fid: int, name: str, typ: object, required: bool = False, doc: str | None = None) -> dict:
    out: dict = {"id": fid, "name": name, "required": required, "type": typ}
    if doc is not None:
        out["doc"] = doc
    return out


def schema(*fields: dict) -> dict:
    return {"type": "struct", "schema-id": 0, "fields": list(fields)}


PRIMITIVES = [
    "boolean",
    "int",
    "long",
    "float",
    "double",
    "decimal(9,2)",
    "decimal(38,10)",
    "decimal(38,0)",
    "date",
    "time",
    "timestamp",
    "timestamptz",
    "timestamp_ns",
    "timestamptz_ns",
    "string",
    "uuid",
    "fixed[16]",
    "binary",
    "variant",
    "unknown",
    "geometry(OGC:CRS84)",
    "geometry(EPSG:4326)",
    "geometry(NOT:A_CRS)",
    "geography(OGC:CRS84, spherical)",
    "geography(OGC:CRS84, vincenty)",
    "geography(EPSG:3857, spherical)",
]


def cases() -> list[tuple[str, dict]]:
    out = []
    for p in PRIMITIVES:
        out.append((f"optional {p}", schema(field(1, "c", p))))
        if p != "unknown":  # Iceberg requires an unknown column to be optional
            out.append((f"required {p}", schema(field(1, "c", p, required=True))))
    out.append(("doc becomes comment", schema(field(1, "c", "int", doc="the doc"))))
    out.append(
        (
            "struct",
            schema(
                field(
                    1,
                    "s",
                    {
                        "type": "struct",
                        "fields": [
                            field(2, "a", "int", required=True),
                            field(3, "b", "string", doc="inner doc"),
                        ],
                    },
                )
            ),
        )
    )
    for elem_req in (False, True):
        out.append(
            (
                f"list, element {'required' if elem_req else 'optional'}",
                schema(
                    field(
                        1,
                        "l",
                        {"type": "list", "element-id": 2, "element": "long", "element-required": elem_req},
                    )
                ),
            )
        )
    for val_req in (False, True):
        out.append(
            (
                f"map, value {'required' if val_req else 'optional'}",
                schema(
                    field(
                        1,
                        "m",
                        {
                            "type": "map",
                            "key-id": 2,
                            "key": "string",
                            "value-id": 3,
                            "value": "double",
                            "value-required": val_req,
                        },
                    )
                ),
            )
        )
    out.append(
        (
            "nested list of struct in map",
            schema(
                field(
                    1,
                    "n",
                    {
                        "type": "map",
                        "key-id": 2,
                        "key": "int",
                        "value-id": 3,
                        "value": {
                            "type": "list",
                            "element-id": 4,
                            "element": {"type": "struct", "fields": [field(5, "x", "timestamptz")]},
                            "element-required": True,
                        },
                        "value-required": False,
                    },
                    required=True,
                )
            ),
        )
    )
    with_default = field(1, "c", "int")
    with_default.update({"initial-default": 5, "write-default": 7})
    out.append(("column default values", schema(with_default)))
    out.append(
        (
            "list of time",
            schema(field(1, "l", {"type": "list", "element-id": 2, "element": "time", "element-required": True})),
        )
    )
    out.append(
        ("time nested in a struct", schema(field(1, "s", {"type": "struct", "fields": [field(2, "t", "time")]})))
    )
    return out


def main(path: str) -> None:
    spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()
    jvm = spark._jvm
    parser = jvm.org.apache.iceberg.SchemaParser
    util = jvm.org.apache.iceberg.spark.SparkSchemaUtil
    results = []
    for name, iceberg in cases():
        text = json.dumps(iceberg, separators=(",", ":"))
        entry: dict = {"case": name, "iceberg": iceberg}
        try:
            parsed = parser.fromJson(text)
        except Exception as e:
            entry["parse_error"] = java_error(e)
            results.append(entry)
            continue
        try:
            entry["spark"] = json.loads(util.convert(parsed).json())
        except Exception as e:
            entry["error"] = java_error(e)
        results.append(entry)
    meta = {
        "spark": spark.version,
        "iceberg": jvm.org.apache.iceberg.IcebergBuild.version(),
        "cases": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
        f.write("\n")
    spark.stop()


def java_error(e: Exception) -> dict:
    java = getattr(e, "java_exception", None)
    if java is None:
        return {"class": type(e).__name__, "message": str(e)}
    return {"class": java.getClass().getName(), "message": java.getMessage()}


if __name__ == "__main__":
    main(sys.argv[1])
