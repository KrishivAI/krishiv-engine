#!/usr/bin/env python3
"""Generate Spark `upper()`/`lower()` golden fixtures for the string kernels (P2-T023a).

Runs a REAL Spark (never from memory) in a throw-away container over every Unicode scalar value (U+0000 to
U+10FFFF without surrogates) and records the code points whose `upper(s)` or `lower(s)` differs from `s`,
with the result as a list of code points. Every other scalar value maps to itself. Usage (from repo root):

    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:4.2.0 \
        /opt/spark/bin/spark-submit --master 'local[2]' /work/gen_case_mapping.py \
        > tests/fixtures/spark-4.2.0/case-mapping.json

The fixture goes to stdout (the container user cannot write the mount); Spark logs go to stderr.
"""

import json
import sys

from pyspark.sql import SparkSession


def main() -> None:
    spark = SparkSession.builder.appName("case-mapping").getOrCreate()
    scalars = [chr(c) for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    df = spark.createDataFrame([(s,) for s in scalars], "s string")
    rows = df.selectExpr("s", "upper(s) u", "lower(s) l").where("s <> u or s <> l").collect()
    upper, lower = {}, {}
    for r in rows:
        cp = ord(r.s)
        if r.u != r.s:
            upper[cp] = [ord(c) for c in r.u]
        if r.l != r.s:
            lower[cp] = [ord(c) for c in r.l]
    json.dump(
        {"spark_version": spark.version, "scalars": len(scalars), "upper": upper, "lower": lower},
        sys.stdout,
        sort_keys=True,
        separators=(",", ":"),
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
