#!/usr/bin/env python3
"""Dump Spark's `UTF8_LCASE` key mapping per code point, for `krishiv_kernels::string::collation_key`.

Calls `CollationAwareUTF8String.lowerCaseCodePoints` of a local Spark on one string holding every Unicode
scalar value (newlines between them), so the table is what Spark's own function produces: ICU's simple lower
case, `U+0130` to `i` + `U+0307`, and final sigma to sigma. Only the code points the function changes are kept.
Usage (from repo root), then rerun `gen_case_table.py`:

    env -u SPARK_HOME uv run --no-project --python 3.12 --with 'pyspark==4.2.0' \
        python tests/scripts/gen_collation_lowercase.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pyspark.sql import SparkSession

OUT = "tests/fixtures/spark-4.2.0/collation-lowercase.json"


def main() -> None:
    spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()
    scalars = [c for c in range(0x110000) if not 0xD800 <= c < 0xE000 and c != 0x0A]
    jvm = spark._jvm
    text = jvm.org.apache.spark.unsafe.types.UTF8String.fromString("".join(chr(c) + "\n" for c in scalars))
    util = jvm.org.apache.spark.sql.catalyst.util.CollationAwareUTF8String
    parts = util.lowerCaseCodePoints(text).toString().split("\n")[:-1]
    assert len(parts) == len(scalars)
    changed = {str(c): [ord(x) for x in p] for c, p in zip(scalars, parts, strict=True) if p != chr(c)}
    Path(OUT).write_text(
        json.dumps({"spark_version": spark.version, "lower": changed}, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{len(changed)} code points -> {OUT}")


if __name__ == "__main__":
    main()
