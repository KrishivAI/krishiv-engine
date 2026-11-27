#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probe the Spark-version differences of TIME in the CSV and JSON sources that Krishiv keys by profile.

Runs a REAL Spark (never from memory) in a throw-away container and prints one JSON object to stdout:
- `infer`: the schema CSV (`inferSchema`) and JSON (`inferTimestamp`) file reads and `schema_of_csv` infer for
  time-only texts, with `spark.sql.timeType.enabled` off and on. Spark 4.2 never infers TIME; 4.3.0 infers
  `time(6)` with the gate on (`krishiv.io.text.inferTime`, set from the profile).
- `read`: `12:34:56.123456789` read from a CSV file into `TIME(6)` with the gate on, under the default grammar and
  under `timeFormat=HH:mm:ss.SSSSSSSSS`, cast to STRING. Spark 4.2 cuts the default grammar to microseconds and
  keeps every digit a pattern reads (`krishiv.io.text.timeNanos`).

Usage (from repo root), once per version:

    v=4.2.1; out=tests/fixtures/spark-$v/time-text-profiles.json; mkdir -p "$(dirname "$out")"
    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:$v \\
        /opt/spark/bin/spark-submit --master 'local[1]' --driver-memory 1g /work/probe_time_text_profiles.py \\
        > "$out.tmp" && python3 -m json.tool "$out.tmp" > /dev/null && mv "$out.tmp" "$out"

Checked by `krishiv-session/tests/time_file_reads.rs` (`the_4_2_profile_never_infers_time`) and
`krishiv-io/src/csv/time_tests.rs` (the 4.2 profile rows).
"""

import json
import os
import sys
import tempfile

from pyspark.sql import SparkSession


def main() -> None:
    spark = SparkSession.builder.master("local[1]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    root = tempfile.mkdtemp()
    csv = os.path.join(root, "a.csv")
    with open(csv, "w") as f:
        f.write("id,t\n0,12:34:56\n1,01:02:03.5\n")
    js = os.path.join(root, "a.json")
    with open(js, "w") as f:
        f.write('{"id":0,"t":"12:34:56"}\n')
    nine = os.path.join(root, "nine.csv")
    with open(nine, "w") as f:
        f.write("t\n12:34:56.123456789\n")

    infer = {}
    for gate in ["false", "true"]:
        spark.conf.set("spark.sql.timeType.enabled", gate)
        c = spark.read.option("header", "true").option("inferSchema", "true").csv(csv)
        j = spark.read.option("inferTimestamp", "true").json(js)
        infer[gate] = {
            "csv": c.schema["t"].dataType.simpleString(),
            "json": j.schema["t"].dataType.simpleString(),
            "schema_of_csv": spark.sql("SELECT schema_of_csv('14:30:45') AS s").collect()[0].s,
        }

    spark.conf.set("spark.sql.timeType.enabled", "true")
    read = {}
    for name, opts in [("default", {}), ("HH:mm:ss.SSSSSSSSS", {"timeFormat": "HH:mm:ss.SSSSSSSSS"})]:
        df = spark.read.option("header", "true").options(**opts).schema("t TIME(6)").csv(nine)
        read[name] = df.selectExpr("CAST(t AS STRING) AS s").collect()[0].s

    json.dump({"spark": spark.version, "infer": infer, "read": read}, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    spark.stop()


if __name__ == "__main__":
    main()
