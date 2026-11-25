# SPDX-License-Identifier: Apache-2.0
"""Writes theta_interop/fixture.json from a real Spark: Theta sketches over 100,000 distinct values per input type.

Run with Spark 4.3.0: spark-submit gen.py <out.json> [<krishiv.json>]
`krishiv.json` (optional, from the Krishiv test's dump `KRISHIV_THETA_INTEROP_DUMP=<path>`) maps type -> hex of
Krishiv's compact sketch; Spark then reads it (`theta_sketch_estimate`, `theta_union` with its own) and the result is
written under "krishiv_read_by_spark".
"""

import json
import sys

from pyspark.sql import SparkSession

N = 100_000
SRC = {
    "int": "CAST(id AS INT)",
    "bigint": "id * 100000 + 7",
    "float": "CAST(id AS FLOAT) + CAST(0.5 AS FLOAT)",
    "double": "CAST(id AS DOUBLE) + 0.25",
    "string": "CONCAT('value-', CAST(id AS STRING))",
    "binary": "CAST(CONCAT('bin-', CAST(id AS STRING)) AS BINARY)",
    "array_int": "ARRAY(CAST(id AS INT), CAST(id % 7 AS INT))",
    "array_bigint": "ARRAY(id * 100000 + 7, id % 7)",
}

spark = (
    SparkSession.builder.master("local[2]")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
out = {"spark_version": spark.version, "distinct": N, "lg_nom": 12, "sketches": {}}
for name, expr in SRC.items():
    spark.range(N).selectExpr(f"{expr} AS v").createOrReplaceTempView("t")
    row = spark.sql(
        "SELECT hex(theta_sketch_agg(v)) h, theta_sketch_estimate(theta_sketch_agg(v)) e FROM t"
    ).first()
    out["sketches"][name] = {"hex": row["h"], "estimate": row["e"]}
# Spark 4.3.0 on a group with no sketch (design 3.3), for the profile rule.
r = spark.sql(
    "SELECT theta_sketch_estimate(theta_intersection_agg(s)) e FROM VALUES (CAST(NULL AS BINARY)) t(s)"
).first()
out["intersection_of_nothing"] = r["e"]
if len(sys.argv) > 2:
    with open(sys.argv[2]) as fh:
        kr = json.load(fh)
    res = {}
    for name, h in kr.items():
        spark.sql(f"SELECT unhex('{h}') AS s").createOrReplaceTempView("k")
        spark_hex = out["sketches"][name]["hex"]
        r = spark.sql(
            "SELECT theta_sketch_estimate(s) e, "
            f"theta_sketch_estimate(theta_union(s, unhex('{spark_hex}'))) u FROM k"
        ).first()
        res[name] = {"estimate": r["e"], "union_with_spark": r["u"]}
    out["krishiv_read_by_spark"] = res
with open(sys.argv[1], "w") as fh:
    json.dump(out, fh, indent=1)
