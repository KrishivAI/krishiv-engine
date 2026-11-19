# SPDX-License-Identifier: Apache-2.0
"""Writes hll_interop/fixture.json from a real Spark: HLL sketches over 100,000 distinct values per input type.

Run in the apache/spark container: spark-submit gen.py <out.json> [<krishiv.json>]
`krishiv.json` (optional, from the Krishiv test's dump) maps type -> hex of Krishiv's compact sketch; Spark then
reads it (`hll_sketch_estimate`, `hll_union` with its own) and the result is written under "krishiv_read_by_spark".
"""

import json
import sys

from pyspark.sql import SparkSession

N = 100_000
SRC = {
    "int": "CAST(id AS INT)",
    "bigint": "id * 100000 + 7",
    "string": "CONCAT('value-', CAST(id AS STRING))",
    "binary": "CAST(CONCAT('bin-', CAST(id AS STRING)) AS BINARY)",
}

spark = (
    SparkSession.builder.master("local[2]")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
out = {"spark_version": spark.version, "distinct": N, "lg_k": 12, "sketches": {}}
for name, expr in SRC.items():
    df = spark.range(N).selectExpr(f"{expr} AS v")
    df.createOrReplaceTempView("t")
    row = spark.sql(
        "SELECT hex(hll_sketch_agg(v)) h, hll_sketch_estimate(hll_sketch_agg(v)) e FROM t"
    ).first()
    out["sketches"][name] = {"hex": row["h"], "estimate": row["e"]}
if len(sys.argv) > 2:
    with open(sys.argv[2]) as fh:
        kr = json.load(fh)
    res = {}
    for name, h in kr.items():
        spark.sql(f"SELECT unhex('{h}') AS s").createOrReplaceTempView("k")
        spark_hex = out["sketches"][name]["hex"]
        r = spark.sql(
            f"SELECT hll_sketch_estimate(s) e, hll_sketch_estimate(hll_union(s, unhex('{spark_hex}'))) u FROM k"
        ).first()
        res[name] = {"estimate": r["e"], "union_with_spark": r["u"]}
    out["krishiv_read_by_spark"] = res
with open(sys.argv[1], "w") as fh:
    json.dump(out, fh, indent=1)
