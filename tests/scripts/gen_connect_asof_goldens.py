#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Writes tests/fixtures/spark-4.3.0/connect-asof.json: Spark 4.3.0's answers for the as-of joins
that Spark Connect's `AsOfJoin` relation (relation 39, PySpark's `joinAsOf`) builds.

The Connect server turns the relation into `Dataset.joinAsOf`, the same logical `AsOfJoin`
PySpark's classic `_joinAsOf` builds, so the answers come from a local Spark 4.3.0 in the stock
image (no Connect client dependencies needed). Krishiv's test,
`crates/krishiv-connect-server/tests/as_of_join.rs`, sends each case as an `AsOfJoin` relation
over the same rows and compares; the nearest direction, a tolerance and `using_columns` are
recorded with Spark's rows but Krishiv refuses them for now (`KRISHIV_UNSUPPORTED_FEATURE`).

Run from the repository root:

    docker run --rm -v "$PWD:/repo" apache/spark:4.3.0 /opt/spark/bin/spark-submit \\
        --master 'local[1]' /repo/tests/scripts/gen_connect_asof_goldens.py \\
        /repo/tests/fixtures/spark-4.3.0/connect-asof.json
"""

import json
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

LEFT = [
    ["x", 1, 1],
    ["x", 5, 2],
    ["x", 10, 3],
    ["y", 4, 4],
    ["x", None, 5],
    [None, 3, 6],
    ["z", 7, 7],
]
RIGHT = [
    ["x", 1, 101],
    ["x", 4, 102],
    ["x", 5, 103],
    ["x", 9, 104],
    ["y", 6, 105],
    ["x", None, 106],
    [None, 3, 107],
]


def cell(v):
    return "NULL" if v is None else str(v)


def main(out):
    spark = SparkSession.builder.getOrCreate()
    schema = "k STRING, t INT, v INT"
    left = spark.createDataFrame(LEFT, schema).toDF("lk", "lt", "a")
    right = spark.createDataFrame(RIGHT, schema).toDF("rk", "rt", "b")
    cases = []

    def run(name, krishiv, **kw):
        swapped = kw.pop("swapped", False)
        keyed = kw.pop("keyed", True)
        using = kw.pop("using", None)
        if using is not None:
            l2 = left.withColumnRenamed("lk", "k")
            r2 = right.withColumnRenamed("rk", "k")
            df = l2._joinAsOf(r2, "lt", "rt", on=using, **kw)
        else:
            on = (left["lk"] == right["rk"]) if keyed else None
            la, ra = (right["rt"], left["lt"]) if swapped else (left["lt"], right["rt"])
            df = left._joinAsOf(right, la, ra, on=on, **kw)
        rows = sorted("|".join(cell(v) for v in r) for r in df.collect())
        cases.append(
            {
                "name": name,
                "how": kw.get("how", "inner"),
                "direction": kw.get("direction", "backward"),
                "allow_exact": kw.get("allowExactMatches", True),
                "keyed": keyed,
                "swapped": swapped,
                "tolerance": None if kw.get("tolerance") is None else 2,
                "using": using,
                "krishiv": krishiv,
                "rows": rows,
            }
        )

    for how in ["inner", "left"]:
        for direction in ["backward", "forward"]:
            for exact in [True, False]:
                run(
                    f"{how}_{direction}_{'exact' if exact else 'strict'}",
                    "rows",
                    how=how,
                    direction=direction,
                    allowExactMatches=exact,
                )
    run("no_key_backward", "rows", keyed=False)
    run("no_key_forward_left", "rows", keyed=False, how="left", direction="forward")
    # `joinAsOf(right, right.rt, left.lt)`: the operands swapped, which is the forward match.
    run("swapped_operands", "rows", swapped=True, how="left")
    run("swapped_operands_forward", "rows", swapped=True, direction="forward")
    run("nearest", "refused", direction="nearest")
    run("tolerance", "refused", tolerance=F.lit(2))
    run("using_columns", "refused", using=["k"])

    doc = {
        "spark": spark.version,
        "generator": "tests/scripts/gen_connect_asof_goldens.py",
        "left": {"columns": ["lk", "lt", "a"], "rows": LEFT},
        "right": {"columns": ["rk", "rt", "b"], "rows": RIGHT},
        "cases": cases,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    spark.stop()


if __name__ == "__main__":
    main(sys.argv[1])
