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

The `overflow` cases (INT values, no key) show where Spark evaluates the distance on the Connect
form: `RewriteAsOfJoin` takes `min_by` over every passing right row, so `L - R` (backward) or
`R - L` (forward) of any passing pair can raise `ARITHMETIC_OVERFLOW` under ANSI, unlike the SQL
scan (`SortMergeAsOfJoinExec`), which evaluates none backward and only two rows forward.

Run from the repository root:

    out=$(mktemp -d) && chmod 777 "$out"
    docker run --rm -v "$PWD:/repo:ro" -v "$out:/out" apache/spark:4.3.0 \\
        /opt/spark/bin/spark-submit --master 'local[1]' \\
        /repo/tests/scripts/gen_connect_asof_goldens.py /out/connect-asof.json \\
      && cp "$out/connect-asof.json" tests/fixtures/spark-4.3.0/
    rm -rf "$out"

(the image's `spark` user cannot write a bind mount owned by another user).
"""

import json
import re
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


OVERFLOW = [
    # (name, direction, left values, right values)
    ("backward_overflow", "backward", [2147483647], [-2]),
    ("backward_overflow_farthest_row", "backward", [2147483647], [-2, 2147483646]),
    ("forward_overflow_later_row", "forward", [-2], [0, 1, 2147483647]),
    ("backward_no_overflow", "backward", [5], [1, 3]),
    ("forward_no_overflow", "forward", [-2], [0, 1]),
]


def condition(e):
    """The root error condition: AQE wraps a failed stage's error in
    `STAGE_MATERIALIZATION_MULTIPLE_FAILURES`, whose causes are the query's own error."""
    found = [
        c for c in re.findall(r"Spark\w*Exception: \[([A-Z_]+)", str(e)) if not c.startswith("STAGE_MATERIALIZATION")
    ]
    if not found:
        raise e
    return found[0]


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

    overflow = []
    for name, direction, lv, rv in OVERFLOW:
        l1 = spark.createDataFrame([[v] for v in lv], "lt INT")
        r1 = spark.createDataFrame([[v] for v in rv], "rt INT")
        df = l1._joinAsOf(r1, l1["lt"], r1["rt"], direction=direction)
        case = {"name": name, "direction": direction, "left": lv, "right": rv}
        try:
            case["rows"] = sorted("|".join(cell(v) for v in r) for r in df.collect())
            case["error"] = None
        except Exception as e:  # Spark's error class is the answer
            case["rows"] = None
            case["error"] = condition(e)
        overflow.append(case)

    doc = {
        "spark": spark.version,
        "generator": "tests/scripts/gen_connect_asof_goldens.py",
        "left": {"columns": ["lk", "lt", "a"], "rows": LEFT},
        "right": {"columns": ["rk", "rt", "b"], "rows": RIGHT},
        "cases": cases,
        "overflow": overflow,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    spark.stop()


if __name__ == "__main__":
    main(sys.argv[1])
