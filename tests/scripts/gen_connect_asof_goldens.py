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

The `tolerance_nearest` section (follow-up (a) of `design/P3-4.3C-asof-join.md`) records the nearest
direction and tolerances over the same rows, the tolerance errors (negative or NULL:
`AS_OF_JOIN.TOLERANCE_IS_NON_NEGATIVE`; nondeterministic: `TOLERANCE_IS_UNFOLDABLE`; a column:
`INTERNAL_ERROR`, unresolved when checked, which a Spark 4.3.0 Connect server answers too), NULL
operands, TIMESTAMP values with an interval tolerance and nearest ties (with every candidate row).

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


def error_class(e):
    """The error condition with its subclass (`AS_OF_JOIN.TOLERANCE_IS_NON_NEGATIVE`), from an
    analysis error or a failed job (AQE's `STAGE_MATERIALIZATION` wrapper skipped)."""
    found = [
        c for c in re.findall(r"\[([A-Z][A-Z_]*(?:\.[A-Z_]+)*)\]", str(e)) if not c.startswith("STAGE_MATERIALIZATION")
    ]
    if not found:
        raise e
    return found[0]


def tolerance_and_nearest(spark, left, right):
    """Follow-up (a) of `design/P3-4.3C-asof-join.md`: the nearest direction and a tolerance over the
    same rows as `cases` (`krishiv: rows` once Krishiv runs them), the tolerance error cases, NULL
    operands, a TIMESTAMP column with an interval tolerance, and nearest ties. Each case records
    Spark's rows or its error class; `exprs` names the operands and tolerance as SQL text, which
    the Connect test sends as `expression_string`s (Spark's `F.expr` of the same text)."""
    out = []

    def record(name, df_fn, meta):
        case = dict(meta, name=name)
        try:
            df = df_fn()
            case["rows"] = sorted("|".join(cell(v) for v in r) for r in df.collect())
            case["error"] = None
        except Exception as e:  # Spark's error class is the answer
            case["rows"] = None
            case["error"] = error_class(e)
            # The message line that carries the class, its SQLSTATE suffix dropped.
            line = next(ln for ln in str(e).splitlines() if f"[{case['error']}]" in ln)
            case["message"] = re.sub(r"\s*SQLSTATE: \w+;?$", "", line[line.index(f"[{case['error']}]") :])
        out.append(case)

    def keyed(name, how, direction, exact, tolerance, keyed=True, left_value="lt", right_value="rt"):
        def build():
            on = (left["lk"] == right["rk"]) if keyed else None
            la = F.expr(left_value) if left_value != "lt" else left["lt"]
            ra = F.expr(right_value) if right_value != "rt" else right["rt"]
            tol = None if tolerance is None else F.expr(tolerance)
            return left._joinAsOf(
                right, la, ra, on=on, how=how, tolerance=tol, allowExactMatches=exact, direction=direction
            )

        record(
            name,
            build,
            {
                "data": "main",
                "how": how,
                "direction": direction,
                "allow_exact": exact,
                "keyed": keyed,
                "left_value": left_value,
                "right_value": right_value,
                "tolerance": tolerance,
            },
        )

    for how in ["inner", "left"]:
        for exact in [True, False]:
            tag = "exact" if exact else "strict"
            keyed(f"nearest_{how}_{tag}", how, "nearest", exact, None)
            for direction in ["backward", "forward", "nearest"]:
                for tol in ["0", "1", "2"]:
                    keyed(f"tolerance{tol}_{direction}_{how}_{tag}", how, direction, exact, tol)
    keyed("nearest_no_key", "inner", "nearest", True, None, keyed=False)
    keyed("nearest_no_key_left_strict_tolerance1", "left", "nearest", False, "1", keyed=False)
    # A foldable tolerance that is not a literal, a BIGINT and a DECIMAL one, and a NULL one.
    keyed("tolerance_folded_expression", "left", "backward", True, "1 + 1")
    keyed("tolerance_bigint", "left", "forward", True, "CAST(2 AS BIGINT)")
    keyed("tolerance_decimal", "left", "nearest", True, "2.5")
    keyed("tolerance_null", "left", "backward", True, "CAST(NULL AS INT)")
    # The error cases: negative and non-foldable tolerances.
    keyed("tolerance_negative", "inner", "backward", True, "-1")
    keyed("tolerance_negative_folded", "inner", "forward", True, "1 - 3")
    # A column as the tolerance. As SQL text it is unresolved when Spark checks it (what a Connect
    # client sends); as `left["lt"]` it is resolved.
    keyed("tolerance_column", "inner", "backward", True, "lt")
    record(
        "tolerance_resolved_column",
        lambda: left._joinAsOf(right, left["lt"], right["rt"], on=left["lk"] == right["rk"], tolerance=left["lt"]),
        {
            "data": "main",
            "how": "inner",
            "direction": "backward",
            "allow_exact": True,
            "keyed": True,
            "left_value": "lt",
            "right_value": "rt",
            "tolerance": "left.lt (resolved, classic API only; Connect sends it unresolved: INTERNAL_ERROR)",
        },
    )
    keyed("tolerance_nondeterministic", "inner", "backward", True, "CAST(rand() * 0 AS INT)")
    # NULL operands only: a NULL literal as one side's as-of value.
    for direction in ["backward", "forward", "nearest"]:
        keyed(f"null_left_operand_{direction}", "left", direction, True, None, left_value="CAST(NULL AS INT)")
        keyed(f"null_right_operand_{direction}", "left", direction, True, None, right_value="CAST(NULL AS INT)")
    keyed("null_both_operands", "left", "backward", True, None, left_value="NULL", right_value="NULL")

    # TIMESTAMP values with an interval tolerance (session time zone UTC).
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    tl = spark.createDataFrame(
        [["x", "2026-01-01 00:00:05", 1], ["x", "2026-01-01 00:00:20", 2], ["y", "2026-01-01 00:00:09", 3]],
        "lk STRING, s STRING, a INT",
    ).selectExpr("lk", "CAST(s AS TIMESTAMP) AS lt", "a")
    tr = spark.createDataFrame(
        [["x", "2026-01-01 00:00:03", 101], ["x", "2026-01-01 00:00:08", 102], ["y", "2026-01-01 00:00:10", 103]],
        "rk STRING, s STRING, b INT",
    ).selectExpr("rk", "CAST(s AS TIMESTAMP) AS rt", "b")
    for direction in ["backward", "forward", "nearest"]:
        for tol in ["INTERVAL 2 SECONDS", "INTERVAL 3 SECONDS"]:
            record(
                f"timestamp_{direction}_{tol.split()[1]}s",
                lambda d=direction, t=tol: tl._joinAsOf(
                    tr, tl["lt"], tr["rt"], on=tl["lk"] == tr["rk"], how="left", tolerance=F.expr(t), direction=d
                ),
                {
                    "data": "timestamp",
                    "how": "left",
                    "direction": direction,
                    "allow_exact": True,
                    "keyed": True,
                    "left_value": "lt",
                    "right_value": "rt",
                    "tolerance": tol,
                },
            )
    record(
        "timestamp_int_tolerance",
        lambda: tl._joinAsOf(tr, tl["lt"], tr["rt"], on=tl["lk"] == tr["rk"], tolerance=F.lit(2)),
        {
            "data": "timestamp",
            "how": "inner",
            "direction": "backward",
            "allow_exact": True,
            "keyed": True,
            "left_value": "lt",
            "right_value": "rt",
            "tolerance": "2",
        },
    )

    # Nearest ties: a left value as far from a lower as from a higher right row. `candidates`
    # lists every right row at the least distance; the rows Spark chose are recorded as well.
    for name, lv, rv in [("tie_between_lower_and_higher", 5, [3, 7]), ("tie_with_exact", 5, [5, 5, 9])]:
        l1 = spark.createDataFrame([[lv, 1]], "lt INT, a INT")
        r1 = spark.createDataFrame([[v, 100 + i] for i, v in enumerate(rv)], "rt INT, b INT")
        best = min(abs(lv - v) for v in rv)
        record(
            name,
            lambda l1=l1, r1=r1: l1._joinAsOf(r1, l1["lt"], r1["rt"], direction="nearest"),
            {
                "data": {"left": [[lv, 1]], "right": [[v, 100 + i] for i, v in enumerate(rv)]},
                "how": "inner",
                "direction": "nearest",
                "allow_exact": True,
                "keyed": False,
                "left_value": "lt",
                "right_value": "rt",
                "tolerance": None,
                "candidates": sorted(f"{lv}|1|{v}|{100 + i}" for i, v in enumerate(rv) if abs(lv - v) == best),
            },
        )
    return out


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

    tolerance_nearest = tolerance_and_nearest(spark, left, right)

    doc = {
        "spark": spark.version,
        "generator": "tests/scripts/gen_connect_asof_goldens.py",
        "left": {"columns": ["lk", "lt", "a"], "rows": LEFT},
        "right": {"columns": ["rk", "rt", "b"], "rows": RIGHT},
        "cases": cases,
        "overflow": overflow,
        "tolerance_nearest": tolerance_nearest,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    spark.stop()


if __name__ == "__main__":
    main(sys.argv[1])
