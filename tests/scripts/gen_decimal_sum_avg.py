#!/usr/bin/env python3
"""Generate Spark expectations for grouped decimal `sum`/`avg` (P2-T160).

Runs a REAL Spark 4.2 (never from memory) over Spark Connect. Each case is a column of one decimal
type, rows in a fixed order (`range(0, n, 1, partitions)`, not a local relation, whose grouped
aggregate tolerates a transient overflow), grouped by `id % groups`, with ANSI on and off. With
several partitions, each partition's partial buffers are merged; the multi-partition cases are
chosen so the outcome does not depend on merge order, except R6 (`M | 1 | -5`), which overflows in
Spark on the first merge. It records each group's result text and type (ordered by group), or the
error condition. The Krishiv test folds the same rows through its accumulators: per partition, then
merging the partial states in partition order (the per-batch proof path included). Usage (from
repo root, with a
Spark 4.2 Connect server at `$SPARK_REMOTE`, default `sc://127.0.0.1:15002`):

    uv run --no-project --python 3.12 --with 'pyspark-client==4.2.0' \
        python tests/scripts/gen_decimal_sum_avg.py > tests/fixtures/spark-4.2.0/decimal-sum-avg.json
"""

import itertools
import json
import os
import random

from pyspark.errors import PySparkException
from pyspark.sql import SparkSession

random.seed(160)
M = 10**38 - 1
# (precision, scale, unscaled values in row order); one group and one partition.
SINGLE = [
    (1, 0, [-1, -2, -2]),  # HALF_UP away from zero: -1.6667
    (5, 0, [1] + [0] * 31),  # 1/32 = 0.03125: an exact half point at scale 4
    (5, 0, [-1] + [0] * 31),
    (15, 2, [random.randrange(-(10**15) + 1, 10**15) for _ in range(7)]),
    (15, 2, [random.randrange(90000, 10500000) for _ in range(13)]),
    (10, 5, [1, 2, 2]),
    (18, 0, [10**18 - 1, 10**18 - 1, 10**18 - 2]),
    (18, 14, [1, 0]),
    (28, 0, [10**28 - 1] * 3 + [5]),
    (30, 5, [1, 2, 3, 4, 5, 6]),  # avg scale 9
    (28, 18, [1, 2, 3, 4, 5, 6]),  # avg scale 22: the digit loop
    (38, 0, [10**33, 10**33 + 1, 7]),  # avg numerator beyond u128
    (38, 0, [M, M]),
    (38, 0, [M, 1, -5]),  # transient overflow: raises
    (38, 0, [M, -5, 1]),  # no prefix leaves the bound
    (34, 0, [10**34 - 1, 10**34 - 1, 1]),
    (15, 15, [random.randrange(-(10**15) + 1, 10**15) for _ in range(9)]),
    (38, 10, [random.randrange(-(10**30), 10**30) for _ in range(5)]),
]
for _ in range(10):
    p = random.randrange(1, 39)
    s = random.randrange(0, p + 1)
    SINGLE.append((p, s, [random.randrange(-(10**p) + 1, 10**p) for _ in range(random.randrange(1, 9))]))

# (precision, scale, values, groups, partitions): group of row i is i % groups; partitions split the
# rows into contiguous ranges. Spark merges partial buffers in shuffle-fetch order, so every case
# must have one outcome whatever the merge order (checked by `order_independent`); merge-order
# shapes such as R6 (`M | 1 | -5`) are covered by the Krishiv accumulator tests instead.
MULTI = [
    (38, 0, [M, 1], 1, 2),  # the total leaves the bound: overflows in any order
    (38, 0, [M - 10, -5, 3], 1, 3),  # no subset of partials leaves the bound
    (38, 0, [M, M, -M, -M], 1, 2),  # each partial overflows
    (38, 0, [M - 10, 7, -(M - 10), 9, 3, -2], 2, 3),  # group 0 cancels; group 1: 7, 9, -2
    (38, 0, [M, M, 1, 2], 2, 2),  # both groups' totals leave the bound
    (15, 2, [random.randrange(90000, 10500000) for _ in range(40)], 4, 3),
    (5, 0, [1] + [0] * 63, 2, 4),  # half points in group 0
    (18, 0, [10**18 - 1] * 24, 3, 4),
    (28, 18, [random.randrange(-(10**28) + 1, 10**28) for _ in range(30)], 5, 3),
]


def order_independent(p, vals, groups, parts):
    """Whether every group's outcome (overflow or not) is the same for every merge order.

    Rows of one partition are folded in order; a partial that overflows stays overflowed. The
    partials are then merged in each possible order with the bound checked after every merge.
    """
    bound = 10 ** min(38, p + 10)
    n = len(vals)
    for g in range(groups):
        partials = []
        for k in range(parts):
            acc, overflowed = 0, False
            for i in range(k * n // parts, (k + 1) * n // parts):
                if i % groups == g:
                    acc += vals[i]
                    overflowed |= abs(acc) >= bound
            partials.append(None if overflowed else acc)
        outcomes = set()
        for order in itertools.permutations(partials):
            acc, overflowed = 0, False
            for x in order:
                overflowed |= x is None or abs(acc + x) >= bound
                acc += x or 0
            outcomes.add(overflowed)
        if len(outcomes) > 1:
            return False
    return True


while len(MULTI) < 15:
    p = random.randrange(1, 39)
    s = random.randrange(0, p + 1)
    vals = [random.randrange(-(10**p) + 1, 10**p) for _ in range(random.randrange(4, 40))]
    groups, parts = random.randrange(1, 5), random.randrange(2, 5)
    if order_independent(p, vals, groups, parts):
        MULTI.append((p, s, vals, groups, parts))
assert all(order_independent(p, v, g, k) for p, _, v, g, k in MULTI)
CASES = [(p, s, vals, 1, 1) for p, s, vals in SINGLE] + MULTI


def text(u, s):
    neg = "-" if u < 0 else ""
    d = str(abs(u)).rjust(s + 1, "0")
    return neg + (d if s == 0 else d[:-s] + "." + d[-s:])


def main():
    remote = os.environ.get("SPARK_REMOTE", "sc://127.0.0.1:15002")
    out = []
    for ansi in [True, False]:
        for fn in ["sum", "avg"]:
            for p, s, vals, groups, parts in CASES:
                spark = SparkSession.builder.remote(remote).getOrCreate()
                spark.conf.set("spark.sql.ansi.enabled", str(ansi).lower())
                when = " ".join(f"WHEN {i} THEN CAST('{text(v, s)}' AS DECIMAL({p},{s}))" for i, v in enumerate(vals))
                spark.range(0, len(vals), 1, parts).selectExpr(
                    f"id % {groups} AS g", f"CASE id {when} END AS v"
                ).createOrReplaceTempView("t")
                case = {
                    "ansi": ansi,
                    "fn": fn,
                    "type": [p, s],
                    "groups": groups,
                    "partitions": parts,
                    "values": [str(v) for v in vals],
                }
                try:
                    rows = spark.sql(
                        f"SELECT g, {fn}(v) AS a, typeof({fn}(v)) AS t FROM t GROUP BY g ORDER BY g"
                    ).collect()
                    case["type_out"] = rows[0].t
                    case["value"] = [None if r.a is None else format(r.a, "f") for r in rows]
                except PySparkException as e:
                    case["error"] = e.getCondition()
                out.append(case)
                spark.stop()
    print(json.dumps(out, indent=1))


main()
