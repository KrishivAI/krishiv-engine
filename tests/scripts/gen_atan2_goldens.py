#!/usr/bin/env python3
"""Generate the `atan2` golden fixture (float-sweep failure group (a), Comet checks).

Every result comes from a REAL Spark 4.2 run. Spark computes `atan2(y + 0.0, x + 0.0)` through
`java.lang.Math.atan2`, i.e. fdlibm. The inputs are a seeded mix of magnitudes, signs, ratios that
overflow or underflow (`|y/x|` beyond 2**60), zeros, infinities and NaN. Values are recorded as the
text Python's `repr` gives for the double (`nan`, `inf`, `-inf`), which parses back exactly: one
case per line, `y<TAB>x<TAB>result`.

    SPARK_HOME=<empty dir> uv run --no-project --python 3.12 --with 'pyspark-client==4.2.0' \\
      python tests/scripts/gen_atan2_goldens.py --remote sc://127.0.0.1:15012 \\
      --out tests/fixtures/spark-4.2.0/atan2-goldens.tsv
"""

from __future__ import annotations

import argparse
import math
import random
import struct
from pathlib import Path

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_atan2_goldens.py"
SPECIAL = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    2.0,
    math.inf,
    -math.inf,
    math.nan,
    5e-324,
    -5e-324,
    1.7976931348623157e308,
    -1.7976931348623157e308,
    2.2250738585072014e-308,
    1e-300,
    1e300,
]


def inputs() -> list[tuple[float, float]]:
    rng = random.Random(20261105)
    pairs = [(y, x) for y in SPECIAL for x in SPECIAL]

    def mag() -> float:
        bits = rng.getrandbits(64)
        v = struct.unpack("<d", struct.pack("<Q", bits))[0]
        return v if math.isfinite(v) else 1.0

    for _ in range(1500):
        kind = rng.randrange(4)
        if kind == 0:
            pairs.append((rng.uniform(-10, 10), rng.uniform(-10, 10)))
        elif kind == 1:
            pairs.append(
                (rng.uniform(-1, 1) * 10 ** rng.randint(-30, 30), rng.uniform(-1, 1) * 10 ** rng.randint(-30, 30))
            )
        elif kind == 2:
            pairs.append((mag(), mag()))
        else:
            pairs.append(
                (
                    rng.choice([-1, 1]) * 2.0 ** rng.randint(-1074, 1023),
                    rng.choice([-1, 1]) * 2.0 ** rng.randint(-1074, 1023),
                )
            )
    return pairs


def lit(v: float) -> str:
    if math.isnan(v):
        return "CAST('NaN' AS DOUBLE)"
    if math.isinf(v):
        return "CAST('Infinity' AS DOUBLE)" if v > 0 else "CAST('-Infinity' AS DOUBLE)"
    return f"CAST('{v!r}' AS DOUBLE)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.remote(a.remote).create()
    pairs = inputs()
    rows = ", ".join(f"({i}, {lit(y)}, {lit(x)})" for i, (y, x) in enumerate(pairs))
    got = spark.sql(f"SELECT i, atan2(y, x) FROM VALUES {rows} AS t(i, y, x) ORDER BY i").collect()
    spark.stop()
    cases = [[repr(pairs[r[0]][0]), repr(pairs[r[0]][1]), repr(r[1])] for r in got]
    text = "".join("\t".join(c) + "\n" for c in cases)
    Path(a.out).write_text(text)
    print("wrote", a.out, len(cases), "cases")


if __name__ == "__main__":
    main()
