#!/usr/bin/env python3
"""Generate Spark differential fixtures for the math kernels (P2-T026).

Runs a REAL Spark (never from memory) in a throw-away container over seeded random inputs and records
Spark's result for every row: the double functions (`sqrt` .. `signum`, `pow`, `atan2`, `hypot`, `log`),
`round`/`bround` and `ceil`/`floor` on doubles, floats, integral types and decimals, `pmod` and
`greatest`/`least`. Each group has the SQL template (columns c0, c1, ...), the column types, the rows
and the results with Spark's result type. Doubles are JSON numbers, or "nan"/"inf"/"-inf"; decimals are
unscaled integers (strings); binary is hex; dates are days. Usage (from repo root):

    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:4.2.0 \
        /opt/spark/bin/spark-submit --master 'local[2]' /work/gen_math_goldens.py \
        > tests/fixtures/spark-4.2.0/math-differential.json

With group names as an argument (`... gen_math_goldens.py asinh,acosh`) only those groups are recorded;
the other draws still happen, so the inputs are unchanged. Spark 4.0 and 4.1 differ from 4.2 in `asinh`
and `acosh`, so their fixtures (`tests/fixtures/spark-4.{0,1}.0/math-differential.json`) hold just those.

The fixture goes to stdout (the container user cannot write the mount); Spark logs go to stderr.
"""

import decimal
import json
import math
import random
import sys

from pyspark.sql import SparkSession

SEED = 20261018
N = 300

UNARY = [
    "sqrt", "cbrt", "exp", "expm1", "ln", "log10", "log2", "log1p", "sin", "cos", "tan", "asin", "acos",
    "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "cot", "csc", "sec", "degrees", "radians",
    "rint", "signum",
]  # fmt: skip
BINARY = ["pow", "atan2", "hypot", "log"]
DOUBLE_EDGES = [
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 10.0, 100.0, 1e-300, 1e300, 710.0, -745.0, 1e-5, math.pi,
    math.pi / 2, -math.pi, 1e18, 123456789.0, float("nan"), float("inf"), float("-inf"),
]  # fmt: skip


def fl(v):
    if v is None or (isinstance(v, float) and math.isfinite(v)):
        return v
    if isinstance(v, float):
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    return v


def rand_double(rng):
    r = rng.random()
    if r < 0.12:
        return rng.choice(DOUBLE_EDGES)
    if r < 0.4:
        return rng.uniform(-10, 10)
    if r < 0.5:
        return rng.uniform(-1, 1)
    mag = 10 ** rng.uniform(-20, 20)
    return mag if rng.random() < 0.5 else -mag


def main() -> None:
    spark = SparkSession.builder.appName("math-goldens").getOrCreate()
    spark.conf.set("spark.sql.ansi.enabled", "false")
    decimal.getcontext().prec = 80
    rng = random.Random(SEED)
    groups = []

    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None

    def group(name, template, schema, rows, enc, dec=None):
        """`template` uses the columns c0, c1, ...; `enc` encodes each input column, `dec` the result."""
        if only is not None and name not in only:
            return
        df = spark.createDataFrame([tuple(r) for r in rows], schema)
        dec = dec or fl

        def error_class(e):
            return getattr(e, "getCondition", lambda: None)() or "UNKNOWN"

        try:
            res = df.selectExpr(f"{template} AS r", f"typeof({template}) AS t").collect()
            out = [dec(r.r) for r in res]
            types = {r.t for r in res}
        except Exception:  # find the failing rows one by one and record their error class
            out, types = [], set()
            for row in rows:
                one = spark.createDataFrame([tuple(row)], schema)
                try:
                    r = one.selectExpr(f"{template} AS r", f"typeof({template}) AS t").collect()[0]
                    out.append(dec(r.r))
                    types.add(r.t)
                except Exception as e:
                    out.append({"error": error_class(e)})
        assert len(types) == 1, (name, types)
        groups.append(
            {
                "name": name,
                "sql": template,
                "types": [f.dataType.simpleString() for f in df.schema.fields],
                "rows": [[enc[i](v) for i, v in enumerate(r)] for r in rows],
                "out": out,
                "out_type": types.pop(),
            }
        )

    ident = lambda v: v  # noqa: E731
    unscaled = lambda scale: lambda v: None if v is None else str(int(v.scaleb(scale)))  # noqa: E731

    def dec_out(v):
        if v is None:
            return None
        if isinstance(v, decimal.Decimal):
            return str(int(v.scaleb(-v.as_tuple().exponent)))
        return fl(v) if isinstance(v, float) else v

    for f in UNARY:
        group(f, f"{f}(c0)", "c0 double", [(nullable(rng, lambda: rand_double(rng), 0.05),) for _ in range(N)], [fl])
    for f in BINARY:
        group(
            f,
            f"{f}(c0, c1)",
            "c0 double, c1 double",
            [
                (nullable(rng, lambda: rand_double(rng), 0.05), nullable(rng, lambda: rand_double(rng), 0.05))
                for _ in range(N)
            ],
            [fl, fl],
        )
    # Mixed-width integral arguments reach the double functions through casts: sign of every type.
    group("sign_int", "sign(c0)", "c0 int", [(v,) for v in [-5, 0, 7, None]], [ident])
    group(
        "sign_decimal",
        "sign(c0)",
        "c0 decimal(10,2)",
        [(decimal.Decimal(v),) for v in ["-1.5", "0", "9.99"]] + [(None,)],
        [unscaled(2)],
    )

    # round / bround
    scales = [-4, -1, 0, 1, 2, 3, 10, 20]
    for fn in ("round", "bround"):
        for sc in scales:
            vals = [
                rng.choice([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.675, 1.005, 8.345, 0.285, 1e15 + 0.5]) for _ in range(30)
            ]
            vals += [rand_double(rng) for _ in range(120)]
            group(f"{fn}_double_{sc}", f"{fn}(c0, {sc})", "c0 double", [(v,) for v in vals] + [(None,)], [fl])
            fvals = [rng.choice([0.5, 1.5, 2.5, 2.675, 8.345, 0.285]) for _ in range(20)] + [
                rng.uniform(-100, 100) for _ in range(80)
            ]
            group(f"{fn}_float_{sc}", f"{fn}(c0, {sc})", "c0 float", [(v,) for v in fvals] + [(None,)], [fl])
            for ty, lo, hi in [
                ("tinyint", -100, 100),
                ("smallint", -30000, 30000),
                ("int", -2_000_000_000, 2_000_000_000),
                ("bigint", -(10**18), 10**18),
            ]:
                vals = [rng.randint(lo, hi) for _ in range(60)] + [
                    rng.choice([5, 15, 25, 35, -5, -15, -25, 125, -125, 0]) for _ in range(40)
                ]
                group(
                    f"{fn}_{ty}_{sc}", f"{fn}(c0, {sc})", f"c0 {ty}", [(v,) for v in vals] + [(None,)], [ident], dec_out
                )
            for p, s in [(10, 2), (5, 5), (20, 10), (38, 18), (38, 0), (3, 1), (18, 6)]:
                digits = p
                vals = []
                for _ in range(50):
                    u = rng.randint(-(10**digits) + 1, 10**digits - 1)
                    if rng.random() < 0.3:
                        u = max(min((u // 10) * 10 + rng.choice([5, 5, 0]), 10**digits - 1), -(10**digits) + 1)  # ties
                    vals.append(decimal.Decimal(u).scaleb(-s))
                group(
                    f"{fn}_decimal_{p}_{s}_{sc}",
                    f"{fn}(c0, {sc})",
                    f"c0 decimal({p},{s})",
                    [(v,) for v in vals] + [(None,)],
                    [unscaled(s)],
                    dec_out,
                )
    # ceil / floor
    for fn in ("ceil", "floor"):
        vals = [rand_double(rng) for _ in range(200)] + [
            1e19,
            -1e19,
            0.5,
            -0.5,
            2.0,
            -2.0,
            float("nan"),
            float("inf"),
            float("-inf"),
            None,
        ]
        group(f"{fn}_double", f"{fn}(c0)", "c0 double", [(v,) for v in vals], [fl])
        group(f"{fn}_float", f"{fn}(c0)", "c0 float", [(rng.uniform(-100, 100),) for _ in range(60)] + [(None,)], [fl])
        for ty in ["tinyint", "int", "bigint"]:
            group(
                f"{fn}_{ty}",
                f"{fn}(c0)",
                f"c0 {ty}",
                [(v,) for v in [0, 1, -1, 5, -5, 100]] + [(None,)],
                [ident],
                dec_out,
            )
        for p, s in [(10, 2), (5, 5), (38, 18), (38, 0), (3, 1), (20, 10)]:
            vals = [decimal.Decimal(rng.randint(-(10**p) + 1, 10**p - 1)).scaleb(-s) for _ in range(80)]
            for sc in [None, -2, 0, 1, 3, 20]:
                template = f"{fn}(c0)" if sc is None else f"{fn}(c0, {sc})"
                group(
                    f"{fn}_decimal_{p}_{s}_{sc}",
                    template,
                    f"c0 decimal({p},{s})",
                    [(v,) for v in vals] + [(None,)],
                    [unscaled(s)],
                    dec_out,
                )
        for sc in [-2, 0, 1, 3]:
            group(
                f"{fn}_int_{sc}",
                f"{fn}(c0, {sc})",
                "c0 int",
                [(v,) for v in [0, 1, -1, 15, -15, 12345, -12345, 2147483647, -2147483648]] + [(None,)],
                [ident],
                dec_out,
            )
            group(
                f"{fn}_double_{sc}",
                f"{fn}(c0, {sc})",
                "c0 double",
                [(rng.uniform(-100, 100),) for _ in range(60)] + [(None,)],
                [fl],
                dec_out,
            )
    print(
        json.dumps(
            {
                "spark_version": spark.version,
                "generator": "tests/scripts/gen_math_goldens.py",
                "seed": SEED,
                "groups": groups,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def nullable(rng, f, rate=0.1):
    return None if rng.random() < rate else f()


if __name__ == "__main__":
    main()
