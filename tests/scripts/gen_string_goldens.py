#!/usr/bin/env python3
"""Generate Spark differential fixtures for the string kernels (P2-T023a).

Runs a REAL Spark (never from memory) in a throw-away container over seeded random inputs drawn from an
alphabet of awkward characters (accents, astral code points, newlines, LIKE wildcards, escapes, sigma,
dotted capital I): `substring`, `lpad`/`rpad`, the `trim` family (both argument orders), `replace`, `like`
(with and without ESCAPE), `split` (literal delimiters), `startswith`/`endswith`/`contains`, `concat`,
`length`, `upper`/`lower`. Each case records the arguments and Spark's result (`null` for SQL NULL).
Usage (from repo root):

    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:4.2.0 \
        /opt/spark/bin/spark-submit --master 'local[2]' /work/gen_string_goldens.py \
        > tests/fixtures/spark-4.2.0/string-differential.json

The fixture goes to stdout (the container user cannot write the mount); Spark logs go to stderr.
"""

import json
import random
import sys

from pyspark.sql import SparkSession

SEED = 20261018
ALPHABET = ["a", "b", "c", "A", "B", "é", "É", "😀", "日", " ", "\n", "%", "_", "\\", "ß", "ǆ", "Σ", "İ", "x"]
INT_EDGES = [-2147483648, -10, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7, 10, 2147483647]
ESCAPES = ["\\", "$", "!"]
DELIMITERS = [",", "ab", "é", "😀", " ", "%", "_", "", "\\|", "\\.", "aa", "x", "日", "\n"]
N = 300


def text(rng: random.Random, null_rate: float = 0.06, alphabet=None, maxlen: int = 8):
    if rng.random() < null_rate:
        return None
    return "".join(rng.choice(alphabet or ALPHABET) for _ in range(rng.randint(0, maxlen)))


def integer(rng: random.Random, null_rate: float = 0.06):
    if rng.random() < null_rate:
        return None
    return rng.choice(INT_EDGES) if rng.random() < 0.5 else rng.randint(-12, 14)


def pad_len(rng: random.Random, null_rate: float = 0.06):
    # Spark raises an integer overflow for lengths near INT_MIN and runs out of memory near INT_MAX.
    if rng.random() < null_rate:
        return None
    return rng.choice([-1000, 1000]) if rng.random() < 0.1 else rng.randint(-12, 14)


def like_pattern(rng: random.Random, esc: str):
    parts = []
    for _ in range(rng.randint(0, 6)):
        r = rng.random()
        if r < 0.25:
            parts.append("%")
        elif r < 0.45:
            parts.append("_")
        elif r < 0.55:
            parts.append(esc + rng.choice(["%", "_", esc]))
        else:
            c = rng.choice([a for a in ALPHABET if a != esc])
            parts.append(c if c not in "%_" else "a")
    return "".join(parts)


def main() -> None:
    spark = SparkSession.builder.appName("string-goldens").getOrCreate()
    rng = random.Random(SEED)
    cases = []

    def run(name, template, rows, types):
        """`template` uses c0, c1, ... for the columns of each row."""
        schema = ", ".join(f"c{i} {t}" for i, t in enumerate(types))
        df = spark.createDataFrame([tuple(r) for r in rows], schema).selectExpr(template + " AS r")
        out = [r.r for r in df.collect()]
        assert len(out) == len(rows)
        for args, res in zip(rows, out):
            if isinstance(res, list):
                res = list(res)
            cases.append({"fn": name, "args": list(args), "out": res})

    s = lambda: text(rng)
    run("upper", "upper(c0)", [(s(),) for _ in range(N)], ["string"])
    run("lower", "lower(c0)", [(s(),) for _ in range(N)], ["string"])
    run("char_length", "length(c0)", [(s(),) for _ in range(N)], ["string"])
    run("substring2", "substring(c0, c1)", [(s(), integer(rng)) for _ in range(N)], ["string", "int"])
    run("substring3", "substring(c0, c1, c2)", [(s(), integer(rng), integer(rng)) for _ in range(N)], ["string", "int", "int"])
    for side in ("lpad", "rpad"):
        run(side + "2", f"{side}(c0, c1)", [(s(), pad_len(rng)) for _ in range(N)], ["string", "int"])
        run(side + "3", f"{side}(c0, c1, c2)", [(s(), pad_len(rng), s()) for _ in range(N)], ["string", "int", "string"])
    run("trim1", "trim(c0)", [(s(),) for _ in range(N)], ["string"])
    run("btrim2", "btrim(c0, c1)", [(s(), text(rng, 0.06, ALPHABET[:6] + [" ", "x"], 4)) for _ in range(N)], ["string", "string"])
    for f in ("trim", "ltrim", "rtrim"):
        # Two-parameter call: trim string first.
        run(f + "2", f"{f}(c0, c1)", [(text(rng, 0.06, ALPHABET[:6] + [" ", "x"], 4), s()) for _ in range(N)], ["string", "string"])
    small = ["a", "b", "ab", "é", "😀", "%", "_", " "]
    run("replace3", "replace(c0, c1, c2)",
        [(s(), text(rng, 0.06, small[:5], 3), text(rng, 0.06, small, 3)) for _ in range(N)], ["string", "string", "string"])
    run("like2", "like(c0, c1)", [(s(), like_pattern(rng, "\\")) for _ in range(N)], ["string", "string"])
    for esc in ESCAPES:
        # The escape must be a literal in the expression.
        lit = "'" + esc.replace("\\", "\\\\") + "'"
        rows = [(s(), like_pattern(rng, esc), esc) for _ in range(N // 3)]
        run("like3", f"like(c0, c1, {lit})", rows, ["string", "string", "string"])
    for f in ("startswith", "endswith", "contains"):
        run(f, f"{f}(c0, c1)", [(s(), text(rng, 0.06, small[:5], 3)) for _ in range(N)], ["string", "string"])
    run("concat3", "concat(c0, c1, c2)", [(s(), s(), s()) for _ in range(N)], ["string", "string", "string"])
    run("split3", "split(c0, c1, c2)",
        [(s(), rng.choice(DELIMITERS), integer(rng)) for _ in range(N)], ["string", "string", "int"])

    json.dump(
        {"spark_version": spark.version, "seed": SEED, "cases": cases},
        sys.stdout,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
