#!/usr/bin/env python3
"""Generate Spark 4.2 goldens for the `vector_*` scalar functions (P3-T076, bit-exact per profile).

Runs a REAL Spark 4.2 (never from memory) over Spark Connect. Spark's `VectorFunctionImplUtils` sums in
blocks of 8 (each block left to right, then added to the accumulator, the tail one element at a time), in
`float`, so the last bits of a 768-dimension result depend on that order. The fixture records every
result as the float's IEEE bits (a u32), so a test can compare bit for bit.

Two kinds of case:
  * `random`: a dimension, a seed and a scale. The vectors are not stored: element `i` of the vector for
    `(seed, role)` is `float32((unit(splitmix64 stream) - 0.5) * 2 * scale)`, the stream seeded with
    `seed * 2 + role` (role 0 = left, 1 = right), one step per element. The Rust test rebuilds them.
  * `edge`: the vectors are stored as float bits (a NaN is its bit pattern too); `null` is a NULL element.

For each case: `ip` (`vector_inner_product`), `l2`, `cos`, `norm1`, `norm2`, `norminf` (`vector_norm(a, d)`),
`normalize1`, `normalize2`, `normalizeinf` (a list of bits, or null), each either `{"bits": u32}`,
`{"null": true}`, `{"bits_list": [...]}` or `{"error": class, "sqlstate": s, "parameters": {...}}`.

Usage (repo root, a Spark 4.2 Connect server at `$SPARK_REMOTE`, default sc://127.0.0.1:15002):

    uv run --no-project --python 3.12 --with 'pyspark-client==4.2.0' \
        python tests/scripts/gen_vector_goldens.py > tests/fixtures/spark-4.2.0/vector-goldens.json
"""

import json
import os
import struct

from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType, StructField, StructType

MASK = (1 << 64) - 1
INF = float("inf")


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return state, z ^ (z >> 31)


def f32(x):
    """`x` rounded to float32, as a Python float."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def from_bits(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def random_vector(dim, seed, role, scale):
    state = seed * 2 + role
    out = []
    for _ in range(dim):
        state, z = splitmix64(state)
        u = (z >> 11) / float(1 << 53)
        out.append(f32((u - 0.5) * 2.0 * scale))
    return out


RANDOM_CASES = (
    [(d, 1, 1.0) for d in (1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 64, 100)]
    + [(d, 7, 1.0) for d in (128, 384, 768, 1024, 1536)]
    + [(768, 11, 1000.0), (768, 12, 0.001), (768, 13, 1e19), (768, 14, 1e-19), (16, 15, 1e20), (16, 16, 1e-22)]
)

# (name, left, right) in float bits; None is a NULL element.
EDGE_CASES = [
    ("empty", [], []),
    ("empty-vs-one", [], [0x3F800000]),
    ("zeros", [0, 0, 0], [0, 0, 0]),
    ("nan-first", [0x7FC00000, 0x3F800000], [0x3F800000, 0x3F800000]),
    ("nan-middle-of-nine", [0x3F800000] * 4 + [0x7FC00000] + [0x3F800000] * 4, [0x3F800000] * 9),
    ("nan-tail", [0x3F800000, 0x7FC00000], [0x3F800000, 0x40000000]),
    ("inf-and-one", [0x7F800000, 0x3F800000], [0x3F800000, 0x3F800000]),
    ("neg-inf", [0xFF800000, 0x3F800000], [0x3F800000, 0x3F800000]),
    ("denormal-pair", [0x00000400, 0x00000400], [0x00000400, 0x00000400]),
    ("tiny-1e-20", [bits(1e-20)] * 2, [bits(1e-20)] * 2),
    ("tiny-1e-30", [bits(1e-30)] * 2, [bits(1e-30)] * 2),
    ("huge-1e30", [bits(1e30)] * 2, [bits(1e30)] * 2),
    ("huge-1e20", [bits(1e20)] * 2, [bits(1e20), 0x3F800000]),
    ("null-element", [0x3F800000, None, 0x40000000], [0x3F800000, 0x40000000, 0x40400000]),
    ("null-element-late", [0x3F800000] * 8 + [None], [0x3F800000] * 9),
    ("null-element-vs-length", [0x3F800000, None], [0x3F800000]),
    ("null-in-right", [0x3F800000, 0x40000000], [None, 0x40000000]),
    ("length-mismatch", [0x3F800000, 0x40000000], [0x3F800000]),
    ("negative-zero", [0x80000000, 0x80000000], [0x00000000, 0x80000000]),
]

DEGREES = (("1", 1.0), ("2", 2.0), ("inf", INF))


def error_of(e):
    err = {"error": getattr(e, "getErrorClass", lambda: None)() or type(e).__name__}
    err["sqlstate"] = getattr(e, "getSqlState", lambda: None)()
    params = getattr(e, "getMessageParameters", lambda: None)()
    err["parameters"] = dict(params) if params else {}
    return err


def run(spark, sql, a, b):
    schema = StructType(
        [StructField("a", ArrayType(FloatType(), True), True), StructField("b", ArrayType(FloatType(), True), True)]
    )
    spark.createDataFrame([(a, b)], schema).createOrReplaceTempView("t")
    try:
        (value,) = spark.sql(sql).collect()[0]
    except Exception as e:  # every Spark failure is recorded as a cell
        return error_of(e)
    if value is None:
        return {"null": True}
    if isinstance(value, list):
        return {"bits_list": [bits(x) for x in value]}
    return {"bits": bits(value)}


def cell(spark, a, b):
    out = {}
    out["ip"] = run(spark, "select vector_inner_product(a, b) from t", a, b)
    out["l2"] = run(spark, "select vector_l2_distance(a, b) from t", a, b)
    out["cos"] = run(spark, "select vector_cosine_similarity(a, b) from t", a, b)
    for name, degree in DEGREES:
        lit = "float('inf')" if degree == INF else f"float({degree})"
        out[f"norm{name}"] = run(spark, f"select vector_norm(a, {lit}) from t", a, b)
        out[f"normalize{name}"] = run(spark, f"select vector_normalize(a, {lit}) from t", a, b)
    return out


def main():
    remote = os.environ.get("SPARK_REMOTE", "sc://127.0.0.1:15002")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    cases = []
    for dim, seed, scale in RANDOM_CASES:
        a = random_vector(dim, seed, 0, scale)
        b = random_vector(dim, seed, 1, scale)
        cases.append({"kind": "random", "dim": dim, "seed": seed, "scale": scale, "result": cell(spark, a, b)})
    for name, left, right in EDGE_CASES:
        a = None if left is None else [None if x is None else from_bits(x) for x in left]
        b = None if right is None else [None if x is None else from_bits(x) for x in right]
        cases.append({"kind": "edge", "name": name, "a": left, "b": right, "result": cell(spark, a, b)})
    # One case per line keeps the fixture reviewable.
    print(f'{{"spark": {json.dumps(spark.version)},\n"cases": [')
    print(",\n".join(json.dumps(c, separators=(",", ":")) for c in cases))
    print("]}")


if __name__ == "__main__":
    main()
