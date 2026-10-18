#!/usr/bin/env python3
"""Generate Spark differential fixtures for the hash kernels (P2-T027).

Runs a REAL Spark (never from memory) in a throw-away container over seeded random inputs: byte strings of
every length from 0 to 100 (all branches of Murmur3's tail and XXH64's 32/8/4/1-byte stages), strings,
every fixed-width type with edge values, wide decimals, arrays and structs with NULLs, and multi-column
chains. Each group records the column types, the rows and Spark's `hash`/`xxhash64`; the digest group
records `md5`/`sha1`/`sha2(.., bits)`/`crc32` of binary and string inputs. Binary is hex, decimals are
unscaled integers (strings), dates are days and timestamps microseconds. Usage (from repo root):

    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:4.2.0 \
        /opt/spark/bin/spark-submit --master 'local[2]' /work/gen_hash_differential.py \
        > tests/fixtures/spark-4.2.0/hash-differential.json

The fixture goes to stdout (the container user cannot write the mount); Spark logs go to stderr.
"""

import datetime
import decimal
import json
import math
import random

from pyspark.sql import SparkSession

SEED = 20261018
N = 60
ALPHABET = ["a", "b", "Z", "é", "日", "😀", " ", "\n", "\u0000", "ß"]
SHA_BITS = [0, 224, 256, 384, 512, 1, -1, 128, 255, None]


def nullable(rng, f, rate=0.1):
    return None if rng.random() < rate else f()


def text(rng):
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 12)))


def raw(rng, n):
    return bytes(rng.randrange(256) for _ in range(n))


def main() -> None:
    spark = SparkSession.builder.appName("hash-differential").getOrCreate()
    rng = random.Random(SEED)
    decimal.getcontext().prec = 80  # exact unscaled values
    groups = []

    def hash_group(name, schema, rows, enc, cols=None, types=None):
        df = spark.createDataFrame([tuple(r) for r in rows], schema)
        cols = cols or ", ".join(f.name for f in df.schema.fields)
        out = df.selectExpr(f"hash({cols}) AS h", f"xxhash64({cols}) AS x").collect()
        groups.append(
            {
                "name": name,
                "types": types or [f.dataType.simpleString() for f in df.schema.fields],
                "rows": [[enc[i](v) for i, v in enumerate(r)] for r in rows],
                "hash": [o.h for o in out],
                "xxhash64": [o.x for o in out],
            }
        )

    ident = lambda v: v  # noqa: E731
    # JSON has no NaN or infinity.
    fl = lambda v: v if v is None or math.isfinite(v) else str(v)  # noqa: E731
    hexs = lambda v: None if v is None else bytes(v).hex()  # noqa: E731
    unscaled = lambda scale: lambda v: None if v is None else str(int(v.scaleb(scale)))  # noqa: E731
    days = lambda v: None if v is None else (v - datetime.date(1970, 1, 1)).days  # noqa: E731
    epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    micros = lambda v: None if v is None else (v - epoch) // datetime.timedelta(microseconds=1)  # noqa: E731

    # Binary of every length 0..100, a few random contents each.
    rows = [(bytearray(raw(rng, n)),) for n in range(101) for _ in range(3)]
    hash_group("binary", "c0 binary", rows, [hexs])
    rows = [(nullable(rng, lambda: text(rng)),) for _ in range(200)]
    hash_group("string", "c0 string", rows, [ident])
    for name, ty, vals in [
        ("tinyint", "tinyint", [-128, -1, 0, 1, 127]),
        ("smallint", "smallint", [-32768, -1, 0, 1, 32767]),
        ("int", "int", [-2147483648, -1, 0, 1, 2147483647, 42, 1 << 20]),
        ("bigint", "bigint", [-(1 << 63), -1, 0, 1, (1 << 63) - 1, 1 << 32, 1 << 40]),
        ("float", "float", [0.0, -0.0, 1.5, -1.25, float("nan"), float("inf"), float("-inf"), 3.4028235e38, 1e-45]),
        ("double", "double", [0.0, -0.0, 1.5, -1.25, float("nan"), float("inf"), float("-inf"), 1.7976931348623157e308, 5e-324]),
        ("boolean", "boolean", [True, False]),
    ]:
        rows = [(v,) for v in vals] + [(None,)]
        hash_group(name, f"c0 {ty}", rows, [fl])
    D = decimal.Decimal
    rows = [(D(s),) for s in ["0", "1.2345", "-1.2345", "99999999999999.9999", "-99999999999999.9999", "0.0001"]] + [(None,)]
    hash_group("decimal18", "c0 decimal(18,4)", rows, [unscaled(4)])
    wide = ["0", "1.234567", "-1.234567", "255", "256", "-256", "-255", "-128", "-129", "127", "128", "32768", "-32769"]
    wide += [str(rng.randint(-(10**31), 10**31)) + ".123456" for _ in range(30)]
    wide += ["9" * 32 + ".999999", "-" + "9" * 32 + ".999999"]
    rows = [(D(s),) for s in wide] + [(None,)]
    hash_group("decimal38", "c0 decimal(38,6)", rows, [unscaled(6)])
    dates = [datetime.date(1970, 1, 1), datetime.date(2024, 2, 29), datetime.date(1969, 12, 31), datetime.date(1, 1, 1), datetime.date(9999, 12, 31)]
    hash_group("date", "c0 date", [(d,) for d in dates] + [(None,)], [days])
    ts = [datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc), datetime.datetime(2024, 2, 29, 12, 34, 56, 789012, tzinfo=datetime.timezone.utc), datetime.datetime(1969, 12, 31, 23, 59, 59, 999999, tzinfo=datetime.timezone.utc)]
    hash_group("timestamp", "c0 timestamp", [(t,) for t in ts] + [(None,)], [micros])
    # Nested values with NULLs.
    rows = [(nullable(rng, lambda: [nullable(rng, lambda: rng.randint(-5, 5)) for _ in range(rng.randint(0, 5))], 0.15),) for _ in range(80)]
    hash_group("array_int", "c0 array<int>", rows, [ident])
    rows = [(nullable(rng, lambda: [nullable(rng, lambda: text(rng), 0.2) for _ in range(rng.randint(0, 3))], 0.15),) for _ in range(60)]
    hash_group("array_string", "c0 array<string>", rows, [ident])
    rows = [(nullable(rng, lambda: (nullable(rng, lambda: rng.randint(-9, 9)), nullable(rng, lambda: text(rng))), 0.15),) for _ in range(80)]
    hash_group("struct", "c0 struct<a:int,b:string>", rows, [lambda v: None if v is None else [v[0], v[1]]])
    rows = [(nullable(rng, lambda: [(nullable(rng, lambda: rng.randint(0, 3)), nullable(rng, lambda: text(rng))) for _ in range(rng.randint(0, 3))], 0.15),) for _ in range(60)]
    hash_group("array_struct", "c0 array<struct<a:int,b:string>>", rows, [lambda v: None if v is None else [[e[0], e[1]] for e in v]])
    # Map columns need the legacy switch in Spark 4.
    spark.conf.set("spark.sql.legacy.allowHashOnMapType", "true")
    # Entries are given as key/value arrays: a Python dict would reach Spark in the JVM's hash order.
    def entries():
        return [(text(rng) + str(i), nullable(rng, lambda: rng.randint(0, 9))) for i in range(rng.randint(0, 4))]

    rows = [(nullable(rng, entries, 0.15),) for _ in range(60)]
    rows = [(None, None) if r[0] is None else ([k for k, _ in r[0]], [v for _, v in r[0]]) for r in rows]
    hash_group(
        "map",
        "k array<string>, v array<int>",
        rows,
        [ident, ident],
        cols="map_from_arrays(k, v)",
        types=["map<string,int>"],
    )
    groups[-1]["rows"] = [[None if k is None else [[a, b] for a, b in zip(k, v, strict=True)]] for k, v in rows]
    spark.conf.unset("spark.sql.legacy.allowHashOnMapType")
    # Multi-column chains.
    rows = [
        (
            nullable(rng, lambda: rng.randint(-100, 100)),
            nullable(rng, lambda: text(rng)),
            nullable(rng, lambda: rng.choice([0.0, -0.0, 1.5, float("nan"), rng.random()])),
            nullable(rng, lambda: rng.randint(-(1 << 40), 1 << 40)),
            nullable(rng, lambda: bytearray(raw(rng, rng.randint(0, 10)))),
        )
        for _ in range(200)
    ]
    hash_group("chain", "c0 int, c1 string, c2 double, c3 bigint, c4 binary", rows, [ident, ident, fl, ident, hexs])

    # Digests.
    def digests(name, ty, values, enc):
        df = spark.createDataFrame([(v,) for v in values], f"c0 {ty}")
        exprs = ["md5(c0) AS md5", "sha1(c0) AS sha1", "crc32(c0) AS crc32"]
        exprs += [f"sha2(c0, {b if b is not None else 'CAST(NULL AS INT)'}) AS sha2_{'null' if b is None else str(b).replace('-', 'm')}" for b in SHA_BITS]
        out = df.selectExpr(*exprs).collect()
        res = {"name": name, "type": ty, "rows": [enc(v) for v in values]}
        for key in out[0].asDict():
            res[key] = [o[key] for o in out]
        groups.append(res)

    digests("digest_binary", "binary", [bytearray(raw(rng, n)) for n in [0, 1, 3, 55, 56, 63, 64, 65, 111, 112, 127, 128, 129, 200]] + [None], hexs)
    digests("digest_string", "string", [text(rng) for _ in range(30)] + ["", "Spark", None], ident)
    print(json.dumps({"spark_version": spark.version, "generator": "tests/scripts/gen_hash_differential.py", "seed": SEED, "sha2_bits": SHA_BITS, "groups": groups}, ensure_ascii=False, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
