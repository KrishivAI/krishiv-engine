#!/usr/bin/env python3
"""Generate Spark differential fixtures for `CAST(string AS TIMESTAMP[_NTZ])` (P2-T025a).

Runs a REAL Spark (never from memory) in a throw-away container over seeded, grammar-shaped and mutated
timestamp strings: sign, year of 1 to 8 digits, month/day/hour/minute/second of 0 to 3 digits, fractions of
0 to 11 digits, separators, zone texts (`Z`, offsets, `UTC+3`, region ids), stray characters, and fixed
strings inside the DST gap and overlap of Los Angeles and Berlin. Each string is cast with `try_cast` to
`TIMESTAMP` under three session time zones (UTC, America/Los_Angeles, Europe/Berlin) and to
`TIMESTAMP_NTZ`; the result is epoch microseconds (`null` for NULL). Strings that depend on the clock
(`now`, `today`, a time of day) are left out. Usage (from repo root):

    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:4.2.0 \
        /opt/spark/bin/spark-submit --master 'local[2]' /work/gen_cast_goldens.py \
        > tests/fixtures/spark-4.2.0/cast-timestamp-differential.json

The fixture goes to stdout (the container user cannot write the mount); Spark logs go to stderr.
"""

import json
import random
import re
import sys

from pyspark.sql import SparkSession

SEED = 20261018
N = 2500
ZONES = {"utc": "UTC", "la": "America/Los_Angeles", "berlin": "Europe/Berlin"}
# fmt: off
FIXED = [
    "2024-03-10 02:30:00", "2024-03-10 01:59:59.999999", "2024-03-10 03:00:00", "2024-11-03 01:30:00",
    "2024-11-03 00:59:59", "2024-11-03 02:00:00", "2024-03-31 02:30:00", "2024-03-31 03:00:00", "2024-10-27 02:30:00",
    "2024-10-27 01:59:59", "2024-10-27 03:00:00", "1900-01-01 00:00:00", "0001-01-01", "9999-12-31 23:59:59.999999",
    "+294247-01-10 04:00:54.775807", "+294247-01-10 04:00:54.775808", "-290308-12-21 19:59:05.224192",
    "-290308-12-21 19:59:05.224191", "+294248-01-01", "-0001-01-01 00:00:00",
    "1970-01-01 00:00:00 America/Los_Angeles", "2020-01-02 03:04:05 Foo/Bar", "2020-01-02 03:04:05 UTC+8",
    "2020-01-02 03:04:05.123456789 +01:00", "2020", "2020-1", "2020-01-02T03:04:05Z", " 2020-01-02 03:04:05 ",
    "\t2020-01-02 03:04:05\n", "epoch", "EPOCH", " Epoch ",
]
ZONE_TEXTS = [
    "", "", "", "Z", " Z", "z", "+08:00", "-0530", "+8", "-8", "+19", "+18:00", "+08:30:15", "UTC", " UTC", "GMT+3",
    " UTC+01:00", "UT-2", "America/New_York", " Europe/Berlin", "Foo/Bar", "[UTC]", "+08:60", " +05:30", "Etc/GMT+5",
    "PST", "utc",
]
# fmt: on
JUNK = [" ", "-", ":", ".", "T", "Z", "+", "0", "9", "x", "\t", "é"]


def digits(rng, lo, hi):
    return "".join(rng.choice("0123456789") for _ in range(rng.randint(lo, hi)))


def number(rng, maximum, lo=1, hi=2):
    """Mostly in-range numbers of `lo..hi` digits, sometimes out of range or too long."""
    r = rng.random()
    if r < 0.8:
        return str(rng.randint(0, maximum)).zfill(rng.randint(lo, hi))
    if r < 0.9:
        return str(rng.randint(maximum + 1, maximum + 40))
    return digits(rng, 0, 4)


def make(rng):
    out = ""
    r = rng.random()
    out += "-" if r < 0.05 else ("+" if r < 0.1 else "")
    out += rng.choice(
        ["2020", "1999", "0001", "9999", "10000", "294247", "202", "12345678", digits(rng, 1, 8), "2024", "1970"]
    )
    parts = rng.choice([1, 2, 3, 3, 3, 3])
    if parts >= 2:
        out += "-" + number(rng, 12)
    if parts >= 3:
        out += "-" + number(rng, 31)
    if parts == 3 and rng.random() < 0.9:
        out += rng.choice([" ", " ", "T", "T", "  ", "t", ""])
        out += number(rng, 23)
        if rng.random() < 0.9:
            out += ":" + number(rng, 59)
            if rng.random() < 0.9:
                out += ":" + number(rng, 59)
                if rng.random() < 0.6:
                    out += "." + digits(rng, 0, 11)
                out += rng.choice(ZONE_TEXTS)
            elif rng.random() < 0.3:
                out += rng.choice(ZONE_TEXTS)
    if rng.random() < 0.15:
        out = rng.choice(["", " ", "\t"]) + out + rng.choice(["", " ", "\n"])
    if rng.random() < 0.2:  # mutate
        i = rng.randint(0, len(out))
        out = out[:i] + (rng.choice(JUNK) if rng.random() < 0.7 else "") + out[i + (1 if rng.random() < 0.5 else 0) :]
    return out


def clock_dependent(s):
    t = s.strip().lower()
    first_non_digit = re.search(r"[^0-9]", t)
    return (
        t.startswith("t")
        or t in ("now", "today", "yesterday", "tomorrow")
        or (first_non_digit is not None and first_non_digit.start() > 0 and t[first_non_digit.start()] == ":")
    )


def main() -> None:
    spark = SparkSession.builder.appName("cast-goldens").getOrCreate()
    rng = random.Random(SEED)
    strings = list(FIXED)
    seen = set(strings)
    while len(strings) < N + len(FIXED):
        s = make(rng)
        if s not in seen and not clock_dependent(s):
            seen.add(s)
            strings.append(s)
    strings = [s for s in strings if not clock_dependent(s)]
    df = spark.createDataFrame([(s,) for s in strings], "s string")
    result = {s: {} for s in strings}
    for key, zone in ZONES.items():
        spark.conf.set("spark.sql.session.timeZone", zone)
        rows = df.selectExpr("s", "unix_micros(try_cast(s as timestamp)) m").collect()
        for r in rows:
            result[r.s][key] = r.m
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    # A local date-time's microseconds: an NTZ value read in a UTC session.
    rows = df.selectExpr("s", "unix_micros(cast(try_cast(s as timestamp_ntz) as timestamp)) m").collect()
    for r in rows:
        result[r.s]["ntz"] = r.m
    cases = [{"s": s, **result[s]} for s in strings]
    json.dump(
        {"spark_version": spark.version, "seed": SEED, "zones": ZONES, "cases": cases},
        sys.stdout,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
