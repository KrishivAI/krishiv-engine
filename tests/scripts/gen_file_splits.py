#!/usr/bin/env python3
"""Generate the Spark file-split planning fixture (P2-T117).

Every expectation comes from a REAL Spark 4.2 run (the `krishiv-bench` container), never from docs or
memory: for each case, sparse files of the given sizes are created inside the container and planned by
Spark's own `FileSourceScanExec` over the `text` source (`splits/plan_splits.scala`, run through
`spark-shell`). The fixture records the listing order Spark saw and the file partitions it produced
(file name, start, length per split), so the Krishiv split planner can be checked for exact parity.
The same script runs the schema-mapping probes (case-insensitive resolution, duplicate fields, type
widening and mismatch, missing columns over Parquet) and records Spark's rows or error conditions.

The case matrix covers file-count/size shapes (empty, single large, many small, mixed around the split
and open-cost boundaries, equal-size ties, splittable and non-splittable codec suffixes) crossed with
`spark.sql.files.maxPartitionBytes`, `openCostInBytes`, `minPartitionNum`, `maxPartitionNum` and
`spark.sql.leafNodeDefaultParallelism`. Output is deterministic and contains no local paths or hostnames.

Usage (from repo root; take the bench lock, start Spark, stop it afterwards; see bench/README.md):

    docker compose -f bench/baselines/docker-compose.yml up -d spark
    flock "$KRISHIV_DATA/.bench.lock" nice -n 19 \
      python3 tests/scripts/gen_file_splits.py --out tests/fixtures/spark-4.2.0/file-splits.json
    docker compose -f bench/baselines/docker-compose.yml stop spark
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gen_function_matrix import container_image, spark_shell

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_file_splits.py"
KIB = 1024
MIB = 1024 * KIB
GIB = 1024 * MIB

SHAPES: dict[str, list[tuple[str, int]]] = {
    "empty": [("e0.txt", 0)],
    "single_large": [("big.txt", GIB + 12345)],
    "many_small": [(f"s{i:02}.txt", (i + 1) * KIB) for i in range(10)],
    "mixed": [
        ("m0.txt", 500 * MIB),
        ("m1.txt", 300 * MIB + 7),
        ("m2.txt", 128 * MIB),
        ("m3.txt", 128 * MIB + 1),
        ("m4.txt", 4 * MIB),
        ("m5.txt", 4 * MIB - 1),
        ("m6.txt", 1),
        ("m7.txt", 0),
        ("m8.txt", 0),
    ],
    "uniform_200": [(f"u{i:03}.txt", MIB) for i in range(200)],
    "ties": [(f"t{i}.txt", 50 * MIB) for i in range(7)] + [("t7.txt", 3 * MIB)],
    "codecs": [
        ("c.txt", 300 * MIB),
        ("c.gz", 300 * MIB),
        ("c.bz2", 300 * MIB),
        ("c.deflate", 300 * MIB),
        ("c.lz4", 300 * MIB),
        ("c.snappy", 300 * MIB),
        ("c.zst", 300 * MIB),
    ],
}

CONFS: dict[str, dict[str, str]] = {
    "default": {},
    "max64m": {"spark.sql.files.maxPartitionBytes": str(64 * MIB)},
    "max1m": {"spark.sql.files.maxPartitionBytes": str(MIB)},
    "open0": {"spark.sql.files.openCostInBytes": "0"},
    "open1m": {"spark.sql.files.openCostInBytes": str(MIB)},
    "open64m": {"spark.sql.files.openCostInBytes": str(64 * MIB)},
    "min8": {"spark.sql.files.minPartitionNum": "8"},
    "min100": {"spark.sql.files.minPartitionNum": "100"},
    "leaf4": {"spark.sql.leafNodeDefaultParallelism": "4"},
    "maxnum2": {"spark.sql.files.maxPartitionNum": "2"},
    "maxnum10_min100": {"spark.sql.files.maxPartitionNum": "10", "spark.sql.files.minPartitionNum": "100"},
    "max1m_open0_min8": {
        "spark.sql.files.maxPartitionBytes": str(MIB),
        "spark.sql.files.openCostInBytes": "0",
        "spark.sql.files.minPartitionNum": "8",
    },
}

LOCAL_PATH = re.compile(r"/home/|/Users/|[A-Za-z]:\\\\")


def cases() -> list[dict]:
    out = []
    for shape, files in SHAPES.items():
        for conf_name, conf in CONFS.items():
            out.append(
                {
                    "id": f"{shape}__{conf_name}",
                    "files": [{"name": n, "size": s} for n, s in files],
                    "conf": conf,
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="krishiv-bench-spark")
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/file-splits.json")
    args = ap.parse_args()

    inputs = cases()
    data = "".join(json.dumps(c) + "\n" for c in inputs)
    script = (Path(__file__).parent / "splits" / "plan_splits.scala").read_text(encoding="utf-8")
    lines = spark_shell(args.container, script, data)
    version = next(line.split(" ", 1)[1] for line in lines if line.startswith("@@VERSION "))
    if f"spark-{version}" not in args.out:
        raise SystemExit(f"Spark {version} does not match the fixture path {args.out}")
    parallelism = int(next(line.split(" ", 1)[1] for line in lines if line.startswith("@@PARALLELISM ")))
    planned = {}
    for line in lines:
        if line.startswith("@@S "):
            s = json.loads(line[4:])
            planned[s["id"]] = s
    probes = [json.loads(line[4:]) for line in lines if line.startswith("@@E ")]
    out_cases = []
    for c in inputs:
        s = planned.get(c["id"])
        if s is None or "error" in s:
            raise SystemExit(f"case {c['id']} failed in Spark: {s}")
        out_cases.append(
            {
                "id": c["id"],
                "conf": c["conf"],
                "listing": s["listing"],
                "partitions": s["partitions"],
            }
        )
    doc = {
        "format_version": FORMAT_VERSION,
        "spark_version": version,
        "image": container_image(args.container),
        "generator": GENERATOR,
        "default_parallelism": parallelism,
        "notes": "Each case lists the files in Spark's listing order as [name, size] and the planned "
        "FilePartitions as lists of [name, start, length]. Files are planned by FileSourceScanExec over the "
        "text source; codec suffixes decide splittability. Probes read Parquet files written by Spark with a "
        "user schema and record the rows (Row.toString) or the error conditions of the exception chain.",
        "cases": out_cases,
        "probes": probes,
    }
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    if LOCAL_PATH.search(text):
        raise SystemExit("refusing to write a fixture that contains a local path")
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {len(out_cases)} split cases and {len(probes)} probes against Spark {version} -> {args.out}")


if __name__ == "__main__":
    main()
