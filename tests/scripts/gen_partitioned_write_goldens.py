#!/usr/bin/env python3
"""Generate the partitioned-write golden fixture (P2-T082b acceptance: Spark reads what Krishiv writes).

Each case of `tests/fixtures/partitioned-writes/cases.json` is written by Krishiv (the testkit binary
`write-partition-cases`, which runs the case's query and `df.write.partitionBy(..).format(..).save(dir)`), then read
back by a REAL Spark 4.2.0 with `spark.read.format(f).options(o).load(dir)`. The fixture records what Spark read: the
schema (`simpleString`, so partition-type inference shows), the rows and the partition directories that hold data
files. Rows are each column cast to STRING (NULL spelled `NULL`) joined by `|`, sorted, the same text the Rust test
builds from Krishiv's own read. Output is deterministic and contains no local paths.

Usage (from the repo root; any Spark 4.2.0 session works, the local one needs only a JDK):

    cargo build -p krishiv-testkit --bin write-partition-cases
    python3 tests/scripts/gen_partitioned_write_goldens.py \\
      --writer "$CARGO_TARGET_DIR/debug/write-partition-cases" \\
      --out tests/fixtures/spark-4.2.0/partitioned-write-goldens.json

On krishiv2 run it through `ops/scripts/spark-local.sh` (in krishiv-internal), which puts PySpark 4.2.0 and a JDK on
the path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_partitioned_write_goldens.py"
CASES = Path("tests/fixtures/partitioned-writes/cases.json")


def partition_dirs(root: Path) -> list[str]:
    """Directories below `root` holding data files, relative, sorted; `_`/`.` names skipped as Spark skips them."""
    out: list[str] = []

    def walk(at: Path) -> None:
        has_file = False
        for entry in sorted(at.iterdir()):
            if entry.name.startswith(("_", ".")):
                continue
            if entry.is_dir():
                walk(entry)
            else:
                has_file = True
        if has_file:
            out.append(at.relative_to(root).as_posix())

    walk(root)
    return sorted(out)


def read_back(spark, case: dict, path: Path) -> dict:
    df = spark.read.format(case["format"]).options(**case["options"]).load(str(path))
    parts = [f"coalesce(CAST(`{f.name}` AS STRING), 'NULL')" for f in df.schema.fields]
    rows = sorted(r[0] for r in df.selectExpr(f"concat_ws('|', {', '.join(parts)}) AS row").collect())
    return {"schema": df.schema.simpleString(), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--writer", required=True, help="the write-partition-cases binary")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[1]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("OFF")
    cases = json.loads(CASES.read_text())["cases"]
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([args.writer, str(CASES), tmp], check=True, stdout=subprocess.DEVNULL)
        for case in cases:
            path = Path(tmp) / case["id"]
            result = {"id": case["id"], "partition_dirs": partition_dirs(path)}
            result.update(read_back(spark, case, path))
            results.append(result)
    doc = {
        "format_version": FORMAT_VERSION,
        "generator": GENERATOR,
        "spark_version": spark.version,
        "cases": results,
    }
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(results)} cases -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
