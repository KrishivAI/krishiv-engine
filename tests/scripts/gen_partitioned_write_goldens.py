#!/usr/bin/env python3
"""Generate (or check) the partitioned-write golden fixture (P2-T082b acceptance: Spark reads what Krishiv writes).

Each case of `tests/fixtures/partitioned-writes/cases.json` is written twice, by Krishiv and by Spark 4.2.0:
`df.write.partitionBy(..).format(..).options(..).save(dir)`, then the case's optional second write
(`then`: an `append` or an `overwrite`). Krishiv writes through the testkit binary `write-partition-cases`. The
fixture records, per case:

- `spark_partition_dirs`: the directories Spark's own writer made. The Rust test compares Krishiv's directory names
  with them, so Krishiv's partition-value escaping is compared with Spark's writer, not only with Spark's lenient
  reader.
- `schema`, `nullable`, `rows`: what Spark's reader (`spark.read.format(f).options(o).load(dir)`) returns from
  Krishiv's directory.
- `spark_rows`: what that reader returns from Spark's own directory. The script fails if those rows differ from the
  rows read from Krishiv's directory, so an output Spark misreads (stale files after an overwrite, say) cannot become
  the golden.

Rows are each column cast to STRING (NULL spelled `NULL`) joined by `|`, sorted, the text the Rust test builds from
Krishiv's own read. Directory listings skip names Spark's file index skips: a `.` prefix, and a `_` prefix without
`=`. Output is deterministic and has no local paths.

Usage (from the repo root; a local Spark 4.2.0 needs only a JDK and `pyspark==4.2.0`):

    cargo build -p krishiv-testkit --bin write-partition-cases
    python3 tests/scripts/gen_partitioned_write_goldens.py --writer target/debug/write-partition-cases \\
      --out tests/fixtures/spark-4.2.0/partitioned-write-goldens.json
    # nightly: regenerate and fail on any difference from the committed fixture
    python3 tests/scripts/gen_partitioned_write_goldens.py --writer target/debug/write-partition-cases \\
      --check tests/fixtures/spark-4.2.0/partitioned-write-goldens.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

FORMAT_VERSION = 1
SPARK_VERSION = "4.2.0"
GENERATOR = "tests/scripts/gen_partitioned_write_goldens.py"
CASES = Path("tests/fixtures/partitioned-writes/cases.json")


def skipped(name: str) -> bool:
    """Spark's `InMemoryFileIndex.shouldFilterOutPathName` for the names here: `.x`, and `_x` without `=`."""
    return name.startswith(".") or (name.startswith("_") and "=" not in name)


def partition_dirs(root: Path) -> list[str]:
    """Directories below `root` holding data files, relative (the root itself is ""), sorted."""
    out: list[str] = []

    def walk(at: Path) -> None:
        has_file = False
        for entry in sorted(at.iterdir()):
            if skipped(entry.name):
                continue
            if entry.is_dir():
                walk(entry)
            else:
                has_file = True
        if has_file:
            rel = at.relative_to(root).as_posix()
            out.append("" if rel == "." else rel)

    walk(root)
    return sorted(out)


def spark_write(spark, case: dict, path: Path) -> None:
    writes = [("errorifexists", case["query"])]
    if "then" in case:
        writes.append((case["then"]["mode"], case["then"]["query"]))
    for mode, query in writes:
        (
            spark.sql(query)
            .write.mode(mode)
            .partitionBy(*case["partition_by"])
            .format(case["format"])
            .options(**case["options"])
            .save(str(path))
        )


def read_back(spark, case: dict, path: Path) -> dict:
    df = spark.read.format(case["format"]).options(**case["options"]).load(str(path))
    parts = [f"coalesce(CAST(`{f.name}` AS STRING), 'NULL')" for f in df.schema.fields]
    rows = sorted(r[0] for r in df.selectExpr(f"concat_ws('|', {', '.join(parts)}) AS row").collect())
    return {
        "schema": df.schema.simpleString(),
        "nullable": [f.nullable for f in df.schema.fields],
        "rows": rows,
    }


def generate(writer: str) -> dict:
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[1]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("OFF")
    if spark.version != SPARK_VERSION:
        sys.exit(f"{GENERATOR}: Spark {spark.version}, want {SPARK_VERSION}")
    doc = json.loads(CASES.read_text())
    if doc.get("format_version") != FORMAT_VERSION:
        sys.exit(f"{GENERATOR}: {CASES} format_version {doc.get('format_version')}, want {FORMAT_VERSION}")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        ours, theirs = Path(tmp) / "krishiv", Path(tmp) / "spark"
        subprocess.run([writer, str(CASES), str(ours)], check=True, stdout=subprocess.DEVNULL)
        for case in doc["cases"]:
            spark_dir = theirs / case["id"]
            spark_write(spark, case, spark_dir)
            read = read_back(spark, case, ours / case["id"])
            own = read_back(spark, case, spark_dir)
            if read["rows"] != own["rows"]:
                sys.exit(
                    f"{GENERATOR}: {case['id']}: Spark reads {read['rows']} from Krishiv's output "
                    f"but {own['rows']} from its own"
                )
            results.append(
                {
                    "id": case["id"],
                    "spark_partition_dirs": partition_dirs(spark_dir),
                    **read,
                    "spark_rows": own["rows"],
                }
            )
    return {
        "format_version": FORMAT_VERSION,
        "generator": GENERATOR,
        "spark_version": SPARK_VERSION,
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--writer", required=True, help="the write-partition-cases binary")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", help="write the fixture here")
    target.add_argument("--check", help="regenerate and fail if this fixture differs")
    args = parser.parse_args()
    text = json.dumps(generate(args.writer), indent=1, ensure_ascii=False) + "\n"
    if args.check:
        if Path(args.check).read_text() != text:
            sys.stdout.write(text)
            print(f"{args.check} differs from a fresh Spark run (the new fixture is on stdout)", file=sys.stderr)
            return 1
        print(f"{args.check} matches a fresh Spark run", file=sys.stderr)
        return 0
    Path(args.out).write_text(text)
    print(f"{args.out} written", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
