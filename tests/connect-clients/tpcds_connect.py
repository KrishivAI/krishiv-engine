# SPDX-License-Identifier: Apache-2.0
"""TPC-DS through Spark Connect: the first run of the 99 queries against ``krishiv server`` (P3 PR-53a leftover).

Report-only. The pinned pyspark client registers the 24 TPC-DS tables with ``CREATE TEMPORARY VIEW ... USING
parquet`` (a SQL command, so it does not need the ``CreateDataFrameView`` Connect command) and runs each query of
``bench/queries/tpcds``. Per query the report has ``ok`` (rows and seconds) or ``error`` (the Spark error class when the
message carries one). With ``--reference`` (a second Connect server, e.g. Spark 4.2 on the same files) the row counts
are compared too.

    cd bench && uv run python ../tests/connect-clients/tpcds_connect.py --remote sc://127.0.0.1:15002 \
        --data ~/krishiv-data/tpcds --queries q03,q07 --out /tmp/tpcds-connect
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

QUERIES = Path(__file__).resolve().parents[2] / "bench" / "queries" / "tpcds"
CLASS = re.compile(r"\[([A-Z][A-Z0-9_]+(?:\.[A-Z0-9_]+)*)\]")


def error_class(message: str) -> str:
    """The first ``[CLASS.SUB]`` in an error message, else ``UNCLASSIFIED``."""
    m = CLASS.search(message)
    return m.group(1) if m else "UNCLASSIFIED"


def statements(sql: str) -> list[str]:
    """A query file's statements (a few TPC-DS queries are two), without empty ones."""
    return [s.strip() for s in sql.split(";") if s.strip()]


def register(spark, data: Path) -> list[str]:
    tables = sorted(p.stem for p in data.glob("*.parquet"))
    for t in tables:
        spark.sql(
            f"CREATE OR REPLACE TEMPORARY VIEW {t} USING parquet OPTIONS (path '{data / (t + '.parquet')}')"
        )
    return tables


def run_one(spark, name: str) -> dict:
    from pyspark.errors import PySparkException

    t0 = time.perf_counter()
    try:
        rows = 0
        for stmt in statements((QUERIES / f"{name}.sql").read_text()):
            rows = len(spark.sql(stmt).collect())
        return {
            "status": "ok",
            "rows": rows,
            "seconds": round(time.perf_counter() - t0, 3),
        }
    except PySparkException as e:
        msg = str(e)
        return {
            "status": "error",
            "class": error_class(msg),
            "message": msg.splitlines()[0][:300] if msg else "",
        }


def summary(results: dict[str, dict], reference: dict[str, dict] | None) -> str:
    ok = [q for q, r in results.items() if r["status"] == "ok"]
    lines = [f"TPC-DS through Connect: {len(ok)}/{len(results)} queries ran.", ""]
    bad = {q: r for q, r in results.items() if r["status"] != "ok"}
    if bad:
        by: dict[str, list[str]] = {}
        for q, r in bad.items():
            by.setdefault(r["class"], []).append(q)
        lines += ["| error class | queries |", "|---|---|"]
        lines += [
            f"| {c} | {', '.join(sorted(qs))} |"
            for c, qs in sorted(by.items(), key=lambda kv: -len(kv[1]))
        ]
    if reference:
        diff = [
            q
            for q in ok
            if reference.get(q, {}).get("status") == "ok"
            and reference[q]["rows"] != results[q]["rows"]
        ]
        lines += [
            "",
            f"Row counts differ from the reference on {len(diff)} queries: {', '.join(sorted(diff)) or 'none'}.",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", required=True)
    ap.add_argument(
        "--reference", help="a second Connect server on the same files, for row counts"
    )
    ap.add_argument("--data", required=True, help="directory of <table>.parquet files")
    ap.add_argument("--queries", help="comma list, e.g. q03,q07 (default: all)")
    ap.add_argument("--out", default="target/tpcds-connect")
    a = ap.parse_args(argv)

    from pyspark.sql import SparkSession  # the bench venv has pyspark[connect]

    names = (
        a.queries.split(",")
        if a.queries
        else sorted(p.stem for p in QUERIES.glob("q*.sql"))
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict[str, dict]] = {}
    for label, remote in [("krishiv", a.remote), ("reference", a.reference)]:
        if not remote:
            continue
        spark = SparkSession.builder.remote(remote).getOrCreate()
        # Eight queries name their output columns in double quotes; the bench's Spark adapter sets this too.
        spark.conf.set("spark.sql.ansi.doubleQuotedIdentifiers", "true")
        register(spark, Path(a.data))
        runs[label] = {q: run_one(spark, q) for q in names}
        spark.stop()
    (out / "tpcds.json").write_text(json.dumps(runs, indent=1, sort_keys=True))
    text = summary(runs["krishiv"], runs.get("reference"))
    (out / "tpcds-summary.md").write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
