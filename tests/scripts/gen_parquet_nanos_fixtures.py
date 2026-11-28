#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate tests/fixtures/parquet/nanos: INT64 TIMESTAMP(NANOS) files (P3-T090, Spark 4.3).

- `pyarrow.parquet`: a foreign writer's nanosecond timestamps (no Spark schema in the footer):
  `utc` (isAdjustedToUTC) and `local`, one row, 2020-01-01 00:00:00.123456789.
- `spark-flat.parquet`, `spark-nested.parquet`: Spark 4.3.0 with `spark.sql.timestampNanosTypes.enabled`
  (`parquet_nanos_fixture.scala`); the footer's Spark schema keeps each column's precision.

What Spark 4.3.0 and 4.2.1 read from them is in `krishiv-session/tests/nanos_parquet.rs`.

Usage (from the repo root):

    uv run --python 3.12 --with pyarrow==21.0.0 python tests/scripts/gen_parquet_nanos_fixtures.py
    docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/tests/fixtures/parquet/nanos:/out" \\
        -v "$PWD/tests/scripts:/s" apache/spark:4.3.0 \\
        /opt/spark/bin/spark-shell --conf spark.ui.enabled=false -i /s/parquet_nanos_fixture.scala
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "parquet" / "nanos"
NANOS = 1_577_836_800_123_456_789


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "utc": pa.array([NANOS], pa.timestamp("ns", "UTC")),
            "local": pa.array([NANOS], pa.timestamp("ns")),
        }
    )
    pq.write_table(table, OUT / "pyarrow.parquet")


if __name__ == "__main__":
    main()
