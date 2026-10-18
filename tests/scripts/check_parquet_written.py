#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the Parquet writer's files with other engines' readers (P2-T082a).

Reads every `tests/fixtures/parquet/written/<variant>.parquet` (written by Krishiv, see
`krishiv-testkit/tests/parquet_write_interop.rs`) with pyarrow and DuckDB, and the copies Spark 4.2
wrote after reading them (`parquet_written.scala`, so the values are exactly what Spark read), and
records in `reads.json`, per engine and file: the row count and the SHA-256 of the canonical rows
(the encoding of the Rust test's `canonical`), plus the Spark types (`<variant>.schema`, written by
the Spark script) and whether pyarrow sees a column and offset index on every column chunk. The
Rust test compares every hash with what the writer was given.

Usage (from the repo root; Python 3.12 for the pinned pyarrow wheel):

    out=$(mktemp -d) && chmod 777 "$out"
    docker run --rm -v "$PWD/tests/fixtures/parquet/written:/in:ro" -v "$out:/out" \
        -v "$PWD/tests/scripts:/s:ro" apache/spark:4.2.0 \
        /opt/spark/bin/spark-shell --conf spark.ui.enabled=false -i /s/parquet_written.scala
    uv run --python 3.12 --with duckdb==1.4.1 --with pyarrow==21.0.0 \
        python tests/scripts/check_parquet_written.py tests/fixtures/parquet/written "$out"
"""

from __future__ import annotations

import datetime
import decimal
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

EPOCH_DATE = datetime.date(1970, 1, 1)
EPOCH = datetime.datetime(1970, 1, 1)
EPOCH_UTC = datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)
WIDE = decimal.Context(prec=100)
# Files DuckDB 1.4 cannot read, for the same reason it cannot read Spark 4.2's (checked there).
DUCKDB_UNREADABLE = {
    "lz4": "no reader for the Hadoop-framed LZ4 codec (Spark's lz4 writes it too)",
    "v2": "no DELTA_BYTE_ARRAY for fixed-length decimals (Spark's v2 writer without dictionary uses it too)",
}


def canonical(value: Any, t: Any) -> Any:
    """The canonical JSON value of `value` of Arrow type `t` (see the Rust test's `canonical`)."""
    import pyarrow as pa

    if value is None:
        return None
    if pa.types.is_boolean(t):
        return bool(value)
    if pa.types.is_integer(t):
        return str(value)
    if pa.types.is_floating(t):
        return struct.pack("<d", float(value)).hex()
    if pa.types.is_decimal(t):
        return str(int(value.scaleb(t.scale, context=WIDE)))
    if pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_string_view(t):
        return value
    if pa.types.is_binary(t) or pa.types.is_large_binary(t) or pa.types.is_binary_view(t):
        return value.hex()
    if pa.types.is_date32(t):
        return str((value - EPOCH_DATE).days)
    if pa.types.is_timestamp(t):
        epoch = EPOCH_UTC if value.tzinfo is not None else EPOCH
        return str((value - epoch) // datetime.timedelta(microseconds=1))
    if pa.types.is_struct(t):
        return {f.name: canonical(value[f.name], f.type) for f in t}
    if pa.types.is_map(t):
        return [[canonical(k, t.key_type), canonical(v, t.item_type)] for k, v in value]
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_list_view(t):
        return [canonical(v, t.value_type) for v in value]
    raise TypeError(f"no canonical form for {t}")


def digest(table: Any) -> dict[str, Any]:
    """Row count and SHA-256 of the canonical rows of `table`, in `id` order."""
    types = {f.name: f.type for f in table.schema}
    h = hashlib.sha256()
    # Sorted in Python: Arrow's `take` has no kernel for every type (string views).
    for row in sorted(table.to_pylist(), key=lambda r: r["id"]):
        line = {name: canonical(v, types[name]) for name, v in row.items()}
        h.update(json.dumps(line, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
        h.update(b"\n")
    return {"rows": table.num_rows, "sha256": h.hexdigest()}


def page_index(path: Path) -> bool:
    import pyarrow.parquet as pq

    md = pq.ParquetFile(path).metadata
    return all(
        md.row_group(g).column(c).has_column_index and md.row_group(g).column(c).has_offset_index
        for g in range(md.num_row_groups)
        for c in range(md.num_columns)
    )


def duckdb_table(path: Path) -> Any:
    import duckdb

    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    return con.sql(f"SELECT * FROM read_parquet('{path}')").fetch_arrow_table()


def main() -> None:
    import pyarrow.parquet as pq

    written, spark_out = Path(sys.argv[1]), Path(sys.argv[2])
    reads: dict[str, dict[str, Any]] = {"pyarrow": {}, "duckdb": {}, "spark": {}}
    for path in sorted(written.glob("*.parquet")):
        variant = path.stem
        reads["pyarrow"][variant] = digest(pq.read_table(path)) | {"page_index": page_index(path)}
        if variant in DUCKDB_UNREADABLE:
            reads["duckdb"][variant] = {"unsupported": DUCKDB_UNREADABLE[variant]}
        else:
            reads["duckdb"][variant] = digest(duckdb_table(path))
        copy = spark_out / f"{variant}.parquet"
        reads["spark"][variant] = digest(pq.read_table(copy)) | {
            "schema": (spark_out / f"{variant}.schema").read_text().strip()
        }
    (written / "reads.json").write_text(json.dumps(reads, indent=2, sort_keys=True) + "\n")
    expected = json.loads((written / "expected.json").read_text())
    bad = [
        f"{engine}/{variant}"
        for engine, files in reads.items()
        for variant, got in files.items()
        if got.get("sha256", expected[variant]["sha256"]) != expected[variant]["sha256"]
    ]
    print("all engines read what was written" if not bad else f"different values: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
