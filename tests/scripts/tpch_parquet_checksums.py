#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TPC-H Parquet files and pyarrow's per-column checksums (P2-T080a interop check).

Generates TPC-H at a scale factor with DuckDB (`CALL dbgen`) and writes one Parquet file per table
with DuckDB's own writer, then reads every file back with pyarrow and records, per column, a SHA-256
over a canonical encoding of every value in row order. The Krishiv Parquet reader computes the same
checksums (`krishiv-testkit/tests/parquet_interop.rs`) and must match exactly.

Canonical value encoding (each value followed by 0x1F; a null is 0x00 'N'):
- integers: decimal ASCII; decimals: the unscaled integer in decimal ASCII; dates: days since
  1970-01-01 in decimal ASCII; doubles: the 8 little-endian IEEE bytes in lowercase hex;
  strings: their UTF-8 bytes; booleans: `1` / `0`.

Usage (from the repo root; data stays outside the repo):

    uv run --with duckdb==1.4.1 --with pyarrow==21.0.0 python tests/scripts/tpch_parquet_checksums.py \
        --out "$KRISHIV_DATA/tpch-duckdb/sf1" --sf 1

`checksums.json` is written next to the files.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import struct
from pathlib import Path

TABLES = ["region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem"]
SEP = b"\x1f"
NULL = b"\x00N"
EPOCH = datetime.date(1970, 1, 1)


def generate(out: Path, sf: float) -> str:
    import duckdb

    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={sf})")
    for t in TABLES:
        con.execute(f"COPY {t} TO '{out / (t + '.parquet')}' (FORMAT parquet)")
    return duckdb.__version__


def encode(value: object, arrow_type: object) -> bytes:
    import pyarrow as pa

    if value is None:
        return NULL
    if pa.types.is_decimal(arrow_type):
        return str(int(value.scaleb(arrow_type.scale))).encode()
    if pa.types.is_date32(arrow_type):
        return str((value - EPOCH).days).encode()
    if pa.types.is_floating(arrow_type):
        return struct.pack("<d", value).hex().encode()
    if pa.types.is_boolean(arrow_type):
        return b"1" if value else b"0"
    if pa.types.is_integer(arrow_type):
        return str(value).encode()
    if isinstance(value, str):
        return value.encode()
    raise TypeError(f"no canonical encoding for {arrow_type}")


def checksums(path: Path) -> dict[str, object]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    columns = {}
    for name, column in zip(table.column_names, table.columns):
        h = hashlib.sha256()
        for chunk in column.chunks:
            h.update(b"".join(encode(v, column.type) + SEP for v in chunk.to_pylist()))
        columns[name] = {"type": str(column.type), "sha256": h.hexdigest()}
    return {"rows": table.num_rows, "columns": columns}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sf", type=float, default=1.0)
    p.add_argument("--skip-generate", action="store_true")
    args = p.parse_args()
    import pyarrow

    generator = "existing files" if args.skip_generate else f"duckdb {generate(args.out, args.sf)}"
    result = {
        "generator": generator,
        "reader": f"pyarrow {pyarrow.__version__}",
        "scale_factor": args.sf,
        "tables": {t: checksums(args.out / f"{t}.parquet") for t in TABLES},
    }
    (args.out / "checksums.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
