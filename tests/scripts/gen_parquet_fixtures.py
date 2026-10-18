#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the Parquet interoperability fixtures (P2-T080a).

Small Parquet files written by other engines' writers, each with the values pyarrow reads from it:
- pyarrow (Arrow C++): every codec (none, snappy, gzip, zstd, lz4, brotli), data page v1 and v2,
  dictionary and plain encodings, several row groups, nulls; `INT96` timestamps; unsigned integers;
- DuckDB: its own writer (`COPY ... (FORMAT parquet)`);
- Spark 4.2 (`spark.parquet`, written in the Spark container by `parquet_fixture.scala` and
  described here with `--describe`): Spark's default `INT96` timestamps, decimals of every physical
  width, nested struct, array and map columns.

For every file, `<name>.json` records the row count and, per column, the Arrow type pyarrow reads and
a SHA-256 over a canonical encoding of the values in row order (the encoding of
`tpch_parquet_checksums.py`, plus: timestamps as microseconds since the epoch, binary as lowercase
hex). The Krishiv reader computes the same checksums in `krishiv-testkit/tests/parquet_interop.rs`.
The files also seed the `parquet_read` fuzz target.

Usage (from the repo root; Python 3.12 for the pinned pyarrow wheel):

    uv run --python 3.12 --with duckdb==1.4.1 --with pyarrow==21.0.0 \
        python tests/scripts/gen_parquet_fixtures.py tests/fixtures/parquet
    docker run --rm -v "$PWD/tests/fixtures/parquet:/out" -v "$PWD/tests/scripts:/s" apache/spark:4.2.0 \
        /opt/spark/bin/spark-shell --conf spark.ui.enabled=false -i /s/parquet_fixture.scala
    uv run --python 3.12 --with pyarrow==21.0.0 \
        python tests/scripts/gen_parquet_fixtures.py --describe tests/fixtures/parquet/spark.parquet
"""

from __future__ import annotations

import datetime
import decimal
import hashlib
import json
import struct
import sys
from pathlib import Path

SEP = b"\x1f"
NULL = b"\x00N"
EPOCH = datetime.date(1970, 1, 1)
ROWS = 600


def encode(value: object, t: object) -> bytes:
    import pyarrow as pa

    if value is None:
        return NULL
    if pa.types.is_decimal(t):
        return str(int(value.scaleb(t.scale))).encode()
    if pa.types.is_date32(t):
        return str((value - EPOCH).days).encode()
    if pa.types.is_timestamp(t):
        return str(value).encode()  # already microseconds (see `canonical`)
    if pa.types.is_floating(t):
        return struct.pack("<d", value).hex().encode()
    if pa.types.is_boolean(t):
        return b"1" if value else b"0"
    if pa.types.is_integer(t):
        return str(value).encode()
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return value.hex().encode()
    if isinstance(value, str):
        return value.encode()
    raise TypeError(f"no canonical encoding for {t}")


def canonical(column: object) -> tuple[object, list[object]]:
    """The column's values for `encode`: timestamps as integer microseconds."""
    import pyarrow as pa
    import pyarrow.compute as pc

    t = column.type
    if pa.types.is_timestamp(t):
        micros = pc.cast(pc.cast(column, pa.timestamp("us", tz=t.tz)), pa.int64())
        return t, micros.to_pylist()
    return t, column.to_pylist()


def describe(path: Path) -> dict[str, object]:
    import pyarrow.parquet as pq

    import pyarrow as pa

    table = pq.read_table(path)
    columns = {}
    for name, column in zip(table.column_names, table.columns):
        if pa.types.is_nested(column.type):
            # Nested values are checked explicitly by the Rust test.
            columns[name] = {"type": str(column.type), "sha256": None}
            continue
        t, values = canonical(column.combine_chunks())
        h = hashlib.sha256(b"".join(encode(v, t) + SEP for v in values))
        columns[name] = {"type": str(column.type), "sha256": h.hexdigest()}
    return {"rows": table.num_rows, "columns": columns}


def sample() -> object:
    import pyarrow as pa

    def maybe(i: int, v: object) -> object:
        return None if i % 7 == 3 else v

    rows = range(ROWS)
    return pa.table(
        {
            "i32": pa.array([maybe(i, i - 1500) for i in rows], pa.int32()),
            "i64": pa.array([maybe(i, i * 1_000_003 - 2**40) for i in rows], pa.int64()),
            "f64": pa.array([maybe(i, i / 7.0 - 100.0) for i in rows], pa.float64()),
            "dec": pa.array(
                [maybe(i, decimal.Decimal(i * 37 - 9000).scaleb(-2)) for i in rows], pa.decimal128(15, 2)
            ),
            "wide": pa.array(
                [maybe(i, decimal.Decimal(i * 10**20 + 7).scaleb(-6)) for i in rows], pa.decimal128(38, 6)
            ),
            "d": pa.array([maybe(i, EPOCH + datetime.timedelta(days=i * 3 - 4000)) for i in rows], pa.date32()),
            "ts": pa.array(
                [maybe(i, i * 86_400_000_123 - 10**15) for i in rows], pa.timestamp("us", tz="UTC")
            ),
            "s": pa.array([maybe(i, f"v{i % 50}-ü{'x' * (i % 13)}") for i in rows], pa.string()),
            "b": pa.array([maybe(i, bytes([i % 256, 0, 255])) for i in rows], pa.binary()),
            "flag": pa.array([maybe(i, i % 3 == 0) for i in rows], pa.bool_()),
        }
    )


def write_json(out: Path, name: str) -> None:
    import pyarrow.parquet as pq

    meta = describe(out / name)
    meta["writer"] = pq.ParquetFile(out / name).metadata.created_by
    (out / f"{Path(name).stem}.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if sys.argv[1] == "--describe":
        path = Path(sys.argv[2])
        write_json(path.parent, path.name)
        return
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    table = sample()
    files: list[str] = []
    for codec in ["none", "snappy", "gzip", "zstd", "lz4", "brotli"]:
        name = f"pyarrow_{codec}.parquet"
        pq.write_table(table, out / name, compression=codec, row_group_size=200)
        files.append(name)
    pq.write_table(
        table,
        out / "pyarrow_v2_plain.parquet",
        data_page_version="2.0",
        use_dictionary=False,
        row_group_size=150,
    )
    files.append("pyarrow_v2_plain.parquet")
    int96 = pa.table({"ts": table.column("ts"), "i32": table.column("i32")})
    pq.write_table(int96, out / "pyarrow_int96.parquet", use_deprecated_int96_timestamps=True)
    files.append("pyarrow_int96.parquet")
    unsigned = pa.table(
        {
            "u8": pa.array([None if i % 5 == 0 else i % 256 for i in range(500)], pa.uint8()),
            "u16": pa.array([i * 131 % 65536 for i in range(500)], pa.uint16()),
            "u32": pa.array([4_294_967_295 - i for i in range(500)], pa.uint32()),
        }
    )
    pq.write_table(unsigned, out / "pyarrow_unsigned.parquet")
    files.append("pyarrow_unsigned.parquet")
    con = duckdb.connect()
    con.register("t", table)
    con.execute(f"COPY (SELECT * FROM t) TO '{out / 'duckdb.parquet'}' (FORMAT parquet, ROW_GROUP_SIZE 256)")
    files.append("duckdb.parquet")
    for name in files:
        write_json(out, name)


if __name__ == "__main__":
    main()
