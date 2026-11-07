#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write the hand-made regression inputs of the `parquet_read` fuzz target for the page-index and Bloom stages.

`design/P2-T080b-parquet-indexes.md` §4 item 6 asks for four inputs, each of which the reader must end as
"ignored" (index or filter not used) within the target's time budget, never by allocating what the bytes claim:

- `hand-column-index-count-2p31`: a column index whose `null_pages` list declares 2^31 entries;
- `hand-offset-index-outside-chunk`: an offset index whose page location points far outside the column chunk;
- `hand-offset-index-rows-decreasing`: an offset index whose `first_row_index` goes down;
- `hand-bloom-bytes-2p31`: a Bloom filter header declaring a bitset of 2^31 - 1 bytes (`numBytes` is an i32).

Each starts from one pyarrow file (page index on, a Bloom filter on column `a`) and replaces one structure in
place: the crafted struct is written at the structure's offset and the rest of its region is left as it was,
so the footer and every other offset stay valid. The footer's offsets are read with a small Thrift compact
protocol reader (the fields of `ColumnChunk` in parquet.thrift).

Usage (from repo root): `python3 tests/scripts/fuzz_regress_parquet_index.py fuzz/regressions/parquet_read`
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Thrift compact protocol type ids.
BOOL_TRUE, BYTE, I16, I32, I64, DOUBLE, BINARY, LIST, SET, MAP, STRUCT = 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
BOOL_FALSE = 2


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def zigzag(n: int) -> int:
    return (n << 1) ^ (n >> 63)


class Reader:
    """Just enough of the Thrift compact protocol to walk a FileMetaData."""

    def __init__(self, data: bytes, pos: int = 0) -> None:
        self.data, self.pos = data, pos

    def byte(self) -> int:
        b = self.data[self.pos]
        self.pos += 1
        return b

    def varint(self) -> int:
        shift = n = 0
        while True:
            b = self.byte()
            n |= (b & 0x7F) << shift
            if not b & 0x80:
                return n
            shift += 7

    def zigzag_int(self) -> int:
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def skip(self, t: int, in_list: bool = False) -> None:
        if t in (BOOL_TRUE, BOOL_FALSE):
            # A struct field's bool is in its type nibble; a list element's is a byte of its own.
            if in_list:
                self.pos += 1
            return
        if t == BYTE:
            self.pos += 1
        elif t in (I16, I32, I64):
            self.varint()
        elif t == DOUBLE:
            self.pos += 8
        elif t == BINARY:
            n = self.varint()  # read first: `self.pos += self.varint()` would lose the length's own bytes
            self.pos += n
        elif t in (LIST, SET):
            size, et = self.list_header()
            for _ in range(size):
                self.skip(et, in_list=True)
        elif t == MAP:
            size = self.varint()
            if size:
                kv = self.byte()
                for _ in range(size):
                    self.skip(kv >> 4)
                    self.skip(kv & 0x0F)
        elif t == STRUCT:
            self.struct(lambda fid, ft: self.skip(ft))
        else:
            raise ValueError(f"thrift type {t}")

    def list_header(self) -> tuple[int, int]:
        b = self.byte()
        size = b >> 4
        if size == 15:
            size = self.varint()
        return size, b & 0x0F

    def int_fields(self) -> dict[int, int]:
        """A struct's i32/i64 fields by field id (everything else skipped)."""
        fields: dict[int, int] = {}

        def f(fid: int, t: int) -> None:
            if t in (I32, I64):
                fields[fid] = self.zigzag_int()
            else:
                self.skip(t)

        self.struct(f)
        return fields

    def struct(self, field) -> None:
        last = 0
        while True:
            b = self.byte()
            if b == 0:
                return
            delta, t = b >> 4, b & 0x0F
            fid = last + delta if delta else self.zigzag_int()
            last = fid
            field(fid, t)


def column_chunks(data: bytes) -> list[dict[int, int]]:
    """The i32/i64 fields of every ColumnChunk (by field id) in the file's footer."""
    footer_len = int.from_bytes(data[-8:-4], "little")
    r = Reader(data, len(data) - 8 - footer_len)
    chunks: list[dict[int, int]] = []

    def row_group(fid: int, t: int) -> None:
        if fid == 1 and t == LIST:  # RowGroup.columns
            size, _ = r.list_header()
            for _ in range(size):
                chunks.append(r.int_fields())
        else:
            r.skip(t)

    def file_meta(fid: int, t: int) -> None:
        if fid == 4 and t == LIST:  # FileMetaData.row_groups
            size, _ = r.list_header()
            for _ in range(size):
                r.struct(row_group)
        else:
            r.skip(t)

    r.struct(file_meta)
    return chunks


def base_file() -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"a": list(range(2000)), "s": [f"v{i % 50}" for i in range(2000)]})
    out = io.BytesIO()
    pq.write_table(
        table,
        out,
        data_page_size=512,
        # Plain pages: about 16 of them per column chunk, so the offset index has room for the crafted ones.
        use_dictionary=False,
        write_batch_size=64,  # the page size is checked once per batch
        write_page_index=True,
        row_group_size=1000,
        bloom_filter_options={"a": {"ndv": 2000, "fpp": 0.05}},
    )
    return out.getvalue()


def patch(data: bytes, offset: int, length: int, new: bytes) -> bytes:
    if len(new) > length:
        raise ValueError(f"crafted struct of {len(new)} bytes does not fit the {length}-byte region")
    return data[:offset] + new + data[offset + len(new) :]


def page_location(offset: int, size: int, first_row: int) -> bytes:
    # PageLocation { 1: i64 offset, 2: i32 compressed_page_size, 3: i64 first_row_index }
    return (
        bytes([0x16])
        + varint(zigzag(offset))
        + bytes([0x15])
        + varint(zigzag(size))
        + bytes([0x16])
        + varint(zigzag(first_row))
        + b"\x00"
    )


def offset_index(locations: list[bytes]) -> bytes:
    # OffsetIndex { 1: list<PageLocation> page_locations }
    assert len(locations) < 15
    return bytes([0x19, (len(locations) << 4) | STRUCT]) + b"".join(locations) + b"\x00"


def inputs(data: bytes) -> dict[str, bytes]:
    col = column_chunks(data)[0]  # row group 0, column `a`
    oi_off, oi_len = col[4], col[5]  # ColumnChunk.offset_index_offset / _length
    ci_off, ci_len = col[6], col[7]  # ColumnChunk.column_index_offset / _length
    out: dict[str, bytes] = {}

    # ColumnIndex { 1: list<bool> null_pages, ... }: a list header declaring 2^31 entries, then nothing.
    out["hand-column-index-count-2p31"] = patch(data, ci_off, ci_len, bytes([0x19, 0xF1]) + varint(1 << 31))

    out["hand-offset-index-outside-chunk"] = patch(data, oi_off, oi_len, offset_index([page_location(1 << 40, 100, 0)]))

    # Two pages inside the chunk's data, rows going down.
    r = Reader(data, oi_off)
    first: list[int] = []

    def locs(fid: int, t: int) -> None:
        if fid == 1 and t == LIST:
            size, _ = r.list_header()
            for _ in range(size):
                first.append(r.int_fields()[1])  # PageLocation.offset
        else:
            r.skip(t)

    r.struct(locs)
    out["hand-offset-index-rows-decreasing"] = patch(
        data, oi_off, oi_len, offset_index([page_location(first[0], 64, 100), page_location(first[1], 64, 50)])
    )

    out["hand-bloom-bytes-2p31"] = bloom_patch(data)
    return out


def bloom_patch(data: bytes) -> bytes:
    """BloomFilterHeader { 1: i32 numBytes, 2: algorithm, 3: hash, 4: compression }: only numBytes changes. The
    header is written again with the largest i32 and the bitset after it is cut so the region keeps its length."""
    import pyarrow.parquet as pq

    md = pq.ParquetFile(io.BytesIO(data)).metadata.row_group(0).column(0)
    off, length = md.bloom_filter_offset, md.bloom_filter_length
    r = Reader(data, off)
    r.byte()  # field 1 header (i32)
    r.varint()  # numBytes
    rest_start = r.pos
    r.pos = off

    def skip_all(fid: int, t: int) -> None:
        r.skip(t)

    r.struct(skip_all)
    rest = data[rest_start : r.pos]  # fields 2-4 and the stop byte
    header = bytes([0x15]) + varint(zigzag(2**31 - 1)) + rest
    return patch(data, off, length, header)


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, data in inputs(base_file()).items():
        (out / name).write_bytes(data)
        print(f"{out / name}: {len(data)} bytes")


if __name__ == "__main__":
    main()
