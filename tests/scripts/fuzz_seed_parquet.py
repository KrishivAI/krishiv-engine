#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write the seed corpus of the `parquet_read` fuzz target.

The Parquet interoperability fixtures (`tests/fixtures/parquet/*.parquet`: pyarrow with every codec and
page version, INT96 and unsigned columns, DuckDB, Spark 4.2 with nested columns), plus each one cut
to its footer region, so mutations reach both the footer decoder and the page decoders.

With pyarrow installed it also writes a page-indexed file (many small pages, a column and an offset
index) and a file with a stored Bloom filter, the inputs the page-index and Bloom stages of
`design/P2-T080b-parquet-indexes.md` decode; without it those two are skipped.

Usage (from repo root): `python3 tests/scripts/fuzz_seed_parquet.py fuzz/corpus/parquet_read`
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def indexed_seeds() -> list[bytes]:
    """A page-indexed file and a Bloom-filter file, or nothing without pyarrow."""
    try:
        import io

        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return []
    table = pa.table({"a": list(range(2000)), "s": [f"v{i % 50}" for i in range(2000)]})
    seeds = []
    paged = io.BytesIO()
    pq.write_table(table, paged, data_page_size=512, write_page_index=True, row_group_size=1000)
    seeds.append(paged.getvalue())
    try:
        bloom = io.BytesIO()
        pq.write_table(
            table,
            bloom,
            bloom_filter_options={"a": {"ndv": 2000, "fpp": 0.05}, "s": True},
            write_page_index=True,
        )
        seeds.append(bloom.getvalue())
    except (TypeError, ValueError):
        # This pyarrow cannot write Bloom filters.
        pass
    return seeds


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1] / "fixtures" / "parquet"
    for path in sorted(root.glob("*.parquet")):
        data = path.read_bytes()
        footer = int.from_bytes(data[-8:-4], "little")
        for seed in (data, b"PAR1" + data[-(footer + 8) :]):
            (out / hashlib.sha256(seed).hexdigest()[:16]).write_bytes(seed)
    for data in indexed_seeds():
        footer = int.from_bytes(data[-8:-4], "little")
        for seed in (data, b"PAR1" + data[-(footer + 8) :]):
            (out / hashlib.sha256(seed).hexdigest()[:16]).write_bytes(seed)


if __name__ == "__main__":
    main()
