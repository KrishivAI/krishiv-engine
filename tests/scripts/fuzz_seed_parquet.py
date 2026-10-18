#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write the seed corpus of the `parquet_read` fuzz target.

The Parquet interoperability fixtures (`tests/fixtures/parquet/*.parquet`: pyarrow with every codec and
page version, INT96 and unsigned columns, DuckDB, Spark 4.2 with nested columns), plus each one cut
to its footer region, so mutations reach both the footer decoder and the page decoders.

Usage (from repo root): `python3 tests/scripts/fuzz_seed_parquet.py fuzz/corpus/parquet_read`
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1] / "fixtures" / "parquet"
    for path in sorted(root.glob("*.parquet")):
        data = path.read_bytes()
        footer = int.from_bytes(data[-8:-4], "little")
        for seed in (data, b"PAR1" + data[-(footer + 8) :]):
            (out / hashlib.sha256(seed).hexdigest()[:16]).write_bytes(seed)


if __name__ == "__main__":
    main()
