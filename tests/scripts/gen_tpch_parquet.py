#!/usr/bin/env python3
r"""Write the TPC-H tables at a scale factor as Parquet files, one per table (P2-T074 test data).

DuckDB's `tpch` extension is dbgen, so the rows are the specification's. Usage (from repo root):

    uv run --project bench python -I tests/scripts/gen_tpch_parquet.py \
        0.01 crates/krishiv-exec/tests/fixtures/tpch-sf0.01-parquet
"""

import os
import sys

import duckdb

TABLES = ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]


def main() -> None:
    sf, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={float(sf)})")
    for t in TABLES:
        path = os.path.join(out, f"{t}.parquet")
        con.execute(f"COPY {t} TO '{path}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 16384)")


if __name__ == "__main__":
    main()
