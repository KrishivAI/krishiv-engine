#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Writes tests/fixtures/parquet/pyarrow_deep_nested.parquet (read by krishiv-testkit parquet_interop).

    uv run --python 3.12 --with pyarrow==21.0.0 python tests/scripts/gen_parquet_nested.py tests/fixtures/parquet/pyarrow_deep_nested.parquet
"""
import sys, pyarrow as pa, pyarrow.parquet as pq
rows = {
 "l": [None, [], [None, {"a": None, "b": None}, {"a": 1, "b": []}, {"a": 2, "b": [None, 3]}], [{"a": 4, "b": [5]}]],
 "o": [None, {"s": None}, {"s": {"i": None}}, {"s": {"i": 5}}],
 "m": [None, {1: None}, {1: [None, 9], 2: []}, {}],
}
st = pa.struct([("a", pa.int32()), ("b", pa.list_(pa.int32()))])
t = pa.table({
 "l": pa.array(rows["l"], pa.list_(st)),
 "o": pa.array(rows["o"], pa.struct([("s", pa.struct([("i", pa.int32())]))])),
 "m": pa.array([None if v is None else list(v.items()) for v in rows["m"]], pa.map_(pa.int32(), pa.list_(pa.int32()))),
})
pq.write_table(t, sys.argv[1], row_group_size=2, compression="snappy")
print(t.to_pylist())
