# SPDX-License-Identifier: Apache-2.0
"""Generates the quickstart datasets (orders, customers) as Parquet files.

    python docs/datasets/generate.py OUT_DIR [--orders N]

Deterministic (fixed seed, no clock), needs only pyarrow. The default is 200,000 orders, about 5 MB;
nothing is downloaded and no data file is stored in the repository.
"""

import argparse
import pathlib
import random

import pyarrow as pa
import pyarrow.parquet as pq

REGIONS = ["east", "west", "north", "south"]
SEGMENTS = ["consumer", "business", "public"]


def generate(out: pathlib.Path, orders: int = 200_000, customers: int = 5_000, seed: int = 42) -> list[pathlib.Path]:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    cust = pa.table(
        {
            "customer_id": list(range(customers)),
            "segment": [rng.choice(SEGMENTS) for _ in range(customers)],
            "region": [rng.choice(REGIONS) for _ in range(customers)],
        }
    )
    ords = pa.table(
        {
            "order_id": list(range(orders)),
            "customer_id": [rng.randrange(customers) for _ in range(orders)],
            "amount": [round(rng.uniform(1, 500), 2) for _ in range(orders)],
            # Day number in 2025 (1..365), as a date32.
            "order_date": pa.array([20089 + rng.randrange(365) for _ in range(orders)], type=pa.int32()).cast(
                pa.date32()
            ),
        }
    )
    paths = [out / "customers.parquet", out / "orders.parquet"]
    pq.write_table(cust, paths[0])
    pq.write_table(ords, paths[1])
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--orders", type=int, default=200_000)
    args = ap.parse_args()
    for p in generate(args.out, args.orders):
        print(p, p.stat().st_size, "bytes")
