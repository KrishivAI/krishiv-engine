# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "adbc-driver-flightsql==1.13.0",
#   "adbc-driver-manager==1.13.0",
#   "pyarrow>=17",
#   "pyspark[connect]==4.2.0",
# ]
# ///
# SPDX-License-Identifier: Apache-2.0
"""ADBC end to end against `krishiv server` (design P3-T101 §11, `adbc_flight_sql_tpch_q1`).

Runs TPC-H q1 over a small `lineitem` through the ADBC Flight SQL driver (cookies on, so the
temporary view of the first statement is there for the second; the DDL also checks that a
command sent as a query runs once) and through Spark Connect on the same port, and compares the
Arrow types (q1's decimal precision and scale) and the rows in q1's ORDER BY order.

    uv run --script tests/scripts/flight_sql_adbc.py --addr 127.0.0.1:15002

Exit 0 when they agree, 1 with a report when they do not.
"""

from __future__ import annotations

import argparse
import sys

LINEITEM = """
CREATE TEMPORARY VIEW lineitem AS SELECT * FROM VALUES
  ('A', 'F', CAST(17 AS DECIMAL(15,2)), CAST(21168.23 AS DECIMAL(15,2)), CAST(0.04 AS DECIMAL(15,2)),
   CAST(0.02 AS DECIMAL(15,2)), DATE'1996-03-13'),
  ('A', 'F', CAST(36 AS DECIMAL(15,2)), CAST(45983.16 AS DECIMAL(15,2)), CAST(0.09 AS DECIMAL(15,2)),
   CAST(0.06 AS DECIMAL(15,2)), DATE'1996-04-12'),
  ('N', 'O', CAST(8 AS DECIMAL(15,2)), CAST(13309.60 AS DECIMAL(15,2)), CAST(0.10 AS DECIMAL(15,2)),
   CAST(0.02 AS DECIMAL(15,2)), DATE'1996-01-29'),
  ('N', 'O', CAST(28 AS DECIMAL(15,2)), CAST(28955.64 AS DECIMAL(15,2)), CAST(0.09 AS DECIMAL(15,2)),
   CAST(0.06 AS DECIMAL(15,2)), DATE'1996-04-21'),
  ('R', 'F', CAST(24 AS DECIMAL(15,2)), CAST(22824.48 AS DECIMAL(15,2)), CAST(0.10 AS DECIMAL(15,2)),
   CAST(0.04 AS DECIMAL(15,2)), DATE'1996-03-30'),
  ('R', 'F', CAST(32 AS DECIMAL(15,2)), CAST(49620.16 AS DECIMAL(15,2)), CAST(0.07 AS DECIMAL(15,2)),
   CAST(0.02 AS DECIMAL(15,2)), DATE'1996-01-30'),
  ('N', 'F', CAST(38 AS DECIMAL(15,2)), CAST(44694.46 AS DECIMAL(15,2)), CAST(0.00 AS DECIMAL(15,2)),
   CAST(0.05 AS DECIMAL(15,2)), DATE'1998-12-01')
AS t(l_returnflag, l_linestatus, l_quantity, l_extendedprice, l_discount, l_tax, l_shipdate)
"""

Q1 = """
SELECT l_returnflag, l_linestatus,
       sum(l_quantity) AS sum_qty,
       sum(l_extendedprice) AS sum_base_price,
       sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
       sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
       avg(l_quantity) AS avg_qty,
       avg(l_extendedprice) AS avg_price,
       avg(l_discount) AS avg_disc,
       count(*) AS count_order
FROM lineitem
WHERE l_shipdate <= date '1998-12-01' - interval '90' day
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus
"""


def via_adbc(addr: str):
    import adbc_driver_flightsql.dbapi as flight_sql

    kwargs = {"adbc.flight.sql.rpc.with_cookie_middleware": "true"}
    with flight_sql.connect(f"grpc://{addr}", db_kwargs=kwargs, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(LINEITEM)
        cur.fetchall()
        cur.execute(Q1)
        return cur.fetch_arrow_table()


def via_connect(addr: str):
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.remote(f"sc://{addr}").getOrCreate()
    try:
        spark.sql(LINEITEM)
        return spark.sql(Q1).toArrow()
    finally:
        spark.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addr", required=True, help="host:port of krishiv server's Connect listener")
    args = parser.parse_args()
    flight = via_adbc(args.addr)
    connect = via_connect(args.addr)
    problems = []
    if flight.schema.names != connect.schema.names:
        problems.append(f"names: {flight.schema.names} != {connect.schema.names}")
    for f, c in zip(flight.schema, connect.schema, strict=False):
        if f.type != c.type:
            problems.append(f"{f.name}: {f.type} != {c.type}")
    if flight.to_pylist() != connect.to_pylist():
        problems.append(f"rows:\n  flight  {flight.to_pylist()}\n  connect {connect.to_pylist()}")
    if flight.num_rows != 3:
        problems.append(f"{flight.num_rows} groups, want 3")
    if problems:
        print("ADBC and Spark Connect disagree:\n" + "\n".join(problems))
        return 1
    print(f"ADBC q1: {flight.num_rows} rows, {len(flight.schema)} columns, equal to Spark Connect")
    print("types: " + ", ".join(f"{f.name}={f.type}" for f in flight.schema))
    return 0


if __name__ == "__main__":
    sys.exit(main())
