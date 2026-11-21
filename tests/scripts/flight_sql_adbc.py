# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "adbc-driver-flightsql==1.13.0",
#   "adbc-driver-manager==1.13.0",
#   "pyarrow>=17",
#   "pyspark[connect]==4.2.0",
# ]
# ///
# SPDX-License-Identifier: Apache-2.0
"""ADBC end to end against `krishiv server` (design P3-T101 §0, §11, §16 m7).

Writes a TPC-H `lineitem` of SF 0.01's row count (60,175 rows, the q1 columns, seeded, so
every run reads the same file) as Parquet, then runs TPC-H q1 over it through the ADBC Flight
SQL driver with a bearer token, and through Spark Connect on the same port, and compares the Arrow
types (q1's decimal precision and scale) and the rows in q1's ORDER BY order. `lineitem` is a
CTE in both runs (Spark Connect's `spark.sql` does not create temporary views here yet). A
separate ADBC-only pair of statements checks the cookie session: a temporary view created by
one statement (a command sent as a query, which runs once) is read by the next.

This compares Krishiv over ADBC with Krishiv over Connect (§11: "rows equal to the Connect run");
it does not prove Spark parity of q1's decimal types, which the TPC-H goldens cover.

    uv run --script tests/scripts/flight_sql_adbc.py --addr 127.0.0.1:15002 --token T \
        --data /tmp/lineitem [--ca ca.pem]

With `--ca`, both clients use TLS and trust that CA (the server's name is `localhost`).
Exit 0 when they agree, 1 with a report when they do not.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from decimal import Decimal
from pathlib import Path

ROWS = 60_175

Q1 = """
WITH lineitem AS (SELECT * FROM parquet.`{path}`)
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


def write_lineitem(directory: Path) -> Path:
    """`lineitem.parquet` with q1's columns and TPC-H's value ranges, seeded."""
    import datetime

    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = random.Random(42)
    d = pa.decimal128(15, 2)
    start = datetime.date(1992, 1, 2)
    cols: dict[str, list] = {k: [] for k in ("rf", "ls", "qty", "price", "disc", "tax", "ship")}
    for _ in range(ROWS):
        ship = start + datetime.timedelta(days=rng.randrange(0, 2526))
        qty = rng.randrange(1, 51)
        cols["rf"].append(rng.choice("RA") if ship <= datetime.date(1995, 6, 17) else "N")
        cols["ls"].append("F" if ship <= datetime.date(1995, 6, 17) else "O")
        cols["qty"].append(Decimal(qty))
        cols["price"].append(Decimal(qty * rng.randrange(90_000, 200_000)) / 100)
        cols["disc"].append(Decimal(rng.randrange(0, 11)) / 100)
        cols["tax"].append(Decimal(rng.randrange(0, 9)) / 100)
        cols["ship"].append(ship)
    table = pa.table(
        {
            "l_returnflag": pa.array(cols["rf"], pa.string()),
            "l_linestatus": pa.array(cols["ls"], pa.string()),
            "l_quantity": pa.array(cols["qty"], d),
            "l_extendedprice": pa.array(cols["price"], d),
            "l_discount": pa.array(cols["disc"], d),
            "l_tax": pa.array(cols["tax"], d),
            "l_shipdate": pa.array(cols["ship"], pa.date32()),
        }
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "lineitem.parquet"
    pq.write_table(table, path)
    return path


def connect_adbc(addr: str, token: str, ca: str | None):
    import adbc_driver_flightsql.dbapi as flight_sql  # ty: ignore[unresolved-import]  # the script's own uv env

    kwargs = {
        "adbc.flight.sql.rpc.with_cookie_middleware": "true",
        "adbc.flight.sql.authorization_header": f"Bearer {token}",
    }
    if ca:
        kwargs["adbc.flight.sql.client_option.tls_root_certs"] = Path(ca).read_text()
        kwargs["adbc.flight.sql.client_option.tls_override_hostname"] = "localhost"
        uri = f"grpc+tls://{addr}"
    else:
        uri = f"grpc://{addr}"
    return flight_sql.connect(uri, db_kwargs=kwargs, autocommit=True)


def via_adbc(addr: str, token: str, ca: str | None, q1: str):
    with connect_adbc(addr, token, ca) as conn, conn.cursor() as cur:
        # The cookie session: a command runs once and leaves its view for the next statement.
        cur.execute("CREATE TEMPORARY VIEW adbc_v AS SELECT 7 AS x")
        cur.fetchall()
        cur.execute("SELECT x FROM adbc_v")
        if cur.fetchall() != [(7,)]:
            raise SystemExit("the temporary view did not reach the next statement")
        cur.execute(q1)
        return cur.fetch_arrow_table()


def via_connect(addr: str, token: str, ca: str | None, q1: str):
    from pyspark.sql import SparkSession  # ty: ignore[unresolved-import]  # the script's own uv env

    host, port = addr.rsplit(":", 1)
    if ca:
        # grpc's Python client trusts the roots in this file.
        os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = ca
        host = "localhost"
    # PySpark's `token` turns TLS on; over plaintext the bearer goes as a plain header
    # (Connect URL parameters are request metadata).
    remote = f"sc://{host}:{port}/;" + (f"token={token};use_ssl=true" if ca else f"authorization=Bearer {token}")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    try:
        return spark.sql(q1).toArrow()
    finally:
        spark.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addr", required=True, help="host:port of krishiv server's Connect listener")
    parser.add_argument("--token", required=True, help="the bearer token of an admin principal")
    parser.add_argument("--data", required=True, help="a directory for lineitem.parquet")
    parser.add_argument("--ca", help="the CA to trust; both clients use TLS")
    args = parser.parse_args()
    q1 = Q1.format(path=write_lineitem(Path(args.data)))
    flight = via_adbc(args.addr, args.token, args.ca, q1)
    connect = via_connect(args.addr, args.token, args.ca, q1)
    problems = []
    if flight.schema.names != connect.schema.names:
        problems.append(f"names: {flight.schema.names} != {connect.schema.names}")
    else:
        for f, c in zip(flight.schema, connect.schema, strict=True):
            if f.type != c.type:
                problems.append(f"{f.name}: {f.type} != {c.type}")
    if flight.to_pylist() != connect.to_pylist():
        problems.append(f"rows:\n  flight  {flight.to_pylist()}\n  connect {connect.to_pylist()}")
    if sum(flight.column("count_order").to_pylist()) == 0 or flight.num_rows < 3:
        problems.append(f"{flight.num_rows} groups: q1 read nothing")
    if problems:
        print("ADBC and Spark Connect disagree:\n" + "\n".join(problems))
        return 1
    tls = "TLS" if args.ca else "plaintext"
    print(f"ADBC q1 ({tls}, token): {flight.num_rows} rows, {len(flight.schema)} columns, equal to Spark Connect")
    print("types: " + ", ".join(f"{f.name}={f.type}" for f in flight.schema))
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # grpc's channel finalizers can hang interpreter shutdown: leave without them.
    os._exit(code)
