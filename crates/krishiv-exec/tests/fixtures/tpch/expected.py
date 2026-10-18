# SPDX-License-Identifier: Apache-2.0
"""Expected answers of the TPC-H join-shape tests (tests/hash_join/tpch.rs).

Loads the tables the test generator dumps and runs the official query text
(bench/queries/tpch) in DuckDB; writes one `|`-separated line per row.
Usage: python expected.py <dump dir>
"""

import pathlib
import sys

import duckdb

SCHEMAS = {
    "nation": "n_nationkey BIGINT, n_name VARCHAR, n_regionkey BIGINT",
    "supplier": "s_suppkey BIGINT, s_name VARCHAR, s_nationkey BIGINT, s_comment VARCHAR",
    "part": "p_partkey BIGINT, p_name VARCHAR, p_brand VARCHAR, p_type VARCHAR, "
    "p_size INTEGER, p_retailprice DECIMAL(15,2)",
    "partsupp": "ps_partkey BIGINT, ps_suppkey BIGINT, ps_supplycost DECIMAL(15,2)",
    "customer": "c_custkey BIGINT, c_nationkey BIGINT, c_phone VARCHAR, c_acctbal DECIMAL(15,2)",
    "orders": "o_orderkey BIGINT, o_custkey BIGINT, o_orderstatus VARCHAR, o_orderdate DATE, "
    "o_orderpriority VARCHAR, o_comment VARCHAR",
    "lineitem": "l_orderkey BIGINT, l_partkey BIGINT, l_suppkey BIGINT, l_quantity DECIMAL(15,2), "
    "l_extendedprice DECIMAL(15,2), l_discount DECIMAL(15,2), l_shipdate DATE, "
    "l_commitdate DATE, l_receiptdate DATE",
}


def main() -> None:
    dump = pathlib.Path(sys.argv[1])
    here = pathlib.Path(__file__).resolve().parent
    queries = here.parents[4] / "bench" / "queries" / "tpch"
    con = duckdb.connect()
    for name, cols in SCHEMAS.items():
        con.execute(f"CREATE TABLE {name} ({cols})")
        con.execute(f"COPY {name} FROM '{dump / (name + '.tbl')}' (DELIMITER '|', HEADER false)")
    for q in ["q04", "q09", "q13", "q16", "q21", "q22"]:
        rows = con.execute((queries / f"{q}.sql").read_text()).fetchall()
        lines = ["|".join(str(v) for v in row) for row in rows]
        (here / f"{q}.txt").write_text("".join(line + "\n" for line in lines))


if __name__ == "__main__":
    main()
