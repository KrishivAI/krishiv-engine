#!/usr/bin/env python3
"""Write the small tables of the planner's expression and join tests (P2-T074) as Parquet.

`t` mixes NULLs into every type the evaluator handles; `u` joins with it. Usage (from repo root):

    uv run --project bench python -I tests/scripts/gen_planner_tables.py crates/krishiv-exec/tests/fixtures/planner
"""

import os
import sys

import duckdb

T = """
SELECT * FROM (VALUES
  (1, 0, 10::BIGINT, 1.50::DECIMAL(10,2), 'apple', DATE '1994-01-01', 0.5::DOUBLE),
  (2, 1, 20, 2.25, 'banana', DATE '1995-06-30', -1.25),
  (3, 2, -30, NULL, 'abc', DATE '1996-12-31', 2.0),
  (4, NULL, 40, 4.00, NULL, DATE '1997-02-28', NULL),
  (5, 3, 9223372036854775807, -5.75, 'ab', DATE '1998-08-02', 1.0e10),
  (6, 4, -50, 6.10, '42', NULL, 3.5),
  (7, NULL, 70, 7.00, 'a_b', DATE '1999-12-31', -0.0),
  (8, 3, 80, 8.88, 'xyz', DATE '2000-02-29', 7.25)
) AS v(id, a, b, d, s, dt, f)
"""
T_TYPES = "id INTEGER, a INTEGER, b BIGINT, d DECIMAL(10,2), s VARCHAR, dt DATE, f DOUBLE"

U = """
SELECT * FROM (VALUES
  (1, 1, 'one'), (2, 2, 'two'), (3, 2, 'two again'), (4, NULL, 'none'), (5, 9, 'nine')
) AS v(uid, k, name)
"""
U_TYPES = "uid INTEGER, k INTEGER, name VARCHAR"


def main() -> None:
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    con = duckdb.connect()
    for name, query, types in [("t", T, T_TYPES), ("u", U, U_TYPES)]:
        con.execute(f"CREATE TABLE {name} ({types})")
        con.execute(f"INSERT INTO {name} {query}")
        con.execute(f"COPY {name} TO '{os.path.join(out, name + '.parquet')}' (FORMAT parquet)")


if __name__ == "__main__":
    main()
