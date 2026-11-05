"""Order-independent checksum of a full query result, taken from its Arrow table outside the timed span.

Per column (by position; names and row order are ignored): the non-NULL count, and then

* floating point: the sum as DOUBLE (compared with a relative tolerance: parallel sums differ in the last bits);
* integers and decimals: the exact sum;
* everything else (text, dates, timestamps, booleans, binary): the sum of ``hash(value)`` as HUGEINT.

Plus the row count and a row hash: the sum of ``hash`` of the non-float columns of each row, which ties values in
different columns to the same row. Two runs of one engine on one query must agree; the checksum is not comparable
across engines (their Arrow types differ), so each engine is held to its own first, DuckDB-verified answer.

Floating-point columns are summed, and also summed weighted by a hash of the row's non-float columns, so a measure
moved to another row changes the checksum (when the row has any non-float column). Blind spots, by design (it is a
guard against a timed run that computes something else, not a proof of equality): a change below about ``REL_TOL``
of a column's sum goes unseen, floats in rows with no non-float column are only summed, and the DECIMAL(38) sum can
overflow HUGEINT on a huge result.
"""

from __future__ import annotations

import math
from typing import Any

import pyarrow as pa
import pyarrow.types as pt

REL_TOL = 1e-9


def _quote(i: int) -> str:
    return f'"c{i}"'


def checksum(table: pa.Table) -> dict[str, Any]:
    import duckdb

    table = table.rename_columns([f"c{i}" for i in range(table.num_columns)])
    exprs: list[str] = ["count(*)"]
    kinds: list[str] = ["rows"]
    row_cols: list[str] = []
    for i, f in enumerate(table.schema):
        c, t = _quote(i), f.type
        exprs.append(f"count({c})")
        kinds.append("count")
        if pt.is_floating(t):
            exprs.append(f"sum(CAST({c} AS DOUBLE))")
            kinds.append("float")
        elif pt.is_integer(t) or pt.is_decimal(t):
            exprs.append(f"CAST(sum({c}) AS VARCHAR)")
            kinds.append("exact")
            row_cols.append(f"c{i} := {c}")
        else:
            exprs.append(f"CAST(sum(CAST(hash({c}) AS HUGEINT)) AS VARCHAR)")
            kinds.append("exact")
            row_cols.append(f"c{i} := {c}")
    if row_cols:
        row_hash = f"hash(struct_pack({', '.join(row_cols)}))"
        exprs.append(f"CAST(sum(CAST({row_hash} AS HUGEINT)) AS VARCHAR)")
        kinds.append("exact")
        # Ties each float to its row: the float times a weight in (0, 1] drawn from the row's other columns, summed
        # (compared with a tolerance like the plain sum). Two floats swapped between rows change it.
        weight = f"(CAST({row_hash} % 1000003 AS DOUBLE) + 1) / 1000003"
        for i, f in enumerate(table.schema):
            if pt.is_floating(f.type):
                exprs.append(f"sum(CAST({_quote(i)} AS DOUBLE) * {weight})")
                kinds.append("float")
    con = duckdb.connect()
    try:
        con.register("t", table)
        row = con.execute(f"SELECT {', '.join(exprs)} FROM t").fetchone()
    finally:
        con.close()
    assert row is not None
    return {"values": list(row), "kinds": kinds}


def same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a["kinds"] != b["kinds"]:
        return False
    for kind, x, y in zip(a["kinds"], a["values"], b["values"], strict=True):
        if kind == "float":
            if x is None or y is None:
                if x is not y:
                    return False
            elif not (x == y or (math.isnan(x) and math.isnan(y)) or math.isclose(x, y, rel_tol=REL_TOL, abs_tol=1e-9)):
                return False
        elif x != y:
            return False
    return True
