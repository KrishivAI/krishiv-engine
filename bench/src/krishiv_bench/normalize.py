"""Result normalisation + checksum so engines can be compared.

TPC-H answers differ in *type* across engines (DuckDB/Spark return DECIMAL for sums of
DECIMAL columns, but averages come back as DOUBLE in one and DECIMAL in the other; sum order
changes the low bits of doubles). Rules:
  * Decimal/float/int that are not integral-typed bigints -> float rounded to 5 significant digits
    (Spark returns AVG over DECIMAL as DECIMAL with scale 6 (e.g. 0.049985) where DuckDB returns DOUBLE (0.0499853), so 5 digits is the most both agree on);
    ints stay ints (so counts must match exactly) unless they came as Decimal/float.
  * date/datetime -> ISO string; str -> rstrip (CHAR padding); None -> "NULL"; bool -> int.
  * Rows are sorted as normalised text so ORDER BY ties / non-deterministic tie order do not matter.
    (Queries with LIMIT + ties on the sort key can legitimately differ; reported as mismatch.)
The checksum is only computed for outputs of <= CHECKSUM_MAX_ROWS rows.
"""
from __future__ import annotations

import datetime as dt
import hashlib
from decimal import Decimal

CHECKSUM_MAX_ROWS = 10_000


def norm_value(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    if isinstance(v, (Decimal, float)):
        f = float(v)
        if f != f:
            return "NaN"
        if f == 0:
            return "0"
        return f"{f:.5g}"
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    return str(v).rstrip()


def normalize_rows(rows) -> list[str]:
    return sorted("|".join(norm_value(c) for c in r) for r in rows)


def checksum(rows) -> str | None:
    if len(rows) > CHECKSUM_MAX_ROWS:
        return None
    h = hashlib.sha256()
    for line in normalize_rows(rows):
        h.update(line.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]
