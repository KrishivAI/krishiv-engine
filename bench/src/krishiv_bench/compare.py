"""Type-aware result comparison (replaces the old rounded-hash approach).

Rules (values are the Python objects the engines return):
  * int / str / date / datetime / bool / None: exact (strings right-trimmed: CHAR padding).
  * Decimal vs Decimal: compared at the *smaller* of the two scales (round half-up), so Spark's
    AVG(DECIMAL) -> DECIMAL(scale 6) matches an exact value that has more digits, but two
    DECIMALs of equal scale must be exactly equal.
  * Decimal vs float: |d - f| <= 0.5 * 10^-scale(d) + 1e-6 * |f|. The double comes from another
    computation, not just another type: DuckDB divides DECIMALs in double precision, while Spark
    rounds each quotient to the result scale (HALF_UP) before an enclosing SUM adds them, so a sum
    of N quotients may differ from the double by up to N half-units of the last place (TPC-DS q66:
    Spark and Krishiv 11.694972, DuckDB 11.6949728724). The half-unit covers the rounding of one
    value; the float-vs-float relative tolerance covers the rest, as for any double.
  * float vs float: relative tolerance 1e-6 (abs 1e-12 near zero).
  * int vs Decimal: exact numeric equality.
Rows are compared *in order* when the query has a top-level ORDER BY, else as sorted multisets.
Official TPC-H answers (DuckDB `tpch_answers()`) are text; `match_answer` parses each cell by the engine value's type.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import math
import re
from decimal import Decimal
from pathlib import Path

REL_FLOAT = 1e-6


# ------------------------------------------------------------ encoding (sidecar JSON)
def enc(v):
    if v is None or (isinstance(v, (str, int)) and not isinstance(v, bool)):
        return v
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, Decimal):
        return {"d": str(v)}
    if isinstance(v, float):
        return {"f": repr(v)}
    if isinstance(v, (dt.datetime, dt.date)):
        return {"t": v.isoformat()}
    return str(v)


def dec(v):
    if isinstance(v, dict):
        if "d" in v:
            return Decimal(v["d"])
        if "f" in v:
            return float(v["f"])
        if "t" in v:
            return v["t"]
    return v


def save_rows(path: Path, rows_by_query: dict[str, list]) -> None:
    with gzip.open(path, "wt") as f:
        json.dump({q: [[enc(c) for c in r] for r in rows] for q, rows in rows_by_query.items()}, f)


def load_rows(path: Path) -> dict[str, list[tuple]]:
    with gzip.open(path, "rt") as f:
        return {q: [tuple(dec(c) for c in r) for r in rows] for q, rows in json.load(f).items()}


# ------------------------------------------------------------ scalar comparison
def _scale(d: Decimal) -> int:
    exponent = d.as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def _round_to(d: Decimal, scale: int) -> Decimal:
    from decimal import ROUND_HALF_UP, localcontext

    with localcontext() as c:
        c.prec = 60
        return d.quantize(Decimal(1).scaleb(-scale), rounding=ROUND_HALF_UP)


def values_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, str) or isinstance(b, str):
        return str(a).rstrip() == str(b).rstrip()
    if isinstance(a, (dt.date, dt.datetime)):
        a = a.isoformat()
    if isinstance(b, (dt.date, dt.datetime)):
        b = b.isoformat()
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    if isinstance(a, bool):
        a = int(a)
    if isinstance(b, bool):
        b = int(b)
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return math.isclose(a, b, rel_tol=REL_FLOAT, abs_tol=1e-12)
    if isinstance(a, float):
        a, b = b, a
    if isinstance(b, float):  # a is Decimal or int, b float
        da = Decimal(a)
        tol = Decimal("0.5") * Decimal(1).scaleb(-_scale(da)) + Decimal(REL_FLOAT) * Decimal(abs(b))
        return abs(da - Decimal(b)) <= tol
    da, db = Decimal(a), Decimal(b)  # Decimal/int mix
    s = min(_scale(da), _scale(db)) if (isinstance(a, Decimal) and isinstance(b, Decimal)) else None
    if s is None:
        return da == db
    return _round_to(da, s) == _round_to(db, s)


def row_equal(r1, r2) -> bool:
    return len(r1) == len(r2) and all(values_equal(x, y) for x, y in zip(r1, r2, strict=False))


def _sort_key(row):
    # coarse key stable across Decimal/float differences: strings kept, numbers rounded to 4 dp,
    # so near-equal rows sort together
    out = []
    for c in row:
        if c is None:
            out.append("")
        elif isinstance(c, (Decimal, float)):
            out.append(f"{float(c):020.4f}")
        elif isinstance(c, (int, bool)):
            out.append(f"{int(c):020.4f}")
        else:
            out.append(str(c).rstrip())
    return tuple(out)


def rows_match(a: list, b: list, ordered: bool) -> tuple[bool, str]:
    """Compare two result sets. Returns (ok, message). ordered=True requires identical row order."""
    if len(a) != len(b):
        return False, f"row count {len(a)} != {len(b)}"
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if not row_equal(x, y):
            if ordered:
                # distinguish a pure ordering difference from a value difference
                sa, sb = sorted(a, key=_sort_key), sorted(b, key=_sort_key)
                if all(row_equal(p, q) for p, q in zip(sa, sb, strict=False)):
                    return False, f"ORDER differs at row {i} (same rows as a set)"
                return False, f"value mismatch at row {i}: {tuple(x)!r} vs {tuple(y)!r}"
            break
    else:
        return True, "ok"
    sa, sb = sorted(a, key=_sort_key), sorted(b, key=_sort_key)
    for i, (x, y) in enumerate(zip(sa, sb, strict=False)):
        if not row_equal(x, y):
            return False, f"value mismatch (unordered, sorted row {i}): {tuple(x)!r} vs {tuple(y)!r}"
    return True, "ok"


# ------------------------------------------------------------ ORDER BY detection
def has_top_level_order_by(sql: str) -> bool:
    s = re.sub(r"'(?:[^']|'')*'", "''", sql)
    s = re.sub(r"--[^\n]*", "", s)
    depth, i, low = 0, 0, s.lower()
    while i < len(low):
        ch = low[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and low.startswith("order", i) and (i == 0 or (not low[i - 1].isalnum() and low[i - 1] != "_")):
            m = re.match(r"order\s+by\b", low[i:])
            if m:
                return True
        i += 1
    return False


# ------------------------------------------------------------ official answers
def load_tpch_answers(sf: float) -> dict[str, list[list[str]]]:
    """{qNN: [row-of-strings...]} from DuckDB's bundled official answer sets (header dropped)."""
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch")
    out = {}
    for nr, ans in con.execute(
        "SELECT query_nr, answer FROM tpch_answers() WHERE scale_factor=? ORDER BY 1", [float(sf)]
    ).fetchall():
        lines = [ln for ln in ans.split("\n") if ln.strip() != ""]
        out[f"q{nr:02d}"] = [ln.split("|") for ln in lines[1:]]
    return out


def match_answer(v, s: str) -> bool:
    """Engine value vs official-answer text cell."""
    if v is None:
        return s in ("", "NULL")
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, int):
        try:
            return Decimal(s) == v
        except Exception:
            return False
    if isinstance(v, (Decimal, float)):
        try:
            a = Decimal(s)
        except Exception:
            return False
        if isinstance(v, float):
            return values_equal(v, float(a))
        # exact-valued answer vs DECIMAL: compare at the coarser *engine* scale only if engine is coarser
        sv, sa = _scale(v), _scale(a)
        return v == a if sv >= sa else _round_to(a, sv) == v
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat() == s
    return str(v).rstrip() == s.rstrip()


def answer_check(rows: list, answer: list[list[str]], ordered: bool) -> tuple[bool, str]:
    if len(rows) != len(answer):
        return False, f"row count {len(rows)} != official {len(answer)}"

    def ok_row(r, a):
        return len(r) == len(a) and all(match_answer(x, y) for x, y in zip(r, a, strict=False))

    if all(ok_row(r, a) for r, a in zip(rows, answer, strict=False)):
        return True, "ok"
    # unordered fallback: greedy match on sorted-by-text keys
    remaining = list(answer)
    for r in rows:
        for j, a in enumerate(remaining):
            if ok_row(r, a):
                del remaining[j]
                break
        else:
            i = next(i for i, (rr, aa) in enumerate(zip(rows, answer, strict=False)) if not ok_row(rr, aa))
            return False, f"value mismatch vs official answer at row {i}: {tuple(rows[i])!r} vs {answer[i]!r}"
    return (False, "ORDER differs from official answer (same rows as a set)") if ordered else (True, "ok")
