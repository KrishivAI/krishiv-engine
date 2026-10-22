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
Rows are compared as sorted multisets, unless the query has a top-level ORDER BY: then the sort-key values must match
position by position and the rows within each run of equal keys compare as a multiset (ties may legally come in any
order). Without known or derivable sort keys, whole rows are compared in order.
Official TPC-H answers (DuckDB `tpch_answers()`) are text; `match_answer` parses each cell by the engine value's type.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Comparison:
    """Outcome of comparing two result sets.

    ``status``: ``ok``, ``tie-at-limit`` (rows differ only inside the tie group cut by LIMIT; ``ok`` is True) or
    ``mismatch``. ``mode``: ``unordered``, ``tie-groups`` (sort keys known) or ``ordered-fallback`` (whole rows
    compared position by position because the sort keys are unknown).
    """

    ok: bool
    status: str
    mode: str
    message: str


def _check_sort_keys(sort_keys: Sequence[int], a: list, b: list) -> None:
    if not sort_keys:
        raise ValueError("sort_keys must be non-empty (use None when unknown)")
    width = min((len(r) for r in (*a, *b)), default=None)
    for k in sort_keys:
        if isinstance(k, bool) or not isinstance(k, int) or k < 0 or (width is not None and k >= width):
            raise ValueError(f"sort key index {k!r} is not a column of the result (row width {width})")


def _unordered_diff(a: list, b: list) -> int | None:
    """Index (in coarse-sorted order) of the first difference between two row multisets, None if they are equal.

    Both sides are sorted by the coarse key, so equal rows normally sit at the same position. Rows that tie on the
    coarse key but are distinct (``(1, None)`` and ``(1, "")``) may sit in either order, so a run of equal coarse
    keys is matched as a multiset with ``row_equal``.
    """
    sa, sb = sorted(a, key=_sort_key), sorted(b, key=_sort_key)
    if len(sa) != len(sb):
        return min(len(sa), len(sb))
    i = 0
    while i < len(sa):
        if row_equal(sa[i], sb[i]):
            i += 1
            continue
        k = _sort_key(sa[i])
        lo = i
        while lo > 0 and _sort_key(sa[lo - 1]) == k:
            lo -= 1
        hi = i + 1
        while hi < len(sa) and _sort_key(sa[hi]) == k:
            hi += 1
        if any(_sort_key(r) != k for r in sb[lo:hi]) or (hi < len(sb) and _sort_key(sb[hi]) == k):
            return i
        remaining = list(sb[lo:hi])
        for r in sa[lo:hi]:
            j = next((j for j, c in enumerate(remaining) if row_equal(r, c)), None)
            if j is None:
                return i
            del remaining[j]
        i = hi
    return None


def _position_failure(a: list, b: list, i: int) -> str:
    # distinguish a pure ordering difference from a value difference
    if _unordered_diff(a, b) is None:
        return f"ORDER differs at row {i} (same rows as a set)"
    return f"value mismatch at row {i}: {tuple(a[i])!r} vs {tuple(b[i])!r}"


def compare_rows(
    a: list,
    b: list,
    ordered: bool,
    *,
    sql: str | None = None,
    sort_keys: Sequence[int] | None = None,
    limit: int | None = None,
    offset: bool | None = None,
) -> Comparison:
    """Compare two result sets.

    Unordered: sorted multisets. Ordered with sort keys: the key values must match position by position, and the
    rows within each run of equal keys compare as a multiset. Ordered without sort keys: whole rows, position by
    position (``ordered-fallback``). ``sort_keys`` (column indexes) and ``limit`` default to what
    ``derive_sort_keys`` / ``derive_limit`` find in ``sql``; ``offset`` (the query has an OFFSET) likewise, so a
    stored ``limit`` keeps its OFFSET. Sort keys derived from ``sql`` that fall outside the rows' width are dropped
    (whole-row comparison); keys passed by the caller must be valid. If a LIMIT cut the result (``len >= limit``),
    the last tie group may legally hold different rows in the two results; so may the first when an OFFSET skipped
    rows, with or without a LIMIT (the skip cuts that group too). That is reported as ``tie-at-limit`` (the message
    names LIMIT or OFFSET), not as a mismatch.
    """
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise ValueError(f"limit must be a non-negative int or None, got {limit!r}")
    if len(a) != len(b):
        return Comparison(False, "mismatch", "unordered", f"row count {len(a)} != {len(b)}")
    if not ordered:
        i = _unordered_diff(a, b)
        if i is None:
            return Comparison(True, "ok", "unordered", "ok")
        sa, sb = sorted(a, key=_sort_key), sorted(b, key=_sort_key)
        msg = f"value mismatch (unordered, sorted row {i}): {tuple(sa[i])!r} vs {tuple(sb[i])!r}"
        return Comparison(False, "mismatch", "unordered", msg)
    derived = False
    if sql is not None:
        if sort_keys is None:
            sort_keys = derive_sort_keys(sql)
            derived = sort_keys is not None
        d_limit, d_offset = derive_limit(sql)
        if limit is None:
            limit = d_limit
        if offset is None:
            offset = d_offset
    if derived and sort_keys:
        width = min((len(r) for r in (*a, *b)), default=None)
        if width is not None and max(sort_keys) >= width:  # the parse and the result disagree: compare whole rows
            sort_keys = None
    offset = bool(offset)
    if sort_keys is None:
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            if not row_equal(x, y):
                return Comparison(False, "mismatch", "ordered-fallback", _position_failure(a, b, i))
        return Comparison(True, "ok", "ordered-fallback", "ok (ordered-fallback: sort keys unknown)")
    _check_sort_keys(sort_keys, a, b)

    def key(r):
        return tuple(r[k] for k in sort_keys)

    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if not row_equal(key(x), key(y)):
            return Comparison(False, "mismatch", "tie-groups", _position_failure(a, b, i))
    limit_cut = limit is not None and len(a) >= limit
    cuts: list[str] = []
    start = 0
    while start < len(a):
        end = start + 1
        while end < len(a) and row_equal(key(a[end]), key(a[start])):
            end += 1
        if _unordered_diff(a[start:end], b[start:end]) is not None:
            where = f"rows {start}..{end - 1} (sort key {key(a[start])!r})"
            by = [
                name for name, hit in (("LIMIT", limit_cut and end == len(a)), ("OFFSET", offset and start == 0)) if hit
            ]
            if not by:
                return Comparison(False, "mismatch", "tie-groups", f"value mismatch in tie group at {where}")
            cuts.append(f"{' and '.join(by)}, {where}")
        start = end
    if cuts:
        msg = f"tie-at-limit: rows differ only in the tie group cut by {'; '.join(cuts)}"
        return Comparison(True, "tie-at-limit", "tie-groups", msg)
    return Comparison(True, "ok", "tie-groups", "ok (tie-groups)")


def rows_match(
    a: list,
    b: list,
    ordered: bool,
    *,
    sql: str | None = None,
    sort_keys: Sequence[int] | None = None,
    limit: int | None = None,
    offset: bool | None = None,
) -> tuple[bool, str]:
    """``compare_rows`` as (ok, message). ordered=True requires the same row order, up to ties on the sort keys."""
    res = compare_rows(a, b, ordered, sql=sql, sort_keys=sort_keys, limit=limit, offset=offset)
    return res.ok, res.message


# ------------------------------------------------------------ ORDER BY detection
_NOISE = re.compile(r"'(?:[^']|'')*'|--[^\n]*|/\*.*?\*/", re.S)


def _top_level(sql: str) -> str:
    """The SQL with literals and comments blanked and everything inside parentheses replaced by spaces."""
    s = _NOISE.sub(lambda m: " " * len(m.group()), sql)
    out, depth = [], 0
    for ch in s:
        if ch == "(":
            depth += 1
            out.append(" ")
        elif ch == ")":
            depth = max(0, depth - 1)
            out.append(" ")
        else:
            out.append(ch if depth == 0 else " ")
    return "".join(out)


def _split_commas(text: str, masked: str) -> list[str]:
    """Split ``text`` at the commas that are outside parentheses (``masked`` is ``text`` through ``_top_level``)."""
    parts, last = [], 0
    for m in re.finditer(",", masked):
        parts.append(text[last : m.start()])
        last = m.end()
    parts.append(text[last:])
    return [p.strip() for p in parts]


def _norm(expr: str) -> str:
    return re.sub(r"\s+", "", expr).lower().replace('"', "")


def _select_names(sql: str, masked: str) -> list[tuple[str | None, str]] | None:
    """(output name, normalised expression) per column of the first top-level SELECT; None if not derivable."""
    m = re.search(r"\bselect\b(?:\s+(?:distinct|all)\b)?", masked, re.I)
    if not m:
        return None
    end = re.search(r"\bfrom\b", masked[m.end() :], re.I)
    stop = m.end() + end.start() if end else len(masked)
    items = _split_commas(sql[m.end() : stop], masked[m.end() : stop])
    out: list[tuple[str | None, str]] = []
    for item in items:
        if not item or item.endswith("*"):
            return None
        alias = re.search(r"\bas\s+(\"[^\"]+\"|[A-Za-z_]\w*)\s*$", item, re.I)
        if alias:
            out.append((_norm(alias.group(1)), _norm(item[: alias.start()])))
            continue
        col = re.fullmatch(r"(?:\w+\.)*(\w+)", item)
        out.append((col.group(1).lower() if col else None, _norm(item)))
    return out


def _final_order_by(masked: str) -> re.Match | None:
    ms = list(re.finditer(r"\border\s+by\b", masked, re.I))
    return ms[-1] if ms else None


def derive_sort_keys(sql: str) -> list[int] | None:
    """Output-column indexes of the final top-level ORDER BY, or None when they cannot be derived.

    A simple parse: each ORDER BY item must be a column ordinal, an output name/alias, or the text of a select-list
    expression. ``SELECT *``, keys outside the select list and the like give None (callers fall back to an ordered
    comparison and say so).
    """
    masked = _top_level(sql)
    ob = _final_order_by(masked)
    if ob is None:
        return None
    names = _select_names(sql, masked)
    if names is None:
        return None
    tail_end = re.search(r"\b(?:limit|offset|fetch)\b", masked[ob.end() :], re.I)
    stop = ob.end() + tail_end.start() if tail_end else len(masked)
    keys: list[int] = []
    for item in _split_commas(sql[ob.end() : stop], masked[ob.end() : stop]):
        item = re.sub(r"(?:\s+(?:asc|desc))?(?:\s+nulls\s+(?:first|last))?\s*$", "", item, flags=re.I).strip()
        if not item:
            return None
        if item.isdigit():
            idx = int(item) - 1
            if not 0 <= idx < len(names):
                return None
            keys.append(idx)
            continue
        n = _norm(item)
        last = n.rsplit(".", 1)[-1]
        idx = next((i for i, (name, expr) in enumerate(names) if n in (name, expr) or last == name), None)
        if idx is None:
            return None
        keys.append(idx)
    return keys


def derive_limit(sql: str) -> tuple[int | None, bool]:
    """(LIMIT n, OFFSET skips rows) of the final top-level ORDER BY clause; (None, False) if there is none.

    A number counts only when it is the whole clause: ``LIMIT 1 + 1`` gives no limit (no cut is inferred from a
    numeric prefix). The second item is true for ``OFFSET n`` with n > 0 and for an OFFSET that is not a plain
    number (``OFFSET 0 + 1``: it may skip rows); ``OFFSET 0`` skips nothing and is false. OFFSET works without a
    LIMIT.
    """
    masked = _top_level(sql)
    ob = _final_order_by(masked)
    if ob is None:
        return None, False
    tail = masked[ob.end() :]
    limit: int | None = None
    skips = False
    marks = list(re.finditer(r"\b(limit|offset|fetch)\b", tail, re.I))
    for n, m in enumerate(marks):
        stop = marks[n + 1].start() if n + 1 < len(marks) else len(tail)
        clause = tail[m.end() : stop].replace(";", " ").strip()
        word = m.group(1).lower()
        if word == "limit":
            whole = re.fullmatch(r"\d+", clause)  # a number only when it is the whole clause (not ``1 + 1``)
            limit = int(clause) if whole else None
        elif word == "offset":
            whole = re.fullmatch(r"(\d+)(?:\s+rows?)?", clause, re.I)
            skips = whole is None or int(whole.group(1)) > 0
    return limit, skips


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
