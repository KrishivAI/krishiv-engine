"""Known, verified dialect or type differences between an engine and the DuckDB reference.

A result that differs from DuckDB's is a ``mismatch`` (a wrong answer until shown otherwise). When the difference is
a documented property of the engine's own SQL dialect or type system and the query text is the engine's published
one, and the observed difference matches it (``_VERIFIERS``), the verdict is ``mismatch-dialect`` instead, with the
reason recorded; any other difference on a listed query stays ``mismatch``. The entry is evidence-backed: each
reason below was reproduced against DuckDB on the benchmark data. The engine still never gets a timing for the
query (only ``ok`` verdicts are timed), and a query that is not listed stays a plain ``mismatch``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from decimal import Decimal

from .compare import values_equal

Rows = Sequence[Sequence]

# (engine, suite, query id) -> reason
KNOWN_DIALECT_MISMATCHES: dict[tuple[str, str, str], str] = {
    ("clickhouse", "tpch", "q08"): (
        "ClickHouse divides DECIMAL by DECIMAL into a DECIMAL(38,4) and truncates (SF1 1996: 0.0414); DuckDB returns "
        "a DOUBLE (0.041485...), so the values differ by more than half a unit of the 4th decimal place"
    ),
    ("clickhouse", "clickbench", "q04"): (
        "ClickHouse avg() of a signed 64-bit integer column sums in a 64-bit integer and wraps on overflow "
        "(UserID values span the whole int64 range); DuckDB accumulates in a wider type"
    ),
    ("clickhouse", "clickbench", "q28"): (
        "ClickHouse length() counts bytes of a String; DuckDB length() counts characters (UTF-8 text)"
    ),
    ("clickhouse", "clickbench", "q29"): (
        "ClickHouse length() counts bytes of a String; DuckDB length() counts characters (UTF-8 text)"
    ),
}


def _is_number(v) -> bool:
    return isinstance(v, (int, float, Decimal)) and not isinstance(v, bool)


def _by_key(rows: Rows, ref: Rows) -> list[tuple[Sequence, Sequence]] | None:
    """The rows paired by their first column (a unique key), or None if the shapes or key sets differ."""
    if not ref or len(rows) != len(ref) or any(len(r) != len(ref[0]) for r in (*rows, *ref)):
        return None
    mine, theirs = {r[0]: r for r in rows}, {r[0]: r for r in ref}
    if len(mine) != len(rows) or mine.keys() != theirs.keys():
        return None
    return [(mine[k], theirs[k]) for k in theirs]


def _truncated_decimals(rows: Rows, ref: Rows) -> bool:
    """Every cell equal, or numeric and less than one unit of the 4th decimal place apart (truncation)."""
    pairs = _by_key(rows, ref)
    return pairs is not None and all(
        values_equal(a, b) or (_is_number(a) and _is_number(b) and abs(float(a) - float(b)) < 1e-4)
        for x, y in pairs
        for a, b in zip(x, y, strict=True)
    )


def _single_number(rows: Rows, ref: Rows) -> bool:
    """One row of one number on both sides (an aggregate over the whole table)."""
    return all(len(r) == 1 and len(r[0]) == 1 and _is_number(r[0][0]) for r in (rows, ref))


def _byte_lengths(rows: Rows, ref: Rows) -> bool:
    """Rows keyed by the first column; the second (a mean length) is not smaller in bytes, the rest are equal."""
    pairs = _by_key(rows, ref)
    return pairs is not None and all(
        len(x) > 1
        and _is_number(x[1])
        and _is_number(y[1])
        and float(x[1]) >= float(y[1])
        and all(values_equal(a, b) for i, (a, b) in enumerate(zip(x, y, strict=True)) if i != 1)
        for x, y in pairs
    )


# (engine, suite, query id) -> does the observed difference match the documented one? (rows, DuckDB rows)
_VERIFIERS: dict[tuple[str, str, str], Callable[[Rows, Rows], bool]] = {
    ("clickhouse", "tpch", "q08"): _truncated_decimals,
    ("clickhouse", "clickbench", "q04"): _single_number,
    ("clickhouse", "clickbench", "q28"): _byte_lengths,
    ("clickhouse", "clickbench", "q29"): _byte_lengths,
}


def dialect_reason(engine: str, suite: str, query_id: str, rows: Rows, ref: Rows) -> str | None:
    """The documented reason, only when the observed difference (``rows`` against DuckDB's ``ref``) is that one."""
    key = (engine, suite, query_id)
    reason, verify = KNOWN_DIALECT_MISMATCHES.get(key), _VERIFIERS.get(key)
    return reason if reason is not None and verify is not None and verify(rows, ref) else None


# (engine, suite) -> why the engine does not run the suite
UNSUPPORTED_SUITES: dict[tuple[str, str], str] = {
    ("clickhouse", "tpcds"): (
        "TPC-DS is not run for ClickHouse: the reference is a ClickHouse benchmark reference (ClickBench and TPC-H "
        "only), and no ClickHouse-published TPC-DS query set is vendored"
    ),
}


def unsupported_reason(engine: str, suite: str) -> str | None:
    return UNSUPPORTED_SUITES.get((engine, suite))
