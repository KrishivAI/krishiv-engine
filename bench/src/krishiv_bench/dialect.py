"""Known, verified dialect or type differences between an engine and the DuckDB reference.

A result that differs from DuckDB's is a ``mismatch`` (a wrong answer until shown otherwise). When the difference is
a documented property of the engine's own SQL dialect or type system and the query text is the engine's published
one, the verdict is ``mismatch-dialect`` instead, with the reason recorded. The entry is evidence-backed: each
reason below was reproduced against DuckDB on the benchmark data. The engine still never gets a timing for the
query (only ``ok`` verdicts are timed), and a query that is not listed stays a plain ``mismatch``.
"""

from __future__ import annotations

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


def dialect_reason(engine: str, suite: str, query_id: str) -> str | None:
    return KNOWN_DIALECT_MISMATCHES.get((engine, suite, query_id))


# (engine, suite) -> why the engine does not run the suite
UNSUPPORTED_SUITES: dict[tuple[str, str], str] = {
    ("clickhouse", "tpcds"): (
        "TPC-DS is not run for ClickHouse: the reference is a ClickHouse benchmark reference (ClickBench and TPC-H "
        "only), and no ClickHouse-published TPC-DS query set is vendored"
    ),
}


def unsupported_reason(engine: str, suite: str) -> str | None:
    return UNSUPPORTED_SUITES.get((engine, suite))
