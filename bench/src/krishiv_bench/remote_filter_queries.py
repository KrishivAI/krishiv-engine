"""The one ``remote-filter`` query: a very selective scan over web-text Parquet.

Source data is the ``sample/10BT`` config of ``HuggingFaceFW/fineweb-edu`` (14 Parquet shards, columns ``id``,
``text``, ``token_count``, ``int_score``, ``language_score`` among others). The query keeps rows with
``int_score >= 4 AND token_count >= 4000 AND language_score >= 0.95`` (about 0.2% of rows) and writes the
result to ONE zstd-compressed Parquet file.

The file-mode spelling is one SQL text every engine runs (no dialect difference today: plain comparisons
and a plain glob view). The S3 spellings differ per engine's reader, so ``s3_from`` gives the SQL engines'
FROM fragment while the native readers (Polars, Spark) build their own scan in the child.
"""

from __future__ import annotations

QUERY = "filter_edu"
QUERIES = [QUERY]
TABLE = "docs"
FILTER = "int_score >= 4 AND token_count >= 4000 AND language_score >= 0.95"

# SELECT *: the checksum needs ``id`` and ``token_count``; the file is the deliverable, so keep every column.
SQL = {QUERY: f"SELECT * FROM {TABLE} WHERE {FILTER}"}


def sql_for(engine: str, query: str, table: str = TABLE) -> str:
    """The SQL text ``engine`` runs for ``query`` in file mode (a view over local Parquet)."""
    if query not in SQL:
        raise ValueError(f"unknown remote-filter query {query!r}; available: {', '.join(QUERIES)}")
    _ = engine  # one spelling for every engine today; the hook stays so a dialect has somewhere to go
    return SQL[query].replace(TABLE, table) if table != TABLE else SQL[query]


def s3_from(engine: str, url: str) -> str:
    """The FROM fragment reading ``url`` (``s3://bucket/prefix/*.parquet``) for the SQL engines.

    DuckDB reads S3 through ``read_parquet`` (httpfs); Krishiv reads it through the ``parquet.`...```
    path table. Polars and Spark use their native readers (storage options / hadoop conf), so they have
    no fragment here.
    """
    if engine == "duckdb":
        quoted = url.replace("'", "''")
        return f"read_parquet('{quoted}')"
    if engine == "krishiv":
        return f"parquet.`{url}`"
    raise ValueError(f"engine {engine!r} reads S3 through its native reader, not a SQL fragment")


def s3_sql(engine: str, query: str, url: str) -> str:
    """The full S3-mode SQL text for the SQL engines."""
    if query not in SQL:
        raise ValueError(f"unknown remote-filter query {query!r}; available: {', '.join(QUERIES)}")
    return f"SELECT * FROM {s3_from(engine, url)} WHERE {FILTER}"


__all__ = ["FILTER", "QUERIES", "QUERY", "SQL", "TABLE", "s3_from", "s3_sql", "sql_for"]
