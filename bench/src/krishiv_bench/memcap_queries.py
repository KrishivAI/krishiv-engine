"""The four ``memcap`` queries: one SQL text for DuckDB, Krishiv and Spark, the same logic as Polars expressions.

Adapted from the MIT-licensed ``engine-bench`` repository by Firat Celik (``queries.py``; notice in
``bench/README.md``). ``CHECKS`` is this harness's own: the cross-engine checksum of each result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

SQL = {
    "q1_filter_agg": """
        SELECT store_id, status, COUNT(*) AS orders, SUM(qty * price) AS revenue
        FROM sales
        WHERE ts >= TIMESTAMP '2026-06-01 00:00:00' AND status IN ('completed', 'shipped')
        GROUP BY store_id, status
    """,
    "q2_join_agg": """
        SELECT p.category, p.brand, COUNT(*) AS orders,
               SUM(s.qty * s.price) AS revenue, AVG(s.price - p.list_price) AS avg_delta
        FROM sales s JOIN products p ON s.product_id = p.product_id
        WHERE s.status <> 'cancelled'
        GROUP BY p.category, p.brand
    """,
    "q3_window_latest": """
        SELECT customer_id, order_id, ts, qty * price AS amount
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_id
                                         ORDER BY ts DESC, order_id DESC) AS rn
            FROM sales
        ) t
        WHERE rn = 1
    """,
    "q4_highcard_agg": """
        SELECT customer_id, COUNT(*) AS orders, SUM(qty * price) AS revenue,
               MIN(ts) AS first_ts, MAX(ts) AS last_ts
        FROM sales
        GROUP BY customer_id
    """,
}
QUERIES = list(SQL)

# Per-engine spelling of the same predicate. Krishiv does not parse `TIMESTAMP '...'` literals yet and rejects a
# cast to the session-zone TIMESTAMP, but takes a cast to TIMESTAMP_NTZ, the type of the (zone-less) `ts` column.
TIMESTAMP_LITERAL = "TIMESTAMP '2026-06-01 00:00:00'"
DIALECT_TIMESTAMP = {"krishiv": "CAST('2026-06-01 00:00:00' AS TIMESTAMP_NTZ)"}


def sql_for(engine: str, query: str) -> str:
    """The SQL text ``engine`` runs for ``query`` (the shared text, with the timestamp literal in its dialect)."""
    return SQL[query].replace(TIMESTAMP_LITERAL, DIALECT_TIMESTAMP.get(engine, TIMESTAMP_LITERAL))


# Checksum expressions over a result (a Parquet glob read by DuckDB). Integers compare exactly, the float sums
# within a relative 1e-9 (summation order differs between engines).
CHECKS = {
    "q1_filter_agg": (
        "COUNT(*) AS n, CAST(SUM(orders) AS BIGINT) AS orders, SUM(revenue) AS revenue,"
        " CAST(SUM(CAST(store_id AS BIGINT) * orders) AS BIGINT) AS store_mix"
    ),
    "q2_join_agg": (
        "COUNT(*) AS n, CAST(SUM(orders) AS BIGINT) AS orders, SUM(revenue) AS revenue, SUM(avg_delta) AS avg_delta"
    ),
    "q3_window_latest": (
        "COUNT(*) AS n, CAST(SUM(order_id) AS BIGINT) AS order_ids, SUM(amount) AS amount,"
        " CAST(SUM(CAST(customer_id AS BIGINT)) AS BIGINT) AS customers,"
        " CAST(SUM(CAST(epoch(ts) AS BIGINT)) AS BIGINT) AS ts_sum"
    ),
    "q4_highcard_agg": (
        "COUNT(*) AS n, CAST(SUM(orders) AS BIGINT) AS orders, SUM(revenue) AS revenue,"
        " CAST(SUM(CAST(customer_id AS BIGINT)) AS BIGINT) AS customers,"
        " CAST(MIN(epoch(first_ts)) AS BIGINT) AS first_ts, CAST(MAX(epoch(last_ts)) AS BIGINT) AS last_ts"
    ),
}


def polars_query(name: str, sales: pl.LazyFrame, products: pl.LazyFrame) -> pl.LazyFrame:
    import polars as pl

    amount = pl.col("qty").cast(pl.Float64) * pl.col("price")
    if name == "q1_filter_agg":
        return (
            sales.filter((pl.col("ts") >= pl.datetime(2026, 6, 1)) & pl.col("status").is_in(["completed", "shipped"]))
            .group_by("store_id", "status")
            .agg(pl.len().alias("orders"), amount.sum().alias("revenue"))
        )
    if name == "q2_join_agg":
        return (
            sales.filter(pl.col("status") != "cancelled")
            .join(products, on="product_id", how="inner")
            .group_by("category", "brand")
            .agg(
                pl.len().alias("orders"),
                amount.sum().alias("revenue"),
                (pl.col("price") - pl.col("list_price")).mean().alias("avg_delta"),
            )
        )
    if name == "q3_window_latest":
        rn = pl.int_range(pl.len()).over("customer_id", order_by=["ts", "order_id"], descending=True)
        return sales.filter(rn == 0).select("customer_id", "order_id", "ts", amount.alias("amount"))
    if name == "q4_highcard_agg":
        return sales.group_by("customer_id").agg(
            pl.len().alias("orders"),
            amount.sum().alias("revenue"),
            pl.col("ts").min().alias("first_ts"),
            pl.col("ts").max().alias("last_ts"),
        )
    raise KeyError(name)
