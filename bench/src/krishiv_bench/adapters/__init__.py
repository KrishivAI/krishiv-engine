"""Engine adapter registry. Add Flink etc. here (see base.Engine)."""

from __future__ import annotations

from .base import Dataset, Engine, Measure, Query


def make_engine(name: str, **kw) -> Engine:
    if name == "duckdb":
        from .duckdb_engine import DuckDBEngine

        return DuckDBEngine(**kw)
    if name == "spark-connect":
        from .spark_connect import SparkConnectEngine

        return SparkConnectEngine(**kw)
    if name == "krishiv":
        from .krishiv import KrishivEngine

        return KrishivEngine(**kw)
    if name == "polars":
        from .polars_engine import PolarsEngine

        return PolarsEngine(**kw)
    if name == "datafusion":
        from .datafusion_engine import DataFusionEngine

        return DataFusionEngine(**kw)
    if name == "spark-local":
        from .spark_local import SparkLocalEngine

        return SparkLocalEngine(**kw)
    if name == "sail":
        from .spark_local import SailEngine

        return SailEngine(**kw)
    if name == "clickhouse":
        from .clickhouse import ClickHouseEngine

        return ClickHouseEngine(**kw)
    raise ValueError(f"unknown engine {name!r}; available: {', '.join(ENGINES)}")


ENGINES = ["duckdb", "spark-connect", "krishiv", "polars", "datafusion", "spark-local", "sail", "clickhouse"]
__all__ = ["ENGINES", "Dataset", "Engine", "Measure", "Query", "make_engine"]
