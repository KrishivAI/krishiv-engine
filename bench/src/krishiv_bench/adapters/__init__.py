"""Engine adapter registry. Add Sail/Polars/Flink here (see base.Engine)."""

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

        return KrishivEngine()
    raise ValueError(f"unknown engine {name!r}; available: duckdb, spark-connect, krishiv (stub)")


ENGINES = ["duckdb", "spark-connect", "krishiv"]
__all__ = ["ENGINES", "Dataset", "Engine", "Measure", "Query", "make_engine"]
