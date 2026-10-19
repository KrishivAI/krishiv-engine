"""Polars 2 (lazy): ``pl.scan_parquet`` per table, queries through ``SQLContext``, which builds the same lazy plan the
native API does (all 22 TPC-H queries parse, so no hand-written native versions are needed).

Threads: ``POLARS_MAX_THREADS`` is read when polars is imported, so the process that hosts this adapter sets it
before the first import (the reproduce worker does; ``config()`` records what was requested and what polars reports).
Polars has no memory limit knob; the cap is recorded as ``None`` (unbounded).
"""

from __future__ import annotations

import os
from typing import Any

from .base import Dataset, Engine, Query, table_path


class PolarsEngine(Engine):
    name = "polars"

    def __init__(self, threads: int | None = None, memory_limit: str | None = None):
        self.threads = threads or os.cpu_count() or 1
        self.memory_limit = memory_limit  # accepted for interface parity; polars cannot enforce it
        os.environ.setdefault("POLARS_MAX_THREADS", str(self.threads))
        self._ctx: Any = None
        self._frames: dict[str, Any] = {}

    def version(self) -> str:
        import polars as pl

        return pl.__version__

    def config(self) -> dict:
        import polars as pl

        return {
            "threads_requested": self.threads,
            "threads_effective": pl.thread_pool_size(),
            "memory_limit": None,
            "api": "lazy: scan_parquet + SQLContext",
        }

    def setup(self, dataset: Dataset) -> None:
        import polars as pl

        self._frames = {t: pl.scan_parquet(table_path(dataset, t)) for t in dataset.tables}
        self._ctx = pl.SQLContext(frames=self._frames, eager=False)

    def _lazy(self, query: Query):
        return self._ctx.execute(query.sql)

    def run(self, query: Query) -> list[tuple]:
        return self._lazy(query).collect().rows()

    def timed_run(self, query: Query) -> tuple[int, float]:
        import time

        t0 = time.perf_counter()
        height = self._lazy(query).collect().height
        return height, time.perf_counter() - t0
