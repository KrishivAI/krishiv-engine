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

    def __init__(self, threads: int | None = None, memory_limit: str | None = None, streaming: bool = False):
        # streaming: Polars' streaming engine instead of the default in-memory one (the fallback when the default
        # engine runs out of memory); the reproduce harness asks for it by the name "polars-streaming".
        self.streaming = streaming
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
            "engine": "streaming" if self.streaming else "default",
        }

    def setup(self, dataset: Dataset) -> None:
        import polars as pl

        self._frames = {t: pl.scan_parquet(table_path(dataset, t)) for t in dataset.tables}
        self._ctx = pl.SQLContext(frames=self._frames, eager=False)

    def _lazy(self, query: Query):
        return self._ctx.execute(query.sql)

    def _collect(self, query: Query):
        lazy = self._lazy(query)
        return lazy.collect(engine="streaming") if self.streaming else lazy.collect()

    def run(self, query: Query) -> list[tuple]:
        return self._collect(query).rows()

    def timed_run(self, query: Query) -> tuple[int, float]:
        self._pending = None  # a result nobody took (callers that do not checksum) is not kept alongside the next
        import time

        t0 = time.perf_counter()
        df = self._collect(query)
        secs = time.perf_counter() - t0
        self._pending = df
        return df.height, secs

    def _as_arrow(self, pending):
        return pending.to_arrow()
