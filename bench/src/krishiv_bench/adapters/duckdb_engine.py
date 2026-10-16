from __future__ import annotations

import os
import threading
import time

import duckdb
import psutil

from .base import Dataset, Engine, Measure, Query


class DuckDBEngine(Engine):
    """In-process DuckDB. Tables are views over Parquet (no load step), matching Spark's behaviour.

    peak RSS = max RSS of this (harness) process sampled every 5 ms during the run; includes the
    Python interpreter baseline (~100 MB). Spill: DuckDB does not expose bytes spilled cheaply -> null.
    """
    name = "duckdb"

    def __init__(self, threads: int | None = None, memory_limit: str | None = None):
        self.threads = threads or os.cpu_count()
        self.memory_limit = memory_limit
        self.con = None
        self._proc = psutil.Process()
        self._stop = threading.Event()

    def version(self) -> str:
        return duckdb.__version__

    def config(self) -> dict:
        return {"threads": self.threads, "memory_limit": self.memory_limit}

    def setup(self, dataset: Dataset) -> None:
        self.con = duckdb.connect()
        self.con.execute(f"SET threads={self.threads}")
        if self.memory_limit:
            self.con.execute(f"SET memory_limit='{self.memory_limit}'")
        for t in dataset.tables:
            self.con.execute(
                f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{dataset.root / (t + '.parquet')}')"
            )

    def run(self, query: Query) -> list[tuple]:
        return self.con.execute(query.sql).fetchall()

    def begin_measure(self) -> None:
        self._peak = self._proc.memory_info().rss
        self._cpu0 = sum(self._proc.cpu_times()[:2])
        self._stop.clear()

        def sample():
            while not self._stop.is_set():
                self._peak = max(self._peak, self._proc.memory_info().rss)
                time.sleep(0.005)

        self._t = threading.Thread(target=sample, daemon=True)
        self._t.start()

    def end_measure(self) -> Measure:
        self._stop.set()
        self._t.join()
        cpu = sum(self._proc.cpu_times()[:2]) - self._cpu0
        return Measure(peak_rss_bytes=self._peak, cpu_seconds=cpu, spill_bytes=None)

    def teardown(self) -> None:
        if self.con:
            self.con.close()
