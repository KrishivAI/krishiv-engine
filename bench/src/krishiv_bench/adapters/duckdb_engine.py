from __future__ import annotations

import os

import duckdb

from .base import Dataset, Engine, Measure, Query
from .proc_mem import LocalProc


class DuckDBEngine(Engine):
    """In-process DuckDB. Tables are views over Parquet (no load step), matching Spark's behaviour.

    memory_limit defaults to 12GB to match the Spark container cap (recorded in the result JSON).
    Resource probes use /proc/self (VmHWM reset via clear_refs) outside the timed region: the number is
    the whole harness process (Python + pyarrow baseline recorded separately). Spill: null (not exposed).
    """
    name = "duckdb"

    def __init__(self, threads: int | None = None, memory_limit: str | None = "12GB"):
        self.threads = threads or os.cpu_count()
        self.memory_limit = memory_limit
        self.con = None
        self._p = LocalProc("self")

    def version(self) -> str:
        return duckdb.__version__

    def config(self) -> dict:
        return {"threads": self.threads, "memory_limit": self.memory_limit}

    def setup(self, dataset: Dataset) -> None:
        self._p.check()
        self.con = duckdb.connect()
        self.con.execute(f"SET threads={self.threads}")
        if self.memory_limit:
            self.con.execute(f"SET memory_limit='{self.memory_limit}'")
        for t in dataset.tables:
            self.con.execute(
                f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{dataset.root / (t + '.parquet')}')")

    def run(self, query: Query) -> list[tuple]:
        return self.con.execute(query.sql).fetchall()

    def cancel(self) -> None:
        if self.con:
            self.con.interrupt()

    def begin_measure(self) -> None:
        self._p.reset()
        self._rss0, _, self._cpu0 = self._p.read()

    def end_measure(self) -> Measure:
        _, hwm, cpu = self._p.read()
        return Measure(peak_rss_bytes=hwm, baseline_rss_bytes=self._rss0, cpu_seconds=cpu - self._cpu0)

    def teardown(self) -> None:
        if self.con:
            self.con.close()
