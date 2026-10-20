from __future__ import annotations

import os
import shutil

import duckdb

from ..paths import data_root
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
        self.con: duckdb.DuckDBPyConnection | None = None
        self._p = LocalProc("self")

    def version(self) -> str:
        return duckdb.__version__

    def config(self) -> dict:
        return {"threads": self.threads, "memory_limit": self.memory_limit}

    def setup(self, dataset: Dataset) -> None:
        self._p.check()
        con = self.con = duckdb.connect()
        con.execute(f"SET threads={self.threads}")
        # An in-memory DuckDB spills to `.tmp` in the working directory (the repo checkout) by
        # default; keep scratch under $KRISHIV_DATA, off the repo and off the small /tmp tmpfs.
        scratch = data_root() / "tmp" / f"duckdb-{os.getpid()}"
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            quoted = str(scratch).replace("'", "''")  # KRISHIV_DATA may contain a quote
            con.execute(f"SET temp_directory='{quoted}'")
            if self.memory_limit:
                con.execute(f"SET memory_limit='{self.memory_limit}'")
            for t in dataset.tables:
                con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{table_source(dataset.root, t)}')")
        except BaseException:
            self.teardown()  # the caller's cleanup scope starts only after setup() returns
            raise

    def run(self, query: Query) -> list[tuple]:
        assert self.con is not None, "setup() not called"
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
        shutil.rmtree(data_root() / "tmp" / f"duckdb-{os.getpid()}", ignore_errors=True)


def table_source(root, table: str) -> str:
    """The `read_parquet` argument for `table`: its file, or a glob over its directory of parts."""
    for candidate in (root / f"{table}.parquet", root / table):
        if candidate.is_dir():
            return str(candidate / "*.parquet")
    return str(root / f"{table}.parquet")
