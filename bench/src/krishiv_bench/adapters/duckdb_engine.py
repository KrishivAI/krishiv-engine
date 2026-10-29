from __future__ import annotations

import os
import shutil
import time

import duckdb

from ..paths import data_root
from .base import Dataset, Engine, Measure, Query
from .proc_mem import LocalProc

# DuckDB's `default_null_order` values; Spark sorts NULLs first ascending and last descending.
NULL_ORDERS = frozenset(
    {"nulls_first", "nulls_last", "nulls_first_on_asc_last_on_desc", "nulls_last_on_asc_first_on_desc"}
)
SPARK_NULL_ORDER = "nulls_first_on_asc_last_on_desc"


class DuckDBEngine(Engine):
    """In-process DuckDB. Tables are views over Parquet (no load step), matching Spark's behaviour.

    memory_limit defaults to 12GB to match the Spark container cap (recorded in the result JSON).
    Resource probes use /proc/self (VmHWM reset via clear_refs) outside the timed region: the number is
    the whole harness process (Python + pyarrow baseline recorded separately). Spill: null (not exposed).

    ``null_order`` sets DuckDB's ``default_null_order``. It defaults to ``SPARK_NULL_ORDER``, Spark's NULLS FIRST
    ascending and NULLS LAST descending: DuckDB is the reference for Spark-semantics engines, so an ``ORDER BY``
    without an explicit null order must sort the same way in both. None keeps DuckDB's own (NULLS LAST).
    """

    name = "duckdb"

    def __init__(
        self, threads: int | None = None, memory_limit: str | None = "12GB", null_order: str | None = SPARK_NULL_ORDER
    ):
        if null_order is not None and null_order not in NULL_ORDERS:
            raise ValueError(f"null_order must be one of {sorted(NULL_ORDERS)}, got {null_order!r}")
        self.threads = threads or os.cpu_count()
        self.memory_limit = memory_limit
        self.null_order = null_order
        self.con: duckdb.DuckDBPyConnection | None = None
        self._p = LocalProc("self")

    def version(self) -> str:
        return duckdb.__version__

    def config(self) -> dict:
        cfg: dict = {"threads": self.threads, "memory_limit": self.memory_limit}
        if self.null_order:
            cfg["default_null_order"] = self.null_order
        return cfg

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
            if self.null_order:
                con.execute(f"SET default_null_order='{self.null_order}'")
            for t in dataset.tables:
                con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{table_source(dataset.root, t)}')")
        except BaseException:
            self.teardown()  # the caller's cleanup scope starts only after setup() returns
            raise

    def run(self, query: Query) -> list[tuple]:
        assert self.con is not None, "setup() not called"
        return self.con.execute(query.sql).fetchall()

    def timed_run(self, query: Query) -> tuple[int, float]:
        """Questions with a checksum query have answers too large to fetch into Python (``h2o-*``): the timed call
        materialises the answer in a temporary table, the benchmark's own method, and counts its rows."""
        if not query.check_sql:
            return super().timed_run(query)
        assert self.con is not None, "setup() not called"
        t0 = time.perf_counter()
        self.con.execute(f"CREATE OR REPLACE TEMP TABLE h2o_ans AS {query.sql}")
        secs = time.perf_counter() - t0
        try:
            return int(self.con.execute("SELECT COUNT(*) FROM h2o_ans").fetchone()[0]), secs  # ty: ignore[not-subscriptable]
        finally:
            self.con.execute("DROP TABLE IF EXISTS h2o_ans")

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
