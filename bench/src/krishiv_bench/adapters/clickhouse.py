"""ClickHouse through ``chdb``: the ClickHouse server engine embedded as a Python library (a reference engine).

Why ``chdb`` and not a downloaded ``clickhouse-local`` binary: it is the same ClickHouse query engine and the
same Parquet reader, built from the same release, but it installs like every other engine here (a pinned PyPI
wheel, hash-locked in ``uv.lock``), so the benchmark needs no separate download, checksum or PATH step and the
version is recorded the same way as the others. The engine version is read from the running library
(``SELECT version()``) and recorded in the result JSON.

Tables are views over ``file('<parquet>', Parquet)`` (no load step, for parity with the other adapters).
Threads: ``max_threads``. Memory: ``max_memory_usage`` (per query; one query runs at a time) set to the limit, with
GROUP BY and ORDER BY spilling to disk once they use half of it (ClickHouse's own guidance), so a memory limit
costs time instead of failing the query. Scratch lives under ``$KRISHIV_DATA/tmp`` (not the small ``/tmp``).

``join_use_nulls=1`` is the SQL-standard outer-join behaviour (missing side is NULL, not the type default); it
is the setting ClickHouse's own TPC-H runs use.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from typing import Any

from ..paths import data_root
from .base import Dataset, Engine, Measure, Query, table_path
from .proc_mem import LocalProc

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_UNITS = {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}


def parse_memory_limit(text: str | int) -> int:
    """``12g`` / ``12GiB`` / ``12GB`` / ``1024`` -> bytes (binary units, as for the other engines). Rejects the rest."""
    m = re.fullmatch(r"([1-9][0-9]*)\s*([kmgt]?)(?:i?b)?", str(text).strip().lower())
    if m is None:
        raise ValueError(f"invalid memory limit {text!r}: expected a positive integer with an optional k/m/g/t unit")
    return int(m.group(1)) * _UNITS[m.group(2)]


def sql_string(text: str) -> str:
    """``text`` as a ClickHouse single-quoted string literal (backslash and quote escaped)."""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


class ClickHouseEngine(Engine):
    name = "clickhouse"

    def __init__(self, threads: int | None = None, memory_limit: str | int | None = "12g"):
        self.threads = os.cpu_count() or 1 if threads is None else threads
        if self.threads < 1:
            raise ValueError(f"invalid thread count {threads!r}: must be at least 1")
        self.memory_limit = memory_limit
        self.memory_bytes = parse_memory_limit(memory_limit) if memory_limit is not None else None
        self._session: Any = None
        self._p = LocalProc("self")

    @property
    def _scratch(self):
        return data_root() / "tmp" / f"clickhouse-{os.getpid()}"

    def version(self) -> str:
        import chdb

        if self._session is not None:
            return f"chdb {chdb.__version__} (ClickHouse {self._scalar('SELECT version()')})"
        return f"chdb {chdb.__version__}"

    def config(self) -> dict:
        spill = self.memory_bytes // 2 if self.memory_bytes else None
        return {
            "threads": self.threads,
            "memory_limit": self.memory_limit,
            "max_memory_usage": self.memory_bytes,
            "spill_threshold_bytes": spill,
            "join_use_nulls": True,
            "api": "chdb Session: views over file(<parquet>, Parquet)",
        }

    def _scalar(self, sql: str) -> str:
        return self._session.query(sql, "CSV").bytes().decode().strip().strip('"')

    def setup(self, dataset: Dataset) -> None:
        from chdb import session

        self._p.check()
        scratch = self._scratch
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            self._session = session.Session(str(scratch))
            self._session.query(f"SET max_threads = {self.threads}")
            self._session.query("SET join_use_nulls = 1")
            if self.memory_bytes:
                half = self.memory_bytes // 2
                for knob, value in (
                    ("max_memory_usage", self.memory_bytes),
                    ("max_bytes_before_external_group_by", half),
                    ("max_bytes_before_external_sort", half),
                ):
                    self._session.query(f"SET {knob} = {value}")
            for t in dataset.tables:
                if _IDENTIFIER.fullmatch(t) is None:  # the queries name tables bare, so it must be a plain identifier
                    raise ValueError(f"invalid table name {t!r}: expected letters, digits and underscores")
                src = sql_string(table_path(dataset, t))
                self._session.query(f"CREATE VIEW {t} AS SELECT * FROM file({src}, Parquet)")
        except BaseException:
            self.teardown()  # the caller's cleanup scope starts only after setup() returns
            raise

    def _table(self, query: Query):
        assert self._session is not None, "setup() not called"
        return self._session.query(query.sql, "ArrowTable")

    def run(self, query: Query) -> list[tuple]:
        table = self._table(query)
        cols = [c.to_pylist() for c in table.columns]
        return list(zip(*cols, strict=True)) if cols else []

    def timed_run(self, query: Query) -> tuple[int, float]:
        t0 = time.perf_counter()
        n = self._table(query).num_rows
        return n, time.perf_counter() - t0

    def begin_measure(self) -> None:
        self._p.reset()
        self._rss0, _, self._cpu0 = self._p.read()

    def end_measure(self) -> Measure:
        _, hwm, cpu = self._p.read()
        return Measure(peak_rss_bytes=hwm, baseline_rss_bytes=self._rss0, cpu_seconds=cpu - self._cpu0)

    def teardown(self) -> None:
        session, self._session = self._session, None
        try:
            if session is not None:
                session.close()
        finally:
            shutil.rmtree(self._scratch, ignore_errors=True)
