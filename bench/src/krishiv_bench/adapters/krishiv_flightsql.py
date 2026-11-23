# SPDX-License-Identifier: Apache-2.0
"""``krishiv-flightsql``: ``krishiv server`` driven over Arrow Flight SQL with the ADBC driver (P3-T101).

The server is started exactly as for ``krishiv-connect`` (same binary, ``--port 0 --connect-port 0`` on loopback, the
runner's settings by default); Flight SQL is served on the Spark Connect port when the binary is built with the
``flight-sql`` feature. The client is ``adbc-driver-flightsql``: one connection (one server session, cookie-bound),
tables registered as temporary views over Parquet in that session, each query timed around ``execute`` +
``fetch_arrow_table`` (the full result as Arrow). Peak RSS, CPU seconds and I/O are those of the server process tree,
as for ``krishiv-connect``.

A timeout cancels the statement in flight with ADBC's ``adbc_cancel`` (the server answers ``CancelFlightInfo``) and
keeps the connection and its views. Only cancel may be called from another thread than the one inside the cursor, so
when it fails the connection is not closed under that thread: the next statement opens a new connection and
registers the views again.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any

from . import query_log
from .base import Dataset, Query
from .krishiv_connect import KrishivConnectEngine


class KrishivFlightSqlEngine(KrishivConnectEngine):
    name = "krishiv-flightsql"

    def __init__(self, *args: Any, **kw: Any):
        super().__init__(*args, **kw)
        self._conn: Any = None
        self._dataset: Dataset | None = None
        self._lock = threading.Lock()
        self._cursor: Any = None  # the cursor of the statement in flight, for `cancel`
        self._stale = False  # a cancel failed: reconnect before the next statement

    def _uri(self) -> str:
        # The server reports `sc://host:port`; Flight SQL listens on the same port.
        return "grpc://" + self.remote.removeprefix("sc://").split("/", 1)[0]

    def version(self) -> str:
        return f"krishiv server ({self.binary.name}), Flight SQL over ADBC"

    def config(self) -> dict:
        out = super().config()
        out["server"] = "krishiv server, loopback, Arrow Flight SQL (adbc-driver-flightsql)"
        return out

    def _connect(self) -> None:
        from adbc_driver_flightsql import dbapi  # ty: ignore[unresolved-import]  # installed for the modes run only

        assert self._dataset is not None, "setup() not called"
        conn = dbapi.connect(self._uri(), autocommit=True)
        with conn.cursor() as cur:
            for t in self._dataset.tables:
                path = str(self._dataset.root / f"{t}.parquet").replace("`", "``")
                cur.execute(f"CREATE OR REPLACE TEMPORARY VIEW {t} AS SELECT * FROM parquet.`{path}`")
        self._conn = conn

    def _connection(self) -> Any:
        if self._stale or self._conn is None:
            old, self._conn, self._stale = self._conn, None, False
            if old is not None:
                with contextlib.suppress(Exception):
                    old.close()
            self._connect()
        return self._conn

    def setup(self, dataset: Dataset) -> None:
        self._start_server()
        self._dataset = dataset
        try:
            self._connect()
        except BaseException:
            with contextlib.suppress(Exception):
                self.teardown()
            raise

    def _execute(self, sql: str, fetch: str) -> tuple[Any, float]:
        """Runs ``sql`` on a cursor ``cancel`` can reach; returns ``cursor.<fetch>()`` and the seconds around it."""
        with self._connection().cursor() as cur:
            with self._lock:
                self._cursor = cur
            try:
                t0 = time.perf_counter()
                cur.execute(sql)
                out = getattr(cur, fetch)()
                return out, time.perf_counter() - t0
            finally:
                with self._lock:
                    self._cursor = None

    def run(self, query: Query) -> list[tuple]:
        rows, _ = self._execute(query.sql, "fetchall")
        return [tuple(r) for r in rows]

    def timed_run(self, query: Query) -> tuple[int, float]:
        self._pending = None
        table, secs = self._execute(query.sql, "fetch_arrow_table")
        self._pending = table
        return table.num_rows, secs

    def cancel(self) -> None:
        with self._lock:
            cur = self._cursor
        if cur is None:
            return
        try:
            cur.adbc_cancel()
        except Exception:
            self._stale = True

    def mark(self, label: str) -> None:
        if query_log.enabled():
            self._execute(query_log.mark_sql(label), "fetch_arrow_table")

    def flush_query_log(self) -> None:
        if self._conn is None or self._proc is None:
            return
        with contextlib.suppress(Exception):
            self._qlog.flush(lambda sql: self._execute(sql, "fetch_arrow_table")[0])

    def teardown(self) -> None:
        self.flush_query_log()
        conn, self._conn = self._conn, None
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
        # KrishivConnectEngine.teardown flushes through pyspark (no session here: nothing) and stops the server.
        super().teardown()
