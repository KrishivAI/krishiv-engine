# SPDX-License-Identifier: Apache-2.0
"""``krishiv-flightsql``: ``krishiv server`` driven over Arrow Flight SQL with the ADBC driver (P3-T101).

The server is started exactly as for ``krishiv-connect`` (same binary, ``--port 0 --connect-port 0`` on loopback, the
runner's settings by default); Flight SQL is served on the Spark Connect port when the binary is built with the
``flight-sql`` feature. The client is ``adbc-driver-flightsql``: one connection (one server session, cookie-bound),
tables registered as catalog tables over Parquet (``session_modes.table_sql``, as in the other
session modes), each query timed around ``execute`` +
``fetch_arrow_table`` (the full result as Arrow). Peak RSS, CPU seconds and I/O are those of the server process tree,
as for ``krishiv-connect``.

The ADBC driver drops the server's ``arrow_flight_session_id`` cookie unless its cookie middleware is on
(``adbc.flight.sql.rpc.with_cookie_middleware``); without it every statement runs in a new session and its state is
gone, so the connection always turns it on.

A timeout cancels every statement in flight with ADBC's ``adbc_cancel`` (the server answers ``CancelFlightInfo``) and
keeps the connection. Only cancel may be called from another thread than the one inside the cursor, so
a connection is never closed while a statement runs on it: when a cancel fails, the next statement opens a new
connection and registers the tables again, and the old one is closed when its last statement returns. Teardown
closes only the idle connections; one with a statement still inside is closed by that statement's return, which the
server's stop (right after) brings about.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any

from . import query_log, session_modes
from .base import Dataset, Query
from .krishiv_connect import KrishivConnectEngine

# The server keeps a session per `arrow_flight_session_id` cookie; the driver sends it back only with this on.
DB_KWARGS = {"adbc.flight.sql.rpc.with_cookie_middleware": "true"}


class _Conn:
    """One ADBC connection, the statements running on it, and whether a reconnect replaced it."""

    def __init__(self, conn: Any):
        self.conn = conn
        self.running = 0
        self.retired = False


class KrishivFlightSqlEngine(KrishivConnectEngine):
    name = "krishiv-flightsql"

    def __init__(self, *args: Any, **kw: Any):
        super().__init__(*args, **kw)
        self._conn: _Conn | None = None
        self._retired: list[_Conn] = []  # replaced while a statement still ran on them; closed when it returns
        self._dataset: Dataset | None = None
        self._lock = threading.Lock()
        self._cursors: dict[int, tuple[Any, _Conn]] = {}  # every cursor inside a statement and its connection
        self._stale = False  # a cancel failed: the next statement opens a new connection

    def _uri(self) -> str:
        # The server reports `sc://host:port`; Flight SQL listens on the same port.
        return "grpc://" + self.remote.removeprefix("sc://").split("/", 1)[0]

    def version(self) -> str:
        return f"krishiv server ({self.binary.name}), Flight SQL over ADBC"

    def config(self) -> dict:
        out = super().config()
        out["server"] = "krishiv server, loopback, Arrow Flight SQL (adbc-driver-flightsql, cookie middleware)"
        return out

    def _open(self) -> Any:
        """A new connection (one server session) with the dataset's tables registered (``IF NOT EXISTS``)."""
        from adbc_driver_flightsql import dbapi  # ty: ignore[unresolved-import]  # installed for the modes run only

        assert self._dataset is not None, "setup() not called"
        conn = dbapi.connect(self._uri(), db_kwargs=dict(DB_KWARGS), autocommit=True)
        try:
            with conn.cursor() as cur:
                for t in self._dataset.tables:
                    cur.execute(session_modes.table_sql(t, str(self._dataset.root / f"{t}.parquet")))
        except BaseException:
            with contextlib.suppress(Exception):
                conn.close()
            raise
        return conn

    def _retire(self, c: _Conn) -> Any:
        """Under ``_lock``: ``c`` takes no more statements; returns its handle to close now, or None while one runs."""
        c.retired = True
        if c.running:
            self._retired.append(c)
            return None
        return c.conn

    def _acquire(self) -> _Conn:
        """The connection for the next statement, counted as running on it."""
        close = None
        with self._lock:
            if self._stale and self._conn is not None:
                close, self._conn = self._retire(self._conn), None
            self._stale = False
            c = self._conn
            if c is not None:
                c.running += 1
        if close is not None:
            with contextlib.suppress(Exception):
                close.close()
        if c is None:
            c = _Conn(self._open())
            c.running = 1
            with self._lock:
                self._conn = c
        return c

    def _release(self, c: _Conn) -> None:
        with self._lock:
            c.running -= 1
            done = c.retired and c.running == 0
            if done and c in self._retired:
                self._retired.remove(c)
        if done:
            with contextlib.suppress(Exception):
                c.conn.close()

    def setup(self, dataset: Dataset) -> None:
        self._start_server()
        self._dataset = dataset
        try:
            self._conn = _Conn(self._open())
        except BaseException:
            with contextlib.suppress(Exception):
                self.teardown()
            raise

    def _execute(self, sql: str, fetch: str) -> tuple[Any, float]:
        """Runs ``sql`` on a cursor ``cancel`` can reach; returns ``cursor.<fetch>()`` and the seconds around it."""
        c = self._acquire()
        try:
            with c.conn.cursor() as cur:
                with self._lock:
                    self._cursors[id(cur)] = (cur, c)
                try:
                    t0 = time.perf_counter()
                    cur.execute(sql)
                    out = getattr(cur, fetch)()
                    return out, time.perf_counter() - t0
                finally:
                    with self._lock:
                        self._cursors.pop(id(cur), None)
        finally:
            self._release(c)

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
            cursors = list(self._cursors.values())
        for cur, c in cursors:  # the timed-out statement, and any earlier one that has not stopped yet
            try:
                cur.adbc_cancel()
            except Exception:
                # Only a statement left running on the current connection makes it unusable; one on a connection
                # already retired is closed by its own return.
                if c is self._conn:
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
        with self._lock:
            # Retire the current connection too: an idle one is closed now; one with a statement still inside (a
            # timed-out statement that did not stop) stays open until that statement returns (`_release`), which the
            # server's stop below brings about, so no connection is closed under a running statement.
            idle = self._retire(self._conn) if self._conn is not None else None
            self._conn = None
        if idle is not None:
            with contextlib.suppress(Exception):
                idle.close()
        # KrishivConnectEngine.teardown flushes through pyspark (no session here: nothing) and stops the server.
        super().teardown()
