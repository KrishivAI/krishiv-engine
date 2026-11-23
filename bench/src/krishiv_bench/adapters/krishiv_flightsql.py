# SPDX-License-Identifier: Apache-2.0
"""``krishiv-flightsql``: ``krishiv server`` driven over Arrow Flight SQL with the ADBC driver (P3-T101).

The server is started exactly as for ``krishiv-connect`` (same binary, ``--port 0 --connect-port 0`` on loopback);
Flight SQL is served on the Spark Connect port when the binary is built with the ``flight-sql`` feature. The client
is ``adbc-driver-flightsql``: one connection (one server session, cookie-bound), tables registered as temporary views
over Parquet in that session, each query timed around ``execute`` + ``fetch_arrow_table`` (the full result as Arrow).
Peak RSS, CPU seconds and I/O are those of the server process tree, as for ``krishiv-connect``.
"""

from __future__ import annotations

import contextlib
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

    def _uri(self) -> str:
        # The server reports `sc://host:port`; Flight SQL listens on the same port.
        return "grpc://" + self.remote.removeprefix("sc://").split("/", 1)[0]

    def version(self) -> str:
        return f"krishiv server ({self.binary.name}), Flight SQL over ADBC"

    def config(self) -> dict:
        out = super().config()
        out["server"] = "krishiv server, loopback, Arrow Flight SQL (adbc-driver-flightsql)"
        return out

    def setup(self, dataset: Dataset) -> None:
        self._start_server()
        try:
            from adbc_driver_flightsql import dbapi  # ty: ignore[unresolved-import]  # installed for the modes run only

            self._conn = dbapi.connect(self._uri(), autocommit=True)
            with self._conn.cursor() as cur:
                for t in dataset.tables:
                    path = str(dataset.root / f"{t}.parquet").replace("`", "``")
                    cur.execute(f"CREATE OR REPLACE TEMPORARY VIEW {t} AS SELECT * FROM parquet.`{path}`")
        except BaseException:
            with contextlib.suppress(Exception):
                self.teardown()
            raise

    def run(self, query: Query) -> list[tuple]:
        assert self._conn is not None, "setup() not called"
        with self._conn.cursor() as cur:
            cur.execute(query.sql)
            return [tuple(r) for r in cur.fetchall()]

    def timed_run(self, query: Query) -> tuple[int, float]:
        assert self._conn is not None, "setup() not called"
        self._pending = None
        with self._conn.cursor() as cur:
            t0 = time.perf_counter()
            cur.execute(query.sql)
            table = cur.fetch_arrow_table()
            secs = time.perf_counter() - t0
        self._pending = table
        return table.num_rows, secs

    def cancel(self) -> None:
        # ADBC has no session-wide interrupt; closing the connection ends the in-flight statement.
        conn, self._conn = self._conn, None
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()

    def _dump_query_log(self) -> None:
        conn = self._conn
        if conn is None:
            return

        def fetch(sql: str) -> Any:
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetch_arrow_table()

        query_log.dump(self.name, fetch)

    def teardown(self) -> None:
        self._dump_query_log()
        conn, self._conn = self._conn, None
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
        # KrishivConnectEngine.teardown stops the pyspark session (none here: spark is None) and the server.
        super().teardown()
