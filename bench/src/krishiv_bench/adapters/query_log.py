# SPDX-License-Identifier: Apache-2.0
"""The engine's own timing of each statement, for the mode comparison (embedded vs server).

With ``$KRISHIV_BENCH_QUERY_LOG`` set to a directory, the session-based Krishiv adapters (``krishiv-python``,
``krishiv-connect``, ``krishiv-flightsql``) read ``system.queries`` after every query of a suite (``run_suite`` calls
``flush_query_log``) and write the rows not written before as ``<engine>-<start ms>-<seq>.parquet``. Flushing per
query keeps the log whole although ``system.queries`` keeps at most 1,000 finished rows per principal.

Before each run ``run_suite`` calls ``mark``: an untimed ``SELECT '<query> <kind> <i>' AS krishiv_bench_mark``, so the
row after a marker is that run's statement and the pairing with the harness's samples is exact (set-up statements,
the untimed answer runs and memory-guard retries are in the log too). ``wall_ns`` of a timed run's row is the engine's
time for it; the harness's seconds for the run minus that is what the client side, the transport and the result
handover add.
"""

from __future__ import annotations

import contextlib
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

SQL = (
    "SELECT query_id, statement_class, state, query_text, submitted_at, started_at, ended_at, wall_ns, queued_ns, "
    "cpu_ns, rows_produced, bytes_read, peak_memory_bytes, result_cache_hit FROM system.queries ORDER BY submitted_at"
)


def enabled() -> bool:
    return bool(os.environ.get("KRISHIV_BENCH_QUERY_LOG"))


def mark_sql(label: str) -> str:
    """The marker statement for ``label`` (letters, digits, space, ``_ . : -`` kept; anything else dropped)."""
    return f"SELECT '{re.sub(r'[^A-Za-z0-9 _.:-]', '', label)}' AS krishiv_bench_mark"


class QueryLog:
    """The rows of ``system.queries`` one engine has written so far, and where the next file goes."""

    def __init__(self, engine: str):
        self.engine = engine
        self._seen: set[str] = set()
        self._seq = 0
        self._start = int(time.time() * 1000)

    def flush(self, fetch: Callable[[str], Any]) -> Path | None:
        """Writes the rows of ``fetch(SQL)`` (a pyarrow table) not written before; None when the log is off, nothing
        is new or the read fails (the measurement never fails the run it describes)."""
        out = os.environ.get("KRISHIV_BENCH_QUERY_LOG")
        if not out:
            return None
        with contextlib.suppress(Exception):
            import pyarrow.parquet as pq

            table = fetch(SQL)
            ids = table.column("query_id").to_pylist()
            keep = [i for i, q in enumerate(ids) if q not in self._seen]
            if not keep:
                return None
            self._seen.update(ids[i] for i in keep)
            path = Path(out) / f"{self.engine}-{self._start}-{self._seq:05d}.parquet"
            self._seq += 1
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table.take(keep), path)
            return path
        return None
