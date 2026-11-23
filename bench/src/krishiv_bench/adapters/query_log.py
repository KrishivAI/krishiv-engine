# SPDX-License-Identifier: Apache-2.0
"""The engine's own timing of each statement, for the mode comparison (embedded vs server).

With ``$KRISHIV_BENCH_QUERY_LOG`` set to a directory, the session-based Krishiv adapters (``krishiv-python``,
``krishiv-connect``, ``krishiv-flightsql``) read ``system.queries`` before they tear down and write it there as
``<engine>-<unix ms>.parquet``. Its ``wall_ns`` is the engine's time for a statement; the harness's seconds for the same
statement minus that is what the client side, the transport and the result handover add.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

SQL = (
    "SELECT query_id, statement_class, state, query_text, submitted_at, started_at, ended_at, wall_ns, queued_ns, "
    "cpu_ns, rows_produced, bytes_read, peak_memory_bytes, result_cache_hit FROM system.queries"
)


def dump(engine: str, fetch: Callable[[str], Any]) -> Path | None:
    """Writes ``fetch(SQL)`` (a pyarrow table) to the log directory; None when it is unset or the read fails."""
    out = os.environ.get("KRISHIV_BENCH_QUERY_LOG")
    if not out:
        return None
    with contextlib.suppress(Exception):  # the measurement never fails the run it describes
        import pyarrow.parquet as pq

        table = fetch(SQL)
        path = Path(out) / f"{engine}-{int(time.time() * 1000)}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        return path
    return None
