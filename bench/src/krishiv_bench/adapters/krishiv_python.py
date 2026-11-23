# SPDX-License-Identifier: Apache-2.0
"""``krishiv-python``: the embedded engine through the ``krishiv`` Python package, in the harness's own process.

Tables are registered and queries timed exactly as ``krishiv-connect`` and ``krishiv-flightsql`` do (a catalog table
over each Parquet file, ``session_modes.table_sql``; ``sql(..).toArrow()``), so the three modes plan the same scans.
Against
``krishiv-connect`` the difference is the Spark Connect round trip only, and against ``krishiv`` (the native runner)
it is the Python layer (plan building in Python, the result's handover to pyarrow). The session starts from the
runner's settings (``session_modes``). The engine runs in this process, so peak RSS, CPU seconds and I/O are this
process's (``TreeProc(sample=False)``: no sampler thread inside the process it measures; the peak is the VmHWM at the
window's end and includes the harness's own few hundred MB). ``setup_s`` covers building the session and registering the
tables; ``import krishiv`` happens earlier, in ``version()``.
"""

from __future__ import annotations

import contextlib
import os
import time

from . import query_log, session_modes
from .base import Dataset, Measure, Query
from .proc_mem import TreeProc, tree_measure
from .spark_connect import SparkConnectEngine


class KrishivPythonEngine(SparkConnectEngine):
    name = "krishiv-python"

    def __init__(self, conf: list[tuple[str, str]] | None = None):
        super().__init__(remote="", container=None, container_data_root=None)
        self.conf = session_modes.with_runner_defaults(conf)
        self._tree: TreeProc | None = None
        self._qlog = query_log.QueryLog(self.name)

    def _session(self):
        if self.spark is None:
            from krishiv import SparkSession  # ty: ignore[unresolved-import]  # the wheel under test, not a bench dep

            builder = SparkSession.builder
            for k, v in self.conf:
                builder = builder.config(k, v)
            self.spark = builder.getOrCreate()
        return self.spark

    def version(self) -> str:
        import krishiv  # ty: ignore[unresolved-import]

        return f"krishiv {krishiv.__version__} ({krishiv.__commit__}, embedded Python)"

    def config(self) -> dict:
        import krishiv  # ty: ignore[unresolved-import]

        return {
            "mode": "embedded Python, in the harness process",
            "conf": self.conf,
            "effective": session_modes.effective(self.conf),
            "allocator": krishiv.__allocator__,
            "commit": krishiv.__commit__,
        }

    def setup(self, dataset: Dataset) -> None:
        spark = self._session()
        for t in dataset.tables:
            spark.sql(session_modes.table_sql(t, str(dataset.root / f"{t}.parquet")))

    def timed_run(self, query: Query) -> tuple[int, float]:
        spark = self._session()
        self._pending = None
        t0 = time.perf_counter()
        table = spark.sql(query.sql).toArrow()
        secs = time.perf_counter() - t0
        self._pending = table
        return table.num_rows, secs

    def cancel(self) -> None:
        if self.spark is not None:
            with contextlib.suppress(Exception):
                self.spark.interruptAll()

    def mark(self, label: str) -> None:
        if self.spark is not None and query_log.enabled():
            self.spark.sql(query_log.mark_sql(label)).toArrow()

    def flush_query_log(self) -> None:
        spark = self.spark
        if spark is not None:
            self._qlog.flush(lambda sql: spark.sql(sql).toArrow())

    def begin_measure(self) -> None:
        self.end_measure()
        self._tree = TreeProc(os.getpid(), sample=False)
        self._tree.start()

    def end_measure(self) -> Measure:
        tree, self._tree = self._tree, None
        return tree_measure(tree)

    def teardown(self) -> None:
        self.end_measure()
        if self.spark is not None:
            self.flush_query_log()
            with contextlib.suppress(Exception):
                self.spark.stop()
            self.spark = None
