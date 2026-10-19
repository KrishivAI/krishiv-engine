"""Spark and Sail as in-process-session adapters sharing one Spark-SQL driver.

* ``SparkLocalEngine`` (``spark-local``): Apache Spark 4.2 ``local[N]`` in this process (``pyspark==4.2.0``, a JVM
  child of the host process). Tuned for a single node: AQE on, ``shuffle.partitions`` = cores, UI off,
  driver heap = the pinned memory limit.
* ``SailEngine`` (``sail``): LakeSail's ``pysail`` Spark Connect server started on a loopback port in this process
  (Rust, no JVM), driven by the pinned ``pyspark`` Connect client.

Both register each Parquet table as a temp view over the same files; Spark ANSI mode keeps its Spark 4 default.
"""

from __future__ import annotations

import os
import time
from typing import Any

from .base import Dataset, Engine, Query, table_path


class _SparkSQLEngine(Engine):
    def __init__(self, threads: int | None = None, memory_limit: str = "32g"):
        self.threads = threads or os.cpu_count() or 1
        self.memory_limit = memory_limit
        self.spark: Any = None

    def _register(self, dataset: Dataset) -> None:
        for t in dataset.tables:
            self.spark.read.parquet(table_path(dataset, t)).createOrReplaceTempView(t)

    def run(self, query: Query) -> list[tuple]:
        return [tuple(r) for r in self.spark.sql(query.sql).collect()]

    def timed_run(self, query: Query) -> tuple[int, float]:
        t0 = time.perf_counter()
        n = len(self.spark.sql(query.sql).collect())
        return n, time.perf_counter() - t0

    def teardown(self) -> None:
        if self.spark is not None:
            self.spark.stop()


class SparkLocalEngine(_SparkSQLEngine):
    name = "spark-local"

    def version(self) -> str:
        import pyspark

        return pyspark.__version__

    def config(self) -> dict:
        cfg: dict = {
            "master": f"local[{self.threads}]",
            "driver_memory": self.memory_limit,
            "shuffle_partitions": self.threads,
            "aqe": True,
            "ui": False,
        }
        if self.spark is not None:
            cfg["java"] = self.spark._jvm.System.getProperty("java.version")
        return cfg

    def setup(self, dataset: Dataset) -> None:
        from pyspark.sql import SparkSession

        self.spark = (
            SparkSession.builder.master(f"local[{self.threads}]")
            .appName("krishiv-bench-spark-local")
            .config("spark.driver.memory", self.memory_limit)
            .config("spark.sql.shuffle.partitions", str(self.threads))
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.ui.enabled", "false")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.sql.ansi.doubleQuotedIdentifiers", "true")
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel("ERROR")
        self._register(dataset)


class SailEngine(_SparkSQLEngine):
    name = "sail"

    def __init__(self, threads: int | None = None, memory_limit: str = "32g"):
        super().__init__(threads, memory_limit)
        self._server: Any = None

    def version(self) -> str:
        from importlib.metadata import version

        return f"pysail {version('pysail')}"

    def config(self) -> dict:
        # Sail sizes its worker pool to the host cores (== the pinned thread count here) and has no memory knob.
        return {
            "threads_requested": self.threads,
            "memory_limit": None,
            "server": "pysail SparkConnectServer, loopback",
        }

    def setup(self, dataset: Dataset) -> None:
        from pysail.spark import SparkConnectServer
        from pyspark.sql import SparkSession

        self._server = SparkConnectServer("127.0.0.1", 0)
        self._server.start(background=True)
        address = self._server.listening_address
        if address is None:
            raise RuntimeError("Sail Spark Connect server did not report a listening address")
        port = address[1]
        self.spark = SparkSession.builder.remote(f"sc://127.0.0.1:{port}").getOrCreate()
        self._register(dataset)

    def teardown(self) -> None:
        super().teardown()
        if self._server is not None:
            self._server.stop()
