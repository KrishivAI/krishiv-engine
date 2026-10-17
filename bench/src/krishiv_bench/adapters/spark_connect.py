from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from .base import Dataset, Engine, Measure, Query
from .proc_mem import DockerProc

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


class SparkConnectEngine(Engine):
    """Spark Connect client (pyspark[connect], version-pinned to the server in pyproject).

    Host data dir is mounted read-only at /data in the container, so Parquet paths are
    rewritten host_root -> container_root. Server-side metrics are read through `docker exec`
    on the JVM (`pgrep -x java`): VmHWM reset before each run via /proc/<pid>/clear_refs=5 (same method
    as DuckDB). A JVM keeps its committed heap, so peaks are cumulative/order-dependent. Failures of the
    probe raise (never silently record a lifetime HWM). Spill: not collected -> null.
    """

    name = "spark-connect"

    def __init__(
        self,
        remote: str = "sc://127.0.0.1:15002",
        container: str | None = "krishiv-bench-spark",
        container_data_root: str = "/data",
        host_data_root: str | None = None,
    ):
        self.remote = remote
        self.container = container
        self.cdata = container_data_root.rstrip("/")
        self.host_data_root = host_data_root
        self.spark: SparkSession | None = None
        self._p: DockerProc | None = None

    def _spark_version_server(self) -> str:
        return self.spark.version if self.spark else "?"

    def version(self) -> str:
        if self.spark is None:
            from pyspark.sql import SparkSession

            self.spark = SparkSession.builder.remote(self.remote).getOrCreate()
        return self.spark.version

    def config(self) -> dict:
        out = {"remote": self.remote, "container": self.container}
        spark = self.spark
        with contextlib.suppress(Exception):
            assert spark is not None
            for k in ("spark.sql.shuffle.partitions", "spark.driver.memory", "spark.master", "spark.sql.ansi.enabled"):
                out[k] = spark.conf.get(k, None)
        return out

    def setup(self, dataset: Dataset) -> None:
        from pyspark.sql import SparkSession

        self.spark = SparkSession.builder.remote(self.remote).getOrCreate()
        host_root = str(self.host_data_root or dataset.root.parents[1])
        for t in dataset.tables:
            p = str(dataset.root / f"{t}.parquet").replace(host_root, self.cdata, 1)
            self.spark.read.parquet(p).createOrReplaceTempView(t)
        if self.container:
            self._p = DockerProc(self.container)
            self._p.check()

    def run(self, query: Query) -> list[tuple]:
        assert self.spark is not None, "setup() not called"
        return [tuple(r) for r in self.spark.sql(query.sql).collect()]

    def cancel(self) -> None:
        if self.spark is not None:
            with contextlib.suppress(Exception):
                self.spark.interruptAll()

    def begin_measure(self) -> None:
        if self._p is not None:
            self._p.reset()
            self._rss0, _, self._cpu0 = self._p.read()

    def end_measure(self) -> Measure:
        if not self.container:
            return Measure()
        assert self._p is not None
        _, hwm, cpu = self._p.read()
        return Measure(peak_rss_bytes=hwm, baseline_rss_bytes=self._rss0, cpu_seconds=cpu - self._cpu0)

    def teardown(self) -> None:
        if self.spark:
            self.spark.stop()
