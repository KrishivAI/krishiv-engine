from __future__ import annotations

import subprocess

from .base import Dataset, Engine, Measure, Query


def _docker(container: str, script: str) -> str:
    r = subprocess.run(["docker", "exec", container, "bash", "-c", script],
                       capture_output=True, text=True, timeout=30)
    return r.stdout.strip()


class SparkConnectEngine(Engine):
    """Spark Connect client (pyspark[connect], version-pinned to the server in pyproject).

    Host data dir is mounted read-only at /data in the container, so Parquet paths are
    rewritten host_root -> container_root. Server-side metrics are read through `docker exec`
    on the JVM (pid of SparkSubmit): VmHWM (peak RSS; reset before each query via
    /proc/<pid>/clear_refs=5) and utime+stime (CPU s). Spill: Spark's spill metrics live in the
    UI/event log, not collected yet -> null.
    """
    name = "spark-connect"

    def __init__(self, remote: str = "sc://127.0.0.1:15002", container: str | None = "krishiv-bench-spark",
                 container_data_root: str = "/data", host_data_root: str | None = None):
        self.remote = remote
        self.container = container
        self.cdata = container_data_root.rstrip("/")
        self.host_data_root = host_data_root
        self.spark = None
        self._pid = None

    def _spark_version_server(self) -> str:
        return self.spark.version if self.spark else "?"

    def version(self) -> str:
        if self.spark is None:
            from pyspark.sql import SparkSession
            self.spark = SparkSession.builder.remote(self.remote).getOrCreate()
        return self.spark.version

    def config(self) -> dict:
        out = {"remote": self.remote, "container": self.container}
        try:
            for k in ("spark.sql.shuffle.partitions", "spark.driver.memory", "spark.master", "spark.sql.ansi.enabled"):
                out[k] = self.spark.conf.get(k, None)
        except Exception:
            pass
        return out

    def setup(self, dataset: Dataset) -> None:
        from pyspark.sql import SparkSession
        self.spark = SparkSession.builder.remote(self.remote).getOrCreate()
        host_root = str(self.host_data_root or dataset.root.parents[1])
        for t in dataset.tables:
            p = str(dataset.root / f"{t}.parquet").replace(host_root, self.cdata, 1)
            self.spark.read.parquet(p).createOrReplaceTempView(t)
        if self.container:
            self._pid = _docker(self.container, "pgrep -f org.apache.spark.deploy.SparkSubmit | head -1") or None

    def run(self, query: Query) -> list[tuple]:
        return [tuple(r) for r in self.spark.sql(query.sql).collect()]

    def _cpu_ticks(self) -> int | None:
        if not self._pid:
            return None
        s = _docker(self.container, f"cut -d' ' -f14,15 /proc/{self._pid}/stat")
        try:
            a, b = s.split()
            return int(a) + int(b)
        except ValueError:
            return None

    def begin_measure(self) -> None:
        if self._pid:
            _docker(self.container, f"echo 5 > /proc/{self._pid}/clear_refs")
            self._ticks0 = self._cpu_ticks()

    def end_measure(self) -> Measure:
        if not self._pid:
            return Measure()
        hwm = _docker(self.container, f"awk '/VmHWM/{{print $2}}' /proc/{self._pid}/status")
        t1 = self._cpu_ticks()
        return Measure(
            peak_rss_bytes=int(hwm) * 1024 if hwm.isdigit() else None,
            cpu_seconds=((t1 - self._ticks0) / 100.0) if t1 is not None and self._ticks0 is not None else None,
            spill_bytes=None,
        )

    def teardown(self) -> None:
        if self.spark:
            self.spark.stop()
