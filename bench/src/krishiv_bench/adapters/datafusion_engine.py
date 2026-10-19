"""DataFusion through ``datafusion-python``: Parquet tables registered as listing tables (no load step).

Threads = ``target_partitions`` (DataFusion's parallelism knob; its Tokio pool is sized to the host cores, which
equal the pinned thread count on the benchmark server). Memory = a fair spill pool of ``memory_limit``.
"""

from __future__ import annotations

import time
from typing import Any

from .base import Dataset, Engine, Query, table_path


def parse_bytes(text: str) -> int:
    units = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}
    t = text.strip().lower().removesuffix("b")
    return int(float(t[:-1]) * units[t[-1]]) if t and t[-1] in units else int(t)


class DataFusionEngine(Engine):
    name = "datafusion"

    def __init__(self, threads: int | None = None, memory_limit: str = "32g"):
        import os

        self.threads = threads or os.cpu_count() or 1
        self.memory_limit = memory_limit
        self._ctx: Any = None

    def version(self) -> str:
        import datafusion

        return datafusion.__version__

    def config(self) -> dict:
        return {"target_partitions": self.threads, "memory_limit": self.memory_limit, "memory_pool": "fair spill"}

    def setup(self, dataset: Dataset) -> None:
        from datafusion import RuntimeEnvBuilder, SessionConfig, SessionContext

        # ClickBench column names are CamelCase and unquoted in the queries (DuckDB/Spark match case-insensitively).
        cfg = (
            SessionConfig()
            .with_target_partitions(self.threads)
            .set("datafusion.sql_parser.enable_ident_normalization", "false")
        )
        rt = RuntimeEnvBuilder().with_fair_spill_pool(parse_bytes(self.memory_limit))
        self._ctx = SessionContext(cfg, rt)
        for t in dataset.tables:
            self._ctx.register_parquet(t, table_path(dataset, t))

    def run(self, query: Query) -> list[tuple]:
        rows: list[tuple] = []
        for batch in self._ctx.sql(query.sql).collect():
            cols = [c.to_pylist() for c in batch.columns]
            rows.extend(zip(*cols, strict=True) if cols else [])
        return rows

    def timed_run(self, query: Query) -> tuple[int, float]:
        t0 = time.perf_counter()
        n = sum(b.num_rows for b in self._ctx.sql(query.sql).collect())
        return n, time.perf_counter() - t0
