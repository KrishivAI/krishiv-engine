"""Engine adapter interface.

Lifecycle: ``setup(dataset)`` once -> ``run(query)`` many times -> ``teardown()``.
``run`` executes the query to completion and returns every result row (the harness
normalises and checksums them). Adapters report engine-side resource use via
``begin_measure()`` / ``end_measure()`` which bracket each timed run (not the timing itself).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Dataset:
    name: str  # "tpch", "tpcds", ...
    sf: float
    root: Path  # directory holding <table>.parquet files (host path)
    tables: list[str] = field(default_factory=list)


@dataclass
class Query:
    id: str  # "q01"
    sql: str
    suite: str


@dataclass
class Measure:
    peak_rss_bytes: int | None = None  # process RSS high-water during the run (VmHWM after clear_refs reset)
    baseline_rss_bytes: int | None = None  # process RSS just before the run
    cpu_seconds: float | None = None
    spill_bytes: int | None = None


def table_path(dataset: Dataset, table: str) -> str:
    """Parquet source of ``table``: ``<root>/<table>.parquet``, or a glob over a directory of part files."""
    f = dataset.root / f"{table}.parquet"
    if f.is_file():
        return str(f)
    for d in (dataset.root / table, f):
        if d.is_dir():
            return str(d / "*.parquet")
    raise FileNotFoundError(f"no Parquet source for table {table!r} under {dataset.root}")


class Engine(ABC):
    name: str = "?"

    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def setup(self, dataset: Dataset) -> None:
        """Register the dataset's tables (as views over Parquet; no pre-loading, for parity)."""

    @abstractmethod
    def run(self, query: Query) -> list[tuple]:
        """Execute to completion, return all rows."""

    def timed_run(self, query: Query) -> tuple[int, float]:
        """Execute to completion; return ``(row_count, seconds)`` measured around the execution only.

        Default: time ``run``. Adapters whose ``run`` converts rows in Python override this so that conversion is
        not part of the measurement.
        """
        t0 = time.perf_counter()
        rows = self.run(query)
        return len(rows), time.perf_counter() - t0

    def teardown(self) -> None:  # noqa: B027  # optional hook, default no-op  # pragma: no cover
        pass

    def cancel(self) -> None:  # noqa: B027  # optional hook, default no-op
        """Best-effort abort of the in-flight query (called from another thread on timeout)."""

    def begin_measure(self) -> None:  # noqa: B027  # optional hook, default no-op
        pass

    def end_measure(self) -> Measure:
        return Measure()

    def config(self) -> dict:
        """Engine configuration recorded in the result JSON."""
        return {}

    def build_info(self) -> dict[str, str]:
        """Build facts of the engine under test (e.g. ``{"allocator": "mimalloc"}``), recorded in the result JSON."""
        return {}
