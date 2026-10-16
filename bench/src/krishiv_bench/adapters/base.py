"""Engine adapter interface.

Lifecycle: ``setup(dataset)`` once -> ``run(query)`` many times -> ``teardown()``.
``run`` executes the query to completion and returns every result row (the harness
normalises and checksums them). Adapters report engine-side resource use via
``begin_measure()`` / ``end_measure()`` which bracket each timed run (not the timing itself).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Dataset:
    name: str            # "tpch", "tpcds", ...
    sf: float
    root: Path           # directory holding <table>.parquet files (host path)
    tables: list[str] = field(default_factory=list)


@dataclass
class Query:
    id: str              # "q01"
    sql: str
    suite: str


@dataclass
class Measure:
    peak_rss_bytes: int | None = None      # process RSS high-water during the run (VmHWM after clear_refs reset)
    baseline_rss_bytes: int | None = None  # process RSS just before the run
    cpu_seconds: float | None = None
    spill_bytes: int | None = None


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

    def teardown(self) -> None:  # pragma: no cover - default no-op
        pass

    def cancel(self) -> None:
        """Best-effort abort of the in-flight query (called from another thread on timeout)."""

    def begin_measure(self) -> None:
        pass

    def end_measure(self) -> Measure:
        return Measure()

    def config(self) -> dict:
        """Engine configuration recorded in the result JSON."""
        return {}
