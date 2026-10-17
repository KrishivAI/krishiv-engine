"""Stub Krishiv adapter. Implemented in P2 once the engine has a SQL entry point.

Planned shape (so P2 only fills in the bodies):
  * connect via Flight SQL / the `krishiv` CLI / in-process PyO3 binding -- whichever P2 lands;
  * setup(): CREATE EXTERNAL TABLE / register Parquet under dataset.root;
  * run(): execute SQL, return all rows as tuples;
  * begin_measure()/end_measure(): read peak RSS + spill bytes from the engine's metrics
    endpoint (or /proc of the spawned process).
Register-name: "krishiv" (already wired in adapters/__init__.py).
"""

from __future__ import annotations

from .base import Dataset, Engine, Query


class KrishivEngine(Engine):
    name = "krishiv"

    def version(self) -> str:
        raise NotImplementedError("Krishiv adapter is implemented in P2 (engine has no SQL entry point in P0)")

    def setup(self, dataset: Dataset) -> None:
        raise NotImplementedError("Krishiv adapter is implemented in P2")

    def run(self, query: Query) -> list[tuple]:
        raise NotImplementedError("Krishiv adapter is implemented in P2")
