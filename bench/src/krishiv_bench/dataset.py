"""Resolve a suite's dataset directory into an adapter ``Dataset`` (shared by bisect and flamediff)."""

from __future__ import annotations

from .abba import discover_tables
from .adapters import Dataset
from .h2o import is_h2o
from .paths import data_root, dataset_dir


def resolve_dataset(suite: str, sf: float, dataset: str | None = None) -> Dataset:
    """The dataset of ``suite`` at ``sf`` (default name per suite).

    # Errors
    ``SystemExit`` when the suite has no default dataset and none is given, or no table is found.
    """
    from .cli import ABBA_DATASET

    name = dataset or ABBA_DATASET.get(suite)
    if name is None:
        raise SystemExit(f"suite {suite} has no default dataset: pass --dataset")
    root = data_root() / name if suite == "clickbench" or is_h2o(suite) else dataset_dir(name, sf)
    tables = discover_tables(root)
    if not tables:
        raise SystemExit(f"no dataset at {root}")
    return Dataset(name, sf, root, tables)
