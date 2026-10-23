"""Locations. Datasets and results live outside the repo under $KRISHIV_DATA."""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    return Path(os.environ.get("KRISHIV_DATA", str(Path.home() / "krishiv-data"))).expanduser()


def split_dataset(name: str) -> tuple[str, str | None]:
    """`base` or `base@<generator version>`: a versioned dataset is never regenerated in place."""
    base, _, version = name.partition("@")
    return base, (version or None)


def dataset_leaf(sf: str | float | int, version: str | None = None) -> str:
    return f"sf{fmt_sf(sf)}" + (f"-dbgen-{version}" if version else "")


def dataset_dir(name: str, sf: str | float | int) -> Path:
    base, version = split_dataset(name)
    return data_root() / base / dataset_leaf(sf, version)


def manifest_stem(name: str, sf: str | float | int) -> str:
    base, version = split_dataset(name)
    return f"{base}-{dataset_leaf(sf, version)}"


def results_dir() -> Path:
    return data_root() / "results"


def fmt_sf(sf) -> str:
    f = float(sf)
    return str(int(f)) if f == int(f) else str(f)


def bench_root() -> Path:
    """The bench/ directory of the repo (queries live here)."""
    return Path(__file__).resolve().parents[2]


def queries_dir(suite: str) -> Path:
    return bench_root() / "queries" / suite
