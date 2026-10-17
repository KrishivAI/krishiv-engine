"""Locations. Datasets and results live outside the repo under $KRISHIV_DATA."""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    return Path(os.environ.get("KRISHIV_DATA", str(Path.home() / "krishiv-data"))).expanduser()


def dataset_dir(name: str, sf: str | float | int) -> Path:
    return data_root() / name / f"sf{fmt_sf(sf)}"


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
