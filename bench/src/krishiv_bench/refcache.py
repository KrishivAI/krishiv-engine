"""Reference-engine cache: DuckDB's answers and timings, stored once and reused.

A reference engine's answer to a query over a given dataset never changes, and at SF1000 running it costs hours.
``reproduce`` and ``abba`` therefore store what the reference engine produced, per suite, scale factor, query, dataset
and engine version, and reuse it on later runs; ``--refresh-reference`` runs the engine again and replaces the entry.
The engine under test is still verified against the cached answer, exactly as against a fresh one.

Two kinds of entry, with different keys:

* **answer**: the result rows. Independent of threads and memory, so any run of the same query over the same data
  reuses it. Used by ``abba`` and the interleaved ``reproduce``, which still time the reference live (interleaving
  exists to share the machine's drift between the engines, and a replayed time cannot do that).
* **timing**: the answer plus the reference's timed samples, CPU and peak RSS, which depend on the threads, the
  memory and the sample count. Used by ``reproduce --sequential``, where the reference runs alone and a replayed time
  is as good as a live one (the machine is the same, which the record states: the entry says when it was measured).

The dataset is identified by its manifest checksum (``$KRISHIV_DATA/manifests``), else by the names and sizes of its
files; the query by the hash of the text the reference runs; the engine by its installed version. A change to any of
them is a different key, so a stale entry is never read, only left behind.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from . import compare
from .adapters import Dataset, Query
from .paths import data_root, manifest_stem

FORMAT = 1  # bumped when what an entry means changes (the reference's SQL dialect settings, the row encoding)
MAX_STORED_ROWS = 1_000_000  # a larger answer is not stored (the reference reruns), as in the suite runner


def _enc(v: Any) -> Any:
    """A cell as JSON, losslessly: unlike the result files' encoding (dates as plain strings, for display), a cached
    answer is compared with live rows, so a date must come back a date."""
    if isinstance(v, dt.datetime):  # before date: a datetime is a date
        return {"ts": v.isoformat()}
    if isinstance(v, dt.date):
        return {"date": v.isoformat()}
    if isinstance(v, dt.time):
        return {"time": v.isoformat()}
    if isinstance(v, dt.timedelta):
        return {"td": [v.days, v.seconds, v.microseconds]}
    if isinstance(v, (bytes, bytearray, memoryview)):
        return {"bytes": bytes(v).hex()}
    if isinstance(v, (list, tuple)):
        return {"list": [_enc(x) for x in v]}
    if isinstance(v, dict):
        return {"map": [[_enc(k), _enc(x)] for k, x in v.items()]}
    return compare.enc(v)


def _dec(v: Any) -> Any:
    if isinstance(v, dict):
        if "ts" in v:
            return dt.datetime.fromisoformat(v["ts"])
        if "date" in v:
            return dt.date.fromisoformat(v["date"])
        if "time" in v:
            return dt.time.fromisoformat(v["time"])
        if "td" in v:
            return dt.timedelta(*v["td"])
        if "bytes" in v:
            return bytes.fromhex(v["bytes"])
        if "list" in v:
            return [_dec(x) for x in v["list"]]
        if "map" in v:
            return {_dec(k): _dec(x) for k, x in v["map"]}
    return compare.dec(v)


def cache_dir() -> Path:
    return data_root() / "refcache"


def dataset_signature(ds: Dataset) -> str:
    """Identifies the data: the manifest checksum when there is one, else the relative names and sizes of the files."""
    man = data_root() / "manifests" / f"{manifest_stem(ds.name, ds.sf)}.sha256"
    if man.exists():
        return "manifest:" + hashlib.sha256(man.read_bytes()).hexdigest()
    files = sorted((str(p.relative_to(ds.root)), p.stat().st_size) for p in ds.root.rglob("*") if p.is_file())
    return "files:" + hashlib.sha256(json.dumps(files).encode()).hexdigest()


def _digest(fields: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:40]


class RefCache:
    """The cache directory plus the policy: ``refresh`` runs the reference again and overwrites its entries."""

    def __init__(self, root: Path | None = None, refresh: bool = False, engine: str = "duckdb", version: str = "?"):
        self.root = root or cache_dir()
        self.refresh = refresh
        self.engine = engine
        self.version = version

    def _fields(self, kind: str, suite: str, ds: Dataset, query: Query, **extra: Any) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "kind": kind,
            "engine": self.engine,
            "version": self.version,
            "suite": suite,
            "sf": ds.sf,
            "dataset": ds.name,
            "data": dataset_signature(ds),
            "query": query.id,
            "sql": hashlib.sha256((query.sql + "\0" + (query.check_sql or "")).encode()).hexdigest(),
            **extra,
        }

    def answer_key(self, suite: str, ds: Dataset, query: Query) -> dict[str, Any]:
        return self._fields("answer", suite, ds, query)

    def timing_key(self, suite: str, ds: Dataset, query: Query, threads: int, memory_gb: int, samples: int) -> dict:
        return self._fields("timing", suite, ds, query, threads=threads, memory_gb=memory_gb, samples=samples)

    def _paths(self, fields: dict[str, Any]) -> tuple[Path, Path]:
        stem = f"{fields['engine']}-{fields['suite']}-sf{fields['sf']}-{fields['query']}-{_digest(fields)}"
        return self.root / f"{stem}.json", self.root / f"{stem}.rows.json.gz"

    def load(self, fields: dict[str, Any]) -> tuple[dict, list[tuple]] | None:
        """``(entry, rows)`` stored under ``fields``, or ``None`` (absent, ``refresh``, or unreadable)."""
        if self.refresh:
            return None
        meta, rows = self._paths(fields)
        try:
            entry = json.loads(meta.read_text())
            if entry.get("fields") != fields:  # a digest collision or a hand-edited file is a miss
                return None
            with gzip.open(rows, "rt") as f:
                return entry, [tuple(_dec(c) for c in r) for r in json.load(f)]
        except (OSError, ValueError, KeyError, EOFError):
            return None

    def store(self, fields: dict[str, Any], rows: list, payload: dict | None = None) -> bool:
        """Writes the entry (rows beside it); ``False`` and nothing written when the answer is too large to keep."""
        if len(rows) > MAX_STORED_ROWS:
            return False
        meta, rows_file = self._paths(fields)
        self.root.mkdir(parents=True, exist_ok=True)
        with gzip.open(rows_file, "wt") as f:
            json.dump([[_enc(c) for c in r] for r in rows], f)
        entry = {"fields": fields, "created_utc": dt.datetime.now(dt.UTC).isoformat(), **(payload or {})}
        meta.write_text(json.dumps(entry, indent=2))  # last: an entry with a meta file always has its rows
        return True
