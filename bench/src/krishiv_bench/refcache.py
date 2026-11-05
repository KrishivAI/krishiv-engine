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

The dataset is identified by its manifest checksum (``$KRISHIV_DATA/manifests``), else by the name, size, modification time
and first and last 64 KiB of each of its files; the query by the hash of the text the reference runs; the engine by its installed version. A change to any of
them is a different key, so a stale entry is never read, only left behind.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from . import compare
from .adapters import Dataset, Query
from .adapters.duckdb_engine import SPARK_NULL_ORDER
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
    """Identifies the data: the manifest checksum when there is one, else a fingerprint of each file."""
    man = data_root() / "manifests" / f"{manifest_stem(ds.name, ds.sf)}.sha256"
    if man.exists():
        return "manifest:" + hashlib.sha256(man.read_bytes()).hexdigest()
    files = [_file_identity(str(p.relative_to(ds.root)), p) for p in sorted(ds.root.rglob("*")) if p.is_file()]
    return "files:" + hashlib.sha256(json.dumps(files).encode()).hexdigest()


FINGERPRINT_BYTES = 64 * 1024


def _file_identity(rel: str, path: Path) -> list:
    """Name, size, modification time and a hash of the first and last 64 KiB of a data file. The last block holds a
    Parquet file's footer (schema, row-group and page statistics), so a rewrite of the same size that changes the
    data changes it too; the modification time covers the rest, at the price of a miss after a copy."""
    st = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(FINGERPRINT_BYTES))
        if st.st_size > FINGERPRINT_BYTES:
            f.seek(max(FINGERPRINT_BYTES, st.st_size - FINGERPRINT_BYTES))
            h.update(f.read(FINGERPRINT_BYTES))
    return [rel, st.st_size, st.st_mtime_ns, h.hexdigest()]


def machine_signature() -> str:
    """Identifies the machine a timing was taken on: CPU model, logical cores, RAM and host name."""
    from .runner import machine_info

    m = machine_info()
    return hashlib.sha256(
        json.dumps([m["cpu_model"], m["cores_logical"], m["ram_bytes"], m["hostname"]]).encode()
    ).hexdigest()[:16]


def _digest(fields: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:40]


class RefCache:
    """The cache directory plus the policy: ``refresh`` runs the reference again and overwrites its entries."""

    def __init__(self, root: Path | None = None, refresh: bool = False, engine: str = "duckdb", version: str = "?"):
        self.root = root or cache_dir()
        self.refresh = refresh
        self.engine = engine
        self.version = version
        self.hits = self.misses = self.stores = 0

    def summary(self) -> dict[str, Any]:
        """What the run record says about the cache: its policy and how many entries were read and written."""
        return {
            "enabled": True,
            "refresh": self.refresh,
            "dir": str(self.root),
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
        }

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
            "null_order": SPARK_NULL_ORDER,  # the reference's session settings that shape its answer
            "sql": hashlib.sha256((query.sql + "\0" + (query.check_sql or "")).encode()).hexdigest(),
            **extra,
        }

    def answer_key(self, suite: str, ds: Dataset, query: Query) -> dict[str, Any]:
        return self._fields("answer", suite, ds, query)

    def timing_key(
        self,
        suite: str,
        ds: Dataset,
        query: Query,
        threads: int,
        memory_gb: int,
        samples: int,
        cap_gb: int | None = None,
        cold: bool = False,
    ) -> dict:
        return self._fields(
            "timing",
            suite,
            ds,
            query,
            threads=threads,
            memory_gb=memory_gb,
            samples=samples,
            cap_gb=cap_gb,
            cold=cold,
            machine=machine_signature(),
        )

    def _paths(self, fields: dict[str, Any]) -> tuple[Path, Path]:
        stem = f"{fields['engine']}-{fields['suite']}-sf{fields['sf']}-{fields['query']}-{_digest(fields)}"
        return self.root / f"{stem}.json", self.root / f"{stem}.rows.json.gz"

    def load(self, fields: dict[str, Any]) -> tuple[dict, list[tuple]] | None:
        """``(entry, rows)`` stored under ``fields``, or ``None`` (absent, ``refresh``, or unreadable)."""
        if self.refresh:
            self.misses += 1
            return None
        meta, rows = self._paths(fields)
        try:
            entry = json.loads(meta.read_text())
            if entry.get("fields") != fields:  # a digest collision or a hand-edited file is a miss
                self.misses += 1
                return None
            with gzip.open(rows, "rt") as f:
                raw = json.load(f)
            if not isinstance(raw, list) or not all(isinstance(r, list) for r in raw):
                raise ValueError("rows file is not a list of rows")
            answer = [tuple(_dec(c) for c in r) for r in raw]
            self.hits += 1
            return entry, answer
        except Exception:  # any damage (truncated gzip, bad JSON, wrong shape) is a miss, never a crash
            self.misses += 1
            return None

    def store(self, fields: dict[str, Any], rows: list, payload: dict | None = None) -> bool:
        """Writes the entry (rows beside it); ``False`` when nothing was kept: the answer is too large, or the write
        failed (a full disk must not abort a run that took hours). Both files are written to a temporary name and
        renamed, the meta file last, so an entry that exists is complete."""
        if len(rows) > MAX_STORED_ROWS:
            return False
        meta, rows_file = self._paths(fields)
        entry = {"fields": fields, "created_utc": dt.datetime.now(dt.UTC).isoformat(), **(payload or {})}
        tmp: list[str] = []
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            for path, write in (
                (rows_file, lambda f: json.dump([[_enc(c) for c in r] for r in rows], f)),
                (meta, lambda f: json.dump(entry, f, indent=2)),
            ):
                fd, name = tempfile.mkstemp(dir=self.root, prefix=".tmp-")
                tmp.append(name)
                os.close(fd)
                opener = gzip.open if path is rows_file else open
                with opener(name, "wt") as f:
                    write(f)
                os.replace(name, path)
            self.stores += 1
            return True
        except (OSError, TypeError, ValueError):
            return False
        finally:
            for name in tmp:
                Path(name).unlink(missing_ok=True)
