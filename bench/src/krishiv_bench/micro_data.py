"""Deterministic data for the operator microbenchmark suite (``micro``, P2-T168), written as Parquet.

One dataset is a directory ``$KRISHIV_DATA/micro/<version>/<size>/`` holding

* ``fact.parquet``: ``rows`` rows with every column the cases need (a case reads only the columns it names, so the
  width costs nothing at run time):

  ========================  ===============================================================================
  ``id``, ``u``             row number; a random permutation of ``0..rows-1`` (unique: ``u < T`` keeps exactly T rows)
  ``v1``, ``v3``, ``dec``   int64 in 1..100; float64 in [0, 100); ``decimal(15,2)`` in -1000.00..100000.00
  ``s``                     16 random lowercase letters; every 1000th row is the needle ``needlehaystack01``
  ``gi<c>``, ``gs<c>``      group key with exactly ``c`` groups of ``rows/c`` rows: ``u % c`` as int64, and as
                            ``id%010d`` text; for every ``c`` in ``groups``
  ``gj<c>``                 a second int key, ``(u // c) % c``: with ``gi<c>`` it forms ``c*c`` exact groups
  ``ji_b<B>_m<pct>``        int64 probe key for the build table of ``B`` keys: ``pct`` % of the rows draw a key
                            uniformly from the build keys, the others a key that is not in the build side
  ``js_b<B>_m<pct>``        the same keys as ``id%010d`` text
  ``jz_b<B>_a<alpha*10>``   int64 probe key, always a build key, Zipf-distributed (key 0 hottest)
  ========================  ===============================================================================

* ``build_<B>.parquet`` per build size ``B``: ``k`` (the unique int64 keys ``0..B-1``, shuffled), ``ks`` (text of the
  same key) and ``bv`` (a payload, ``k * 7919 % 100003``).

Counts that decide a case's shape (groups, build sizes, matches) are therefore exact, not sampled, except the
match rate and the Zipf skew, which are drawn.

Determinism: every random column of every fixed-size chunk (``CHUNK_ROWS``) draws from its own
``numpy.random.default_rng([seed, crc32(column), chunk])`` stream and the permutations from per-name streams, so the
output depends only on the size set, the seed and the NumPy version (recorded), never on thread count or call order.

Checksums and layout follow ``h2o_data``: ``MANIFEST.json`` records the parameters, library versions, per-column
content digests (independent of the Parquet encoder), the Parquet file hashes and one dataset digest
(``h2o_data.verify`` recomputes them). A dataset directory is written under a temporary name and renamed into place;
an existing one is never touched (a generator change is a new version directory).

Memory: one chunk of the fact table (5M rows, about 2 GB at the default size set), the permutation of the fact rows
and the Zipf CDFs (8 bytes per row or key each; about 2.5 GB more at the full size set).
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import zlib
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import h2o_data
from .h2o_data import CHUNK_ROWS, MANIFEST, ROW_GROUP_ROWS, TableDigest, content_digest, read_manifest, sha256_file

GENERATOR_VERSION = "v1"
SEED = 168
NEEDLE = "needlehaystack01"  # the text of every NEEDLE_EVERY-th row of ``fact.s``
NEEDLE_EVERY = 1000
MATCH_PCTS_INT = (100, 50, 10)  # share of probe rows whose key is in the build side
MATCH_PCTS_STR = (100, 10)
ZIPF_ALPHAS = (0.8, 1.2)
KEY_DIGITS = 10  # text keys are ``id`` + ``KEY_DIGITS`` digits


def sci(n: int) -> str:
    """``1000`` -> ``1e3``, ``5000`` -> ``5e3`` (the harness's usual spelling, see ``h2o_data.sci``)."""
    return h2o_data.sci(n)


def col_gi(c: int) -> str:
    return f"gi{sci(c)}"


def col_gs(c: int) -> str:
    return f"gs{sci(c)}"


def col_gj(c: int) -> str:
    return f"gj{sci(c)}"


def col_ji(build: int, pct: int) -> str:
    return f"ji_b{sci(build)}_m{pct}"


def col_js(build: int, pct: int) -> str:
    return f"js_b{sci(build)}_m{pct}"


def col_jz(build: int, alpha: float) -> str:
    return f"jz_b{sci(build)}_a{round(alpha * 10):02d}"


def build_table(build: int) -> str:
    return f"build_{sci(build)}"


@dataclass(frozen=True)
class MicroSizes:
    """One size set: ``rows`` fact rows, the build-side sizes to join against, the group counts to aggregate by.

    ``pairs`` are the group counts ``c`` that also get a second key (``c * c`` groups); each must be in ``groups``.
    """

    name: str
    rows: int
    builds: tuple[int, ...]
    groups: tuple[int, ...]
    pairs: tuple[int, ...] = ()

    def validate(self) -> None:
        if self.rows < 10_000 or self.rows % 1000:
            raise ValueError("rows must be a multiple of 1000 and at least 10000")
        if not self.builds or any(b < 10 for b in self.builds) or len(set(self.builds)) != len(self.builds):
            raise ValueError("builds must be distinct sizes of at least 10 keys")
        if self.rows + 2 * max(self.builds) >= 10**KEY_DIGITS:
            raise ValueError("a key needs more digits than KEY_DIGITS")
        if (
            not self.groups
            or any(not 1 <= c <= self.rows for c in self.groups)
            or len(set(self.groups)) != len(self.groups)
        ):
            raise ValueError("groups must be distinct counts in 1..rows")
        if any(c not in self.groups or c * c > self.rows for c in self.pairs):
            raise ValueError("pairs must be group counts whose square is at most rows")


# ``quick`` is the CI smoke set; ``default`` is meant to finish the whole suite in under 30 minutes on a 12-thread
# machine (an estimate, see bench/README.md); ``full`` reaches the 1e8 end of the ranges (hours, tens of GB) and is
# run on request only.
SIZES: dict[str, MicroSizes] = {
    "quick": MicroSizes("quick", 1_000_000, (10**3, 10**5), (10**2, 10**4, 10**6), (10**2,)),
    "default": MicroSizes(
        "default", 10_000_000, (10**3, 10**5, 10**7), (10**2, 10**3, 10**4, 10**6, 10**7), (10**2, 10**3)
    ),
    "full": MicroSizes(
        "full", 100_000_000, (10**3, 10**5, 10**7, 10**8), (10**2, 10**3, 10**4, 10**6, 10**8), (10**2, 10**3, 10**4)
    ),
}


def sizes_from_params(params: dict[str, Any]) -> MicroSizes:
    """The size set recorded in a manifest (JSON turns the tuples into lists)."""
    return MicroSizes(
        params["name"],
        params["rows"],
        tuple(params["builds"]),
        tuple(params["groups"]),
        tuple(params["pairs"]),
    )


# ---------------------------------------------------------------- column builders
def _rng(seed: int, name: str, chunk: int = 0) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(name.encode()), chunk])


def _fixed_width_strings(m: np.ndarray) -> pa.Array:
    """A string array from an ``(n, width)`` matrix of ASCII bytes (one row per string)."""
    n, width = m.shape
    offsets = np.arange(n + 1, dtype=np.int32) * np.int32(width)
    return pa.Array.from_buffers(pa.string(), n, [None, pa.py_buffer(offsets), pa.py_buffer(np.ascontiguousarray(m))])


def key_strings(values: np.ndarray) -> pa.Array:
    """``id`` followed by the 10-digit zero-padded decimal of each non-negative integer."""
    n = len(values)
    out = np.empty((n, 2 + KEY_DIGITS), dtype=np.uint8)
    out[:, :2] = np.frombuffer(b"id", dtype=np.uint8)
    rest = values.astype(np.int64)
    for i in range(KEY_DIGITS - 1, -1, -1):
        out[:, 2 + i] = rest % 10 + 48
        rest = rest // 10
    return _fixed_width_strings(out)


def _letters(rng: np.random.Generator, n: int, start: int) -> pa.Array:
    """16 random lowercase letters per row; the rows whose number is a multiple of ``NEEDLE_EVERY`` hold the needle."""
    m = rng.integers(97, 123, size=(n, len(NEEDLE)), dtype=np.uint8)
    first = (-start) % NEEDLE_EVERY
    m[first::NEEDLE_EVERY] = np.frombuffer(NEEDLE.encode(), dtype=np.uint8)
    return _fixed_width_strings(m)


def _decimal_cents(cents: np.ndarray, precision: int, scale: int) -> pa.Array:
    """``decimal(precision, scale)`` from int64 unscaled values (two's complement into the 128-bit words)."""
    words = np.empty((len(cents), 2), dtype="<i8")
    words[:, 0] = cents
    words[:, 1] = cents >> 63
    return pa.Array.from_buffers(pa.decimal128(precision, scale), len(cents), [None, pa.py_buffer(words)])


class _Fact:
    """State shared by the chunks of the fact table: the permutation and the Zipf CDFs."""

    def __init__(self, p: MicroSizes, seed: int):
        self.p, self.seed = p, seed
        self.u = _rng(seed, "u").permutation(p.rows).astype(np.int64)
        self._cdf: dict[tuple[int, float], np.ndarray] = {}

    def cdf(self, build: int, alpha: float) -> np.ndarray:
        key = (build, alpha)
        if key not in self._cdf:
            pmf = 1.0 / np.arange(1, build + 1, dtype=np.float64) ** alpha
            cdf = np.cumsum(pmf)
            self._cdf[key] = cdf / cdf[-1]
        return self._cdf[key]

    def chunk(self, chunk: int, start: int, n: int) -> pa.Table:
        p, seed = self.p, self.seed
        u = self.u[start : start + n]
        cols: dict[str, pa.Array] = {
            "id": pa.array(np.arange(start, start + n, dtype=np.int64)),
            "u": pa.array(u),
            "v1": pa.array(_rng(seed, "v1", chunk).integers(1, 101, n, dtype=np.int64)),
            "v3": pa.array(_rng(seed, "v3", chunk).random(n) * 100.0),
            "dec": _decimal_cents(_rng(seed, "dec", chunk).integers(-100_000, 10_000_001, n, dtype=np.int64), 15, 2),
            "s": _letters(_rng(seed, "s", chunk), n, start),
        }
        for c in p.groups:
            cols[col_gi(c)] = pa.array(u % c)
            cols[col_gs(c)] = key_strings(u % c)
        for c in p.pairs:
            cols[col_gj(c)] = pa.array((u // c) % c)
        for b in p.builds:
            for pct in MATCH_PCTS_INT:
                cols[col_ji(b, pct)] = pa.array(self._probe_keys(b, pct, chunk, n))
            for pct in MATCH_PCTS_STR:
                cols[col_js(b, pct)] = key_strings(self._probe_keys(b, pct, chunk, n))
            for alpha in ZIPF_ALPHAS:
                draw = _rng(seed, col_jz(b, alpha), chunk).random(n)
                cols[col_jz(b, alpha)] = pa.array(
                    np.searchsorted(self.cdf(b, alpha), draw, side="right").clip(max=b - 1)
                )
        return pa.table(cols)

    def _probe_keys(self, build: int, pct: int, chunk: int, n: int) -> np.ndarray:
        """Probe keys of one match rate; the int and the text column of a (build, rate) pair share them."""
        rng = _rng(self.seed, f"match_b{build}_m{pct}", chunk)
        key = rng.integers(0, build, n, dtype=np.int64)
        miss = rng.integers(0, 100, n) >= pct
        return np.where(miss, key + build, key)


def fact_chunks(p: MicroSizes, seed: int = SEED) -> Iterator[pa.Table]:
    fact = _Fact(p, seed)
    for chunk, start in enumerate(range(0, p.rows, CHUNK_ROWS)):
        yield fact.chunk(chunk, start, min(CHUNK_ROWS, p.rows - start))


def build_chunks(build: int, seed: int = SEED) -> Iterator[pa.Table]:
    keys = _rng(seed, f"build_{build}").permutation(build).astype(np.int64)
    for start in range(0, build, CHUNK_ROWS):
        k = keys[start : start + CHUNK_ROWS]
        yield pa.table({"k": pa.array(k), "ks": key_strings(k), "bv": pa.array(k * 7919 % 100003)})


# ---------------------------------------------------------------- writing
def _write(path: Path, chunks: Iterator[pa.Table]) -> dict[str, Any]:
    """Writes the chunks to ``path`` (zstd, ``ROW_GROUP_ROWS`` rows per row group); returns the table digest."""
    td: TableDigest | None = None
    with contextlib.ExitStack() as stack:
        w = None
        for t in chunks:
            if w is None:
                td = TableDigest(t.schema.names)
                w = stack.enter_context(pq.ParquetWriter(path, t.schema, compression="zstd"))
            assert td is not None
            td.update(t)
            w.write_table(t, row_group_size=ROW_GROUP_ROWS)
    assert td is not None
    return td.result()


def generate(p: MicroSizes, root: Path, seed: int = SEED, on_table: Callable[[str], None] | None = None) -> Path:
    """Generates the dataset of ``p`` into ``root/<size name>/`` and returns that directory.

    Refuses to touch an existing non-empty directory. Written under a hidden temporary name, renamed when complete.
    """
    p.validate()
    final = root / p.name
    if final.exists() and any(final.iterdir()):
        raise FileExistsError(f"{final} already exists; datasets are immutable (use a new --micro-version)")
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".{p.name}.partial-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        sources = {"fact": fact_chunks(p, seed)} | {build_table(b): build_chunks(b, seed) for b in p.builds}
        tables: dict[str, dict[str, Any]] = {}
        for name, chunks in sources.items():
            f = tmp / f"{name}.parquet"
            if on_table:
                on_table(name)
            digest = _write(f, chunks)
            tables[name] = {"file": f.name, "file_sha256": sha256_file(f), **digest}
        manifest = {
            "schema": 1,
            "generator": "krishiv-bench micro_data",
            "generator_version": GENERATOR_VERSION,
            "params": {**asdict(p), "seed": seed},
            "libs": {"numpy": str(np.__version__), "pyarrow": str(pa.__version__)},
            "tables": tables,
            "content_sha256": content_digest(tables),
        }
        (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if final.exists():  # an empty leftover directory
            final.rmdir()
        tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


__all__ = [
    "GENERATOR_VERSION",
    "NEEDLE",
    "SIZES",
    "MicroSizes",
    "generate",
    "read_manifest",
    "sizes_from_params",
]
