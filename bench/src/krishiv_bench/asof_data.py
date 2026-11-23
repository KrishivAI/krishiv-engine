"""Dataset of the ``asof`` suite (Spark 4.3's ``ASOF JOIN``): seeded ``trades`` and ``quotes`` over one trading day.

Scale ``sf`` N: ``N`` million trades and ``5 N`` million quotes over ``SYMBOLS`` symbols (``S0000`` .. ``S0999``),
timestamps in microseconds over one day. Prices are integer cents, so the queries' sums are exact in every engine.

Ties: an as-of join's pick among right rows with the same value is unspecified (Spark documents it so), so every
quote has its own timestamp: quote ``i`` falls in its own slot ``[START + i * step, START + (i + 1) * step)`` of the
day. Rows are shuffled within each chunk, so no engine reads them in time order. Trades draw their timestamps
uniformly over the day; a trade may equal a quote (the exact matches the queries test).

Determinism: each chunk of ``CHUNK`` rows of a table draws from ``numpy.random.default_rng([SEED, table, chunk])``,
so the rows depend only on the scale, the seed and the NumPy version (recorded in ``MANIFEST.json``). The manifest
holds per table the row count and a content digest (SHA-256 over each chunk's column values, independent of the
Parquet encoder). Output: ``$KRISHIV_DATA/asof/sf<N>/{trades,quotes}.parquet``, written under a temporary name and
renamed into place; an existing directory is never touched (a generator change is a new ``GENERATOR_VERSION`` and
needs a new directory).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .paths import data_root, fmt_sf

GENERATOR_VERSION = "v1"
SEED = 20261119
CHUNK = 1_000_000  # rows per RNG stream and per row group
SYMBOLS = 1000
TRADES_PER_SF = 1_000_000
QUOTES_PER_SF = 5_000_000
START_US = int(np.datetime64("2026-06-29T00:00:00", "us").astype(np.int64))
DAY_US = 24 * 3600 * 1_000_000
MANIFEST = "MANIFEST.json"
_TRADES, _QUOTES = 1, 2

TRADES_SCHEMA = pa.schema(
    [
        ("ts", pa.timestamp("us")),
        ("symbol", pa.string()),
        ("qty", pa.int64()),
        ("price_cents", pa.int64()),
    ]
)
QUOTES_SCHEMA = pa.schema(
    [
        ("ts", pa.timestamp("us")),
        ("symbol", pa.string()),
        ("bid_cents", pa.int64()),
        ("ask_cents", pa.int64()),
    ]
)
_SYMBOL_NAMES = np.array([f"S{i:04d}" for i in range(SYMBOLS)], dtype=object)


def rows_of(sf: int) -> dict[str, int]:
    if sf <= 0:
        raise ValueError(f"--sf must be a positive whole number (got {sf})")
    return {"trades": sf * TRADES_PER_SF, "quotes": sf * QUOTES_PER_SF}


def asof_dir(sf: int) -> Path:
    return data_root() / "asof" / f"sf{fmt_sf(sf)}"


def trades_chunk(chunk: int, rows: int) -> pa.RecordBatch:
    """Chunk ``chunk`` (``rows`` rows) of ``trades``."""
    rng = np.random.default_rng([SEED, _TRADES, chunk])
    ts = START_US + rng.integers(0, DAY_US, rows, dtype=np.int64)
    sym = rng.integers(0, SYMBOLS, rows)
    qty = rng.integers(1, 1000, rows, dtype=np.int64)
    price = rng.integers(1_000, 100_000, rows, dtype=np.int64)
    return _batch(TRADES_SCHEMA, ts, sym, qty, price)


def quotes_chunk(chunk: int, rows: int, total: int) -> pa.RecordBatch:
    """Chunk ``chunk`` (``rows`` rows) of ``quotes``, of ``total`` rows in all: one timestamp slot per quote."""
    rng = np.random.default_rng([SEED, _QUOTES, chunk])
    step = DAY_US // total
    first = chunk * CHUNK
    index = np.arange(first, first + rows, dtype=np.int64)
    ts = START_US + index * step + rng.integers(0, step, rows, dtype=np.int64)
    sym = rng.integers(0, SYMBOLS, rows)
    bid = rng.integers(1_000, 100_000, rows, dtype=np.int64)
    ask = bid + rng.integers(1, 50, rows, dtype=np.int64)
    order = rng.permutation(rows)
    return _batch(QUOTES_SCHEMA, ts[order], sym[order], bid[order], ask[order])


def _batch(schema: pa.Schema, ts: np.ndarray, sym: np.ndarray, a: np.ndarray, b: np.ndarray) -> pa.RecordBatch:
    return pa.record_batch(
        [
            pa.array(ts, type=pa.int64()).cast(pa.timestamp("us")),
            pa.array(_SYMBOL_NAMES[sym], type=pa.string()),
            pa.array(a, type=pa.int64()),
            pa.array(b, type=pa.int64()),
        ],
        schema=schema,
    )


def batch_digest(h: Any, batch: pa.RecordBatch) -> None:
    """Folds ``batch``'s values into ``h``: the integers' bytes and the strings' UTF-8 with their lengths."""
    for col in batch.columns:
        if pa.types.is_string(col.type):
            for s in col.to_pylist():
                b = s.encode()
                h.update(len(b).to_bytes(4, "little"))
                h.update(b)
        else:
            h.update(col.cast(pa.int64()).to_numpy(zero_copy_only=False).astype("<i8").tobytes())


def chunks(table: str, rows: int):
    """The chunks of ``table``, in order."""
    n = -(-rows // CHUNK)
    for c in range(n):
        size = min(CHUNK, rows - c * CHUNK)
        yield trades_chunk(c, size) if table == "trades" else quotes_chunk(c, size, rows)


def generate(sf: int, out: Path | None = None, rows: dict[str, int] | None = None) -> Path:
    """Writes the ``sf`` dataset (``rows`` overrides its table sizes, for tests); refuses an existing directory."""
    rows = rows or rows_of(sf)
    final = out or asof_dir(sf)
    if final.exists():
        raise FileExistsError(f"{final} exists; a dataset is never regenerated in place")
    final.parent.mkdir(parents=True, exist_ok=True)
    # A directory of this invocation's own, next to the destination (same filesystem, so the rename is atomic):
    # two generators for one destination never share or remove each other's files.
    tmp = Path(tempfile.mkdtemp(prefix=final.name + ".", suffix=".tmp", dir=final.parent))
    try:
        _write(sf, rows, tmp)
        if final.exists():
            raise FileExistsError(f"{final} was created by another generator while this one ran")
        tmp.rename(final)  # fails too if another generator published a non-empty `final` since the check
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def _write(sf: int, rows: dict[str, int], tmp: Path) -> None:
    """Writes the tables, the manifest and the session settings into ``tmp``."""
    tables = {}
    for table, n in rows.items():
        schema = TRADES_SCHEMA if table == "trades" else QUOTES_SCHEMA
        h = hashlib.sha256()
        with pq.ParquetWriter(tmp / f"{table}.parquet", schema, compression="zstd") as w:
            for batch in chunks(table, n):
                batch_digest(h, batch)
                w.write_batch(batch, row_group_size=CHUNK)
        tables[table] = {"rows": n, "content_sha256": h.hexdigest()}
    manifest = {
        "generator": "krishiv_bench.asof_data",
        "generator_version": GENERATOR_VERSION,
        "seed": SEED,
        "sf": sf,
        "libs": {"numpy": np.__version__, "pyarrow": pa.__version__},
        "tables": tables,
    }
    (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # The settings every engine run of this suite needs (read by the Krishiv adapter).
    (tmp / "session.conf").write_text("spark.sql.join.asofJoin.enabled=true\n")


def verify(root: Path) -> list[str]:
    """Recomputes the content digests of ``root``'s tables against its manifest; the problems found."""
    man = json.loads((root / MANIFEST).read_text())
    problems = []
    for table, want in man["tables"].items():
        h = hashlib.sha256()
        rows = 0
        for batch in pq.ParquetFile(root / f"{table}.parquet").iter_batches(batch_size=CHUNK):
            batch_digest(h, batch)
            rows += batch.num_rows
        if rows != want["rows"]:
            problems.append(f"{table}: {rows} rows, manifest says {want['rows']}")
        elif h.hexdigest() != want["content_sha256"]:
            problems.append(f"{table}: content digest differs from the manifest")
    return problems
