"""Deterministic generator for the db-benchmark style groupby (G1) and join (J1) datasets, written as Parquet.

Reference: the open-source ``db-benchmark`` project (originally h2oai, now duckdblabs; MPL-2.0), whose R scripts
``groupby-datagen.R`` and ``join-datagen.R`` define the schemas, key cardinalities, the NA procedure and the sorted
variant. This module is an independent reimplementation of those *distributions* in NumPy/PyArrow; no upstream code
or query text is copied. It is **not** byte-identical to the R output (R's ``sample()`` stream is not reproduced), so
results are comparable in shape and cardinality, not row for row. What pins a dataset is the content checksum in
``MANIFEST.json`` (see below).

Determinism: every column of every fixed-size chunk (``CHUNK_ROWS``) draws from its own
``numpy.random.default_rng([seed, task, column, chunk])`` stream, so the output depends only on the parameters, the seed
and the NumPy version (recorded in the manifest), never on thread count, memory or call order. Join datasets draw whole
columns from per-stage streams (the key permutations are global). Sorted variants break ties by generation order, so
they are deterministic too.

Checksums: ``MANIFEST.json`` holds, per table, the row count, a per-column *content digest* (SHA-256 over the
canonical values, validity and string lengths, independent of the Parquet encoder and chunking) and the SHA-256 of the
Parquet file (which also depends on the PyArrow version). ``verify`` recomputes the digests from the files.

Layout: ``<root>/<label>/<table>.parquet`` with ``label`` = ``G1_<rows>_<K>_<na%>_<sorted>`` (tables: ``x``) or
``J1_<rows>_NA_<na%>_<sorted>`` (tables: ``x``, ``small``, ``medium``, ``big``), as in the upstream file names.
A dataset directory is written under a temporary name and renamed into place, and an existing one is never touched.

Memory: groupby is chunked (a few hundred MB). Join holds the key arrays of the whole table in memory
(about 4 bytes x rows x ~6 for ``x``): ~3 GB at 1e8 rows, ~35 GB at 1e9.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

GENERATOR_VERSION = "v1"
UPSTREAM_SEED = 108  # the seed the upstream R scripts use (the stream itself differs)
CHUNK_ROWS = 5_000_000  # part of the generator definition: the RNG streams are per chunk
ROW_GROUP_ROWS = 1_000_000
JOIN_DIV_SMALL = 1_000_000  # rows(small) = rows / this
JOIN_DIV_MEDIUM = 1_000  # rows(medium) = rows / this
MANIFEST = "MANIFEST.json"
RN = "__rn"  # generation-order column used only to make sorting deterministic

_G1_TAG, _J1_TAG = 1, 2


def sci(n: int) -> str:
    """The upstream ``pretty_sci``: 10000000 -> ``1e7``, 2 -> ``2e0``."""
    s = str(n)
    return f"{s[0]}e{len(s) - 1}"


@dataclass(frozen=True)
class G1Params:
    rows: int
    k: int = 100
    na_pct: int = 0
    sorted: bool = False
    seed: int = UPSTREAM_SEED
    chunk_rows: int = CHUNK_ROWS

    def validate(self) -> None:
        if self.rows < 1:
            raise ValueError("rows must be >= 1")
        if not 1 <= self.k <= 999:
            raise ValueError("k must be in 1..999 (ids are zero-padded to 3 digits)")
        if self.rows % self.k:
            raise ValueError("rows must be a multiple of k (id3 and id6 have rows/k groups)")
        if not 0 <= self.na_pct <= 100:
            raise ValueError("na_pct must be in 0..100")
        if self.chunk_rows < 1:
            raise ValueError("chunk_rows must be >= 1")

    @property
    def label(self) -> str:
        return f"G1_{sci(self.rows)}_{sci(self.k)}_{self.na_pct}_{int(self.sorted)}"


@dataclass(frozen=True)
class J1Params:
    rows: int
    na_pct: int = 0
    sorted: bool = False
    seed: int = UPSTREAM_SEED
    div_small: int = JOIN_DIV_SMALL
    div_medium: int = JOIN_DIV_MEDIUM
    chunk_rows: int = CHUNK_ROWS

    @property
    def n_small(self) -> int:
        return self.rows // self.div_small

    @property
    def n_medium(self) -> int:
        return self.rows // self.div_medium

    def validate(self) -> None:
        if not 0 <= self.na_pct <= 100:
            raise ValueError("na_pct must be in 0..100")
        if self.chunk_rows < 1:
            raise ValueError("chunk_rows must be >= 1")
        for name, div in (("div_small", self.div_small), ("div_medium", self.div_medium)):
            if div < 1 or self.rows % div:
                raise ValueError(f"rows must be a multiple of {name}")
        for n in (self.n_small, self.n_medium, self.rows):
            if n < 10 or n % 10:
                raise ValueError("every key set must have a multiple of 10 (>= 10) keys (0.9 / 0.1 split)")
        if self.rows >= 2**31 // 11 * 10:
            raise ValueError("rows too large for 32-bit keys")

    @property
    def label(self) -> str:
        return f"J1_{sci(self.rows)}_NA_{self.na_pct}_{int(self.sorted)}"


# ---------------------------------------------------------------- content digests
class ColumnDigest:
    """Order-sensitive SHA-256 over a column's canonical content; independent of chunking and of the encoder."""

    def __init__(self) -> None:
        self.valid, self.values, self.lengths = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
        self.rows = 0

    def update(self, arr: pa.Array | pa.ChunkedArray) -> None:
        if isinstance(arr, pa.ChunkedArray):
            for chunk in arr.chunks:
                self.update(chunk)
            return
        self.rows += len(arr)
        self.valid.update(np.asarray(arr.is_valid().to_numpy(zero_copy_only=False), dtype=np.uint8).tobytes())
        if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
            a = arr.fill_null("").cast(pa.large_string())
            off = np.frombuffer(a.buffers()[1], dtype=np.int64)[a.offset : a.offset + len(a) + 1]
            self.lengths.update(np.diff(off).astype("<u4").tobytes())
            data = a.buffers()[2]
            if data is not None and len(a):
                self.values.update(memoryview(data)[int(off[0]) : int(off[-1])])
        elif pa.types.is_integer(arr.type):
            self.values.update(arr.fill_null(0).to_numpy(zero_copy_only=False).astype("<i8").tobytes())
        elif pa.types.is_floating(arr.type):
            self.values.update(arr.fill_null(0.0).to_numpy(zero_copy_only=False).astype("<f8").tobytes())
        else:
            raise TypeError(f"unsupported column type {arr.type}")

    def hexdigest(self) -> str:
        return hashlib.sha256(
            ("|".join((self.valid.hexdigest(), self.values.hexdigest(), self.lengths.hexdigest()))).encode()
        ).hexdigest()


class TableDigest:
    def __init__(self, names: list[str]) -> None:
        self.cols = {n: ColumnDigest() for n in names}

    def update(self, table: pa.Table | pa.RecordBatch) -> None:
        for n, d in self.cols.items():
            d.update(table.column(n))

    def result(self) -> dict[str, Any]:
        rows = {d.rows for d in self.cols.values()}
        assert len(rows) <= 1
        return {"rows": rows.pop() if rows else 0, "columns": {n: d.hexdigest() for n, d in self.cols.items()}}


def digest_parquet(path: Path, batch_rows: int = CHUNK_ROWS) -> dict[str, Any]:
    pf = pq.ParquetFile(path)
    td = TableDigest(pf.schema_arrow.names)
    for batch in pf.iter_batches(batch_size=batch_rows):
        td.update(batch)
    return td.result()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def content_digest(tables: dict[str, dict[str, Any]]) -> str:
    """One digest over all tables' row counts and column digests (the dataset's identity)."""
    canon = {t: {"rows": v["rows"], "columns": v["columns"]} for t, v in sorted(tables.items())}
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- column builders
def _ids_to_str(values: np.ndarray, mask: np.ndarray | None, width: int | None) -> pa.Array:
    """``id%0<width>d`` (zero padded) or ``id%d`` of an integer array; masked entries are null."""
    s = pc.cast(pa.array(values, mask=mask), pa.string())
    if width:
        s = pc.utf8_lpad(s, width, "0")  # ty: ignore[unresolved-attribute]
    return pc.binary_join_element_wise(pa.scalar("id"), s, pa.scalar(""))  # ty: ignore[unresolved-attribute]


def _na_value_mask(values: np.ndarray, domain: int, pct: int, rng: np.random.Generator) -> np.ndarray | None:
    """Rows whose value is in a random ``pct`` % of the value domain 1..domain (upstream's value-level NA)."""
    nna = int(domain * pct / 100)
    if not nna:
        return None
    chosen = np.zeros(domain + 1, dtype=bool)
    chosen[rng.choice(domain, nna, replace=False) + 1] = True
    return chosen[values]


def _na_present_mask(values: np.ndarray, pct: int, rng: np.random.Generator) -> np.ndarray | None:
    """Rows whose value is in a random ``pct`` % of the *distinct values present* (upstream's ``unique(col)``)."""
    present = np.zeros(int(values.max()) + 1, dtype=bool)
    present[values] = True
    cand = np.flatnonzero(present)
    nna = int(len(cand) * pct / 100)
    if not nna:
        return None
    chosen = np.zeros(len(present), dtype=bool)
    chosen[cand[rng.choice(len(cand), nna, replace=False)]] = True
    return chosen[values]


def _na_row_mask(n: int, pct: int, rng: np.random.Generator) -> np.ndarray | None:
    nna = int(n * pct / 100)
    if not nna:
        return None
    m = np.zeros(n, dtype=bool)
    m[rng.choice(n, nna, replace=False)] = True
    return m


def _rng(*key: int) -> np.random.Generator:
    return np.random.default_rng(list(key))


G1_SCHEMA = pa.schema(
    [
        ("id1", pa.string()),
        ("id2", pa.string()),
        ("id3", pa.string()),
        ("id4", pa.int32()),
        ("id5", pa.int32()),
        ("id6", pa.int32()),
        ("v1", pa.int32()),
        ("v2", pa.int32()),
        ("v3", pa.float64()),
    ]
)


def g1_chunk(p: G1Params, chunk: int, n: int) -> pa.Table:
    """Rows of one chunk of the groupby table. NA value sets (``_na_sets``) are the same for every chunk."""
    k, groups = p.k, p.rows // p.k

    def draw(col: int, hi: int) -> np.ndarray:
        return _rng(p.seed, _G1_TAG, col, chunk).integers(1, hi + 1, n, dtype=np.int32)

    ids = [draw(0, k), draw(1, k), draw(2, groups), draw(3, k), draw(4, k), draw(5, groups)]
    v1, v2 = draw(6, 5), draw(7, 15)
    v3 = np.round(_rng(p.seed, _G1_TAG, 8, chunk).random(n) * 100.0, 6)
    masks: list[np.ndarray | None] = [None] * 9
    if p.na_pct:
        for c, domain in enumerate((k, k, groups, k, k, groups)):
            masks[c] = _na_value_mask(ids[c], domain, p.na_pct, _rng(p.seed, _G1_TAG, 100, c))
        for c in range(6, 9):  # row level, an exact share of this chunk
            masks[c] = _na_row_mask(n, p.na_pct, _rng(p.seed, _G1_TAG, 101, c, chunk))
    arrays = [
        _ids_to_str(ids[0], masks[0], 3),
        _ids_to_str(ids[1], masks[1], 3),
        _ids_to_str(ids[2], masks[2], 10),
        pa.array(ids[3], mask=masks[3]),
        pa.array(ids[4], mask=masks[4]),
        pa.array(ids[5], mask=masks[5]),
        pa.array(v1, mask=masks[6]),
        pa.array(v2, mask=masks[7]),
        pa.array(v3, mask=masks[8]),
    ]
    return pa.Table.from_arrays(arrays, schema=G1_SCHEMA)


def g1_chunks(p: G1Params) -> Iterator[pa.Table]:
    for chunk, start in enumerate(range(0, p.rows, p.chunk_rows)):
        yield g1_chunk(p, chunk, min(p.chunk_rows, p.rows - start))


def _split_xlr(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upstream ``split_xlr``: a permutation of 1..1.1n split into common (0.9n), left-only, right-only (0.1n each)."""
    key = rng.permutation(np.arange(1, n + n // 10 + 1, dtype=np.int32))
    nx = n - n // 10
    return key[:nx], key[nx:n], key[n:]


def _sample_all(keys: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Upstream ``sample_all``: every key at least once, the rest drawn with replacement, shuffled."""
    assert len(keys) <= size
    extra = keys[rng.integers(0, len(keys), size - len(keys))]
    y = np.concatenate([keys, extra])
    rng.shuffle(y)
    return y


@dataclass
class _JoinKeys:
    lhs: list[np.ndarray]  # id1..id3 of x
    rhs: dict[str, list[np.ndarray]]  # table -> its id columns (id1[, id2[, id3]])
    v1: np.ndarray
    v2: dict[str, np.ndarray]


def _join_keys(p: J1Params) -> _JoinKeys:
    sizes = (p.n_small, p.n_medium, p.rows)
    parts = [_split_xlr(n, _rng(p.seed, _J1_TAG, 0, i)) for i, n in enumerate(sizes)]
    lhs = [
        _sample_all(np.concatenate([xs, ls]), p.rows, _rng(p.seed, _J1_TAG, 1, i))
        for i, (xs, ls, _) in enumerate(parts)
    ]
    rkeys = [np.concatenate([xs, rs]) for xs, _, rs in parts]
    del parts
    rhs: dict[str, list[np.ndarray]] = {}
    for stage, (name, n, ncols) in enumerate((("small", p.n_small, 1), ("medium", p.n_medium, 2), ("big", p.rows, 3))):
        rhs[name] = [_sample_all(rkeys[i], n, _rng(p.seed, _J1_TAG, 2, stage, i)) for i in range(ncols)]
    v1 = np.round(_rng(p.seed, _J1_TAG, 3, 0).random(p.rows) * 100.0, 6)
    v2 = {
        name: np.round(_rng(p.seed, _J1_TAG, 3, 1 + i).random(n) * 100.0, 6)
        for i, (name, n) in enumerate((("small", p.n_small), ("medium", p.n_medium), ("big", p.rows)))
    }
    return _JoinKeys(lhs, rhs, v1, v2)


J1_COLUMNS = {
    "x": ["id1", "id2", "id3", "id4", "id5", "id6", "v1"],
    "small": ["id1", "id4", "v2"],
    "medium": ["id1", "id2", "id4", "id5", "v2"],
    "big": ["id1", "id2", "id3", "id4", "id5", "id6", "v2"],
}


def _j1_schema(table: str) -> pa.Schema:
    def typ(c: str) -> pa.DataType:
        return pa.string() if c in ("id4", "id5", "id6") else pa.float64() if c in ("v1", "v2") else pa.int32()

    return pa.schema([(c, typ(c)) for c in J1_COLUMNS[table]])


def j1_chunks(p: J1Params, keys: _JoinKeys, table: str) -> Iterator[pa.Table]:
    """Chunks of one join table; only ``x`` gets NAs (upstream: only the left-hand table has them)."""
    ids = keys.lhs if table == "x" else keys.rhs[table]
    n_total = p.rows if table in ("x", "big") else p.n_small if table == "small" else p.n_medium
    v = keys.v1 if table == "x" else keys.v2[table]
    id_masks: list[np.ndarray | None] = [None] * len(ids)
    v_mask_all: np.ndarray | None = None
    if table == "x" and p.na_pct:
        for c, col in enumerate(ids):
            id_masks[c] = _na_present_mask(col, p.na_pct, _rng(p.seed, _J1_TAG, 100, c))
        v_mask_all = _na_row_mask(n_total, p.na_pct, _rng(p.seed, _J1_TAG, 101, 0))
    schema = _j1_schema(table)
    for start in range(0, n_total, p.chunk_rows):
        sl = slice(start, min(start + p.chunk_rows, n_total))
        cols: dict[str, pa.Array] = {}
        for c, col in enumerate(ids):
            mask = id_masks[c]
            m = None if mask is None else mask[sl]
            cols[f"id{c + 1}"] = pa.array(col[sl], mask=m)
            cols[f"id{c + 4}"] = _ids_to_str(col[sl], m, None)
        cols["v1" if table == "x" else "v2"] = pa.array(v[sl], mask=None if v_mask_all is None else v_mask_all[sl])
        yield pa.Table.from_arrays([cols[n] for n in schema.names], schema=schema)


# ---------------------------------------------------------------- writing
def _write_parquet(path: Path, chunks: Iterator[pa.Table], sort_by: list[str] | None, scratch: Path) -> dict[str, Any]:
    """Writes the chunks to ``path`` (sorted by ``sort_by``, ties in generation order) and returns the table digest."""
    if not sort_by:
        schema = None
        td = None
        with contextlib.ExitStack() as stack:
            w = None
            for t in chunks:
                if w is None:
                    schema = t.schema
                    td = TableDigest(schema.names)
                    w = stack.enter_context(pq.ParquetWriter(path, schema, compression="zstd"))
                assert td is not None
                td.update(t)
                w.write_table(t, row_group_size=ROW_GROUP_ROWS)
        assert td is not None
        return td.result()
    import duckdb

    tmp = scratch / (path.name + ".unsorted")
    rn = 0
    with contextlib.ExitStack() as stack:
        w = None
        for t in chunks:
            t = t.append_column(RN, pa.array(np.arange(rn, rn + t.num_rows, dtype=np.int64)))
            rn += t.num_rows
            if w is None:
                w = stack.enter_context(pq.ParquetWriter(tmp, t.schema, compression="zstd"))
            w.write_table(t, row_group_size=ROW_GROUP_ROWS)
    con = duckdb.connect()
    try:
        spill = scratch / "duckdb-spill"
        spill.mkdir(exist_ok=True)
        con.execute(f"SET temp_directory='{str(spill).replace(chr(39), chr(39) * 2)}'")
        order = ", ".join([*(f'"{c}" ASC NULLS FIRST' for c in sort_by), RN])
        src, dst = str(tmp).replace("'", "''"), str(path).replace("'", "''")
        con.execute(
            f"COPY (SELECT * EXCLUDE ({RN}) FROM read_parquet('{src}') ORDER BY {order}) TO '{dst}' "
            f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {ROW_GROUP_ROWS})"
        )
    finally:
        con.close()
        tmp.unlink(missing_ok=True)
        shutil.rmtree(scratch / "duckdb-spill", ignore_errors=True)
    return digest_parquet(path)


def _numpy_pyarrow() -> dict[str, str]:
    return {"numpy": str(np.__version__), "pyarrow": str(pa.__version__)}


def generate(
    task: str,
    params: G1Params | J1Params,
    root: Path,
    on_chunk: Callable[[str], None] | None = None,
) -> Path:
    """Generates one dataset into ``root/<label>/`` and returns that directory.

    Refuses to touch an existing non-empty directory (a dataset is immutable; new generator semantics get a new
    version directory). Written under a hidden temporary name, renamed into place when complete.
    """
    params.validate()
    if (task == "groupby") != isinstance(params, G1Params):
        raise ValueError(f"params {type(params).__name__} do not match task {task!r}")
    final = root / params.label
    if final.exists() and any(final.iterdir()):
        raise FileExistsError(f"{final} already exists; datasets are immutable (use a new --version to regenerate)")
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".{params.label}.partial-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        tables: dict[str, dict[str, Any]] = {}
        if isinstance(params, G1Params):
            specs = {"x": (g1_chunks(params), [f"id{i}" for i in range(1, 7)] if params.sorted else None)}
        else:
            keys = _join_keys(params)
            sort_cols = {"x": ["id1", "id2", "id3"], "small": ["id1"], "medium": ["id2"], "big": ["id3"]}
            specs = {t: (j1_chunks(params, keys, t), sort_cols[t] if params.sorted else None) for t in J1_COLUMNS}
        for name, (chunks, sort_by) in specs.items():
            f = tmp / f"{name}.parquet"
            if on_chunk:
                on_chunk(name)
            d = _write_parquet(f, chunks, sort_by, tmp)
            tables[name] = {"file": f.name, "file_sha256": sha256_file(f), **d}
        manifest = {
            "schema": 1,
            "generator": "krishiv-bench h2o_data",
            "generator_version": GENERATOR_VERSION,
            "upstream": "db-benchmark groupby-datagen.R / join-datagen.R (MPL-2.0); distributions reimplemented, "
            "not byte-identical",
            "task": task,
            "label": params.label,
            "params": asdict(params),
            "libs": _numpy_pyarrow(),
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


def read_manifest(directory: Path) -> dict[str, Any] | None:
    f = directory / MANIFEST
    return json.loads(f.read_text()) if f.is_file() else None


def verify(directory: Path) -> list[str]:
    """Recomputes the content digests from the Parquet files; returns the problems found (empty = verified)."""
    man = read_manifest(directory)
    if man is None:
        return [f"{directory}: no {MANIFEST}"]
    problems = []
    for name, rec in man["tables"].items():
        f = directory / rec["file"]
        if not f.is_file():
            problems.append(f"{name}: {f} is missing")
            continue
        try:
            got = digest_parquet(f)
        except (pa.ArrowException, OSError) as e:
            problems.append(f"{name}: cannot read {f}: {e}")
            continue
        if got["rows"] != rec["rows"]:
            problems.append(f"{name}: {got['rows']} rows, manifest says {rec['rows']}")
        for col, dig in rec["columns"].items():
            if got["columns"].get(col) != dig:
                problems.append(f"{name}.{col}: content digest differs from the manifest")
    if content_digest(man["tables"]) != man["content_sha256"]:
        problems.append("manifest content_sha256 does not match its table digests")
    return problems
