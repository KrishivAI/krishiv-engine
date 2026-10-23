"""Dataset of the ``memcap`` suite: a seeded synthetic star schema (``sales`` fact + ``products`` dimension).

The generator is adapted from the MIT-licensed ``engine-bench`` repository by Firat Celik
(https://github.com/firfircelik/engine-bench, ``gen_data.py``); the notice is in ``bench/README.md``. Column
names, types, cardinalities, the seed and the chunking are the published ones, so the rows are the same.
What differs: the output layout (``<root>/sales/part-NNNN.parquet`` plus ``<root>/products.parquet``, the layout
the Krishiv runner registers), a content digest that does not depend on the Parquet encoder, and a refusal to
write into an existing directory.

Output goes to ``$KRISHIV_DATA/memcap/v1/<rows>m/`` and is never regenerated in place: a change to the generator
is a new version directory (``v2``), because recorded results are tied to the exact rows they ran on.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .paths import data_root

VERSION = "v1"
SEED = 20261019
CHUNK = 5_000_000  # rows per part file and per independent RNG stream
ROW_GROUP = 1_000_000
N_PRODUCTS = 50_000
N_STORES = 500
START = int(np.datetime64("2026-01-01T00:00:00", "s").astype(np.int64))
SPAN_S = 270 * 24 * 3600
STATUSES = np.array(["completed", "shipped", "processing", "cancelled", "returned"])
STATUS_P = [0.55, 0.20, 0.12, 0.08, 0.05]
CATEGORIES = np.array(
    ["electronics", "books", "home", "clothing", "grocery", "toys", "sports", "beauty", "garden", "auto"]
)
DIGEST_FILE = "digest.json"
_ROWS_RE = re.compile(r"^([1-9]\d*)m$")


def parse_rows(label: str) -> int:
    """``10m`` -> 10_000_000. Only whole millions: a part file holds ``CHUNK`` rows and the label names the dir."""
    m = _ROWS_RE.match(label.strip().lower())
    if not m:
        raise ValueError(f"--rows must be a whole number of millions such as 10m, 50m or 150m (got {label!r})")
    return int(m.group(1)) * 1_000_000


def rows_label(rows: int) -> str:
    if rows <= 0 or rows % 1_000_000:
        raise ValueError(f"rows must be a positive multiple of 1,000,000 (got {rows})")
    return f"{rows // 1_000_000}m"


def memcap_dir(label: str) -> Path:
    parse_rows(label)
    return data_root() / "memcap" / VERSION / label.strip().lower()


def n_customers(rows: int) -> int:
    """About 10 orders per customer (15M customers at 150M rows)."""
    return max(1, rows // 10)


def products_table() -> pa.Table:
    rng = np.random.default_rng(SEED)
    return pa.table(
        {
            "product_id": np.arange(N_PRODUCTS, dtype=np.int32),
            "category": rng.choice(CATEGORIES, N_PRODUCTS),
            "brand": np.char.add("brand_", rng.integers(0, 2_000, N_PRODUCTS).astype(str)),
            "list_price": np.round(rng.uniform(2.0, 500.0, N_PRODUCTS), 2),
        }
    )


def sales_chunk(index: int, lo: int, n: int, customers: int) -> pa.Table:
    """Rows ``lo .. lo + n`` of ``sales``; chunk ``index`` draws from its own stream ``[SEED, index]``."""
    rng = np.random.default_rng([SEED, index])
    return pa.table(
        {
            "order_id": np.arange(lo, lo + n, dtype=np.int64),
            "ts": pa.array(START + rng.integers(0, SPAN_S, n), pa.timestamp("s")),
            "customer_id": rng.integers(0, customers, n, dtype=np.int32),
            "product_id": rng.integers(0, N_PRODUCTS, n, dtype=np.int32),
            "store_id": rng.integers(0, N_STORES, n, dtype=np.int16),
            "qty": rng.integers(1, 10, n, dtype=np.int16),
            "price": np.round(rng.uniform(1.0, 500.0, n), 2),
            "status": rng.choice(STATUSES, n, p=STATUS_P),
        }
    )


def update_digest(h: hashlib._Hash, table: pa.Table) -> None:
    """Feeds the logical content of ``table`` (not its Parquet bytes) into ``h``: column names, types, values."""
    for name, col in zip(table.column_names, table.columns, strict=True):
        h.update(f"{name}:{col.type}:{len(col)}\n".encode())
        if pa.types.is_string(col.type):
            h.update("\x00".join(col.to_pylist()).encode())
        else:
            arr = col.combine_chunks()
            if pa.types.is_timestamp(col.type):
                arr = arr.cast(pa.int64())
            h.update(arr.to_numpy(zero_copy_only=False).tobytes())


def gen_memcap(rows: int, out: Path, chunk: int = CHUNK) -> dict:
    """Writes the dataset for ``rows`` rows to the new directory ``out`` and returns its digest record.

    ``out`` must not exist (an existing dataset is never touched). The files are written to a sibling temporary
    directory and renamed into place at the end, so ``out`` is either absent or complete.
    """
    if rows <= 0:
        raise ValueError(f"rows must be positive (got {rows})")
    if chunk <= 0:
        raise ValueError(f"chunk must be positive (got {chunk})")
    if out.exists():
        raise FileExistsError(
            f"{out} exists: a dataset is never regenerated in place (a generator change is a new version)"
        )
    tmp = out.with_name(f"{out.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "sales").mkdir(parents=True)
    try:
        customers = n_customers(rows)
        products = products_table()
        pq.write_table(products, tmp / "products.parquet", compression="snappy")
        ph = hashlib.sha256()
        update_digest(ph, products)
        sh = hashlib.sha256()
        files = 0
        for i, lo in enumerate(range(0, rows, chunk)):
            t = sales_chunk(i, lo, min(chunk, rows - lo), customers)
            update_digest(sh, t)
            pq.write_table(t, tmp / "sales" / f"part-{i:04d}.parquet", row_group_size=ROW_GROUP, compression="snappy")
            files += 1
        record = {
            "generator": VERSION,
            "seed": SEED,
            "rows": rows,
            "chunk_rows": chunk,
            "sales_files": files,
            "customers": customers,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "sales_content_sha256": sh.hexdigest(),
            "products_content_sha256": ph.hexdigest(),
        }
        (tmp / DIGEST_FILE).write_text(json.dumps(record, indent=2) + "\n")
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return record


def load_digest(directory: Path) -> dict:
    """The digest record of a generated dataset; says how to generate it when it is missing."""
    f = directory / DIGEST_FILE
    if not f.is_file():
        raise FileNotFoundError(
            f"{directory} is not a generated memcap dataset ({DIGEST_FILE} missing): "
            f"uv run krishiv-datasets memcap --rows {directory.name}"
        )
    return json.loads(f.read_text())


def generate(label: str) -> Path:
    """Generates ``--rows <label>`` under ``$KRISHIV_DATA/memcap/v1`` plus its manifest; skips an existing one."""
    from .datasets import write_manifest

    out = memcap_dir(label)
    if out.exists():
        print(f"[memcap] {out} exists, not regenerating (a dataset is never rewritten in place)")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    record = gen_memcap(parse_rows(label), out)
    print(f"[memcap] wrote {out}: sales sha256 {record['sales_content_sha256']}")
    print("manifest:", write_manifest(out, f"memcap-{VERSION}-{label.lower()}"))
    return out
