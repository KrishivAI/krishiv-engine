"""Dataset access of the ``remote-filter`` suite: the FineWeb-Edu sample mirror and a tiny synthetic stand-in.

The workload reads the ``sample/10BT`` config of ``HuggingFaceFW/fineweb-edu`` (14 Parquet shards, pinned to
``REVISION``). ``fetch`` downloads the shards (or copies a pre-fetched mirror directory) into the pinned local
mirror ``$KRISHIV_DATA/remote-filter/v1/``: ``docs/<original shard names>`` plus a ``digest.json`` with the
dataset revision and the SHA-256 of every file. The same SHA-256 manifest convention as the other suites
(``sha256sum -c`` compatible, relative paths) is written under ``$KRISHIV_DATA/manifests/``. A mirror is never
regenerated in place: an existing directory with a digest is reused, anything else is refused.

``gen_synthetic`` writes a few thousand rows with the same column names and types (seeded, a handful of
survivors under the filter) for the harness tests, which must not need the network or the 28 GB dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .paths import data_root

SOURCE_REPO = "HuggingFaceFW/fineweb-edu"
SOURCE_CONFIG = "sample/10BT"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
VERSION = "v1"
SHARD_FILES = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(14)]
MANIFEST_NAME = f"remote-filter-{VERSION}"
DIGEST_FILE = "digest.json"
DOCS_DIR = "docs"
SEED = 20261027
SYNTHETIC_SURVIVORS = 7

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("token_count", pa.int64()),
        ("int_score", pa.int64()),
        ("language_score", pa.float64()),
    ]
)


def mirror_dir() -> Path:
    return data_root() / "remote-filter" / VERSION


def shard_url(name: str) -> str:
    """The pinned download URL of a shard (redirects to signed storage URLs; nothing secret in it)."""
    return f"https://huggingface.co/datasets/{SOURCE_REPO}/resolve/{REVISION}/{name}"


def local_name(name: str) -> str:
    """The shard's file name inside the mirror (``sample/10BT/000_00000.parquet`` -> ``000_00000.parquet``)."""
    return name.rsplit("/", 1)[-1]


def sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "krishiv-bench remote-filter"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        if resp.status != 200:
            raise OSError(f"{url} returned HTTP {resp.status}")
        shutil.copyfileobj(resp, f, 1 << 20)


def load_digest(directory: Path) -> dict:
    """The digest record of a fetched mirror; says how to fetch it when it is missing."""
    f = directory / DIGEST_FILE
    if not f.is_file():
        raise FileNotFoundError(
            f"{directory} is not a fetched remote-filter mirror ({DIGEST_FILE} missing): "
            "uv run krishiv-datasets remote-filter [--mirror <pre-fetched dir>]"
        )
    return json.loads(f.read_text())


def fetch(dest: Path | None = None, mirror: Path | None = None) -> Path:
    """Fetches the shards (or copies ``mirror``) into ``dest`` and returns it.

    ``dest`` defaults to the pinned mirror. An existing digest means the mirror is complete and is reused;
    an existing directory without one is refused (a dataset is never regenerated in place). Files land in a
    sibling temporary directory that is renamed into place, so ``dest`` is either absent or complete.
    """
    from .datasets import write_manifest

    out = dest or mirror_dir()
    if out.exists():
        if (out / DIGEST_FILE).is_file():
            return out
        raise FileExistsError(
            f"{out} exists without {DIGEST_FILE}: a dataset is never regenerated in place "
            "(remove it or fetch elsewhere)"
        )
    if mirror is not None and not mirror.is_dir():
        raise FileNotFoundError(f"pre-fetched mirror {mirror} is not a directory")
    tmp = out.with_name(f"{out.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        docs = tmp / DOCS_DIR
        docs.mkdir(parents=True)
        files = []
        for name in SHARD_FILES:
            target = docs / local_name(name)
            if mirror is not None:
                src = mirror / local_name(name)
                if not src.is_file():
                    raise FileNotFoundError(f"pre-fetched mirror {mirror} lacks {local_name(name)}")
                shutil.copyfile(src, target)
            else:
                print(f"[remote-filter] {shard_url(name)} -> {target}", flush=True)
                _download(shard_url(name), target)
            files.append(
                {"name": f"{DOCS_DIR}/{target.name}", "bytes": target.stat().st_size, "sha256": sha256_file(target)}
            )
        record = {
            "generator": VERSION,
            "source": SOURCE_REPO,
            "config": SOURCE_CONFIG,
            "revision": REVISION,
            "files": files,
        }
        (tmp / DIGEST_FILE).write_text(json.dumps(record, indent=2) + "\n")
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print("manifest:", write_manifest(out, MANIFEST_NAME))
    return out


def gen_synthetic(rows: int, out: Path, seed: int = SEED) -> dict:
    """Writes ``out/docs.parquet``: ``rows`` seeded rows, exactly ``SYNTHETIC_SURVIVORS`` passing the filter.

    Non-survivors can never pass: ``token_count`` stays below 4000 and ``language_score`` below 0.95. ``out``
    must not exist (a dataset is never regenerated in place).
    """
    if rows <= SYNTHETIC_SURVIVORS:
        raise ValueError(f"rows must exceed the survivor count ({SYNTHETIC_SURVIVORS}, got {rows})")
    if out.exists():
        raise FileExistsError(f"{out} exists: a dataset is never regenerated in place")
    rng = np.random.default_rng(seed)
    ids = np.array([f"synth-{i:07d}" for i in range(rows)])
    text = np.array([f"synthetic document {i} " + "lorem ipsum " * 8 for i in range(rows)])
    token_count = rng.integers(100, 4000, rows).astype(np.int64)
    int_score = rng.integers(1, 4, rows).astype(np.int64)
    language_score = rng.uniform(0.50, 0.949, rows)
    for i in range(SYNTHETIC_SURVIVORS):
        token_count[i] = 5000 + i
        int_score[i] = 5
        language_score[i] = 0.99
    table = pa.table(
        {
            "id": ids,
            "text": text,
            "token_count": token_count,
            "int_score": int_score,
            "language_score": language_score,
        },
        schema=SCHEMA,
    )
    out.mkdir(parents=True)
    try:
        pq.write_table(table, out / "docs.parquet", compression="zstd")
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    h = hashlib.sha256()
    for col in table.columns:
        arr = col.combine_chunks()
        h.update(str(arr.type).encode() + b"\x00")
        h.update(b"\x00".join(str(v.as_py()).encode() for v in arr))
    record = {
        "generator": "synthetic-v1",
        "seed": seed,
        "rows": rows,
        "survivors": SYNTHETIC_SURVIVORS,
        "content_sha256": h.hexdigest(),
    }
    (out / DIGEST_FILE).write_text(json.dumps(record, indent=2) + "\n")
    return record


__all__ = [
    "DIGEST_FILE",
    "DOCS_DIR",
    "MANIFEST_NAME",
    "REVISION",
    "SCHEMA",
    "SEED",
    "SHARD_FILES",
    "SOURCE_CONFIG",
    "SOURCE_REPO",
    "SYNTHETIC_SURVIVORS",
    "VERSION",
    "fetch",
    "gen_synthetic",
    "load_digest",
    "local_name",
    "mirror_dir",
    "sha256_file",
    "shard_url",
]
