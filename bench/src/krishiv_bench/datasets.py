"""Dataset generators (P0-T18). CLI: `uv run krishiv-datasets <tpch|tpcds|clickbench|nexmark|zipf|h2o-*|dsb|manifest>`.

All output goes to $KRISHIV_DATA/<dataset>/sf<N>/ as Parquet, plus a sha256 manifest in
$KRISHIV_DATA/manifests/. Nothing here is ever written inside the repo.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .paths import data_root, dataset_dir, fmt_sf, manifest_stem

TPCH_TABLES = ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]
CLICKBENCH_URL = "https://datasets.clickhouse.com/hits_compatible/hits.parquet"
CLICKBENCH_PARTITIONED_URL = "https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_{i}.parquet"


# ---------------------------------------------------------------- manifests
def sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(directory: Path, name: str) -> Path:
    """sha256sum-compatible manifest (relative paths) at $KRISHIV_DATA/manifests/<name>.sha256."""
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    out = data_root() / "manifests" / f"{name}.sha256"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p in files:
            f.write(f"{sha256_file(p)}  {p.relative_to(directory)}\n")
    return out


# ---------------------------------------------------------------- TPC-H
def gen_tpch(sf: float, force: bool = False) -> Path:
    """TPC-H via tpchgen-cli (cargo install tpchgen-cli). Large SFs are split into parts."""
    out = dataset_dir("tpch", sf)
    if out.exists() and any(out.iterdir()) and not force:
        print(f"[tpch] {out} exists, skipping (use --force)")
        return out
    if shutil.which("tpchgen-cli") is None:
        sys.exit("tpchgen-cli not found: `cargo install tpchgen-cli`")
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["tpchgen-cli", "parquet", "-s", fmt_sf(sf), "--output-dir", str(out)]
    print("[tpch]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out


def gen_tpch_dbgen(sf: float, force: bool = False) -> Path:
    """TPC-H via DuckDB's tpch extension (the reference dbgen). Output dir `tpch-dbgen/sf<N>`.

    Why both generators: tpchgen-cli is fast and scales well, but its random text columns (addresses, comments)
    differ from dbgen, so queries touching them (q02/q10/q13/q15/q20) do not reproduce the *official* answer sets.
    The dbgen data reproduces all 22 official answers at SF1 and is the correctness reference dataset.
    """
    import duckdb

    # Versioned by the generator that writes it: the dbgen text columns differ between DuckDB versions, and each
    # version's official answers match only its own output. A dataset is never regenerated in place.
    out = dataset_dir(f"tpch-dbgen@{duckdb.__version__}", sf)
    if out.exists() and any(out.iterdir()) and not force:
        print(f"[tpch-dbgen] {out} exists, skipping (use --force)")
        return out
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={float(sf)})")
    for t in TPCH_TABLES:
        con.execute(f"COPY {t} TO '{out / (t + '.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    return out


# ---------------------------------------------------------------- TPC-DS
def gen_tpcds(sf: float, force: bool = False) -> Path:
    """TPC-DS via DuckDB's tpcds extension (dsdgen), exported table-by-table to Parquet."""
    import duckdb

    out = dataset_dir("tpcds", sf)
    if out.exists() and any(out.iterdir()) and not force:
        print(f"[tpcds] {out} exists, skipping (use --force)")
        return out
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL tpcds; LOAD tpcds")
    con.execute(f"CALL dsdgen(sf={float(sf)})")
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    for t in tables:
        con.execute(f"COPY {t} TO '{out / (t + '.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    print(f"[tpcds] wrote {len(tables)} tables to {out}")
    return out


# ---------------------------------------------------------------- ClickBench
def fetch_clickbench(partitioned: bool = False, force: bool = False) -> Path:
    """Download hits.parquet (~14 GB single file). NOT run automatically anywhere."""
    out = data_root() / "clickbench"
    out.mkdir(parents=True, exist_ok=True)
    urls = (
        [(CLICKBENCH_PARTITIONED_URL.format(i=i), f"hits_{i}.parquet") for i in range(100)]
        if partitioned
        else [(CLICKBENCH_URL, "hits.parquet")]
    )
    for url, name in urls:
        dest = out / name
        if dest.exists() and not force:
            continue
        print(f"[clickbench] {url} -> {dest}")
        # curl -C - resumes; -f fails on HTTP errors
        subprocess.run(["curl", "-fL", "-C", "-", "-o", str(dest), url], check=True)
    return out


# ---------------------------------------------------------------- Zipf
def zipf_keys(rows: int, keys: int, alpha: float, seed: int = 0) -> np.ndarray:
    """Draw `rows` keys in [0, keys) with P(k) ∝ 1/(k+1)^alpha (finite Zipf; alpha may be < 1).

    numpy's Generator.zipf requires alpha > 1 and is unbounded, so we sample the bounded
    distribution by inverse-CDF over the explicit pmf. Key 0 is the hottest key.
    """
    if not alpha > 0.0:
        raise ValueError("alpha must be > 0")
    if keys < 1 or rows < 0:
        raise ValueError("keys must be >= 1 and rows >= 0")
    rng = np.random.default_rng(seed)
    pmf = 1.0 / np.arange(1, keys + 1, dtype=np.float64) ** alpha
    cdf = np.cumsum(pmf)
    cdf /= cdf[-1]
    u = rng.random(rows)
    return np.searchsorted(cdf, u, side="right").clip(max=keys - 1).astype(np.int64)


def gen_zipf(
    rows: int,
    keys: int,
    alpha: float,
    out: Path | None = None,
    seed: int = 0,
    payload_cols: int = 2,
    chunk: int = 5_000_000,
) -> Path:
    """Parquet with columns: key (int64, Zipf-skewed), id (int64), v0..vN (float64 payload)."""
    out = out or (data_root() / "zipf" / f"a{alpha}_k{keys}_r{rows}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [("key", pa.int64()), ("id", pa.int64())] + [(f"v{i}", pa.float64()) for i in range(payload_cols)]
    )
    rng = np.random.default_rng(seed + 1)
    pmf = 1.0 / np.arange(1, keys + 1, dtype=np.float64) ** alpha
    cdf = np.cumsum(pmf)
    cdf /= cdf[-1]
    krng = np.random.default_rng(seed)
    with pq.ParquetWriter(out, schema, compression="zstd") as w:
        done = 0
        while done < rows:
            n = min(chunk, rows - done)
            k = np.searchsorted(cdf, krng.random(n), side="right").clip(max=keys - 1)
            cols = [pa.array(k, pa.int64()), pa.array(np.arange(done, done + n), pa.int64())]
            cols += [pa.array(rng.random(n)) for _ in range(payload_cols)]
            w.write_table(pa.Table.from_arrays(cols, schema=schema))
            done += n
    return out


# ---------------------------------------------------------------- Nexmark
def gen_nexmark(
    events: int,
    seed: int = 0,
    out_dir: Path | None = None,
    num_people: int = 1000,
    num_auctions: int = 5000,
    hot_ratio: float = 0.5,
) -> Path:
    """Small Python Nexmark-style generator -> person/auction/bid Parquet (batch form).

    Event mix follows the Nexmark default proportions (person 2%, auction 6%, bid 92%).
    It is a *simplified* generator (uniform + hot-key skew, monotonic event time at 1 ms
    steps), sufficient for P0 smoke use; E05/E08 may need the reference Java generator or a
    Rust port (planned: P2 'krishiv-nexmark' source). Documented in bench/README.md.
    """
    out_dir = out_dir or (data_root() / "nexmark" / f"e{events}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    kind = rng.choice(3, size=events, p=[0.02, 0.06, 0.92])
    ts = np.arange(events, dtype=np.int64)  # event time, ms since epoch 0
    base = 1_700_000_000_000

    def pick(n, universe):
        hot = rng.random(n) < hot_ratio
        k = rng.integers(0, universe, n)
        k[hot] = rng.integers(0, max(1, universe // 100), hot.sum())
        return k.astype(np.int64)

    n_p, n_a, n_b = (kind == 0).sum(), (kind == 1).sum(), (kind == 2).sum()
    person = pa.table(
        {
            "id": np.arange(n_p, dtype=np.int64),
            "name": [f"person_{i}" for i in range(n_p)],
            "state": rng.choice(["OR", "ID", "CA", "WA", "AZ"], n_p),
            "city": rng.choice(["Phoenix", "Portland", "Boise", "Seattle", "LA"], n_p),
            "date_time": base + ts[kind == 0],
        }
    )
    auction = pa.table(
        {
            "id": np.arange(n_a, dtype=np.int64),
            "seller": pick(n_a, max(1, num_people)),
            "category": rng.integers(0, 20, n_a).astype(np.int64),
            "initial_bid": rng.integers(100, 10_000, n_a).astype(np.int64),
            "date_time": base + ts[kind == 1],
            "expires": base + ts[kind == 1] + rng.integers(1_000, 100_000, n_a),
        }
    )
    bid = pa.table(
        {
            "auction": pick(n_b, max(1, num_auctions)),
            "bidder": pick(n_b, max(1, num_people)),
            "price": rng.integers(100, 100_000, n_b).astype(np.int64),
            "channel": rng.choice(["apple", "google", "facebook", "baidu"], n_b),
            "date_time": base + ts[kind == 2],
        }
    )
    for name, t in (("person", person), ("auction", auction), ("bid", bid)):
        pq.write_table(t, out_dir / f"{name}.parquet", compression="zstd")
    return out_dir


# ---------------------------------------------------------------- db-benchmark (groupby / join)
def gen_h2o(task: str, a: argparse.Namespace) -> Path:
    """``$KRISHIV_DATA/h2o/<version>/<label>/`` (see ``h2o_data``); an existing directory is never overwritten."""
    from . import h2o, h2o_data

    rows = h2o.SIZES[a.size]
    params = (
        h2o_data.G1Params(rows, a.k, a.na, a.sorted, a.seed)
        if task == "groupby"
        else h2o_data.J1Params(rows, a.na, a.sorted, a.seed)
    )
    root = data_root() / "h2o" / a.h2o_version
    print(f"[h2o-{task}] {params.label} -> {root / params.label}")
    try:
        d = h2o_data.generate(task, params, root, on_chunk=lambda t: print(f"[h2o-{task}] writing {t}", flush=True))
    except (ValueError, FileExistsError) as e:
        sys.exit(str(e))
    man = h2o_data.read_manifest(d)
    assert man is not None
    print(f"[h2o-{task}] content sha256 {man['content_sha256']}")
    return d


# ---------------------------------------------------------------- CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="krishiv-datasets")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tpch", "tpch-dbgen", "tpcds"):
        p = sub.add_parser(name)
        p.add_argument("--sf", type=float, default=1)
        p.add_argument("--force", action="store_true")
        p.add_argument("--no-manifest", action="store_true")
    p = sub.add_parser("clickbench", help="download hits.parquet (~14 GB)")
    p.add_argument("--partitioned", action="store_true", help="100-file variant")
    p.add_argument("--yes", action="store_true", help="confirm the 14 GB download")
    p = sub.add_parser("nexmark")
    p.add_argument("--events", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("zipf")
    p.add_argument("--rows", type=int, default=10_000_000)
    p.add_argument("--keys", type=int, default=1_000_000)
    p.add_argument("--alpha", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=0)
    from .dsb import DEFAULT_SEED as DEFAULT_QUERY_SEED
    from .h2o import DEFAULT_KEYS, SIZES
    from .h2o_data import GENERATOR_VERSION, UPSTREAM_SEED

    for task in ("groupby", "join"):
        p = sub.add_parser(f"h2o-{task}", help=f"db-benchmark {task} data (G1/J1) at 1e7/1e8/1e9 rows, deterministic")
        p.add_argument("--size", required=True, choices=list(SIZES))
        p.add_argument("--na", type=int, default=0, help="percent NAs (published variants: 0 and 5)")
        p.add_argument("--sorted", action="store_true")
        p.add_argument("--seed", type=int, default=UPSTREAM_SEED)
        p.add_argument("--h2o-version", default=GENERATOR_VERSION, help="output version directory (never overwritten)")
        if task == "groupby":
            p.add_argument("--k", type=int, default=DEFAULT_KEYS, help="number of id1/id2 groups")
    p = sub.add_parser(
        "h2o-verify", help="recompute a generated h2o dataset's content digests against its MANIFEST.json"
    )
    p.add_argument("directory", type=Path)
    p = sub.add_parser(
        "dsb",
        help="DSB (many-join TPC-DS derivative; TPC EULA tools fetched at run time): data, queries, DuckDB oracle",
    )
    p.add_argument("--sf", type=int, default=1, help="integer scale (dsdgen), about 1 GB of raw data per unit")
    p.add_argument("--seed", type=int, default=DEFAULT_QUERY_SEED, help="dsqgen seed for the query parameters")
    p.add_argument("--threads", type=int, default=1, help="DuckDB threads for the Parquet conversion (1: byte-stable)")
    p.add_argument("--no-oracle", action="store_true", help="skip the DuckDB answers")
    p = sub.add_parser("dsb-verify", help="recompute a generated DSB dataset's checksums against its MANIFEST.json")
    p.add_argument("directory", type=Path)
    p = sub.add_parser("memcap", help="memcap suite data (seeded star schema) under $KRISHIV_DATA/memcap/v1/<rows>")
    p.add_argument("--rows", nargs="+", default=["10m"], metavar="N", help="whole millions, e.g. 10m 50m 150m")
    p = sub.add_parser("manifest", help="(re)write sha256 manifest for a dataset dir")
    p.add_argument("name", choices=["tpch", "tpch-dbgen", "tpcds"])
    p.add_argument("--sf", type=float, default=1)
    a = ap.parse_args(argv)

    t0 = time.time()
    if a.cmd in ("h2o-groupby", "h2o-join"):
        print(gen_h2o(a.cmd.removeprefix("h2o-"), a))
    elif a.cmd == "dsb":
        from . import dsb

        try:
            d = dsb.generate(a.sf, a.seed, a.threads, oracle=not a.no_oracle)
        except (ValueError, FileExistsError) as e:
            sys.exit(str(e))
        print(d)
        print("manifest:", write_manifest(d, f"{d.parent.name}-{d.name}"))
        man = dsb.read_manifest(d)
        assert man is not None
        print("relations per query (count: queries):", man["relation_count_distribution"])
    elif a.cmd == "dsb-verify":
        from . import dsb

        problems = dsb.verify(a.directory)
        if problems:
            sys.exit("\n".join(problems))
        print(f"{a.directory}: verified")
    elif a.cmd == "h2o-verify":
        from . import h2o_data

        problems = h2o_data.verify(a.directory)
        if problems:
            sys.exit("\n".join(problems))
        print(f"{a.directory}: verified")
    elif a.cmd in ("tpch", "tpch-dbgen", "tpcds"):
        d = {"tpch": gen_tpch, "tpch-dbgen": gen_tpch_dbgen, "tpcds": gen_tpcds}[a.cmd](a.sf, a.force)
        if not a.no_manifest:
            print("manifest:", write_manifest(d, f"{d.parent.name}-{d.name}"))
    elif a.cmd == "memcap":
        from .memcap_data import generate

        try:
            for label in a.rows:
                generate(label)
        except (ValueError, FileExistsError) as e:
            ap.error(str(e))
    elif a.cmd == "manifest":
        print(write_manifest(dataset_dir(a.name, a.sf), manifest_stem(a.name, a.sf)))
    elif a.cmd == "clickbench":
        if not a.yes:
            sys.exit("ClickBench is ~14 GB; re-run with --yes to download.")
        print(fetch_clickbench(a.partitioned))
    elif a.cmd == "nexmark":
        print(gen_nexmark(a.events, a.seed))
    elif a.cmd == "zipf":
        print(gen_zipf(a.rows, a.keys, a.alpha, seed=a.seed))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
