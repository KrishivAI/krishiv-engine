"""One ``remote-filter`` run in a fresh process: filter, write the result as Parquet, print one JSON line.

``python -m krishiv_bench.remote_filter_child <engine> <input> <out dir> <scratch dir> [krishiv memory_limit]``

``input`` is either a local directory holding the ``fw`` table (``fw.parquet`` or an ``fw/`` directory of part
files, the layout ``krishiv-datasets remote-filter`` writes) or a JSON list of Parquet sources (local files or
remote HTTPS URLs). The Krishiv runner registers local Parquet views, so it needs the local directory; a URL
list for Krishiv is reported, not attempted. DuckDB reads the list with ``read_parquet`` and ``COPY ... (FORMAT
parquet, COMPRESSION zstd)`` to one file; Polars runs
``scan_parquet(...).filter(...).sink_parquet(..., compression="zstd")`` to one file; Krishiv writes the
runner's native output directory with the runner's default Parquet codec (the DUMP protocol carries no
per-write compression option).

The timed span (``query_s``) covers reading the sources, filtering and writing the result; ``setup_s`` is the
time from this module's import to the start of the timed span (imports and session start). Reports one line
``RESULT {json}`` with the row count, the exact sum of ``token_count``, a digest of the sorted ``id`` values,
the file codecs of what was written and the process's peak resident memory (itself and its children).
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

from .remote_filter import FILTER_SQL, TABLE

T0 = time.perf_counter()
THREADS = 8  # every engine gets the same worker count


def parse_input(arg: str) -> tuple[str, list[str]]:
    """``(kind, items)``: ``files`` (local Parquet paths) or ``urls`` (``https://``, ``s3://``, ``hf://`` sources)."""
    if arg.strip().startswith("["):
        items = json.loads(arg)
        if not isinstance(items, list) or not items or not all(isinstance(x, str) for x in items):
            raise SystemExit(f"input JSON list must be a non-empty list of strings (got {arg[:80]!r})")
        if all(x.startswith(("https://", "s3://", "hf://")) for x in items):
            return "urls", list(items)
        return "files", expand_files(items)
    p = Path(arg)
    if p.is_dir():
        files = sorted(str(f) for f in p.rglob("*.parquet"))
        if not files:
            raise SystemExit(f"no Parquet files under {arg}")
        return "files", files
    if p.is_file():
        return "files", [str(p)]
    raise SystemExit(f"input is neither a directory, a file nor a JSON list: {arg!r}")


def expand_files(items: list[str]) -> list[str]:
    out: list[str] = []
    for x in items:
        p = Path(x)
        if p.is_dir():
            out.extend(sorted(str(f) for f in p.rglob("*.parquet")))
        else:
            out.append(x)
    if not out:
        raise SystemExit("no Parquet sources in the input list")
    return out


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def output_stats(out: Path) -> tuple[int, int, list[str], str | None]:
    """Row count, exact sum of ``token_count``, sorted file codecs and the sha256 of the sorted ``id`` values
    (None when the output has no ``id`` column) over the Parquet files under ``out``."""
    import pyarrow.parquet as pq

    rows, total, codecs = 0, 0, set()
    ids: list[str] = []
    has_ids = True
    for f in sorted(out.rglob("*.parquet")):
        pf = pq.ParquetFile(f)
        md = pf.metadata
        rows += md.num_rows
        for rg in range(md.num_row_groups):
            for c in range(md.num_columns):
                codecs.add(md.row_group(rg).column(c).compression)
        columns = ["token_count"] + (["id"] if "id" in pf.schema_arrow.names else [])
        table = pf.read(columns=columns)
        total += sum(v for v in table["token_count"].to_pylist() if v is not None)
        if "id" in table.column_names:
            ids.extend(str(v) for v in table["id"].to_pylist())
        else:
            has_ids = False
    digest = None
    if has_ids and rows:
        h = hashlib.sha256()
        for v in sorted(ids):
            h.update(v.encode("utf-8"))
            h.update(b"\n")
        digest = h.hexdigest()
    return rows, total, sorted(codecs), digest


def peak_rss_bytes() -> int:
    """This process's and its finished children's peak resident memory (Linux ``ru_maxrss`` is in KiB)."""
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(own, kids) * 1024


def s3_settings() -> dict[str, str] | None:
    """The throw-away S3 server's endpoint and credentials from the environment, or None for other sources."""
    endpoint = os.environ.get("KRISHIV_BENCH_S3_ENDPOINT")
    if not endpoint:
        return None
    return {
        "endpoint": endpoint,
        "key": os.environ.get("KRISHIV_BENCH_S3_KEY", ""),
        "secret": os.environ.get("KRISHIV_BENCH_S3_SECRET", ""),
    }


def run_duckdb(items: list[str], out: Path) -> dict:
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET threads = {THREADS}")
    if any(i.startswith("s3://") for i in items):
        s3 = s3_settings()
        if s3 is None:
            raise SystemExit("s3:// sources need KRISHIV_BENCH_S3_ENDPOINT (run with --s3)")
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
        for k, v in (
            ("s3_endpoint", s3["endpoint"]),
            ("s3_access_key_id", s3["key"]),
            ("s3_secret_access_key", s3["secret"]),
            ("s3_url_style", "path"),
        ):
            con.execute(f"SET {k} = {sql_literal(v)}")
        con.execute("SET s3_use_ssl = false")
    sources_sql = ", ".join(sql_literal(u) for u in items)
    target = sql_literal(str(out / "result.parquet"))
    t1 = time.perf_counter()
    con.execute(
        f"COPY (SELECT * FROM read_parquet([{sources_sql}]) WHERE {FILTER_SQL})"
        f" TO {target} (FORMAT parquet, COMPRESSION zstd)"
    )
    return {"version": duckdb.__version__, "t1": t1}


def run_polars(items: list[str], out: Path) -> dict:
    import polars as pl

    options = None
    if any(i.startswith("s3://") for i in items):
        s3 = s3_settings()
        if s3 is None:
            raise SystemExit("s3:// sources need KRISHIV_BENCH_S3_ENDPOINT (run with --s3)")
        options = {
            "aws_endpoint_url": f"http://{s3['endpoint']}",
            "aws_access_key_id": s3["key"],
            "aws_secret_access_key": s3["secret"],
            "aws_region": "us-east-1",
            "aws_allow_http": "true",
        }
    frame = pl.scan_parquet(items, storage_options=options)
    filt = (pl.col("int_score") >= 4) & (pl.col("token_count") >= 4000) & (pl.col("language_score") >= 0.95)
    t1 = time.perf_counter()
    frame.filter(filt).sink_parquet(str(out / "result.parquet"), compression="zstd")
    return {"version": pl.__version__, "t1": t1}


def run_krishiv(data: Path, out: Path, scratch: Path, memory_limit: str | None = None) -> dict:
    from .adapters.base import Dataset, Query
    from .adapters.krishiv import KrishivEngine

    if not data.is_dir() or (not (data / f"{TABLE}.parquet").is_file() and not (data / TABLE).is_dir()):
        raise SystemExit(
            f"krishiv reads the local mirror directory with a {TABLE!r} table "
            f"({TABLE}.parquet or {TABLE}/ of part files under {data})"
        )
    kwargs: dict = {"threads": THREADS, "conf": [("krishiv.spill.dir", str(scratch))]}
    if memory_limit is not None:
        kwargs["memory_limit"] = memory_limit
    eng = KrishivEngine(**kwargs)
    try:
        eng.setup(Dataset("remote-filter", 0, data, [TABLE]))
        t1 = time.perf_counter()
        eng.dump(Query("q1", f"SELECT * FROM {TABLE} WHERE {FILTER_SQL}", "remote-filter"), out / "krishiv")
        return {"version": eng.version(), "t1": t1, "engine_memory_limit": eng.memory_limit}
    finally:
        eng.teardown()


def run_krishiv_urls(items: list[str], out: Path, scratch: Path, memory_limit: str | None = None) -> dict:
    """Krishiv reading the same URLs the other engines read (``s3://`` through the throw-away server's
    endpoint and keys, ``https://`` as is), registered as the ``--table`` view."""
    from .adapters.base import Dataset, Query
    from .adapters.krishiv import KrishivEngine

    conf: list[tuple[str, str]] = [("krishiv.spill.dir", str(scratch))]
    if any(x.startswith("s3://") for x in items):
        s3 = s3_settings()
        if s3 is None:
            raise SystemExit("s3:// sources need KRISHIV_BENCH_S3_ENDPOINT (run with --s3)")
        conf += [
            ("fs.s3a.endpoint", f"http://{s3['endpoint']}"),
            ("fs.s3a.access.key", s3["key"]),
            ("fs.s3a.secret.key", s3["secret"]),
            ("fs.s3a.path.style.access", "true"),
            ("fs.s3a.endpoint.region", "us-east-1"),
        ]
    kwargs: dict = {"threads": THREADS, "conf": conf, "tables": {TABLE: items}}
    if memory_limit is not None:
        kwargs["memory_limit"] = memory_limit
    eng = KrishivEngine(**kwargs)
    try:
        eng.setup(Dataset("remote-filter", 0, scratch, [TABLE]))
        t1 = time.perf_counter()
        eng.dump(Query("q1", f"SELECT * FROM {TABLE} WHERE {FILTER_SQL}", "remote-filter"), out / "krishiv")
        return {"version": eng.version(), "t1": t1, "engine_memory_limit": eng.memory_limit}
    finally:
        eng.teardown()


def main(argv: list[str]) -> int:
    if len(argv) == 4:
        engine, input_arg, out_s, scratch_s = argv
        krishiv_limit: str | None = None
    elif len(argv) == 5 and argv[0] == "krishiv":
        engine, input_arg, out_s, scratch_s, krishiv_limit = argv
    else:
        raise SystemExit(
            f"expected <engine> <input> <out dir> <scratch dir> [krishiv memory_limit] (got {len(argv)} args)"
        )
    out, scratch = Path(out_s), Path(scratch_s)
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    scratch.chmod(0o700)  # Krishiv refuses a spill directory that group or others can write
    if engine == "krishiv":
        if input_arg.strip().startswith("["):
            kind, items = parse_input(input_arg)
            if kind != "urls":
                raise SystemExit("krishiv reads the local mirror directory or a list of URLs")
            info = run_krishiv_urls(items, out, scratch, krishiv_limit)
        else:
            info = run_krishiv(Path(input_arg), out, scratch, krishiv_limit)
    else:
        _, items = parse_input(input_arg)
        if engine == "duckdb":
            info = run_duckdb(items, out)
        elif engine == "polars":
            info = run_polars(items, out)
        else:
            raise SystemExit(f"unknown remote-filter engine {engine!r}")
    t2 = time.perf_counter()
    rows_out, sum_tokens, compression, ids_digest = output_stats(out)
    res = {
        "engine": engine,
        "version": info["version"],
        "setup_s": round(info["t1"] - T0, 3),
        "query_s": round(t2 - info["t1"], 3),
        "rows_out": rows_out,
        "sum_tokens": sum_tokens,
        "ids_digest": ids_digest,
        "compression": compression,
        "peak_rss_bytes": peak_rss_bytes(),
    }
    if "engine_memory_limit" in info:
        res["engine_memory_limit"] = info["engine_memory_limit"]
    print("RESULT " + json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
