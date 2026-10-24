"""One ``memcap`` run in a fresh process: one query on one engine, result written as Parquet, one JSON line printed.

``python -m krishiv_bench.memcap_child <engine> <data dir> <query> <engine memory> <out dir> <scratch dir>``

The parent (``memcap.py``) starts this under a cgroup memory cap and measures wall time, peak memory and the
scratch directory from outside. The timed span (``query_s``) covers reading the Parquet, computing and writing the
result; ``setup_s`` is the interpreter start, imports and session start. Reports one line ``RESULT {json}``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from .sizes import parse_size

T0 = time.perf_counter()
THREADS = 8


def size_mib(text: str) -> int:
    """``3g`` / ``1024m`` -> MiB."""
    return parse_size(text) >> 20


def parquet_rows(out: Path) -> int:
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(f).metadata.num_rows for f in sorted(out.rglob("*.parquet")))


def run_duckdb(data: Path, query: str, memory: str, out: Path, scratch: Path) -> dict:
    import duckdb

    from .memcap_queries import SQL

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{size_mib(memory)}MiB'")
    con.execute(f"SET threads = {THREADS}")
    quoted = str(scratch).replace("'", "''")
    con.execute(f"SET temp_directory = '{quoted}'")
    sales = str(data / "sales" / "*.parquet").replace("'", "''")
    products = str(data / "products.parquet").replace("'", "''")
    con.execute(f"CREATE VIEW sales AS SELECT * FROM read_parquet('{sales}')")
    con.execute(f"CREATE VIEW products AS SELECT * FROM read_parquet('{products}')")
    t1 = time.perf_counter()
    target = str(out / "result.parquet").replace("'", "''")
    con.execute(f"COPY ({SQL[query]}) TO '{target}' (FORMAT parquet)")
    return {"version": duckdb.__version__, "t1": t1}


def run_polars(data: Path, query: str, out: Path) -> dict:
    os.environ.setdefault("POLARS_MAX_THREADS", str(THREADS))
    import polars as pl

    from .memcap_queries import polars_query

    sales = pl.scan_parquet(str(data / "sales" / "*.parquet"))
    products = pl.scan_parquet(str(data / "products.parquet"))
    t1 = time.perf_counter()
    polars_query(query, sales, products).sink_parquet(str(out / "result.parquet"))
    return {"version": pl.__version__, "t1": t1}


def run_spark(data: Path, query: str, memory: str, out: Path, scratch: Path) -> dict:
    import pyspark
    from pyspark.sql import SparkSession

    from .memcap_queries import SQL

    spark = (
        SparkSession.builder.master(f"local[{THREADS}]")
        .config("spark.driver.memory", f"{size_mib(memory)}m")
        .config("spark.local.dir", str(scratch))
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    try:
        spark.sparkContext.setLogLevel("ERROR")
        spark.read.parquet(str(data / "sales")).createOrReplaceTempView("sales")
        spark.read.parquet(str(data / "products.parquet")).createOrReplaceTempView("products")
        t1 = time.perf_counter()
        spark.sql(SQL[query]).write.mode("overwrite").parquet(str(out / "spark"))
    finally:
        spark.stop()
    return {"version": pyspark.__version__, "t1": t1}


def run_krishiv(data: Path, query: str, memory: str, out: Path, scratch: Path) -> dict:
    from .adapters.base import Dataset, Query
    from .adapters.krishiv import KrishivEngine
    from .memcap_queries import sql_for

    eng = KrishivEngine(
        threads=THREADS,
        memory_limit=memory,
        conf=[("krishiv.spill.dir", str(scratch))],
    )
    try:
        eng.setup(Dataset("memcap", 0, data, ["sales", "products"]))
        t1 = time.perf_counter()
        eng.dump(Query(query, sql_for("krishiv", query), "memcap"), out / "krishiv")
        return {"version": eng.version(), "t1": t1}
    finally:
        eng.teardown()


def main(argv: list[str]) -> int:
    engine, data_s, query, memory, out_s, scratch_s = argv
    data, out, scratch = Path(data_s), Path(out_s), Path(scratch_s)
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    scratch.chmod(0o700)  # Krishiv refuses a spill directory that group or others can write
    if engine == "duckdb":
        info = run_duckdb(data, query, memory, out, scratch)
    elif engine == "polars":
        info = run_polars(data, query, out)
    elif engine == "spark":
        info = run_spark(data, query, memory, out, scratch)
    elif engine == "krishiv":
        info = run_krishiv(data, query, memory, out, scratch)
    else:
        raise SystemExit(f"unknown memcap engine {engine!r}")
    t2 = time.perf_counter()
    res = {
        "engine": engine,
        "query": query,
        "version": info["version"],
        "setup_s": round(info["t1"] - T0, 3),
        "query_s": round(t2 - info["t1"], 3),
        "rows_out": parquet_rows(out),
    }
    print("RESULT " + json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
