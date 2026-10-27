"""One ``remote-filter`` run in a fresh process: the filter on one engine, result as ONE zstd Parquet file.

``python -m krishiv_bench.remote_filter_child <engine> <input> <query> <engine memory> <out> <scratch>
[--s3-endpoint URL] [--cred-file PATH]``

``input`` is a local directory (a fetched mirror or the synthetic stand-in: ``docs.parquet`` or ``docs/``)
or ``s3://bucket/prefix`` for the object-store mode. The mode follows the input's scheme. The timed span
(``query_s``) covers reading the Parquet, filtering and writing the result; ``setup_s`` is the interpreter
start, imports and session start. Reports one line ``RESULT {json}``.

Credentials (S3 mode) travel only through the environment or ``--cred-file`` (``KEY=VALUE`` lines applied to
the environment, never printed, never recorded): never on the command line. The endpoint URL is not secret.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .sizes import parse_size

T0 = time.perf_counter()
THREADS = 8
RESULT_FILE = "result.parquet"
CRED_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")


def apply_cred_file(path: str) -> None:
    """Applies a ``KEY=VALUE`` conf file to the environment (allowlisted keys only; values never leave)."""
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key not in CRED_KEYS:
            raise SystemExit(f"cred file {path}: key {key!r} is not an accepted S3 credential")
        os.environ[key] = value


def is_s3(input: str) -> bool:
    return input.startswith("s3://")


def docs_glob(data: Path) -> str:
    """The local ``docs`` source: the single file, or the glob over the shard directory."""
    if (data / "docs.parquet").is_file():
        return str(data / "docs.parquet")
    if (data / "docs").is_dir():
        return str(data / "docs" / "*.parquet")
    raise FileNotFoundError(f"no docs.parquet or docs/ under {data}")


def parquet_rows(out: Path) -> int:
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(f).metadata.num_rows for f in sorted(out.rglob("*.parquet")))


def coalesce_single(out: Path) -> None:
    """Renames the output to one ``result.parquet`` when the engine wrote exactly one part file."""
    target = out / RESULT_FILE
    if target.is_file():
        return
    parts = sorted(out.rglob("*.parquet"))
    if len(parts) == 1 and parts[0] != target:
        parts[0].rename(target)


def run_duckdb(input: str, query: str, memory: str, out: Path, scratch: Path, endpoint: str | None) -> dict:
    import duckdb

    from .remote_filter_queries import s3_sql, sql_for

    con = duckdb.connect()
    mib = parse_size(memory) >> 20
    con.execute(f"SET memory_limit = '{mib}MiB'")
    con.execute(f"SET threads = {THREADS}")
    quoted = str(scratch).replace("'", "''")
    con.execute(f"SET temp_directory = '{quoted}'")
    if is_s3(input):
        if not endpoint:
            raise SystemExit("S3 input needs --s3-endpoint")
        hostport = endpoint.split("://", 1)[-1]
        user, password = os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"]
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
        q = lambda s: s.replace("'", "''")  # noqa: E731  # in-process SQL text only, never recorded
        con.execute(
            "CREATE OR REPLACE SECRET s3 (TYPE S3, "
            f"KEY_ID '{q(user)}', SECRET '{q(password)}', ENDPOINT '{q(hostport)}', "
            "URL_STYLE 'path', USE_SSL false)"
        )
        sql = s3_sql("duckdb", query, input + "/*.parquet")
    else:
        glob = docs_glob(Path(input)).replace("'", "''")
        con.execute(f"CREATE VIEW docs AS SELECT * FROM read_parquet('{glob}')")
        sql = sql_for("duckdb", query)
    t1 = time.perf_counter()
    target = str(out / RESULT_FILE).replace("'", "''")
    con.execute(f"COPY ({sql}) TO '{target}' (FORMAT parquet, COMPRESSION zstd)")
    return {"version": duckdb.__version__, "t1": t1}


def run_polars(input: str, query: str, out: Path, endpoint: str | None) -> dict:
    os.environ.setdefault("POLARS_MAX_THREADS", str(THREADS))
    import polars as pl

    from .remote_filter_queries import sql_for

    if is_s3(input):
        if not endpoint:
            raise SystemExit("S3 input needs --s3-endpoint")
        storage_options = {
            "aws_endpoint_url": endpoint,
            "aws_region": "us-east-1",
            "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
            "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
            "aws_allow_http": "true",
        }
        docs = pl.scan_parquet(input + "/*.parquet", storage_options=storage_options)
    else:
        docs = pl.scan_parquet(docs_glob(Path(input)))
    t1 = time.perf_counter()
    pl.SQLContext(docs=docs).execute(sql_for("polars", query)).sink_parquet(str(out / RESULT_FILE), compression="zstd")
    return {"version": pl.__version__, "t1": t1}


def run_spark(input: str, query: str, memory: str, out: Path, scratch: Path, endpoint: str | None) -> dict:
    import pyspark
    from pyspark.sql import SparkSession

    from .remote_filter_queries import sql_for

    mib = parse_size(memory) >> 20
    builder = (
        SparkSession.builder.master(f"local[{THREADS}]")
        .config("spark.driver.memory", f"{mib}m")
        .config("spark.local.dir", str(scratch))
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
    )
    if is_s3(input):
        if not endpoint:
            raise SystemExit("S3 input needs --s3-endpoint")
        hostport = endpoint.split("://", 1)[-1]
        # Needs the hadoop-aws + aws-sdk jars on the classpath (maintainer machine); values come from the
        # environment / cred file, never from the command line.
        builder = (
            builder.config("spark.hadoop.fs.s3a.endpoint", hostport)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
            .config("spark.hadoop.fs.s3a.access.key", os.environ["AWS_ACCESS_KEY_ID"])
            .config("spark.hadoop.fs.s3a.secret.key", os.environ["AWS_SECRET_ACCESS_KEY"])
        )
        source = "s3a://" + input[len("s3://") :] + "/*.parquet"
    else:
        source = docs_glob(Path(input))
    spark = builder.getOrCreate()
    try:
        spark.sparkContext.setLogLevel("ERROR")
        df = spark.read.parquet(source)
        df.createOrReplaceTempView("docs")
        t1 = time.perf_counter()
        spark.sql(sql_for("spark", query)).coalesce(1).write.mode("overwrite").option("compression", "zstd").parquet(
            str(out / "spark")
        )
    finally:
        spark.stop()
    coalesce_single(out)
    return {"version": pyspark.__version__, "t1": t1}


def _krishiv_stub(work_parent: Path) -> Path:
    """A one-row local table so the runner starts in S3 mode (the query itself reads S3 directly)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    stub = work_parent / "stub"
    stub.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"stub": [1]}), stub / "stub.parquet")
    return stub


def run_krishiv(
    input: str, query: str, memory: str, out: Path, scratch: Path, work: Path, endpoint: str | None
) -> dict:
    from .adapters.base import Dataset, Query
    from .adapters.krishiv import KrishivEngine
    from .remote_filter_queries import s3_sql, sql_for

    conf = [("krishiv.spill.dir", str(scratch))]
    if is_s3(input):
        if not endpoint:
            raise SystemExit("S3 input needs --s3-endpoint")
        # Endpoint and path style are session settings (not secrets); the credentials stay in the
        # environment, where the object-store builder picks them up.
        conf += [("fs.s3a.endpoint", endpoint), ("fs.s3a.path.style.access", "true")]
        eng = KrishivEngine(threads=THREADS, memory_limit=memory, conf=conf)
        try:
            # The stub table only starts the runner (it needs one local table); the query reads S3 directly.
            eng.setup(Dataset("remote-filter", 0, _krishiv_stub(work), ["stub"]))
            t1 = time.perf_counter()
            eng.dump(Query(query, s3_sql("krishiv", query, input + "/*.parquet"), "remote-filter"), out)
            return {"version": eng.version(), "t1": t1}
        finally:
            eng.teardown()
    eng = KrishivEngine(threads=THREADS, memory_limit=memory, conf=conf)
    try:
        eng.setup(Dataset("remote-filter", 0, Path(input), ["docs"]))
        t1 = time.perf_counter()
        eng.dump(Query(query, sql_for("krishiv", query), "remote-filter"), out)
        return {"version": eng.version(), "t1": t1}
    finally:
        eng.teardown()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="remote_filter_child")
    ap.add_argument("engine")
    ap.add_argument("input")
    ap.add_argument("query")
    ap.add_argument("memory")
    ap.add_argument("out")
    ap.add_argument("scratch")
    ap.add_argument("--s3-endpoint", default=None)
    ap.add_argument("--cred-file", default=None)
    a = ap.parse_args(argv)
    if a.cred_file:
        apply_cred_file(a.cred_file)
    out, scratch, work = Path(a.out), Path(a.scratch), Path(a.out).parent
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    scratch.chmod(0o700)  # Krishiv refuses a spill directory that group or others can write
    if a.engine == "duckdb":
        info = run_duckdb(a.input, a.query, a.memory, out, scratch, a.s3_endpoint)
    elif a.engine == "polars":
        info = run_polars(a.input, a.query, out, a.s3_endpoint)
    elif a.engine == "spark":
        info = run_spark(a.input, a.query, a.memory, out, scratch, a.s3_endpoint)
    elif a.engine == "krishiv":
        info = run_krishiv(a.input, a.query, a.memory, out, scratch, work, a.s3_endpoint)
    else:
        raise SystemExit(f"unknown remote-filter engine {a.engine!r}")
    coalesce_single(out)
    t2 = time.perf_counter()
    res = {
        "engine": a.engine,
        "query": a.query,
        "mode": "s3" if is_s3(a.input) else "file",
        "version": info["version"],
        "setup_s": round(info["t1"] - T0, 3),
        "query_s": round(t2 - info["t1"], 3),
        "rows_out": parquet_rows(out),
    }
    print("RESULT " + json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
