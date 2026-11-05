# SPDX-License-Identifier: Apache-2.0
"""Nested TPC-H: the TPC-H tables with every non-key column folded into one ``<table>_data`` array of structs.

Adapted from Apache DataFusion Comet's ``benchmarks/nested/`` (Apache-2.0; apache/datafusion-comet PR #7044,
``scripts/nest_tpch.scala`` and ``queries/q01..q22.sql``). The queries are Comet's, unchanged apart from the licence
header kept at the top of each file (``bench/queries/tpch-nested/``); they are derived from TPC-H under the TPC Fair
Use Policy. Only the shape changes: each array holds exactly one element, so exploding it never changes a row count
and every nested query returns the same answer as its flat TPC-H counterpart.

Layout (as Comet's): join keys and the partition columns (``p_brand``, ``c_mktsegment``, ``o_orderdate``,
``l_shipdate``) stay at the top level; every other column moves into ``<table>_data``. Source columns are cast to
the types below so the output schema is the same whichever generator made the flat data. Divergences from Comet's
conversion, deliberate: the tables are written as one ``<table>.parquet`` file each (the harness layout), not
Hive-partitioned directories, and row groups use DuckDB's default size, not 512 MiB.

``krishiv-datasets tpch-nested --sf N`` writes ``$KRISHIV_DATA/tpch-nested/sf<N>/`` from the flat ``tpch`` dataset
and verifies it; ``krishiv-bench nested-check`` compares an engine's nested answers with DuckDB's flat ones.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import duckdb

from .paths import dataset_dir

MONEY = "DECIMAL(12,2)"


@dataclass(frozen=True)
class Table:
    name: str
    top: tuple[tuple[str, str], ...]  # (column, DuckDB type) kept at the top level
    nested: tuple[tuple[str, str], ...]  # (column, DuckDB type) folded into `<name>_data`

    @property
    def struct(self) -> str:
        return f"{self.name}_data"


TABLES: tuple[Table, ...] = (
    Table("region", (("r_regionkey", "BIGINT"),), (("r_name", "VARCHAR"), ("r_comment", "VARCHAR"))),
    Table(
        "nation",
        (("n_nationkey", "BIGINT"), ("n_regionkey", "BIGINT")),
        (("n_name", "VARCHAR"), ("n_comment", "VARCHAR")),
    ),
    Table(
        "supplier",
        (("s_suppkey", "BIGINT"), ("s_nationkey", "BIGINT")),
        (
            ("s_name", "VARCHAR"),
            ("s_address", "VARCHAR"),
            ("s_phone", "VARCHAR"),
            ("s_acctbal", MONEY),
            ("s_comment", "VARCHAR"),
        ),
    ),
    Table(
        "part",
        (("p_partkey", "BIGINT"), ("p_brand", "VARCHAR")),
        (
            ("p_name", "VARCHAR"),
            ("p_mfgr", "VARCHAR"),
            ("p_type", "VARCHAR"),
            ("p_size", "BIGINT"),
            ("p_container", "VARCHAR"),
            ("p_retailprice", MONEY),
            ("p_comment", "VARCHAR"),
        ),
    ),
    Table(
        "partsupp",
        (("ps_partkey", "BIGINT"), ("ps_suppkey", "BIGINT")),
        (("ps_availqty", "INTEGER"), ("ps_supplycost", MONEY), ("ps_comment", "VARCHAR")),
    ),
    Table(
        "customer",
        (("c_custkey", "BIGINT"), ("c_nationkey", "BIGINT"), ("c_mktsegment", "VARCHAR")),
        (
            ("c_name", "VARCHAR"),
            ("c_address", "VARCHAR"),
            ("c_phone", "VARCHAR"),
            ("c_acctbal", MONEY),
            ("c_comment", "VARCHAR"),
        ),
    ),
    Table(
        "orders",
        (("o_orderkey", "BIGINT"), ("o_custkey", "BIGINT"), ("o_orderdate", "DATE")),
        (
            ("o_orderstatus", "VARCHAR"),
            ("o_totalprice", MONEY),
            ("o_orderpriority", "VARCHAR"),
            ("o_clerk", "VARCHAR"),
            ("o_shippriority", "INTEGER"),
            ("o_comment", "VARCHAR"),
        ),
    ),
    Table(
        "lineitem",
        (("l_orderkey", "BIGINT"), ("l_partkey", "BIGINT"), ("l_suppkey", "BIGINT"), ("l_shipdate", "DATE")),
        (
            ("l_linenumber", "INTEGER"),
            ("l_quantity", MONEY),
            ("l_extendedprice", MONEY),
            ("l_discount", MONEY),
            ("l_tax", MONEY),
            ("l_returnflag", "VARCHAR"),
            ("l_linestatus", "VARCHAR"),
            ("l_commitdate", "DATE"),
            ("l_receiptdate", "DATE"),
            ("l_shipinstruct", "VARCHAR"),
            ("l_shipmode", "VARCHAR"),
            ("l_comment", "VARCHAR"),
        ),
    ),
)
BY_NAME = {t.name: t for t in TABLES}


def scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    """The single integer a one-row query returns (a count)."""
    row = con.execute(sql).fetchone()
    assert row is not None
    return int(row[0])


def source_path(root: Path, table: str) -> str:
    """The flat table's Parquet source: ``<root>/<table>.parquet`` or a glob over a directory of part files."""
    f = root / f"{table}.parquet"
    if f.is_file():
        return str(f)
    d = root / table
    if d.is_dir():
        return str(d / "*.parquet")
    raise FileNotFoundError(f"no flat Parquet source for {table!r} under {root}")


def _cast(col: str, ty: str) -> str:
    return f"CAST({col} AS {ty})"


def nest_select(t: Table, src: str) -> str:
    """The query that reads the flat table at ``src`` and returns its nested shape (one element per array)."""
    top = ", ".join(f"{_cast(c, ty)} AS {c}" for c, ty in t.top)
    fields = ", ".join(f"{c} := {_cast(c, ty)}" for c, ty in t.nested)
    return f"SELECT {top}, [struct_pack({fields})] AS {t.struct} FROM read_parquet('{src}')"


def flatten_select(t: Table, src: str) -> str:
    """The inverse: the nested table at ``src`` back to the flat columns, in the flat column order."""
    top = ", ".join(c for c, _ in t.top)
    inner = ", ".join(f"{t.struct}[1].{c} AS {c}" for c, _ in t.nested)
    return f"SELECT {top}, {inner} FROM read_parquet('{src}')"


def flat_select(t: Table, src: str) -> str:
    """The flat table at ``src`` as it is, uncast, in the column order of ``flatten_select``.

    Uncast on purpose: ``verify`` compares it with the flattened nested table, and DuckDB coerces the two sides to a
    common type, so a conversion that lost precision (a double rounded to a decimal) shows as a difference."""
    cols = ", ".join(c for c, _ in (*t.top, *t.nested))
    return f"SELECT {cols} FROM read_parquet('{src}')"


def nest(flat_root: Path, out: Path, *, threads: int | None = None) -> dict[str, int]:
    """Write the nested tables under ``out`` (zstd Parquet), one file per table; returns the row count of each."""
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    if threads:
        con.execute(f"SET threads = {int(threads)}")
    rows: dict[str, int] = {}
    for t in TABLES:
        dest = out / f"{t.name}.parquet"
        con.execute(
            f"COPY ({nest_select(t, source_path(flat_root, t.name))}) TO '{dest}' (FORMAT parquet, COMPRESSION zstd)"
        )
        rows[t.name] = scalar(con, f"SELECT count(*) FROM read_parquet('{dest}')")
    return rows


def verify(flat_root: Path, nested_root: Path) -> list[str]:
    """Problems with a nested dataset (empty when sound): row counts, one-element arrays, and the data itself.

    The nested table flattened back (first element of each array) must equal the flat source, cast to the same
    types, as a multiset in both directions.
    """
    problems: list[str] = []
    con = duckdb.connect()
    for t in TABLES:
        nested = nested_root / f"{t.name}.parquet"
        if not nested.is_file():
            problems.append(f"{t.name}: {nested} is missing")
            continue
        src = source_path(flat_root, t.name)
        n_flat = scalar(con, f"SELECT count(*) FROM read_parquet('{src}')")
        n_nested = scalar(con, f"SELECT count(*) FROM read_parquet('{nested}')")
        if n_flat != n_nested:
            problems.append(f"{t.name}: {n_nested} nested rows, {n_flat} flat rows")
        bad = scalar(con, f"SELECT count(*) FROM read_parquet('{nested}') WHERE len({t.struct}) <> 1")
        if bad:
            problems.append(f"{t.name}: {bad} rows whose {t.struct} does not hold exactly one element")
        for left, right, label in (
            (flatten_select(t, str(nested)), flat_select(t, src), "nested but not flat"),
            (flat_select(t, src), flatten_select(t, str(nested)), "flat but not nested"),
        ):
            diff = scalar(con, f"SELECT count(*) FROM (({left}) EXCEPT ALL ({right}))")
            if diff:
                problems.append(f"{t.name}: {diff} rows {label}")
    return problems


def digest(root: Path) -> str:
    """Content digest of a nested dataset independent of the Parquet encoder: the rows of every table, sorted."""
    con = duckdb.connect()
    h = hashlib.sha256()
    for t in TABLES:
        sel = flatten_select(t, str(root / f"{t.name}.parquet"))
        for row in con.execute(f"SELECT * FROM ({sel}) ORDER BY ALL").fetchall():
            h.update(repr(row).encode())
        h.update(t.name.encode())
    return h.hexdigest()


def generate(sf: float, source: str = "tpch", force: bool = False) -> Path:
    """Build ``$KRISHIV_DATA/tpch-nested/sf<N>`` from the flat ``source`` dataset and verify it."""
    flat = dataset_dir(source, sf)
    out = dataset_dir("tpch-nested", sf)
    if out.exists() and any(out.iterdir()) and not force:
        print(f"[tpch-nested] {out} exists, skipping (use --force)")
        return out
    if not flat.is_dir():
        raise SystemExit(f"no flat dataset at {flat}; run `uv run krishiv-datasets {source} --sf {sf}` first")
    rows = nest(flat, out)
    problems = verify(flat, out)
    if problems:
        raise SystemExit("nested dataset does not match its source:\n" + "\n".join(problems))
    print(f"[tpch-nested] {out}: " + ", ".join(f"{k} {v}" for k, v in rows.items()) + " rows, verified")
    return out


def check(
    engine_name: str,
    sf: float,
    only: list[str] | None = None,
    *,
    flat_dataset: str = "tpch",
    engine_kwargs: dict | None = None,
    make=None,
) -> list[str]:
    """Run the nested queries on ``engine_name`` and the flat TPC-H queries on DuckDB; return the mismatches.

    The nested answer must equal the flat one (the arrays hold one element), so DuckDB over the flat tables is
    the reference. Rows are compared in order when the flat query has a top-level ORDER BY, as sets otherwise.
    ``make`` replaces ``make_engine`` (tests).
    """
    from .adapters import Dataset, make_engine
    from .compare import has_top_level_order_by, rows_match
    from .runner import load_queries

    make = make or make_engine
    flat_root, nested_root = dataset_dir(flat_dataset, sf), dataset_dir("tpch-nested", sf)
    tables = [t.name for t in TABLES]
    duck = make("duckdb")
    eng = make(engine_name, **(engine_kwargs or {}))
    nested_queries = {q.id: q for q in load_queries("tpch-nested", engine_name, only)}
    flat_queries = {q.id: q for q in load_queries("tpch", "duckdb", only)}
    bad: list[str] = []
    duck.setup(Dataset(flat_dataset, sf, flat_root, tables))
    try:
        eng.setup(Dataset("tpch-nested", sf, nested_root, tables))
        try:
            for qid, nq in nested_queries.items():
                fq = flat_queries[qid]
                try:
                    got = eng.run(nq)
                except Exception as e:
                    bad.append(f"{qid}: {engine_name} failed: {str(e)[:300]}")
                    continue
                ok, msg = rows_match(got, duck.run(fq), has_top_level_order_by(fq.sql), sql=fq.sql)
                if not ok:
                    bad.append(f"{qid}: {msg}")
                print(f"{qid}: {'ok' if ok else 'MISMATCH'} ({len(got)} rows)")
        finally:
            eng.teardown()
    finally:
        duck.teardown()
    return bad
