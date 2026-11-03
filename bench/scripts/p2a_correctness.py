# SPDX-License-Identifier: Apache-2.0
"""TPC-H correctness from the three entry points (P2a exit gate).

Each entry point runs the 22 queries over a dbgen dataset and dumps every result as text:
``{"qNN": {"rows": [[cell|null, ...], ...], "seconds": s} | {"error": "..."}}``.

* ``run-python``: ``import krishiv`` (a wheel), ``spark.sql(...).collect()``.
* ``run-cli``: ``krishiv sql --table NAME=PATH -f qNN.sql --format csv``, one process per query.
* the Rust entry point is ``crates/krishiv/examples/tpch_correctness.rs`` (same JSON).
* ``check``: compares a dump with the official answers (SF1, DuckDB's ``tpch_answers()``) and with
  DuckDB run over the same Parquet (any scale); prints a Markdown table.

Numbers are compared as decimals. Against the official answers they must agree at the coarser of the
two scales (the answers hold doubles where Spark's decimal semantics give a decimal of scale 6). Against
DuckDB they must agree within a relative tolerance of 1e-6 (Spark's ``avg`` over decimals keeps fewer
digits than DuckDB's double); the largest relative difference seen is reported. Rows are compared in
order when the query has a top-level ORDER BY (rows that tie on its keys may swap), as sets otherwise.

    uv run python scripts/p2a_correctness.py run-python --data D --queries Q --out py.json
    uv run python scripts/p2a_correctness.py check --sf 1 --data D --queries Q --results py.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from krishiv_bench.compare import derive_limit, derive_sort_keys, has_top_level_order_by, load_tpch_answers

TABLES = ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]
REL_TOL = 1e-6


def query_ids(only: list[str] | None) -> list[str]:
    ids = [f"q{n:02d}" for n in range(1, 23)]
    if only is None:
        return ids
    unknown = sorted(set(only) - set(ids))
    if unknown or not only:
        raise SystemExit(f"unknown query ids {unknown or only!r}: expected some of q01..q22")
    return [q for q in ids if q in only]


def cell_text(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return str(v).lower()
    if hasattr(v, "isoformat"):
        return v.isoformat(sep=" ") if hasattr(v, "hour") else v.isoformat()
    return str(v)


# ------------------------------------------------------------------ runners
def run_python(args: argparse.Namespace) -> None:
    import krishiv  # ty: ignore[unresolved-import]  # the wheel under test, not a bench dependency

    spark = krishiv.session()
    for t in TABLES:
        spark.read.parquet(str(Path(args.data) / f"{t}.parquet")).createOrReplaceTempView(t)
    out: dict[str, dict] = {}
    for q in query_ids(args.only):
        sql = (Path(args.queries) / f"{q}.sql").read_text()
        start = time.perf_counter()
        try:
            rows = spark.sql(sql).collect()
            out[q] = {
                "rows": [[cell_text(v) for v in r] for r in rows],
                "seconds": time.perf_counter() - start,
            }
        except Exception as e:
            out[q] = {"error": f"{type(e).__name__}: {e}"}
        print(q, "ERROR" if "error" in out[q] else f"{len(out[q]['rows'])} rows", file=sys.stderr)
    Path(args.out).write_text(json.dumps(out))


def run_cli(args: argparse.Namespace) -> None:
    tables = [f"--table={t}={Path(args.data) / (t + '.parquet')}" for t in TABLES]
    out: dict[str, dict] = {}
    for q in query_ids(args.only):
        start = time.perf_counter()
        p = subprocess.run(
            [args.krishiv, "sql", *tables, "--format", "csv", "-f", str(Path(args.queries) / f"{q}.sql")],
            capture_output=True,
            text=True,
            check=False,
        )
        seconds = time.perf_counter() - start
        if p.returncode != 0:
            out[q] = {"error": f"exit {p.returncode}: {p.stderr.strip()[:500]}"}
        else:
            out[q] = {"rows": csv_cells(p.stdout)[1:], "seconds": seconds}  # [1:]: the header row
        print(q, "ERROR" if "error" in out[q] else f"{len(out[q]['rows'])} rows", file=sys.stderr)
    Path(args.out).write_text(json.dumps(out))


def csv_cells(text: str) -> list[list[str | None]]:
    """Rows of the CLI's CSV output: an unquoted empty field is NULL, a quoted one (`""`) the empty string.

    `csv.reader` cannot tell the two apart. Every line ends a row, so a one-column NULL row is an empty line.
    """
    rows: list[list[str | None]] = []
    row: list[str | None] = []
    field: list[str] = []
    quoted = in_quotes = False
    i, n = 0, len(text)

    def end_field() -> None:
        nonlocal quoted
        row.append("".join(field) if field or quoted else None)
        field.clear()
        quoted = False

    while i < n:
        c = text[i]
        if in_quotes:
            if c == '"' and i + 1 < n and text[i + 1] == '"':
                field.append('"')
                i += 1
            elif c == '"':
                in_quotes = False
            else:
                field.append(c)
        elif c == '"':
            in_quotes = quoted = True
        elif c == ",":
            end_field()
        elif c in "\r\n":
            if c == "\r" and i + 1 < n and text[i + 1] == "\n":
                i += 1
            end_field()
            rows.append(row)
            row = []
        else:
            field.append(c)
        i += 1
    if field or quoted or row:
        end_field()
        rows.append(row)
    return rows


# ------------------------------------------------------------------ comparison
def as_decimal(s: str | None) -> Decimal | None:
    if s is None:
        return None
    try:
        d = Decimal(s.strip())
    except InvalidOperation:
        return None
    return d if d.is_finite() else None


def decimal_places(d: Decimal) -> int:
    exponent = d.as_tuple().exponent
    return max(-exponent, 0) if isinstance(exponent, int) else 0


def cells_equal(a: str | None, b: str | None, answer: bool, worst: list[float], text: bool = False) -> bool:
    """`a` is the engine's cell, `b` the reference's (official answer text, or DuckDB's value).

    `text` marks a string column (DuckDB's result type): its cells compare as text even when they look
    numeric, so `"013"` is not `"13"`.
    """
    if a is None or b is None:
        # The official answer files cannot tell an empty field from NULL; DuckDB can.
        return (a in (None, "")) and (b in (None, "")) if answer else a is None and b is None
    da, db = (None, None) if text else (as_decimal(a), as_decimal(b))
    if da is None or db is None:
        # Text is exact; only CHAR padding at the end may differ, never leading characters.
        return a.rstrip() == b.rstrip()
    if answer:
        # The official answers hold doubles at full precision for averages and ratios; under Spark's
        # decimal semantics the engine returns them as decimals of a smaller scale. Equal means: the same
        # value at the coarser of the two scales (the engine's value is the answer correctly rounded).
        scale = min(decimal_places(da), decimal_places(db))
        return round(da, scale) == round(db, scale)
    if da == db:
        return True
    if decimal_places(da) == decimal_places(db):
        # Counts, keys and sums of decimals have the same scale in both engines: they must be equal.
        # The tolerance below is only for values whose scale differs (Spark's decimal `avg` against
        # DuckDB's double).
        return False
    rel = float(abs(da - db) / max(abs(da), abs(db), Decimal(1)))
    worst[0] = max(worst[0], rel)
    return rel <= REL_TOL


def rows_equal(
    got,
    ref,
    ordered: bool,
    answer: bool,
    worst: list[float],
    sort_keys: list[int] | None = None,
    text_cols: frozenset[int] = frozenset(),
    limit: int | None = None,
) -> str | None:
    """None when equal, else a one-line difference.

    An ordered result may differ from the reference in the order of rows that tie on the ORDER BY keys: it
    passes when the sets of rows are equal and the key columns agree position by position. When the query's
    LIMIT cut the result (`len(ref) >= limit`), the last tie group, when it holds two or more rows, may hold
    other rows in each result: only its keys must agree. Columns in `text_cols` compare as text.
    """
    first = [0.0]
    diff = _rows_diff(got, ref, ordered, answer, first, text_cols)
    if diff is None:
        worst[0] = max(worst[0], first[0])
        return None
    if not ordered or not sort_keys or len(got) != len(ref):
        return diff
    tie_worst = [0.0]

    def same_keys(g, r) -> bool:
        return all(cells_equal(g[k], r[k], answer, tie_worst, k in text_cols) for k in sort_keys)

    if not all(same_keys(g, r) for g, r in zip(got, ref, strict=True)):
        return diff
    end = len(ref)
    if limit is not None and ref and len(ref) >= limit:
        start = end
        while start > 0 and same_keys(ref[start - 1], ref[-1]):
            start -= 1
        # Only a visible tie (two or more rows on the last keys) may hide
        # other rows past the cut; a single last row compares whole.
        if end - start >= 2:
            end = start
    if _rows_diff(got[:end], ref[:end], False, answer, tie_worst, text_cols) is not None:
        return diff
    worst[0] = max(worst[0], tie_worst[0])
    return None


def _rows_diff(got, ref, ordered: bool, answer: bool, worst: list[float], text_cols: frozenset[int]) -> str | None:
    if len(got) != len(ref):
        return f"{len(got)} rows, reference has {len(ref)}"
    if not ordered:

        def key(r):
            return tuple((f"{d:.4f}" if (d := as_decimal(c)) is not None else (c or "")) for c in r)

        got, ref = sorted(got, key=key), sorted(ref, key=key)
    for i, (g, r) in enumerate(zip(got, ref, strict=True)):
        if len(g) != len(r):
            return f"row {i}: {len(g)} columns, reference has {len(r)}"
        for j, (a, b) in enumerate(zip(g, r, strict=True)):
            if not cells_equal(a, b, answer, worst, j in text_cols):
                return f"row {i} column {j}: {a!r}, reference {b!r}"
    return None


def duckdb_rows(data: Path, sql: str):
    import duckdb

    con = duckdb.connect()
    for t in TABLES:
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{data / (t + '.parquet')}')")
    cur = con.execute(sql)
    text_cols = frozenset(i for i, d in enumerate(cur.description or []) if str(d[1]).upper() == "VARCHAR")
    return [[cell_text(v) for v in r] for r in cur.fetchall()], text_cols


def check(args: argparse.Namespace) -> int:
    results = json.loads(Path(args.results).read_text())
    answers = load_tpch_answers(args.sf) if args.official else {}
    worst = [0.0]
    ok = 0
    lines = ["| query | rows | official | DuckDB | detail |", "|---|---:|---|---|---|"]
    for q in query_ids(args.only):
        sql = (Path(args.queries) / f"{q}.sql").read_text()
        res = results.get(q)
        if res is None or "error" in res:
            lines.append(f"| {q} | - | ERROR | ERROR | {(res or {}).get('error', 'missing')[:160]} |")
            continue
        got, ordered = res["rows"], has_top_level_order_by(sql)
        keys, limit = derive_sort_keys(sql), derive_limit(sql)[0]
        official = "n/a"
        detail = ""
        duck_rows, text_cols = duckdb_rows(Path(args.data), sql)
        if q in answers:
            d = rows_equal(got, answers[q], ordered, True, worst, keys, text_cols, limit)
            official = "ok" if d is None else "MISMATCH"
            detail += "" if d is None else f"official: {d}; "
        d2 = rows_equal(got, duck_rows, ordered, False, worst, keys, text_cols, limit)
        duck = "ok" if d2 is None else "MISMATCH"
        detail += "" if d2 is None else f"DuckDB: {d2}"
        if official in ("ok", "n/a") and duck == "ok":
            ok += 1
        lines.append(f"| {q} | {len(got)} | {official} | {duck} | {detail} |")
    total = len(query_ids(args.only))
    print("\n".join(lines))
    print(f"\n{ok}/{total} queries correct; largest relative difference vs DuckDB {worst[0]:.2e}")
    return 0 if ok == total else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run-python", "run-cli"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--queries", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--only", nargs="*")
        if name == "run-cli":
            p.add_argument("--krishiv", required=True, help="path of the krishiv binary")
    p = sub.add_parser("check")
    p.add_argument("--sf", type=float, required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--official", action="store_true", help="also compare with the official answers (SF1)")
    p.add_argument("--only", nargs="*")
    args = ap.parse_args()
    if args.cmd == "run-python":
        run_python(args)
    elif args.cmd == "run-cli":
        run_cli(args)
    else:
        return check(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
