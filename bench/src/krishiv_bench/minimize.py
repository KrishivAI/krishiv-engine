"""Shrinks a query that Krishiv answers differently from a reference engine to a minimal case.

``krishiv-bench minimize --suite S --query qN`` (the reference is DuckDB, or ``--reference spark-connect`` for the
Spark oracle) takes the query and its data and reduces both by delta debugging while the two engines keep
disagreeing (different rows, or Krishiv failing where the reference answers):

1. **Data**: every table the query names is first cut to a slice (``--seed-rows`` rows from the start, else evenly
   strided), then ``ddmin`` removes row ranges until no single chunk can go.
2. **Query**: structural reductions over the SQL text, applied greedily until a fixpoint: drop ``ORDER BY``,
   ``LIMIT``, ``HAVING``, ``WHERE``, ``GROUP BY``, ``DISTINCT``; drop each select-list item, each ``AND`` conjunct,
   each ``JOIN`` segment, each comma-separated ``FROM`` item and each set-operation branch; replace a subquery by
   ``NULL`` or recurse into it. Candidates that fail on the reference are never interesting.
3. **Data again** on the smaller query.

The result is written as ``<out>/<suite>_<query>/``: ``query.sql``, ``setup.sql`` (one
``CREATE TEMPORARY VIEW ... AS SELECT * FROM VALUES`` per table, for the golden runner), ``expected.json`` (the
reference rows), ``actual.json`` (Krishiv's rows or error) and the slice as Parquet under ``data/``.

Limits: a bug that needs rows the slice does not contain is not reproduced (``--full-data`` shrinks only the query);
the reductions work on the SQL text, not a parse tree, so a clause shape they do not know stays whole.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from . import compare
from .adapters import Dataset
from .adapters.krishiv import KrishivEngine, RunnerCrashed, RunnerError
from .errors import describe_error
from .paths import data_root

DEFAULT_SEED_ROWS = 2000
DEFAULT_MAX_RUNS = 1500
DEFAULT_MAX_SECONDS = 1200.0
DEFAULT_WHOLE_ROWS = 250_000  # tables up to this size stay whole in the first slice
ESCALATION = (1, 10, 100)  # the first slice's row count is seed_rows times each, until the difference reproduces


# ------------------------------------------------------------------------------------------ SQL text scanning
def mask(sql: str) -> str:
    """``sql`` with string literals, quoted identifiers and comments blanked (same length, same offsets)."""
    out = list(sql)
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c in "'\"`":
            j = i + 1
            while j < n:
                if sql[j] == c:
                    if j + 1 < n and sql[j + 1] == c:  # doubled quote
                        j += 2
                        continue
                    break
                j += 1
            for k in range(i + 1, min(j, n)):
                out[k] = " "
            i = j + 1
        elif sql.startswith("--", i):
            j = sql.find("\n", i)
            j = n if j < 0 else j
            out[i:j] = " " * (j - i)
            i = j
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out[i:j] = " " * (j - i)
            i = j
        else:
            i += 1
    return "".join(out)


def depths(masked: str) -> list[int]:
    """Parenthesis depth at every offset of a masked string."""
    d, cur = [], 0
    for c in masked:
        if c == ")":
            cur -= 1
        d.append(cur)
        if c == "(":
            cur += 1
    return d


def top_level(masked: str, pattern: str, dep: list[int] | None = None) -> list[re.Match[str]]:
    """Matches of ``pattern`` (case-insensitive, word-bounded) outside every parenthesis."""
    dep = dep if dep is not None else depths(masked)
    return [m for m in re.finditer(rf"\b(?:{pattern})\b", masked, re.IGNORECASE) if dep[m.start()] == 0]


def split_top(text: str, sep: str) -> list[str]:
    """Splits ``text`` at top-level ``sep`` (a regex of whole words or ``,``), keeping the pieces' text."""
    m = mask(text)
    dep = depths(m)
    pat = sep if sep == "," else rf"\b(?:{sep})\b"
    cuts = [x for x in re.finditer(pat, m, re.IGNORECASE) if dep[x.start()] == 0]
    parts, last = [], 0
    for x in cuts:
        parts.append(text[last : x.start()])
        last = x.end()
    parts.append(text[last:])
    return [p.strip() for p in parts]


_CLAUSE = r"select|from|where|group\s+by|having|order\s+by|limit|window|qualify"
_SETOP = r"union(?:\s+all)?|intersect|except"


def _clauses(sql: str) -> list[tuple[str, int, int]]:
    """Top-level clauses of one SELECT as ``(keyword, body_start, end)``."""
    m = mask(sql)
    hits = top_level(m, _CLAUSE)
    out = []
    for k, h in enumerate(hits):
        end = hits[k + 1].start() if k + 1 < len(hits) else len(sql)
        out.append((re.sub(r"\s+", " ", h.group(0).lower()), h.end(), end))
    return out


def _replace(sql: str, start: int, end: int, text: str) -> str:
    return (sql[:start] + text + sql[end:]).strip()


def _join_list(parts: list[str], sep: str) -> str:
    return sep.join(parts)


# -------------------------------------------------------------------------------------------- query reductions
def subqueries(sql: str) -> list[tuple[int, int]]:
    """Spans ``(start, end)`` of the content of every outermost parenthesised SELECT (parens excluded)."""
    m = mask(sql)
    dep = depths(m)
    spans = []
    for i, c in enumerate(m):
        if c == "(" and dep[i] == 0:
            j, level = i + 1, 1
            while j < len(m) and level:
                level += (m[j] == "(") - (m[j] == ")")
                j += 1
            if re.match(r"\s*(with|select)\b", sql[i + 1 : j - 1], re.IGNORECASE):
                spans.append((i + 1, j - 1))
    return spans


def candidates(sql: str) -> Iterator[str]:
    """Every one-step reduction of ``sql`` (not necessarily valid SQL), biggest reductions first."""
    sql = sql.strip().rstrip(";").strip()
    m = mask(sql)
    dep = depths(m)
    # Set operations: keep a single branch.
    ops = top_level(m, _SETOP, dep)
    if ops:
        cuts = [0, *[x.start() for x in ops], len(sql)]
        branches = []
        for k in range(len(cuts) - 1):
            lo = cuts[k] + (len(ops[k - 1].group(0)) if k else 0)
            branches.append(sql[lo : cuts[k + 1]].strip())
        for b in branches:
            if b != sql:
                yield b
    clauses = _clauses(sql)
    by = {k: (s, e) for k, s, e in clauses}
    # Whole clauses.
    for key in ("order by", "limit", "having", "where", "group by", "qualify", "window"):
        hit = [h for h in top_level(m, key.replace(" ", r"\s+"), dep)]
        if key in by and hit:
            yield _replace(sql, hit[0].start(), by[key][1], "")
    if re.search(r"\bselect\s+distinct\b", m, re.IGNORECASE):
        yield re.sub(r"\bselect\s+distinct\b", "select", sql, count=1, flags=re.IGNORECASE)
    # Select-list items.
    if "select" in by:
        s, e = by["select"]
        body = sql[s:e]
        distinct = re.match(r"\s*distinct\b", body, re.IGNORECASE)
        prefix = distinct.group(0) if distinct else ""
        body = body[len(prefix) :]
        items = split_top(body, ",")
        if len(items) > 1:
            for k in range(len(items)):
                yield _replace(sql, s, e, f" {prefix} {_join_list(items[:k] + items[k + 1 :], ', ')} ")
    # Conjuncts of WHERE / HAVING.
    for key in ("where", "having"):
        if key in by:
            s, e = by[key]
            conj = split_top(sql[s:e], "and")
            if len(conj) > 1:
                half = len(conj) // 2
                if half > 1:
                    yield _replace(sql, s, e, f" {_join_list(conj[:half], ' AND ')} ")
                    yield _replace(sql, s, e, f" {_join_list(conj[half:], ' AND ')} ")
                for k in range(len(conj)):
                    yield _replace(sql, s, e, f" {_join_list(conj[:k] + conj[k + 1 :], ' AND ')} ")
    # GROUP BY items and ORDER BY items.
    for key in ("group by", "order by"):
        if key in by:
            s, e = by[key]
            items = split_top(sql[s:e], ",")
            if len(items) > 1:
                for k in range(len(items)):
                    yield _replace(sql, s, e, f" {_join_list(items[:k] + items[k + 1 :], ', ')} ")
    # FROM: comma items and JOIN segments.
    if "from" in by:
        s, e = by["from"]
        frm = sql[s:e]
        comma = split_top(frm, ",")
        if len(comma) > 1:
            for k in range(len(comma)):
                yield _replace(sql, s, e, f" {_join_list(comma[:k] + comma[k + 1 :], ', ')} ")
        fm = mask(frm)
        fdep = depths(fm)
        joins = top_level(fm, r"(?:(?:inner|left|right|full|cross|natural)\s+(?:outer\s+)?)?join", fdep)
        if joins:
            starts = [x.start() for x in joins]
            bounds = [*starts, len(frm)]
            for k in range(len(joins)):
                yield _replace(sql, s, e, f" {frm[: bounds[k]]} {frm[bounds[k + 1] :]} ")
    # Subqueries: replace by NULL, or reduce inside.
    for start, end in subqueries(sql):
        yield _replace(sql, start - 1, end + 1, "NULL")
        inner = sql[start:end]
        for c in candidates(inner):
            yield _replace(sql, start, end, c)


def ddmin(items: list[Any], test: Callable[[list[Any]], bool]) -> list[Any]:
    """Zeller's ddmin: a locally minimal sub-list of ``items`` for which ``test`` still holds."""
    n = 2
    while len(items) >= 2:
        chunk = max(1, len(items) // n)
        subsets = [items[i : i + chunk] for i in range(0, len(items), chunk)]
        reduced = False
        for k, sub in enumerate(subsets):
            comp = [x for j, s in enumerate(subsets) if j != k for x in s]
            if len(subsets) > 2 and comp and test(comp):
                items, n, reduced = comp, max(n - 1, 2), True
                break
            if test(sub):
                items, n, reduced = sub, 2, True
                break
        if not reduced:
            if n >= len(items):
                break
            n = min(len(items), n * 2)
    if len(items) == 1 and test([]):
        return []
    return items


def shrink_query(sql: str, interesting: Callable[[str], bool], budget: Callable[[], bool] = lambda: True) -> str:
    """Greedy fixpoint over :func:`candidates`: take the first interesting smaller candidate and start again."""
    best = sql.strip().rstrip(";").strip()
    changed = True
    while changed and budget():
        changed = False
        for cand in candidates(best):
            if not cand or len(cand) >= len(best):
                continue
            if not budget():
                break
            if interesting(cand):
                best, changed = cand, True
                break
    return best


# ------------------------------------------------------------------------------------------------ data slices
def referenced_tables(sql: str, tables: list[str]) -> list[str]:
    m = mask(sql)
    return [t for t in tables if re.search(rf"\b{re.escape(t)}\b", m, re.IGNORECASE)]


def seed_slice(table: pa.Table, rows: int) -> pa.Table:
    """The first ``rows`` rows, or an even stride over the table when it has more."""
    if table.num_rows <= rows:
        return table
    return table.slice(0, rows)


def strided_slice(table: pa.Table, rows: int) -> pa.Table:
    if table.num_rows <= rows:
        return table
    step = table.num_rows / rows
    return table.take(pa.array([int(i * step) for i in range(rows)]))


def shrink_rows(tables: dict[str, pa.Table], interesting: Callable[[dict[str, pa.Table]], bool]) -> dict[str, pa.Table]:
    """ddmin over the row indices of each table in turn."""
    cur = dict(tables)
    for name in sorted(cur, key=lambda n: -cur[n].num_rows):
        idx = list(range(cur[name].num_rows))

        def test(sub: list[int], name: str = name) -> bool:
            cand = {**cur, name: cur[name].take(pa.array(sub, type=pa.int64()))}
            return interesting(cand)

        kept = ddmin(idx, test)
        cur[name] = cur[name].take(pa.array(kept, type=pa.int64()))
    return cur


# --------------------------------------------------------------------------------------------------- oracle
class Oracle:
    """Runs the reference and Krishiv on ``(sql, tables)``; counts runs; True when they still disagree."""

    def __init__(
        self,
        reference: Any,
        krishiv: Any,
        workdir: Path,
        max_runs: int = DEFAULT_MAX_RUNS,
        max_seconds: float = DEFAULT_MAX_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.reference, self.krishiv, self.workdir = reference, krishiv, workdir
        self.runs = 0
        self.max_runs = max_runs
        self.clock = clock
        self.deadline = clock() + max_seconds
        self._data_key: tuple | None = None
        self._sig: str | None = None

    def within_budget(self) -> bool:
        """False once the run count or the time is used up: the shrink stops and keeps the best case so far."""
        return self.runs < self.max_runs and self.clock() < self.deadline

    def _setup(self, ds: Dataset) -> None:
        for eng in (self.reference, self.krishiv):
            eng.teardown()
            eng.setup(ds)

    def pin(self, ds: Dataset, sql: str) -> dict[str, Any]:
        """Runs ``sql`` on the real dataset and pins the kind of difference every later check must keep.

        Without it the reduction can slip onto a different, easier difference (a division by zero that only
        appears once the data is tiny, say) and report that instead of the bug the query was chosen for.
        """
        self._setup(ds)
        self._data_key = None
        self.runs += 1
        out = self._compare(sql)
        if not out["differs"]:
            raise ValueError("the query does not differ from the reference on the dataset")
        self._sig = signature(out)
        return out

    def _load(self, tables: dict[str, pa.Table]) -> None:
        key = tuple(
            (n, t.num_rows, hash(tuple(t.column(0).to_pylist()[:50])) if t.num_columns else 0)
            for n, t in sorted(tables.items())
        )
        if key == self._data_key:
            return
        shutil.rmtree(self.workdir, ignore_errors=True)
        self.workdir.mkdir(parents=True)
        for n, t in tables.items():
            pq.write_table(t, self.workdir / f"{n}.parquet")
        self._setup(Dataset("minimize", 0, self.workdir, sorted(tables)))
        self._data_key = key

    def outcome(self, sql: str, tables: dict[str, pa.Table]) -> dict[str, Any]:
        """``{"differs": bool, "reference": rows | error, "krishiv": rows | error}``."""
        self._load(tables)
        self.runs += 1
        return self._compare(sql)

    def _compare(self, sql: str) -> dict[str, Any]:
        q = _Q(sql)
        try:
            ref_rows: Any = self.reference.run(q)
        except Exception as e:
            return {"differs": False, "reference": f"error: {describe_error(e)}", "krishiv": None}
        try:
            k_rows: Any = self.krishiv.run(q)
        except (RunnerError, RunnerCrashed) as e:
            return {"differs": True, "reference": ref_rows, "krishiv": f"error: {describe_error(e)}"}
        ordered = compare.has_top_level_order_by(sql)
        ok, _ = compare.rows_match(k_rows, ref_rows, ordered, sql=sql)
        # The same rows in another order: a tie-order or ordering difference, not a wrong value.
        order_only = not ok and ordered and compare.rows_match(k_rows, ref_rows, False, sql=sql)[0]
        return {"differs": not ok, "reference": ref_rows, "krishiv": k_rows, "order_only": order_only}

    def differs(self, sql: str, tables: dict[str, pa.Table]) -> bool:
        """True when the engines disagree in the pinned way (any way, when nothing is pinned)."""
        out = self.outcome(sql, tables)
        return bool(out["differs"]) and (self._sig is None or signature(out) == self._sig)


def signature(outcome: dict[str, Any]) -> str:
    """The kind of difference: the error class when Krishiv failed, ``order`` when only the row order differs,
    else ``rows`` (the answers differ)."""
    k = outcome["krishiv"]
    if isinstance(k, str):
        m = re.search(r"\[([A-Z][A-Z0-9_.]+)\]", k)
        return f"error:{m.group(1)}" if m else "error"
    return "order" if outcome.get("order_only") else "rows"


class _Q:
    """The part of ``adapters.base.Query`` an engine's ``run`` reads."""

    def __init__(self, sql: str):
        self.id, self.sql, self.suite, self.check_sql = "min", sql, "minimize", None


# ------------------------------------------------------------------------------------------------- the driver
def minimize(
    sql: str,
    tables: dict[str, pa.Table],
    oracle: Oracle,
    seed_rows: int = DEFAULT_SEED_ROWS,
    full_data: bool = False,
    whole_rows: int = DEFAULT_WHOLE_ROWS,
    big: set[str] | None = None,
) -> tuple[str, dict[str, pa.Table]]:
    """The minimal ``(sql, tables)`` that still differs; ``ValueError`` when the start does not."""
    used = {n: tables[n] for n in referenced_tables(sql, list(tables))}
    if not full_data:
        used = _first_slice(sql, used, oracle, seed_rows, whole_rows, big)
        _progress("first slice", oracle, used)
    elif not oracle.differs(sql, used):
        raise ValueError("the query does not differ on the full data")
    # The query first: with the data unchanged the engines are not restarted, so a step is cheap, and a smaller
    # query needs fewer rows to keep the difference, which is what makes the row shrink fast.
    sql = shrink_query(sql, lambda s: oracle.within_budget() and oracle.differs(s, used), oracle.within_budget)
    _progress("query shrunk", oracle, used, sql)
    used = {n: used[n] for n in referenced_tables(sql, list(used))}
    if not full_data:
        used = shrink_rows(used, lambda t: oracle.within_budget() and oracle.differs(sql, t))
        _progress("rows shrunk", oracle, used)
        sql = shrink_query(sql, lambda s: oracle.within_budget() and oracle.differs(s, used), oracle.within_budget)
    return sql, used


def _progress(stage: str, oracle: Oracle, tables: dict[str, pa.Table], sql: str = "") -> None:
    rows = ", ".join(f"{n} {t.num_rows}" for n, t in sorted(tables.items()))
    print(f"minimize: {stage} after {oracle.runs} runs: {rows} {sql[:100]}", flush=True)


def _first_slice(
    sql: str, used: dict[str, pa.Table], oracle: Oracle, seed_rows: int, whole_rows: int, big: set[str] | None = None
) -> dict[str, pa.Table]:
    """The first slice that still differs: big tables cut to ``seed_rows`` x 1, 10, 100 rows (head, then strided).

    Tables of at most ``whole_rows`` rows (and any table not in ``big``) stay whole, so joins from a sliced fact
    table keep their dimension rows.
    """
    sliced_names = {n for n, t in used.items() if t.num_rows > whole_rows} if big is None else big & set(used)
    tried: list[int] = []
    for factor in ESCALATION:
        n = seed_rows * factor
        if tried and all(used[name].num_rows <= n for name in sliced_names):
            break  # the previous round already kept every big table whole
        tried.append(n)
        for make in (seed_slice, strided_slice):
            sliced = {name: make(t, n) if name in sliced_names else t for name, t in used.items()}
            if oracle.differs(sql, sliced):
                return sliced
    raise ValueError(
        f"the difference does not reproduce on slices of up to {tried[-1]} rows of the tables over {whole_rows} rows; "
        "use --full-data (query reduction only) or a larger --seed-rows"
    )


def load_tables(root: Path, names: list[str], whole_rows: int, pool_rows: int) -> tuple[dict[str, pa.Table], set[str]]:
    """The tables of ``names`` and the set of big ones.

    A table of at most ``whole_rows`` rows is read whole. A bigger one is represented by a pool of about
    ``pool_rows`` rows made of evenly spaced row groups, so a 100-million-row table is never read in full.
    """
    tables: dict[str, pa.Table] = {}
    big: set[str] = set()
    for t in names:
        f = root / f"{t}.parquet"
        src = f if f.is_file() else root / t
        files = [src] if src.is_file() else sorted(src.glob("*.parquet"))
        if len(files) == 1 and pq.ParquetFile(files[0]).metadata.num_rows > whole_rows:
            tables[t] = _pool(files[0], pool_rows)
            big.add(t)
        else:
            tables[t] = pa.concat_tables([pq.read_table(x) for x in files])
            if tables[t].num_rows > whole_rows:
                big.add(t)
    return tables, big


def _pool(path: Path, rows: int) -> pa.Table:
    """Evenly spaced whole row groups of ``path`` until about ``rows`` rows are collected."""
    pf = pq.ParquetFile(path)
    groups = pf.metadata.num_row_groups
    sizes = [pf.metadata.row_group(i).num_rows for i in range(groups)]
    want = max(1, min(groups, round(rows / max(1, max(sizes)))))
    chosen = sorted({int(i * groups / want) for i in range(want)})
    return pf.read_row_groups(chosen)


def sql_literal(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


def setup_sql(tables: dict[str, pa.Table]) -> str:
    """One ``CREATE TEMPORARY VIEW`` per table, with the rows as a ``VALUES`` list (the golden runner's setup)."""
    out = []
    for name, t in sorted(tables.items()):
        cols = t.column_names
        rows = zip(*[c.to_pylist() for c in t.columns], strict=True) if t.num_rows else []
        values = ",\n  ".join("(" + ", ".join(sql_literal(v) for v in r) + ")" for r in rows)
        if not values:
            values = "(" + ", ".join("NULL" for _ in cols) + ")"
            out.append(
                f"CREATE TEMPORARY VIEW {name} AS SELECT * FROM VALUES\n  {values}\n"
                f"AS {name}({', '.join(cols)}) WHERE false;"
            )
        else:
            out.append(
                f"CREATE TEMPORARY VIEW {name} AS SELECT * FROM VALUES\n  {values}\nAS {name}({', '.join(cols)});"
            )
    return "\n".join(out) + "\n"


def write_case(out: Path, sql: str, tables: dict[str, pa.Table], outcome: dict[str, Any], note: str) -> Path:
    """Writes the golden case directory (see the module docstring)."""
    shutil.rmtree(out, ignore_errors=True)
    (out / "data").mkdir(parents=True)
    for n, t in tables.items():
        pq.write_table(t, out / "data" / f"{n}.parquet")
    (out / "query.sql").write_text(sql.rstrip(";") + ";\n")
    (out / "setup.sql").write_text(setup_sql(tables))
    (out / "expected.json").write_text(json.dumps(_jsonable(outcome["reference"]), indent=2))
    (out / "actual.json").write_text(json.dumps(_jsonable(outcome["krishiv"]), indent=2))
    (out / "README.txt").write_text(note + "\n")
    return out


def _jsonable(v: Any) -> Any:
    if isinstance(v, list):
        return [[compare.enc(c) for c in r] for r in v]
    return v


def run_minimize(
    suite: str,
    query: str,
    sf: float,
    dataset: str | None = None,
    reference: str = "duckdb",
    runner: str | None = None,
    seed_rows: int = DEFAULT_SEED_ROWS,
    max_runs: int = DEFAULT_MAX_RUNS,
    full_data: bool = False,
    out_dir: str | None = None,
    whole_rows: int = DEFAULT_WHOLE_ROWS,
    pin: bool = True,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    threads: int = 2,
    spark_remote: str = "sc://127.0.0.1:15012",
) -> Path:
    from .abba import discover_tables
    from .adapters import make_engine
    from .cli import ABBA_DATASET
    from .paths import dataset_dir
    from .runner import load_queries

    name = dataset or ABBA_DATASET[suite]
    root = data_root() / name if suite == "clickbench" else dataset_dir(name, sf)
    names = discover_tables(root)
    if not names:
        raise SystemExit(f"no dataset at {root}")
    (q,) = load_queries(suite, "krishiv", [query])
    used_names = referenced_tables(q.sql, names)
    tables, big = load_tables(root, used_names, whole_rows, seed_rows * ESCALATION[-1])
    kwargs: dict[str, Any] = {"remote": spark_remote} if reference == "spark-connect" else {}
    ref = make_engine(reference, **kwargs)
    krishiv = KrishivEngine(threads=threads, binary=runner)
    krishiv.timeout_s = 600
    work = Path(tempfile.mkdtemp(prefix="minimize-"))
    try:
        oracle = Oracle(ref, krishiv, work / "data", max_runs, max_seconds)
        try:
            if pin:
                oracle.pin(Dataset(name, sf, root, names), q.sql)
            sql, small = minimize(q.sql, tables, oracle, seed_rows, full_data, whole_rows, big)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        outcome = oracle.outcome(sql, small)
    finally:
        krishiv.teardown()
        ref.teardown()
        shutil.rmtree(work, ignore_errors=True)
    out = Path(out_dir) if out_dir else data_root() / "minimized" / f"{suite}_{query}"
    note = (
        f"Minimised from {suite} {query} against {reference} in {oracle.runs} runs"
        f"{' (budget exhausted: not fully minimal)' if not oracle.within_budget() else ''}; "
        f"tables: {', '.join(f'{n} ({t.num_rows} rows)' for n, t in sorted(small.items()))}."
    )
    write_case(out, sql, small, outcome, note)
    print(f"minimal query ({len(sql)} chars):\n{sql}\n{note}\ncase: {out}")
    return out


__all__ = ["Oracle", "candidates", "ddmin", "minimize", "run_minimize", "shrink_query", "shrink_rows"]
