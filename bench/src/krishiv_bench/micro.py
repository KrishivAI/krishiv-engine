"""Operator microbenchmark suite (``micro``, P2-T168): Krishiv, DuckDB and Polars run the same single-operator
queries over the same Parquet files, to tell "our operator is slow" from "our plan is slow".

Data: ``micro_data.py`` (``krishiv-datasets micro --size quick|default|full``) under
``$KRISHIV_DATA/micro/<version>/<size>/``. The cases are generated from the size set recorded in the dataset's
manifest, so the queries always match the data they run on.

A case is one SQL text for every engine, in a family: ``join`` (key type, build size, match rate, Zipf skew),
``groupby`` (the db-benchmark shapes over 1e2..1e8 groups, int and text keys), ``sort``, ``topn``, ``filter``
(selectivities from 0.1 % to 90 %), ``decimal`` (sum and avg), ``string`` (compare and LIKE) and ``scan`` (the
read-only floor to subtract in your head). The queries are shaped so that the operator is most of the work: joins
and filters return a count and a sum, groupbys and sorts return their whole answer (materialised inside the engine).

Method: ``reproduce``'s interleaved ``ours`` method (``reproduce.ours_query``): per case, every engine runs a
checksum query once, untimed (the answer's row count and sums of its columns; for ``topn`` the rows themselves), and
must equal DuckDB's; a mismatch, error or timeout marks the case failed for that engine and it is never timed. Then
``reps`` blocks of A B C C B A give ``2 * reps`` samples per engine, with a bootstrap CI of each ratio to DuckDB,
the process tree's CPU seconds, its peak memory, and, for Krishiv, one ``PROFILE`` run whose per-operator compute
seconds (summed over drivers, so CPU time not wall time) separate the operator from the scan.

Ordering is verified for ``topn`` (rows compared in order); a full ``sort`` is checked for its content (row count and
column sums), not row by row, because a checksum query cannot observe the order of a sorted subquery.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import micro_data
from .abba import LOAD_INDICATIVE, bootstrap_ci, geomean
from .adapters import Query
from .micro_data import (
    MATCH_PCTS_INT,
    MATCH_PCTS_STR,
    ZIPF_ALPHAS,
    MicroSizes,
    build_table,
    col_gi,
    col_gj,
    col_gs,
    col_ji,
    col_js,
    col_jz,
    sci,
)
from .paths import results_dir

SUITE = "micro"
FAMILIES = ("scan", "filter", "string", "sort", "topn", "groupby", "decimal", "join")
MATERIALISED_PER_MILLE = (1, 10, 100, 500, 900)  # share of the rows a materialising filter keeps, per mille
COUNT_PER_MILLE = (10, 500)
DEFAULT_ENGINES = ["krishiv", "duckdb", "polars"]
SCAN_OPERATORS = ("FileScan", "MemoryScan")


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    sql: str
    rows_in: int  # rows the operator consumes (the fact rows; the probe side of a join)
    ops: tuple[str, ...]  # Krishiv operators whose compute seconds are "the operator" of this case
    twin: str  # the criterion bench of krishiv-exec that measures this operator alone, or ``stub: <proposed id>``
    params: dict[str, Any] = field(default_factory=dict)
    # Expressions summed over the answer's columns to check it (None: the answer is small, its rows are compared)
    check: tuple[str, ...] | None = None


def _int(col: str) -> str:
    return f"SUM({col})"


def _float(col: str) -> str:
    return f"SUM(CAST({col} AS DOUBLE))"


def _text_key(col: str) -> str:
    """The number in an ``id%010d`` key, summed: sees every key, exactly."""
    return f"SUM(CAST(SUBSTRING({col}, 3, 10) AS BIGINT))"


def _decimal(col: str) -> str:
    return f"SUM(CAST({col} AS DECIMAL(38,2)))"


def check_sql(sql: str, exprs: Iterable[str]) -> str:
    """The checksum query of a case: the answer's row count and one aggregate per answer column."""
    body = sql.strip().rstrip(";").strip()
    cols = ", ".join(f"{e} AS c{i}" for i, e in enumerate(exprs))
    return f"SELECT COUNT(*) AS n_rows, {cols} FROM ({body}) AS ans"


# ---------------------------------------------------------------- the cases
AGG_OPS = ("HashAggregate", "PartialAggregate")
JOIN_OPS = ("HashJoinBuild", "HashJoinProbe")


def _case(
    p: MicroSizes,
    cid: str,
    family: str,
    sql: str,
    ops: tuple[str, ...],
    twin: str,
    check: tuple[str, ...] | None = None,
    **params: Any,
) -> Case:
    return Case(cid, family, sql, p.rows, ops, twin, params, check)


def _agg_twin(groups: int) -> str:
    return "aggregate/grouped_sum_64_keys" if groups <= 1000 else "aggregate/grouped_sum_262144_keys"


def _scan_cases(p: MicroSizes) -> list[Case]:
    return [
        _case(
            p,
            "scan_int",
            "scan",
            "SELECT SUM(u) AS s FROM fact",
            SCAN_OPERATORS,
            "stub: scan/parquet_int_1m_rows",
            (_int("s"),),
        ),
        _case(
            p,
            "scan_str",
            "scan",
            "SELECT SUM(LENGTH(s)) AS s FROM fact",
            SCAN_OPERATORS,
            "stub: scan/parquet_string_1m_rows",
            (_int("s"),),
        ),
    ]


def _filter_cases(p: MicroSizes) -> list[Case]:
    out = []
    for pm in MATERIALISED_PER_MILLE:
        limit = p.rows * pm // 1000
        twin = "operators/filter_1m_rows/50_percent" if pm == 500 else "stub: operators/filter_1m_rows/selectivity"
        sql = f"SELECT id, v1, v3 FROM fact WHERE u < {limit}"
        checks = (_int("id"), _int("v1"), _float("v3"))
        out.append(
            _case(p, f"filter_rows_pm{pm}", "filter", sql, ("Filter",), twin, checks, per_mille=pm, rows_out=limit)
        )
    for pm in COUNT_PER_MILLE:
        limit = p.rows * pm // 1000
        sql = f"SELECT COUNT(*) AS n, SUM(v1) AS s FROM fact WHERE u < {limit}"
        twin = "stub: operators/filter_count_1m_rows/selectivity"
        out.append(
            _case(
                p,
                f"filter_count_pm{pm}",
                "filter",
                sql,
                ("Filter",),
                twin,
                (_int("n"), _int("s")),
                per_mille=pm,
                rows_out=limit,
            )
        )
    return out


def _string_cases(p: MicroSizes) -> list[Case]:
    shapes = {
        "str_eq": (f"s = '{micro_data.NEEDLE}'", "the needle: 0.1 % of the rows"),
        "str_lt": ("s < 'n'", "a range compare keeping about half"),
        "str_like_prefix": ("s LIKE 'ab%'", "prefix, about 0.15 %"),
        "str_like_suffix": ("s LIKE '%ab'", "suffix, about 0.15 %"),
        "str_like_contains": ("s LIKE '%ab%'", "contains, about 2 %"),
        "str_like_rare": ("s LIKE '%stack0%'", "contains a rare literal: the needle, 0.1 %"),
    }
    twin = "stub: operators/filter_string_1m_rows/predicate"
    return [
        _case(
            p,
            cid,
            "string",
            f"SELECT COUNT(*) AS n FROM fact WHERE {pred}",
            ("Filter",),
            twin,
            (_int("n"),),
            predicate=pred,
            note=note,
        )
        for cid, (pred, note) in shapes.items()
    ]


def _sort_cases(p: MicroSizes) -> list[Case]:
    gi = col_gi(p.groups[0])
    sort = ("Sort",)
    topn = ("TopN",)
    return [
        _case(
            p,
            "sort_int",
            "sort",
            "SELECT u, v1 FROM fact ORDER BY u",
            sort,
            "sort/in_memory_1m_rows/int",
            (_int("u"), _int("v1")),
        ),
        _case(
            p,
            "sort_str",
            "sort",
            "SELECT s, id FROM fact ORDER BY s, id",
            sort,
            "sort/in_memory_1m_rows/string",
            ("SUM(LENGTH(s))", _int("id")),
        ),
        _case(
            p,
            "sort_2key",
            "sort",
            f"SELECT {gi} AS g, u FROM fact ORDER BY {gi}, u",
            sort,
            "stub: sort/in_memory_1m_rows/two_int_keys",
            (_int("g"), _int("u")),
        ),
        _case(
            p,
            "topn_k100_int",
            "topn",
            "SELECT u, v1 FROM fact ORDER BY u LIMIT 100",
            topn,
            "sort/topn_k100_1m_rows",
            k=100,
        ),
        _case(
            p,
            "topn_k1000_float",
            "topn",
            "SELECT id, v3 FROM fact ORDER BY v3 DESC, id LIMIT 1000",
            topn,
            "stub: sort/topn_k1000_float_1m_rows",
            k=1000,
        ),
        _case(
            p,
            "topn_k100_str",
            "topn",
            "SELECT id, s FROM fact ORDER BY s, id LIMIT 100",
            topn,
            "stub: sort/topn_k100_string_1m_rows",
            k=100,
        ),
    ]


def _groupby_cases(p: MicroSizes) -> list[Case]:
    out = []
    for c in p.groups:
        for kind, key, key_check in (("int", col_gi(c), _int), ("str", col_gs(c), _text_key)):
            sum_twin = _agg_twin(c) if kind == "int" else "stub: aggregate/grouped_sum_string_keys"
            by = f"FROM fact GROUP BY {key}"
            out.append(
                _case(
                    p,
                    f"groupby_{kind}_sum_g{sci(c)}",
                    "groupby",
                    f"SELECT {key} AS g, SUM(v1) AS s {by}",
                    AGG_OPS,
                    sum_twin,
                    (key_check("g"), _int("s")),
                    groups=c,
                    key=kind,
                    shape="sum",
                )
            )
            out.append(
                _case(
                    p,
                    f"groupby_{kind}_multi_g{sci(c)}",
                    "groupby",
                    f"SELECT {key} AS g, SUM(v1) AS s, AVG(v3) AS a, COUNT(*) AS n {by}",
                    AGG_OPS,
                    "stub: aggregate/grouped_multi_agg",
                    (key_check("g"), _int("s"), _float("a"), _int("n")),
                    groups=c,
                    key=kind,
                    shape="sum, avg, count",
                )
            )
    for c in p.pairs:
        keys = f"{col_gi(c)}, {col_gj(c)}"
        out.append(
            _case(
                p,
                f"groupby_2key_g{sci(c)}",
                "groupby",
                f"SELECT {col_gi(c)} AS g1, {col_gj(c)} AS g2, SUM(v1) AS s FROM fact GROUP BY {keys}",
                AGG_OPS,
                "stub: aggregate/grouped_sum_two_keys",
                (_int("g1"), _int("g2"), _int("s")),
                groups=c * c,
                key="int, int",
                shape="sum",
            )
        )
    return out


def _decimal_cases(p: MicroSizes) -> list[Case]:
    out = [
        _case(
            p,
            "decimal_global",
            "decimal",
            "SELECT SUM(dec) AS s, AVG(dec) AS a FROM fact",
            AGG_OPS,
            "aggregate/decimal_sum_1_keys",
            (_decimal("s"), _float("a")),
            groups=1,
        )
    ]
    twins = ("aggregate/decimal_sum_64_keys", "aggregate/decimal_sum_262144_keys")
    for c, twin in zip(p.groups[:2], twins, strict=False):
        sql = f"SELECT {col_gi(c)} AS g, SUM(dec) AS s, AVG(dec) AS a FROM fact GROUP BY {col_gi(c)}"
        out.append(
            _case(
                p,
                f"decimal_g{sci(c)}",
                "decimal",
                sql,
                AGG_OPS,
                twin,
                (_int("g"), _decimal("s"), _float("a")),
                groups=c,
            )
        )
    return out


def _join_cases(p: MicroSizes) -> list[Case]:
    out = []
    for b in p.builds:
        table = build_table(b)
        twin = "hash_join/probe_1m_rows" if b <= 10**5 else "hash_join/build_1m_rows"

        def join(cid: str, probe: str, col: str, tw: str, table: str = table, b: int = b, **params: Any) -> Case:
            sql = f"SELECT COUNT(*) AS n, SUM(b.bv) AS sb FROM fact f JOIN {table} b ON f.{probe} = b.{col}"
            return _case(p, cid, "join", sql, JOIN_OPS, tw, (_int("n"), _int("sb")), build=b, **params)

        for pct in MATCH_PCTS_INT:
            out.append(join(f"join_int_b{sci(b)}_m{pct}", col_ji(b, pct), "k", twin, key="int64", match_pct=pct))
        for pct in MATCH_PCTS_STR:
            str_twin = "stub: hash_join/probe_string_keys_1m_rows"
            out.append(join(f"join_str_b{sci(b)}_m{pct}", col_js(b, pct), "ks", str_twin, key="string", match_pct=pct))
        for alpha in ZIPF_ALPHAS:
            cid = f"join_zipf_b{sci(b)}_a{round(alpha * 10):02d}"
            zipf_twin = "stub: hash_join/probe_zipf_1m_rows"
            out.append(join(cid, col_jz(b, alpha), "k", zipf_twin, key="int64", match_pct=100, zipf_alpha=alpha))
    return out


def cases(p: MicroSizes) -> list[Case]:
    """Every case of a size set, family by family (``FAMILIES`` order)."""
    return [
        *_scan_cases(p),
        *_filter_cases(p),
        *_string_cases(p),
        *_sort_cases(p),
        *_groupby_cases(p),
        *_decimal_cases(p),
        *_join_cases(p),
    ]


def select(all_cases: list[Case], only: list[str] | None) -> list[Case]:
    """The cases named by ``only`` (case ids or family names); unknown names are an error, not an empty run."""
    if not only:
        return all_cases
    ids = {c.id for c in all_cases}
    unknown = [t for t in only if t not in ids and t not in FAMILIES]
    if unknown:
        raise SystemExit(f"unknown case or family: {', '.join(unknown)} (families: {', '.join(FAMILIES)})")
    return [c for c in all_cases if c.id in only or c.family in only]


def sizes_of(root: Path) -> MicroSizes:
    man = micro_data.read_manifest(root)
    if man is None:
        raise SystemExit(f"no micro dataset at {root}; generate it with `uv run krishiv-datasets micro --size ...`")
    return micro_data.sizes_from_params(man["params"])


def load_queries(root: Path, engine: str, only: list[str] | None = None) -> list[Query]:
    """The suite's queries for the dataset at ``root``. One SQL text for every engine (``engine`` is accepted for
    symmetry with the other suites: there are no per-engine overrides, the point is the same operation)."""
    del engine
    out = []
    for c in select(cases(sizes_of(root)), only):
        out.append(Query(c.id, c.sql, SUITE, check_sql(c.sql, c.check) if c.check else None))
    return out


def dataset_name(size: str, version: str = micro_data.GENERATOR_VERSION) -> str:
    """The dataset's path relative to ``$KRISHIV_DATA``."""
    return f"micro/{version}/{size}"


def record_extra(root: Path, only: list[str] | None = None) -> dict[str, Any]:
    """What a result needs to be read without the code: the size set, the dataset digest and every case's metadata."""
    man = micro_data.read_manifest(root)
    sizes = sizes_of(root)
    return {
        "micro": {
            "size": sizes.name,
            "sizes": asdict(sizes),
            "dataset_content_sha256": man["content_sha256"] if man else None,
            "generator_version": man["generator_version"] if man else None,
            "cases": {c.id: _case_json(c) for c in select(cases(sizes), only)},
        }
    }


def _case_json(c: Case) -> dict[str, Any]:
    return {
        "family": c.family,
        "rows_in": c.rows_in,
        "operators": list(c.ops),
        "twin": c.twin,
        "params": c.params,
        "sql": c.sql,
    }


# ---------------------------------------------------------------- derived numbers
def operator_seconds(profile: dict | None, ops: Iterable[str]) -> dict[str, float | None]:
    """Compute seconds of Krishiv's operators in a ``PROFILE`` reply: the case's operators, the scan, everything.

    Seconds are summed over the query's drivers (CPU time of the operator, not wall time). ``None`` when the run has
    no per-operator metrics.
    """
    operators = (profile or {}).get("operators")
    if not operators:
        return {"operator": None, "scan": None, "total": None}
    want, scan = set(ops), set(SCAN_OPERATORS)
    return {
        "operator": sum(o["compute_seconds"] for o in operators if o["name"] in want),
        "scan": sum(o["compute_seconds"] for o in operators if o["name"] in scan),
        "total": sum(o["compute_seconds"] for o in operators),
    }


def _engine(record: dict, case: str, engine: str) -> dict:
    return record["queries"].get(case, {}).get("engines", {}).get(engine, {})


def _ratio(a: dict, b: dict) -> tuple[float, tuple[float, float]] | None:
    """Median ratio of two engines' samples with its bootstrap CI, when both have samples."""
    if "samples_s" not in a or "samples_s" not in b:
        return None
    return statistics.median(a["samples_s"]) / statistics.median(b["samples_s"]), bootstrap_ci(
        a["samples_s"], b["samples_s"]
    )


# ---------------------------------------------------------------- report
def _ms(e: dict) -> str:
    return f"{e['median_s'] * 1000:,.1f}" if "median_s" in e else str(e.get("status", "-")).upper()


def _x(r: tuple[float, tuple[float, float]] | None) -> str:
    return "-" if r is None else f"{r[0]:.2f}x ({r[1][0]:.2f}-{r[1][1]:.2f})"


def _mrows(rows: int, e: dict) -> str:
    return f"{rows / e['median_s'] / 1e6:,.1f}" if e.get("median_s") else "-"


def _cpu(e: dict) -> str:
    return f"{e['cpu_seconds_median']:.2f}" if e.get("cpu_seconds_median") is not None else "-"


def render(record: dict, engines: list[str] | None = None) -> str:
    """The parity table (case -> Krishiv / DuckDB / Polars), per-engine throughput and CPU, Krishiv's operator time."""
    names = engines or list(record["engines"])
    m = record["micro"]
    cs: dict[str, dict] = m["cases"]
    others = [e for e in names if e != "duckdb"]
    lines = [
        f"# Operator microbenchmarks: size `{m['size']}` ({m['sizes']['rows']:,} fact rows)",
        "",
        f"- Dataset `{record['dataset']['name']}`, content sha256 `{(m.get('dataset_content_sha256') or 'n/a')[:16]}`; "
        f"{record['pinned']['threads']} threads, {record['ours']['samples_per_engine']} interleaved samples per engine "
        "(median shown). Every engine's answer checksum must equal DuckDB's before a case is timed.",
        "- Ratios are engine / duckdb (above 1 = slower), with a bootstrap 95 % CI; `k/p` is krishiv / polars.",
        "",
        "## Parity (wall time, ms)",
        "",
        "| case | "
        + " | ".join(f"{e} ms" for e in names)
        + " | "
        + " | ".join(f"{e}/duckdb" for e in others)
        + " | k/p |",
        "|---|" + "---:|" * (len(names) + len(others) + 1),
    ]
    for cid in cs:
        per = {e: _engine(record, cid, e) for e in names}
        row = [_ms(per[e]) for e in names]
        row += [_x(_ratio(per[e], per.get("duckdb", {}))) if e != "duckdb" else "" for e in names if e != "duckdb"]
        kp = _ratio(per.get("krishiv", {}), per.get("polars", {}))
        lines.append(f"| {cid} | " + " | ".join(row) + f" | {'-' if kp is None else f'{kp[0]:.2f}x'} |")
    lines += ["", "## Per family: geomean of engine / duckdb", "", "| family | cases | " + " | ".join(others) + " |"]
    lines.append("|---|---:|" + "---:|" * len(others))
    for fam in FAMILIES:
        ids = [c for c, d in cs.items() if d["family"] == fam]
        if not ids:
            continue
        cells = []
        for e in others:
            rs = [r[0] for c in ids if (r := _ratio(_engine(record, c, e), _engine(record, c, "duckdb")))]
            cells.append(f"{geomean(rs):.2f}x (n={len(rs)})" if rs else "-")
        lines.append(f"| {fam} | {len(ids)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Throughput (million input rows per second) and CPU seconds",
        "",
        "| case | input rows | "
        + " | ".join(f"{e} Mrows/s" for e in names)
        + " | "
        + " | ".join(f"{e} CPU s" for e in names)
        + " |",
        "|---|---:|" + "---:|" * (2 * len(names)),
    ]
    for cid, d in cs.items():
        per = {e: _engine(record, cid, e) for e in names}
        lines.append(
            f"| {cid} | {d['rows_in']:,} | "
            + " | ".join(_mrows(d["rows_in"], per[e]) for e in names)
            + " | "
            + " | ".join(_cpu(per[e]) for e in names)
            + " |"
        )
    lines += [
        "",
        "## Krishiv: the operator against the scan (`PROFILE`; compute seconds summed over drivers)",
        "",
        "| case | operators | operator CPU s | scan CPU s | all operators CPU s | operator share | wall ms | twin |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for cid, d in cs.items():
        k = _engine(record, cid, "krishiv")
        sec = operator_seconds(k.get("profile"), d["operators"])
        if sec["operator"] is None or not sec["total"]:
            continue
        lines.append(
            f"| {cid} | {', '.join(d['operators'])} | {sec['operator']:.3f} | {sec['scan']:.3f} | {sec['total']:.3f} "
            f"| {sec['operator'] / sec['total']:.0%} | {_ms(k)} | {d['twin']} |"
        )
    lines += ["", "## Correctness", ""]
    bad = [
        f"- **{e}**: not run, setup failed: {f['reason'][:200]}" for e, f in record.get("setup_failures", {}).items()
    ]
    bad += [
        f"- `{cid}` **{e}**: {str(_engine(record, cid, e).get('reason', _engine(record, cid, e).get('status')))[:200]}"
        for cid in cs
        for e in names
        if _engine(record, cid, e).get("status") != "ok"
    ]
    lines += (
        ["Cases that failed the checksum comparison, errored or timed out (never timed for that engine):", "", *bad]
        if bad
        else [f"All {len(cs)} cases: every engine's answer checksum equals the DuckDB reference."]
    )
    s = record["summary"]
    tag = "INDICATIVE ONLY" if s["indicative"] else "within bound"
    lines += [
        "",
        f"Load at start {record['load_at_start']:.1f} (bound {LOAD_INDICATIVE:g}): {tag}; max 1-min load during the "
        f"run {s['max_load']:.1f} (includes the benchmark itself).",
    ]
    return "\n".join(lines) + "\n"


def latest_result(size: str | None = None) -> Path:
    """The newest ``reproduce`` result of the micro suite (of one size set when given)."""
    found = []
    for f in sorted(results_dir().glob("*_reproduce_micro_*.json")):
        rec = json.loads(f.read_text())
        if rec.get("suite") == SUITE and (size is None or rec.get("micro", {}).get("size") == size):
            found.append(f)
    if not found:
        raise SystemExit(f"no micro result under {results_dir()}; run `krishiv-bench micro` first")
    return found[-1]


def write_report(files: list[Path] | None = None, size: str | None = None) -> Path:
    """Writes ``<result>.md`` next to the newest (or the given) result JSON and returns its path."""
    src = files[0] if files else latest_result(size)
    out = src.with_suffix(".md")
    out.write_text(render(json.loads(src.read_text())))
    return out


def failed_cases(record: dict) -> list[tuple[str, str]]:
    """(case, engine) pairs whose verdict is not ok; an engine that failed to start fails every case (``*``)."""
    failed = [("*", e) for e in record.get("setup_failures", {})]
    return failed + [
        (cid, e)
        for cid in record["micro"]["cases"]
        for e in record["engines"]
        if _engine(record, cid, e).get("status") != "ok"
    ]


__all__ = [
    "DEFAULT_ENGINES",
    "FAMILIES",
    "Case",
    "cases",
    "dataset_name",
    "failed_cases",
    "load_queries",
    "operator_seconds",
    "record_extra",
    "render",
    "write_report",
]
