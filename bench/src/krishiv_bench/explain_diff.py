"""``krishiv-bench explain-diff``: Krishiv's executed plan beside DuckDB's, with the differences that explain a gap.

Per query, both engines run once with their operator profiles:

* Krishiv: the runner's ``PLAN`` command, the ``krishiv.query-profile`` v1 document (physical plan tree, rows in and
  out and compute time per operator; ``docs/reference/query-profile-v1.schema.json``);
* DuckDB: ``EXPLAIN (ANALYZE, FORMAT json)`` (rows, estimated cardinality and time per operator).

Both trees are normalised to :class:`PNode` (kind, base tables underneath, actual rows, seconds) and compared:

* **join order**: a join is named by the multiset of base tables under it. Join sets one engine forms and the other
  does not are a different join order; for the joins both form, very different row counts mean different predicates
  were applied, and a build side much larger than DuckDB's is a build-side choice to look at;
* **scans**: a table scan that returns far more rows than DuckDB's scan of the same table did not get a filter or a
  runtime filter into the scan;
* **row blow-ups**: an intermediate far larger than anything DuckDB materialises (an ``Expand``, an exploding join);
* **repeated subtrees**: the same subplan (a subquery or a CTE) computed more than once.

Estimated rows: DuckDB's estimates are kept in the trees. Krishiv's profile carries no per-operator estimates, so
estimate errors on our side cannot be flagged yet; the report says so instead of guessing.

Operator seconds are summed over threads in both engines (CPU time per operator, not wall time). The Krishiv run has
detailed metrics on: its seconds are not a benchmark timing (use ``abba``).
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A scan returning more than this many times DuckDB's rows (and at least MIN_EXCESS more) is flagged.
SCAN_RATIO = 2.0
MIN_EXCESS = 10_000
# A matched join whose rows differ by this factor (and both above MIN_JOIN_ROWS) is flagged.
JOIN_RATIO = 2.0
MIN_JOIN_ROWS = 1_000
# A Krishiv intermediate this many times larger than DuckDB's largest (and at least MIN_BLOWUP rows) is flagged.
BLOWUP_RATIO = 2.0
MIN_BLOWUP = 100_000


@dataclass
class PNode:
    """One operator of an executed plan, engine-neutral."""

    engine: str
    name: str
    kind: str
    detail: str
    rows: int | None
    seconds: float
    tables: tuple[str, ...] = ()
    children: list[PNode] = field(default_factory=list)
    est: float | None = None
    build_rows: int | None = None
    rows_in: int | None = None
    signature: str = ""

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


# ----------------------------------------------------------------------------------------------- Krishiv


def _krishiv_kind(name: str) -> str:
    n = name.lower()
    for key, kind in (
        ("scan", "scan"),
        ("join", "join"),
        ("aggregate", "aggregate"),
        ("topn", "topn"),
        ("sort", "sort"),
        ("filter", "filter"),
        ("exchange", "exchange"),
        ("project", "project"),
        ("expand", "expand"),
        ("window", "window"),
        ("union", "union"),
        ("limit", "limit"),
    ):
        if key in n:
            return kind
    return "other"


def _krishiv_table(detail: str) -> str:
    """``spark_catalog.default.inventory [cols], ...`` -> ``inventory``."""
    m = re.match(r"\s*([^\s\[,]+)", detail)
    return m.group(1).split(".")[-1].strip("`").lower() if m else "?"


def _strip_ids(text: str) -> str:
    return re.sub(r"#\d+", "", text)


def krishiv_tree(doc: dict[str, Any]) -> PNode:
    """The plan of a ``krishiv.query-profile`` v1 document as a :class:`PNode` tree."""
    if doc.get("format") != "krishiv.query-profile" or doc.get("version") != 1:
        raise ValueError("not a krishiv.query-profile v1 document")
    plan = doc["plan"]  # a flat pre-order list; `children` are positions in it, `id` is null for a heading node
    ops: dict[Any, list[dict]] = collections.defaultdict(list)
    for o in sorted(doc["operators"], key=lambda o: o["id"]):
        ops[o.get("node")].append(o)

    def build(i: int) -> PNode:
        n = plan[i]
        kids = [build(c) for c in n["children"]]
        mine = ops.get(n["id"], []) if n["id"] is not None else []
        out = [o for o in mine if o["role"] != "sink"]
        sinks = [o for o in mine if o["role"] == "sink"]
        if out:
            rows: int | None = out[-1]["metrics"]["rows_out"]
        elif sinks:
            rows = sinks[-1]["metrics"]["rows_out"]
        else:  # an exchange: what passes through it
            rows = kids[0].rows if len(kids) == 1 else None
        kind = _krishiv_kind(n["name"])
        tables = (_krishiv_table(n["detail"]),) if kind == "scan" else tuple(sorted(t for k in kids for t in k.tables))
        p = PNode(
            engine="krishiv",
            name=n["name"],
            kind=kind,
            detail=n["detail"],
            rows=rows,
            seconds=sum(o["metrics"]["compute_time_ns"] for o in mine) / 1e9,
            tables=tables,
            children=kids,
            build_rows=sinks[0]["metrics"]["rows_in"] if kind == "join" and sinks else None,
            rows_in=mine[0]["metrics"]["rows_in"] if mine else None,
        )
        p.signature = f"{n['name']}({_strip_ids(n['detail'])})[{','.join(k.signature for k in kids)}]"
        return p

    child_pos = {c for n in plan for c in n["children"]}
    roots = [i for i in range(len(plan)) if i not in child_pos]
    trees = [build(r) for r in roots]
    if len(trees) == 1:
        return trees[0]
    return PNode("krishiv", "Plan", "other", "", trees[0].rows, 0.0, (), trees)


# ------------------------------------------------------------------------------------------------ DuckDB

_DUCK_WRAPPERS = {"RESULT_COLLECTOR", "EXPLAIN_ANALYZE", "EXPLAIN"}


def _duck_kind(t: str) -> str:
    if t == "CTE_SCAN":
        return "cte_scan"
    if t == "CTE":
        return "cte"
    if t.endswith("_SCAN") and t not in ("DELIM_SCAN", "DUMMY_SCAN", "COLUMN_DATA_SCAN", "CHUNK_SCAN"):
        return "scan"
    if "JOIN" in t or t == "CROSS_PRODUCT":
        return "join"
    if "GROUP_BY" in t or "AGGREGATE" in t:
        return "aggregate"
    return {
        "FILTER": "filter",
        "ORDER_BY": "sort",
        "TOP_N": "topn",
        "PROJECTION": "project",
        "WINDOW": "window",
        "STREAMING_WINDOW": "window",
        "UNION": "union",
        "LIMIT": "limit",
        "STREAMING_LIMIT": "limit",
    }.get(t, "other")


def _duck_table(extra: dict[str, Any]) -> str:
    if extra.get("Table"):
        return str(extra["Table"]).lower()
    files = extra.get("Filename(s)") or extra.get("File Filters") or ""
    first = files[0] if isinstance(files, list) and files else str(files)
    first = first.split(",")[0].strip()
    if not first:
        return "?"
    p = Path(first)
    stem = p.stem if p.suffix == ".parquet" and "*" not in p.name else p.parent.name
    return stem.removesuffix(".parquet").lower()


def duckdb_tree(doc: dict[str, Any]) -> PNode:
    """The plan of DuckDB's ``EXPLAIN (ANALYZE, FORMAT json)`` document as a :class:`PNode` tree."""
    ops = doc.get("operator")
    if ops is None and "children" in doc:  # older documents: the root holds the operators
        ops = doc["children"]
    if not ops:
        raise ValueError("not a DuckDB EXPLAIN ANALYZE json document")
    root = ops[0]
    while root.get("type") in _DUCK_WRAPPERS and len(root.get("children", [])) == 1:
        root = root["children"][0]
    ctes: dict[str, PNode] = {}

    def build(n: dict[str, Any]) -> PNode:
        kids = [build(c) for c in n.get("children", [])]
        extra = dict(n.get("extra_info") or {})
        est_text = extra.pop("Estimated Cardinality", None)
        t = n.get("type") or n.get("operator_type") or "?"
        kind = _duck_kind(t)
        if kind == "scan":
            tables: tuple[str, ...] = (_duck_table(extra),)
        else:
            tables = tuple(sorted(x for k in kids for x in k.tables))
        p = PNode(
            engine="duckdb",
            name=t,
            kind=kind,
            detail=json.dumps(extra, sort_keys=True),
            rows=n.get("intermediate_rows", n.get("operator_cardinality")),
            seconds=float(n.get("timing", n.get("operator_timing", 0.0)) or 0.0),
            tables=tables,
            children=kids,
            est=float(est_text) if est_text not in (None, "") else None,
            build_rows=kids[1].rows if kind == "join" and len(kids) > 1 else None,
        )
        if kind == "cte" and kids:
            ctes[str(extra.get("Table Index", extra.get("CTE Name", "")))] = kids[0]
        if kind == "cte_scan":
            p.detail = json.dumps({k: v for k, v in extra.items() if k != "CTE Index"}, sort_keys=True)
            p.signature = f"CTE_SCAN#{extra.get('CTE Index', '?')}"
            p.tables = ("cte:" + str(extra.get("CTE Index", "?")),)
        else:
            sig_extra = {k: v for k, v in extra.items() if k not in ("Filename(s)", "Total Files Read")}
            p.signature = f"{t}({json.dumps(sig_extra, sort_keys=True)})[{','.join(k.signature for k in kids)}]"
        return p

    tree = build(root)
    # A CTE scan reads its CTE's base tables, so joins over CTE results compare by base tables. CTEs nest, so
    # resolve to a fixed point (each round resolves one more level; a reference to an unknown CTE stays as is).
    refs = {id(n): n.tables[0].removeprefix("cte:") for n in tree.walk() if n.kind == "cte_scan"}
    for _ in range(len(ctes) + 1):
        before = [n.tables for n in tree.walk()]
        for node in _post_order(tree):
            if node.kind == "cte_scan":
                idx = refs[id(node)]
                if idx in ctes:
                    node.tables = ctes[idx].tables
            elif node.kind == "cte" and node.children:
                node.tables = node.children[-1].tables  # its output is the consumer's
            elif node.kind != "scan" and node.children:
                node.tables = tuple(sorted(x for k in node.children for x in k.tables))
        if before == [n.tables for n in tree.walk()]:
            break
    return tree


def _post_order(n: PNode):
    for c in n.children:
        yield from _post_order(c)
    yield n


# ------------------------------------------------------------------------------------------------- diff


@dataclass
class Diff:
    """What differs between the two executed plans."""

    flags: list[str]
    joins: list[dict[str, Any]]
    scans: list[dict[str, Any]]
    top_time: dict[str, list[tuple[str, float, int | None]]]


def _ratio(a: int | None, b: int | None) -> float:
    if a is None or b is None:
        return 1.0
    return (a + 1) / (b + 1)


def _tables(ts: tuple[str, ...]) -> str:
    c = collections.Counter(ts)
    return "{" + ", ".join(f"{t}x{n}" if n > 1 else t for t, n in sorted(c.items())) + "}"


def _joins(tree: PNode) -> dict[tuple[str, ...], list[PNode]]:
    out: dict[tuple[str, ...], list[PNode]] = collections.defaultdict(list)
    for n in tree.walk():
        if n.kind == "join":
            out[n.tables].append(n)
    return out


def _scan_above(tree: PNode) -> dict[int, PNode]:
    """id(scan) -> the Filter directly above it (through projections), if any."""
    above: dict[int, PNode] = {}

    def visit(n: PNode, filt: PNode | None) -> None:
        for c in n.children:
            if n.kind == "filter":
                visit(c, n)
            elif n.kind == "project":
                visit(c, filt)
            else:
                visit(c, None)
        if n.kind == "scan" and filt is not None:
            above[id(n)] = filt

    visit(tree, None)
    return above


def _repeated(tree: PNode) -> list[tuple[PNode, int]]:
    """Maximal subtrees (with a scan, at least two operators) whose signature occurs more than once."""
    count = collections.Counter(n.signature for n in tree.walk())
    seen: set[str] = set()
    out: list[tuple[PNode, int]] = []

    def visit(n: PNode, parent_repeated: bool) -> None:
        rep = count[n.signature] > 1 and bool(n.children) and any(d.kind == "scan" for d in n.walk())
        if rep and not parent_repeated and n.signature not in seen:
            seen.add(n.signature)
            out.append((n, count[n.signature]))
        for c in n.children:
            visit(c, rep or parent_repeated)

    visit(tree, False)
    return out


def diff(k: PNode, d: PNode) -> Diff:
    """Compares Krishiv's tree ``k`` with DuckDB's ``d``."""
    flags: list[str] = []

    # Joins, keyed by their base tables.
    kj, dj = _joins(k), _joins(d)
    joins: list[dict[str, Any]] = []
    for key in sorted(set(kj) | set(dj), key=lambda t: (len(t), t)):
        kn, dn = kj.get(key, []), dj.get(key, [])
        row = {
            "tables": _tables(key),
            "krishiv_rows": [n.rows for n in kn],
            "duckdb_rows": [n.rows for n in dn],
            "krishiv_build": [n.build_rows for n in kn],
            "duckdb_build": [n.build_rows for n in dn],
            "krishiv_build_side": [_tables(n.children[1].tables) if len(n.children) > 1 else "" for n in kn],
            "duckdb_build_side": [_tables(n.children[1].tables) if len(n.children) > 1 else "" for n in dn],
        }
        joins.append(row)
        if kn and not dn:
            flags.append(
                f"join order: Krishiv joins {_tables(key)} ({_fmt(kn[0].rows)} rows); DuckDB never forms this "
                "intermediate"
            )
        elif dn and not kn:
            flags.append(f"join order: DuckDB joins {_tables(key)} ({_fmt(dn[0].rows)} rows); Krishiv does not")
        else:
            for a, b in zip(kn, dn, strict=False):
                big = max(a.rows or 0, b.rows or 0) >= MIN_JOIN_ROWS
                r = _ratio(a.rows, b.rows)
                if big and (r >= JOIN_RATIO or r <= 1 / JOIN_RATIO):
                    flags.append(
                        f"join {_tables(key)}: Krishiv {_fmt(a.rows)} rows vs DuckDB {_fmt(b.rows)} "
                        "(different predicates applied at this join)"
                    )
                if (
                    a.build_rows is not None
                    and b.build_rows is not None
                    and a.build_rows >= MIN_JOIN_ROWS
                    and _ratio(a.build_rows, b.build_rows) >= JOIN_RATIO
                ):
                    side_k = _tables(a.children[1].tables) if len(a.children) > 1 else "?"
                    side_d = _tables(b.children[1].tables) if len(b.children) > 1 else "?"
                    flags.append(
                        f"build side of {_tables(key)}: Krishiv builds {side_k} ({_fmt(a.build_rows)} rows), "
                        f"DuckDB builds {side_d} ({_fmt(b.build_rows)} rows)"
                    )

    # Scans, paired per table by row count (largest with largest).
    ks: dict[str, list[PNode]] = collections.defaultdict(list)
    ds: dict[str, list[PNode]] = collections.defaultdict(list)
    for n in k.walk():
        if n.kind == "scan":
            ks[n.tables[0]].append(n)
    for n in d.walk():
        if n.kind == "scan":
            ds[n.tables[0]].append(n)
    above = _scan_above(k)
    scans: list[dict[str, Any]] = []
    for t in sorted(set(ks) | set(ds)):
        a = sorted(ks.get(t, []), key=lambda n: -(n.rows or 0))
        b = sorted(ds.get(t, []), key=lambda n: -(n.rows or 0))
        scans.append({"table": t, "krishiv_rows": [n.rows for n in a], "duckdb_rows": [n.rows for n in b]})
        if len(a) > len(b):
            flags.append(f"table {t} is scanned {len(a)} times by Krishiv, {len(b)} by DuckDB")
        for x, y in zip(a, b, strict=False):
            if (x.rows or 0) - (y.rows or 0) >= MIN_EXCESS and _ratio(x.rows, y.rows) >= SCAN_RATIO:
                f = above.get(id(x))
                after = (
                    f" ({_fmt(f.rows)} after the Filter above it)"
                    if f is not None
                    else " (no Filter above it: a predicate or runtime filter DuckDB applies is missing)"
                )
                flags.append(
                    f"scan of {t}: Krishiv reads out {_fmt(x.rows)} rows, DuckDB {_fmt(y.rows)}{after}: "
                    "filter not pushed into the scan"
                )

    # Row blow-ups.
    dmax = max((n.rows or 0 for n in d.walk()), default=0)
    for n in k.walk():
        widest_input = max((c.rows or 0 for c in n.children), default=0)
        grows = n.kind != "scan" and (n.rows or 0) > widest_input  # where rows grow (scans have their own flag)
        if grows and (n.rows or 0) >= MIN_BLOWUP and (n.rows or 0) >= BLOWUP_RATIO * max(dmax, 1):
            src = f" from {_fmt(n.rows_in)} input rows" if n.rows_in and n.rows_in != n.rows else ""
            flags.append(
                f"row blow-up: Krishiv {n.name} over {_tables(n.tables)} outputs {_fmt(n.rows)} rows{src}; "
                f"DuckDB's largest intermediate is {_fmt(dmax)}"
            )

    # Repeated subtrees.
    for engine, tree in (("Krishiv", k), ("DuckDB", d)):
        for n, times in _repeated(tree):
            flags.append(
                f"repeated subtree: {engine} computes {n.name} over {_tables(n.tables)} {times} times "
                f"({_fmt(n.rows)} rows each)"
            )

    counted = collections.Counter(flags)
    flags = [f if counted[f] == 1 else f"{f} (x{counted[f]})" for f in dict.fromkeys(flags)]
    top = {
        eng: [
            (n.name + " " + _tables(n.tables), n.seconds, n.rows)
            for n in sorted(t.walk(), key=lambda n: -n.seconds)[:6]
        ]
        for eng, t in (("krishiv", k), ("duckdb", d))
    }
    return Diff(flags, joins, scans, top)


def _fmt(v: float | int | None) -> str:
    if v is None:
        return "?"
    return f"{v:,.0f}"


# ----------------------------------------------------------------------------------------------- report


def render_tree(n: PNode, depth: int = 0) -> list[str]:
    est = f" est {_fmt(n.est)}" if n.est is not None else ""
    build = f" build {_fmt(n.build_rows)}" if n.build_rows is not None else ""
    tables = f" {_tables(n.tables)}" if n.kind in ("scan", "join") else ""
    detail = _strip_ids(n.detail)
    detail = (detail[:90] + "…") if len(detail) > 90 else detail
    line = f"{'  ' * depth}{n.name}{tables} rows {_fmt(n.rows)}{est}{build} {n.seconds:.3f}s  {detail}".rstrip()
    out = [line]
    for c in n.children:
        out += render_tree(c, depth + 1)
    return out


def markdown(qid: str, k: PNode, d: PNode, res: Diff, k_seconds: float, d_seconds: float) -> str:
    lines = [
        f"## {qid}",
        "",
        f"Wall: Krishiv {k_seconds:.3f} s (profiled run, detailed metrics on), DuckDB {d_seconds:.3f} s (EXPLAIN "
        "ANALYZE). Operator seconds are summed over threads. Krishiv's profile has no per-operator estimates.",
        "",
        "**Flags**",
        "",
        *([f"- {f}" for f in res.flags] or ["- none"]),
        "",
        "**Joins** (by the base tables under them)",
        "",
        "| tables | Krishiv rows | DuckDB rows | Krishiv build | DuckDB build |",
        "|---|---:|---:|---|---|",
    ]
    for j in res.joins:
        kb = ", ".join(f"{s} {_fmt(r)}" for s, r in zip(j["krishiv_build_side"], j["krishiv_build"], strict=False))
        db = ", ".join(f"{s} {_fmt(r)}" for s, r in zip(j["duckdb_build_side"], j["duckdb_build"], strict=False))
        lines.append(
            f"| {j['tables']} | {', '.join(map(_fmt, j['krishiv_rows'])) or '-'} | "
            f"{', '.join(map(_fmt, j['duckdb_rows'])) or '-'} | {kb or '-'} | {db or '-'} |"
        )
    lines += ["", "**Scans**", "", "| table | Krishiv rows out | DuckDB rows out |", "|---|---:|---:|"]
    for s in res.scans:
        lines.append(
            f"| {s['table']} | {', '.join(map(_fmt, s['krishiv_rows'])) or '-'} | "
            f"{', '.join(map(_fmt, s['duckdb_rows'])) or '-'} |"
        )
    lines += ["", "**Most operator time**", ""]
    for eng, items in res.top_time.items():
        lines.append(f"- {eng}: " + "; ".join(f"{name} {secs:.3f}s ({_fmt(rows)} rows)" for name, secs, rows in items))
    lines += ["", "<details><summary>Krishiv plan</summary>", "", "```", *render_tree(k), "```", "", "</details>"]
    lines += ["", "<details><summary>DuckDB plan</summary>", "", "```", *render_tree(d), "```", "", "</details>", ""]
    return "\n".join(lines)


def run_explain_diff(
    suite: str,
    sf: float,
    queries: list[str],
    dataset: str | None = None,
    threads: int = 8,
    memory_limit: str = "12g",
    runner: str | None = None,
    out_dir: Path | None = None,
    timeout_s: float = 600,
) -> Path:
    """Profiles ``queries`` on both engines and writes one Markdown report (and the raw profiles as JSON).

    A query that fails (an unsupported feature, a crash, a timeout, an unreadable profile) is recorded in the
    report and the run goes on. ``timeout_s`` bounds each Krishiv query (0 disables it).
    """
    import time

    import duckdb

    from .abba import bench_lock, discover_tables
    from .adapters.base import Dataset
    from .adapters.duckdb_engine import DuckDBEngine
    from .adapters.krishiv import KrishivEngine, RunnerCrashed, RunnerError
    from .paths import data_root, dataset_dir, results_dir
    from .runner import SUITE_DATASET, load_queries

    ds_name = dataset or {"tpch": "tpch-duckdb", "clickbench": "clickbench"}.get(suite, SUITE_DATASET.get(suite, suite))
    root = data_root() / ds_name if suite == "clickbench" else dataset_dir(ds_name, sf)
    tables = discover_tables(root)
    if not tables:
        raise SystemExit(f"no dataset at {root}")
    ds = Dataset(ds_name, sf, root, tables)
    kq = {q.id: q for q in load_queries(suite, "krishiv", queries)}
    dq = {q.id: q for q in load_queries(suite, "duckdb", queries)}
    missing = [q for q in queries if q not in kq]
    if missing:
        raise SystemExit(f"unknown queries for {suite}: {', '.join(missing)}")
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_dir or results_dir() / "explain-diff"
    out_dir.mkdir(parents=True, exist_ok=True)
    # A full suite run would put every query id in the name and pass the filesystem's limit.
    tag = "-".join(queries) if len(queries) <= 5 else f"{queries[0]}-to-{queries[-1]}-{len(queries)}q"
    report = out_dir / f"{stamp}_{suite}_sf{sf:g}_{tag}.md"
    raw: dict[str, Any] = {}
    parts = [f"# explain-diff {suite} SF{sf:g} ({stamp})", ""]
    with bench_lock():
        krishiv = KrishivEngine(threads=threads, memory_limit=memory_limit.lower(), binary=runner)
        krishiv.timeout_s = timeout_s or None  # 0 disables the timeout
        duck = DuckDBEngine(threads=threads, memory_limit=memory_limit.upper())
        try:
            duck.setup(ds)
            krishiv.setup(ds)
            for qid in queries:
                try:
                    kp = krishiv.plan_profile(kq[qid])
                    assert duck.con is not None
                    t0 = time.perf_counter()
                    row = duck.con.execute(
                        "EXPLAIN (ANALYZE, FORMAT json) " + dq[qid].sql.rstrip().rstrip(";")
                    ).fetchone()
                    d_secs = time.perf_counter() - t0
                    assert row is not None
                    ddoc = json.loads(row[1])
                    k_tree, d_tree = krishiv_tree(kp["profile"]), duckdb_tree(ddoc)
                    parts.append(markdown(qid, k_tree, d_tree, diff(k_tree, d_tree), kp["seconds"], d_secs))
                    raw[qid] = {"krishiv": kp, "duckdb": ddoc}
                except (RunnerError, RunnerCrashed, ValueError, KeyError, duckdb.Error) as e:
                    msg = f"{type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ''}"
                    parts += [f"## {qid}", "", f"**failed** - {msg}", ""]
                    raw[qid] = {"error": msg}
        finally:
            krishiv.teardown()
            duck.teardown()
    report.write_text("\n".join(parts))
    report.with_suffix(".json").write_text(json.dumps(raw))
    print("\n".join(parts))
    print(f"report: {report}")
    return report
