"""``krishiv-bench join-trees``: how each join enumerator plans every inner-join tree of a suite (P2-T113).

Per *arm* (a set of session settings, e.g. ``krishiv.optimizer.joinEnumerator=dphyp``) one runner is started over the
dataset, so the optimizer reads the dataset's real statistics. Per query:

* ``EXPLAIN VERBOSE`` once: the optimizer states, per inner-join tree it planned, its inputs, edges and hyperedges, the
  enumerator, why it fell back to greedy (if it did), the csg-cmp pairs costed and the candidates looked at, the
  estimated ``C_out`` of the plan and its shape (the plan fingerprint);
* ``EXPLAIN`` ``--plan-reps`` times: the runner's seconds for parsing, analysis, optimisation and physical planning.
  The arms differ only in join enumeration, so their difference is the enumerator's planning time, and the total is an
  upper bound on it.

Nothing is executed; ``ab`` times the queries whose plans differ. The record goes to ``results/`` and a summary
(trees by size, fallbacks, plans that differ between arms, planning-time percentiles) is printed.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import statistics
from pathlib import Path
from typing import Any

from .abba import discover_tables, table_paths
from .adapters.base import Dataset, Query
from .adapters.krishiv import KrishivEngine, RunnerCrashed, RunnerError
from .paths import dataset_dir, fmt_sf, results_dir
from .runner import git_info, load_queries, machine_info

# An arm is NAME=K=V[,K=V...]; these are the experiment brief's (T113 section 3, Fable review arm A').
DEFAULT_ARMS = [
    "dpsub=krishiv.optimizer.joinEnumerator=dpsub",
    "dphyp=krishiv.optimizer.joinEnumerator=dphyp",
    "dpsub12=krishiv.optimizer.joinEnumerator=dpsub,krishiv.optimizer.joinDpLimit=12",
    "dpsub13=krishiv.optimizer.joinEnumerator=dpsub,krishiv.optimizer.joinDpLimit=13",
]
TREE = re.compile(r"JoinOrder: (tree inputs=\S+ .*)$")
INTS = ("inputs", "edges", "hyperedges", "pairs", "considered")


def parse_arm(text: str) -> tuple[str, list[tuple[str, str]]]:
    """``NAME=K=V,K=V`` -> (NAME, [(K, V), ...])."""
    name, sep, rest = text.partition("=")
    if not sep or not name:
        raise SystemExit(f"bad arm {text!r}: expected NAME=KEY=VALUE[,KEY=VALUE...]")
    conf = []
    for item in filter(None, rest.split(",")):
        k, eq, v = item.partition("=")
        if not eq:
            raise SystemExit(f"bad arm setting {item!r} in {text!r}")
        conf.append((k.strip(), v.strip()))
    return name, conf


def parse_tree(note: str) -> dict[str, Any]:
    """One ``tree ...`` note of ``EXPLAIN VERBOSE`` as a dict (``shape`` is the rest of the line)."""
    head, _, shape = note.partition(" shape=")
    out: dict[str, Any] = {}
    for field in head.split()[1:]:
        k, _, v = field.partition("=")
        out[k] = int(v) if k in INTS else float(v) if k == "cost" else v
    out["shape"] = shape
    return out


def trees_of(explain_text: str) -> list[dict[str, Any]]:
    """The trees the last join ordering of the plan described, in plan order.

    The optimizer orders joins again after eager aggregation when that moved an aggregate; each ordering describes all
    trees, so the last block is the plan's own.
    """
    blocks: list[list[dict[str, Any]]] = []
    previous_was_tree = False
    for line in explain_text.splitlines():
        m = TREE.search(line)
        if m:
            if not previous_was_tree:
                blocks.append([])
            blocks[-1].append(parse_tree(m.group(1)))
        previous_was_tree = bool(m)
    return blocks[-1] if blocks else []


def percentile(xs: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..1)."""
    s = sorted(xs)
    if not s:
        return float("nan")
    rank = max(1, min(len(s), int(-(-p * len(s) // 1))))
    return s[rank - 1]


def plan_arm(eng: KrishivEngine, queries: list[Query], reps: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for q in queries:
        entry: dict[str, Any] = {}
        try:
            rows = eng.run(Query(q.id, f"EXPLAIN VERBOSE {q.sql.strip().rstrip(';')}", q.suite))
            text = "\n".join(str(c) for r in rows for c in r)
            entry["trees"] = trees_of(text)
            secs = []
            for _ in range(reps):
                _, s = eng.timed_run(Query(q.id, f"EXPLAIN {q.sql.strip().rstrip(';')}", q.suite))
                secs.append(s)
            entry["plan_seconds"] = secs
            entry["plan_p50"] = statistics.median(secs) if secs else None
            entry["plan_p99"] = percentile(secs, 0.99) if secs else None
        except (RunnerError, RunnerCrashed) as e:  # a query the engine cannot plan is recorded, not fatal
            entry["error"] = str(e).splitlines()[0][:300] if str(e) else type(e).__name__
        out[q.id] = entry
    return out


def summarize(record: dict[str, Any]) -> dict[str, Any]:
    """Trees by size and fallback, plans that differ from the first arm, and planning-time extremes per arm."""
    arms = list(record["arms"])
    base = arms[0]
    summary: dict[str, Any] = {"base": base, "arms": {}}
    for name in arms:
        qs = record["arms"][name]["queries"]
        trees = [t for e in qs.values() for t in e.get("trees", [])]
        sizes: dict[int, int] = {}
        for t in trees:
            sizes[t["inputs"]] = sizes.get(t["inputs"], 0) + 1
        fallbacks: dict[str, int] = {}
        for t in trees:
            fallbacks[t["fallback"]] = fallbacks.get(t["fallback"], 0) + 1
        p99s = {q: e["plan_p99"] for q, e in qs.items() if e.get("plan_p99") is not None}
        worst = max(p99s, key=lambda q: p99s[q]) if p99s else None
        differ = []
        for q, e in qs.items():
            b = record["arms"][base]["queries"].get(q, {})
            if "trees" not in e or "trees" not in b:
                continue
            shapes = [t["shape"] for t in e["trees"]]
            if shapes != [t["shape"] for t in b["trees"]]:
                cost = sum(t["cost"] for t in e["trees"])
                base_cost = sum(t["cost"] for t in b["trees"])
                differ.append({"query": q, "cost_ratio": cost / base_cost if base_cost else None})
        summary["arms"][name] = {
            "queries": len(qs),
            "errors": sorted(q for q, e in qs.items() if "error" in e),
            "trees": len(trees),
            "trees_by_inputs": dict(sorted(sizes.items())),
            "fallbacks": fallbacks,
            "large_or_hyperedge_trees": sum(1 for t in trees if t["inputs"] > 10 or t["hyperedges"] > 0),
            "max_pairs": max((t["pairs"] for t in trees), default=0),
            "max_considered": max((t["considered"] for t in trees), default=0),
            "worst_plan_p99": {"query": worst, "seconds": p99s[worst]} if worst else None,
            "plans_differing_from_base": differ,
        }
    return summary


def markdown(record: dict[str, Any]) -> str:
    s = record["summary"]
    lines = [
        f"# join-trees {record['suite']} sf{fmt_sf(record['dataset']['scale_factor'])} (base arm: {s['base']})",
        "",
        "| arm | trees | >10 inputs or hyperedge | fallbacks | max pairs | max considered | worst p99 plan | "
        "plans differing |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, a in s["arms"].items():
        worst = a["worst_plan_p99"]
        worst_s = f"{worst['query']} {worst['seconds'] * 1e3:.2f} ms" if worst else "-"
        fb = ", ".join(f"{k} {v}" for k, v in sorted(a["fallbacks"].items()))
        lines.append(
            f"| {name} | {a['trees']} | {a['large_or_hyperedge_trees']} | {fb} | {a['max_pairs']} | "
            f"{a['max_considered']} | {worst_s} | {len(a['plans_differing_from_base'])} |"
        )
    base = s["arms"][s["base"]]
    lines += ["", f"trees by inputs ({s['base']}): {base['trees_by_inputs']}"]
    for name, a in s["arms"].items():
        if a["plans_differing_from_base"]:
            diffs = ", ".join(
                f"{d['query']} ({d['cost_ratio']:.3f}x)" if d["cost_ratio"] else d["query"]
                for d in a["plans_differing_from_base"]
            )
            lines.append(f"{name} differs on: {diffs}")
        if a["errors"]:
            lines.append(f"{name} errors: {', '.join(a['errors'])}")
    return "\n".join(lines)


def run_join_trees(
    suite: str,
    sf: float,
    dataset: str,
    arms: list[str],
    runner: str | None = None,
    only: list[str] | None = None,
    reps: int = 50,
    threads: int = 4,
    memory_limit: str = "8g",
    out_dir: Path | None = None,
) -> Path:
    root = dataset_dir(dataset, sf)
    tables = discover_tables(root)
    if not tables:
        raise SystemExit(f"no dataset at {root}")
    queries = load_queries(suite, "krishiv", only, dataset_root=root)
    ds = Dataset(dataset, sf, root, tables)
    started = dt.datetime.now(dt.UTC)
    record: dict[str, Any] = {
        "schema": 1,
        "kind": "join-trees",
        "suite": suite,
        "started_utc": started.isoformat(),
        "machine": machine_info(),
        "git": git_info(),
        "dataset": {"name": ds.name, "scale_factor": ds.sf, "path": str(ds.root)},
        "plan_reps": reps,
        "arms": {},
    }
    for text in arms:
        name, conf = parse_arm(text)
        eng = KrishivEngine(
            threads=threads, memory_limit=memory_limit, binary=runner, conf=conf, tables=table_paths(root, tables)
        )
        try:
            eng.setup(ds)
            record["arms"][name] = {
                "conf": [f"{k}={v}" for k, v in conf],
                "runner": eng.config(),
                "queries": plan_arm(eng, queries, reps),
            }
        finally:
            eng.teardown()
    record["finished_utc"] = dt.datetime.now(dt.UTC).isoformat()
    record["summary"] = summarize(record)
    out = out_dir or results_dir()
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{started.strftime('%Y%m%dT%H%M%SZ')}_join-trees_{suite}_sf{fmt_sf(ds.sf)}.json"
    f.write_text(json.dumps(record, indent=2))
    print(markdown(record))
    print(f"result: {f}")
    return f


__all__ = ["DEFAULT_ARMS", "markdown", "parse_arm", "parse_tree", "run_join_trees", "summarize", "trees_of"]
