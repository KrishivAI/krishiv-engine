"""Scale-ladder report: which queries get relatively slower, against the reference engine, as the data grows."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from . import h2o
from .paths import fmt_sf, results_dir

GROWTH_FLAG = 1.5  # ratio-to-reference growth between consecutive scale factors that is flagged
REFERENCE = "duckdb"


def load_ladder(
    suite: str, files: list[Path] | None = None, dataset: str | None = None
) -> dict[float, dict[str, dict]]:
    """``{scale factor: {engine: newest result}}`` of the single-engine ``suite`` results (explicit ``files``, else
    every result in the results directory)."""
    if h2o.is_h2o(suite):
        raise SystemExit(f"{suite} has no scale factor: its size is part of the dataset name, so it has no ladder")
    paths = list(files) if files else sorted(results_dir().glob("*.json"))
    found: list[dict] = []
    for f in paths:
        r = json.loads(Path(f).read_text())
        if r.get("suite") != suite or "engine" not in r:
            continue
        if dataset and r["dataset"]["name"] != dataset:
            continue
        found.append(r)
    names = sorted({r["dataset"]["name"] for r in found})
    if len(names) > 1:
        raise SystemExit(
            f"results of {suite} come from different datasets {names}: a ladder compares one; pass --dataset NAME"
        )
    out: dict[float, dict[str, dict]] = {}
    for r in found:
        sf = float(r["dataset"]["scale_factor"])
        name = r["engine"]["name"]
        have = out.setdefault(sf, {}).get(name)
        if have is None or r["started_utc"] > have["started_utc"]:
            out[sf][name] = r
    return out


def _ratios(by_engine: dict[str, dict], engine: str, reference: str) -> dict[str, float]:
    eng, ref = by_engine.get(engine), by_engine.get(reference)
    if not eng or not ref:
        return {}
    out = {}
    for q, d in eng["queries"].items():
        b = ref["queries"].get(q, {})
        if "p50_s" in d and "p50_s" in b and b["p50_s"] > 0:
            out[q] = d["p50_s"] / b["p50_s"]
    return out


def compared(ladder: dict[float, dict[str, dict]], engine: str, reference: str = REFERENCE) -> int:
    """How many (query, consecutive scale pair) ratios could be compared for ``engine``."""
    sfs = sorted(ladder)
    per = {sf: _ratios(ladder[sf], engine, reference) for sf in sfs}
    return sum(len(per[lo].keys() & per[hi].keys()) for lo, hi in itertools.pairwise(sfs))


def flagged(ladder: dict[float, dict[str, dict]], engine: str, reference: str = REFERENCE) -> list[dict]:
    """Queries whose ``engine``/``reference`` ratio grew by more than ``GROWTH_FLAG`` between consecutive scale
    factors, each as ``{"query", "from_sf", "to_sf", "from_ratio", "to_ratio", "growth"}``."""
    sfs = sorted(ladder)
    per = {sf: _ratios(ladder[sf], engine, reference) for sf in sfs}
    out = []
    for lo, hi in itertools.pairwise(sfs):
        for q in sorted(per[lo].keys() & per[hi].keys()):
            growth = per[hi][q] / per[lo][q]
            if growth > GROWTH_FLAG:
                out.append(
                    {
                        "query": q,
                        "from_sf": lo,
                        "to_sf": hi,
                        "from_ratio": per[lo][q],
                        "to_ratio": per[hi][q],
                        "growth": growth,
                    }
                )
    return sorted(out, key=lambda d: -d["growth"])


def render_ladder(ladder: dict[float, dict[str, dict]], reference: str = REFERENCE) -> str:
    sfs = sorted(ladder)
    if len(sfs) < 2:
        raise SystemExit("a ladder needs results at two or more scale factors")
    engines = sorted({e for by in ladder.values() for e in by} - {reference})
    colds = {bool(r.get("cold", {}).get("enabled")) for by in ladder.values() for r in by.values()}
    L = [f"# Scale ladder: SF {', '.join(fmt_sf(s) for s in sfs)}", ""]
    if len(colds) > 1:
        L += ["**Warning: cold and warm results are mixed; their ratios are not comparable.**", ""]
    L += [
        f"Ratio = engine p50 / `{reference}` p50 per query. A query is flagged when its ratio grows by more than "
        f"{GROWTH_FLAG}x between consecutive scale factors.",
        "",
    ]
    for e in engines:
        rows = flagged(ladder, e, reference)
        L += [f"## {e}", ""]
        n = compared(ladder, e, reference)
        if n == 0:
            L += [
                f"**No comparison was possible:** no query has a successful `{e}` and `{reference}` result at two "
                "consecutive scale factors.",
                "",
            ]
            continue
        if not rows:
            L += [f"No query is flagged ({n} comparisons).", ""]
            continue
        L += ["| Query | From SF | To SF | Ratio before | Ratio after | Growth |", "|---|---|---|---|---|---|"]
        L += [
            f"| {d['query']} | {fmt_sf(d['from_sf'])} | {fmt_sf(d['to_sf'])} | {d['from_ratio']:.2f}x | "
            f"{d['to_ratio']:.2f}x | {d['growth']:.2f}x |"
            for d in rows
        ]
        L.append("")
    return "\n".join(L) + "\n"


def write_ladder(suite: str, files: list[Path] | None = None, dataset: str | None = None) -> Path:
    md = render_ladder(load_ladder(suite, files, dataset))
    out = results_dir() / f"ladder_{dataset or suite}.md"
    out.write_text(md)
    return out
