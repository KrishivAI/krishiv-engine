"""Markdown comparison report from result JSONs."""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import compare
from .paths import fmt_sf, results_dir


def latest_results(suite: str, sf: float, directory: Path | None = None) -> dict[str, dict]:
    """Newest result JSON per engine for (suite, sf)."""
    best: dict[str, tuple[str, dict]] = {}
    for f in sorted((directory or results_dir()).glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("suite") != suite or float(r["dataset"]["scale_factor"]) != float(sf):
            continue
        best[r["engine"]["name"]] = (f.name, r)
    return {k: v[1] | {"_file": v[0]} for k, v in best.items()}


def _ms(x):
    return "-" if x is None else f"{x*1000:,.1f}"


def _gb(x):
    return "n/a" if x is None else f"{x/2**30:.2f}"


def render(results: dict[str, dict], baseline: str | None = None) -> str:
    engines = list(results)
    if not engines:
        raise SystemExit("no results to report")
    baseline = baseline if baseline in results else engines[0]
    first = results[baseline]
    qids = sorted({q for r in results.values() for q in r["queries"]})
    L = [f"# Krishiv benchmark report: {first['suite']} SF{fmt_sf(first['dataset']['scale_factor'])}", ""]
    L += ["## Environment", ""]
    m = first["machine"]
    L += [f"- Machine: {m['cpu_model']}, {m['cores_logical']} logical cores, {m['ram_bytes']/2**30:.0f} GiB RAM, {m['os']}",
          f"- Git: krishiv-engine `{(first['git']['commit'] or '?')[:12]}` ({first['git']['branch']}, dirty={first['git']['dirty']})",
          f"- Dataset: {first['dataset']['name']} SF{fmt_sf(first['dataset']['scale_factor'])}, generator `{first['dataset'].get('generator')}`, manifest sha256 `{(first['dataset'].get('manifest_sha256') or 'n/a')[:16]}`",
          f"- Iterations: {first['iterations']} timed + {first['warmup']} discarded warm-up runs per query; p99 is nearest-rank (== max for < 100 samples)",
          "- Absolute numbers from a shared laptop are for relative A/B only (see krishiv-benchmark skill).", ""]
    L += ["| Engine | Version | Commit (dirty) | Result file | Config |", "|---|---|---|---|---|"]
    for e in engines:
        r = results[e]
        L.append(f"| {e} | {r['engine']['version']} | `{(r['git']['commit'] or '?')[:10]}` ({r['git']['dirty']}) | `{r['_file']}` | `{json.dumps(r['engine']['config'])}` |")
    L += ["", "## Per-query p50 latency (ms)", ""]
    hdr = "| Query | " + " | ".join(f"{e} p50 | {e} p99" for e in engines) + " | " + " | ".join(
        f"{e} / {baseline}" for e in engines if e != baseline) + " |"
    L += [hdr, "|" + "---|" * (1 + 2 * len(engines) + len(engines) - 1)]
    totals = {e: 0.0 for e in engines}
    ratios = {e: [] for e in engines if e != baseline}
    all_ok = {e: True for e in engines}
    for q in qids:
        cells, rat = [], []
        for e in engines:
            d = results[e]["queries"].get(q, {})
            if "p50_s" in d:
                totals[e] += d["p50_s"]
                cells += [_ms(d["p50_s"]), _ms(d["p99_s"])]
            else:
                all_ok[e] = False
                cells += ["ERR" if "error" in d else "-", "-"]
        b = results[baseline]["queries"].get(q, {}).get("p50_s")
        for e in engines:
            if e == baseline:
                continue
            x = results[e]["queries"].get(q, {}).get("p50_s")
            if b and x:
                ratios[e].append(x / b)
                rat.append(f"{x/b:.2f}x")
            else:
                rat.append("-")
        L.append(f"| {q} | " + " | ".join(cells) + " | " + " | ".join(rat) + " |")
    L.append("| **sum p50** | " + " | ".join(f"**{_ms(totals[e])}** | " for e in engines) + " | ".join(
        [""] + [f"**geomean {math.exp(sum(map(math.log, ratios[e]))/len(ratios[e])):.2f}x**" if ratios[e] else "-"
                for e in engines if e != baseline]) + " |")
    L += ["", f"Ratios are engine / `{baseline}` (>1 = slower than `{baseline}`). Sum covers queries that succeeded for that engine"
          + ("; **some engines have failed/missing queries, sums are not comparable**." if not all(all_ok.values()) else "."), ""]

    L += ["## Resource use", "",
          "Peak RSS = process RSS high-water during the query (VmHWM, reset via `clear_refs` before each run; same method for both "
          "engines, probes outside the timed region). DuckDB's value is the whole harness process (Python + pyarrow baseline in "
          "the baseline column). Spark's is the JVM: it keeps its committed heap between queries, so the value is cumulative and "
          "order-dependent, not an isolated per-query footprint. CPU-s is not strictly comparable (DuckDB includes Python row conversion).", "",
          "| Engine | max peak RSS GiB | max baseline RSS GiB | mean of per-query median CPU-s | spill |", "|---|---|---|---|---|"]
    for e in engines:
        qs = [d for d in results[e]["queries"].values() if "p50_s" in d]
        rss = [d["peak_rss_bytes"] for d in qs if d.get("peak_rss_bytes")]
        base = [d["baseline_rss_bytes"] for d in qs if d.get("baseline_rss_bytes")]
        cpu = [d["cpu_seconds_median"] for d in qs if d.get("cpu_seconds_median") is not None]
        sp = [d["spill_bytes"] for d in qs if d.get("spill_bytes") is not None]
        L.append(f"| {e} | {_gb(max(rss) if rss else None)} | {_gb(max(base) if base else None)} | "
                 f"{('%.2f' % (sum(cpu)/len(cpu))) if cpu else 'n/a'} | {_gb(max(sp)) if sp else 'n/a (not collected)'} |")
    L.append("")

    L += ["## Correctness", "",
          "Two independent checks, using type-aware comparison (`compare.py`): integers/strings/dates exact; DECIMAL compared at the "
          "smaller declared scale; DECIMAL vs double within half a unit of the decimal scale; double vs double rel 1e-6. "
          "(1) each engine's result is validated at run time against the official TPC-H answer sets (DuckDB `tpch_answers()`), "
          "(2) engines are compared against each other from the stored full result rows. Row order is checked for queries with a "
          "top-level ORDER BY.", ""]
    bad, skipped = [], []
    rows_by_engine = {}
    for e in engines:
        rf = results[e].get("rows_file")
        path = results_dir() / rf if rf else None
        rows_by_engine[e] = compare.load_rows(path) if path and path.exists() else {}
    n_checked = 0
    for q in qids:
        per = {e: results[e]["queries"].get(q, {}) for e in engines}
        for e, d in per.items():
            if "error" in d:
                bad.append(f"- `{q}` **{e}**: error `{d['error'][:200]}`")
            elif d.get("answer_check", {}).get("status") == "fail":
                bad.append(f"- `{q}` **{e}**: official-answer check failed: {d['answer_check']['detail']}")
            elif d.get("answer_check", {}).get("status") != "pass":
                skipped.append(f"- `{q}` **{e}**: official-answer check not performed ({d.get('answer_check', {}).get('detail', 'n/a')})")
        ok = [e for e, d in per.items() if "rows_returned" in d]
        for i, e1 in enumerate(ok):
            for e2 in ok[i + 1:]:
                r1, r2 = rows_by_engine[e1].get(q), rows_by_engine[e2].get(q)
                if r1 is None or r2 is None:
                    skipped.append(f"- `{q}` **{e1} vs {e2}**: cross-engine comparison skipped (rows not stored)")
                    continue
                n_checked += 1
                good, msg = compare.rows_match(r1, r2, per[e1].get("has_order_by", False))
                if not good:
                    bad.append(f"- `{q}` **{e1} vs {e2}**: {msg}")
    if bad:
        L += ["**Mismatches / failures:**", ""] + bad + [""]
    if skipped:
        L += ["**Checks skipped (NOT verified):**", ""] + skipped + [""]
    if not bad and not skipped:
        L.append(f"All {len(qids)} queries pass the official-answer check on every engine and {n_checked} pairwise cross-engine comparisons agree "
                 "(full result sets, nothing skipped).")
    return "\n".join(L) + "\n"


def write_report(suite: str, sf: float, baseline: str | None = None, files: list[Path] | None = None) -> Path:
    if files:
        res = {}
        for f in files:
            r = json.loads(Path(f).read_text()); res[r["engine"]["name"]] = r | {"_file": Path(f).name}
    else:
        res = latest_results(suite, sf)
    md = render(res, baseline)
    out = results_dir() / f"report_{suite}_sf{fmt_sf(sf)}.md"
    out.write_text(md)
    return out
