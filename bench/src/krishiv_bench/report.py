"""Markdown comparison report from result JSONs."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .paths import results_dir


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
    L = [f"# Krishiv benchmark report: {first['suite']} SF{first['dataset']['scale_factor']}", ""]
    L += ["## Environment", ""]
    m = first["machine"]
    L += [f"- Machine: {m['cpu_model']}, {m['cores_logical']} logical cores, {m['ram_bytes']/2**30:.0f} GiB RAM, {m['os']}",
          f"- Git: krishiv-engine `{(first['git']['commit'] or '?')[:12]}` ({first['git']['branch']}, dirty={first['git']['dirty']})",
          f"- Iterations: {first['iterations']} timed + {first['warmup']} warm-up per query; p99 is nearest-rank (== max for < 100 samples)",
          "- Absolute numbers from a shared laptop are for relative A/B only (see krishiv-benchmark skill).", ""]
    L += ["| Engine | Version | Result file | Config |", "|---|---|---|---|"]
    for e in engines:
        r = results[e]
        L.append(f"| {e} | {r['engine']['version']} | `{r['_file']}` | `{json.dumps(r['engine']['config'])}` |")
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

    L += ["## Resource use (per engine, peak over queries)", "",
          "| Engine | peak RSS GiB (max query) | median CPU-s per query run | spill |", "|---|---|---|---|"]
    for e in engines:
        qs = [d for d in results[e]["queries"].values() if "p50_s" in d]
        rss = [d["peak_rss_bytes"] for d in qs if d.get("peak_rss_bytes")]
        cpu = [d["cpu_seconds_median"] for d in qs if d.get("cpu_seconds_median") is not None]
        sp = [d["spill_bytes"] for d in qs if d.get("spill_bytes") is not None]
        L.append(f"| {e} | {_gb(max(rss) if rss else None)} | "
                 f"{('%.2f' % (sum(cpu)/len(cpu))) if cpu else 'n/a'} | {_gb(max(sp)) if sp else 'n/a (not collected)'} |")
    L.append("")

    L += ["## Correctness", "",
          "Row counts are checked against the TPC-H reference answers (`expected_rows_*.json`) and, for outputs of "
          "<= 10 000 rows, a checksum of the sorted, normalised rows is compared across engines "
          "(floats/decimals compared to 5 significant digits; see `normalize.py`).", ""]
    bad = []
    for q in qids:
        per = {e: results[e]["queries"].get(q, {}) for e in engines}
        for e, d in per.items():
            if "error" in d:
                bad.append(f"- `{q}` **{e}**: error `{d['error'][:200]}`")
            elif d.get("rows_match_expected") is False:
                bad.append(f"- `{q}` **{e}**: returned {d['rows_returned']} rows, expected {d['expected_rows']}")
        ok = {e: d for e, d in per.items() if "rows_returned" in d}
        if len(ok) > 1:
            if len({d["rows_returned"] for d in ok.values()}) > 1:
                bad.append(f"- `{q}`: row counts differ across engines: " + ", ".join(f"{e}={d['rows_returned']}" for e, d in ok.items()))
            cs = {e: d["result_checksum"] for e, d in ok.items() if d.get("result_checksum")}
            if len(set(cs.values())) > 1:
                bad.append(f"- `{q}`: result checksums differ: " + ", ".join(f"{e}={c}" for e, c in cs.items()))
    L += bad if bad else ["All queries agree across engines (row counts and checksums) and match expected row counts."]
    return "\n".join(L) + "\n"


def write_report(suite: str, sf: float, baseline: str | None = None, files: list[Path] | None = None) -> Path:
    if files:
        res = {}
        for f in files:
            r = json.loads(Path(f).read_text()); res[r["engine"]["name"]] = r | {"_file": Path(f).name}
    else:
        res = latest_results(suite, sf)
    md = render(res, baseline)
    out = results_dir() / f"report_{suite}_sf{int(sf) if float(sf).is_integer() else sf}.md"
    out.write_text(md)
    return out
