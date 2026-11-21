"""Nightly trend page: per-query Krishiv / DuckDB ratios over time, from the ``abba`` results on disk.

``krishiv-bench trend [--out FILE] [--last N]`` reads ``$KRISHIV_DATA/results/*_abba_*.json`` and writes one
Markdown section per suite and scale factor: a row per query with its ratio in each of the last ``N`` runs
(oldest left), a sparkline, and the change from the previous run; a failed query shows ``x`` and a run whose
load average exceeded the bound is starred (indicative only). The first row is the geomean the run recorded
(computed over its passing queries when the record has none). The last column is the latest run's bootstrap 95% CI
(``ratio_ci95`` per query, ``geomean_ci95`` for the geomean, as ``abba`` records them), empty when that run failed the
query or recorded no CI. Results of other kinds, unreadable files and runs
without a ratio are ignored. The default output is
``$KRISHIV_DATA/results/trend.md``; the nightly script passes ``--out`` to write it into the docs repo.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .paths import fmt_sf, results_dir

BARS = "▁▂▃▄▅▆▇█"
DEFAULT_LAST = 10
# A change below this (relative) is shown as flat.
FLAT = 0.03


def _ratio(entry: dict[str, Any]) -> float | None:
    """The query's ratio, or None when it failed or the value is not a positive finite number."""
    r = entry.get("ratio") if entry.get("status") == "ok" else None
    return r if isinstance(r, int | float) and r > 0 and math.isfinite(r) else None


def _ci(value: Any) -> tuple[float, float] | None:
    """A recorded ``[lo, hi]`` CI, or None when it is missing or not a finite positive interval."""
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    lo, hi = value
    try:
        ok = all(
            isinstance(v, int | float) and not isinstance(v, bool) and v > 0 and math.isfinite(v) for v in (lo, hi)
        )
        return (float(lo), float(hi)) if ok and lo <= hi else None
    except OverflowError:  # an integer bound too large for a double (`math.isfinite(10**400)` raises)
        return None


def load_runs(directory: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Valid ``abba`` runs of ``directory`` grouped by ``(suite, sf)``, oldest first.

    A run is ``{"when": iso timestamp, "indicative": bool, "ratios": {query: ratio or None}, "geomean": float,
    "cis": {query: (lo, hi) or None}, "geomean_ci": (lo, hi) or None}``; a CI is kept only for a passing query.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(directory.glob("*_abba_*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec.get("kind") != "abba":
                continue
            queries = rec["queries"]
            ratios = {q: _ratio(e) for q, e in queries.items()}
            cis = {q: _ci(e.get("ratio_ci95")) if ratios[q] is not None else None for q, e in queries.items()}
            good = [r for r in ratios.values() if r is not None]
            if not good:
                continue
            summary = rec.get("summary", {})
            when = rec["started_utc"]
            if not isinstance(when, str) or not when:
                continue
            geo = summary.get("geomean")
            geo_ci = _ci(summary.get("geomean_ci95"))
            if not isinstance(geo, int | float) or isinstance(geo, bool) or not (geo > 0 and math.isfinite(geo)):
                geo = math.exp(sum(math.log(r) for r in good) / len(good))
                geo_ci = None  # the recorded interval belongs to a geomean this page does not show
            key = (rec["suite"], fmt_sf(rec["dataset"]["scale_factor"]))
            groups[key].append(
                {
                    "when": when,
                    "indicative": bool(summary.get("indicative", False)),
                    "ratios": ratios,
                    "geomean": geo,
                    "cis": cis,
                    "geomean_ci": geo_ci,
                }
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue  # not a result we understand: a trend page must not die on one bad file
    for runs in groups.values():
        runs.sort(key=lambda r: r["when"])
    return dict(groups)


def sparkline(values: list[float | None]) -> str:
    """One block per value on a log scale shared by the row; a gap for a missing value."""
    present = [math.log(v) for v in values if v is not None and v > 0]
    if not present:
        return ""
    lo, hi = min(present), max(present)
    out = []
    for v in values:
        if v is None or v <= 0:
            out.append(" ")
        elif hi == lo:
            out.append(BARS[len(BARS) // 2])
        else:
            out.append(BARS[round((math.log(v) - lo) / (hi - lo) * (len(BARS) - 1))])
    return "".join(out)


def change(values: list[float | None]) -> str:
    """Latest vs the previous measured value: ``+x%`` slower, ``-x%`` faster, ``=`` within ``FLAT``, else ``-``."""
    measured = [v for v in values if v is not None]
    if values[-1] is None or len(measured) < 2:
        return "-"
    rel = measured[-1] / measured[-2] - 1
    return "=" if abs(rel) < FLAT else f"{rel:+.0%}"


def _cell(v: float | None, starred: bool) -> str:
    return "x" if v is None else f"{v:.2f}{'*' if starred else ''}"


def _ci_cell(ci: tuple[float, float] | None) -> str:
    return "" if ci is None else f"{ci[0]:.2f}-{ci[1]:.2f}"


def render_section(suite: str, sf: str, runs: list[dict[str, Any]]) -> str:
    dates = [r["when"][:10] for r in runs]
    lines = [
        f"## {suite} sf{sf}",
        "",
        "| query | " + " | ".join(dates) + " | trend | change | latest 95% CI |",
        "|---|" + "---:|" * len(runs) + "---|---:|---|",
    ]
    geo: list[float | None] = [r["geomean"] for r in runs]
    lines.append(
        "| **geomean** | "
        + " | ".join(_cell(g, r["indicative"]) for g, r in zip(geo, runs, strict=True))
        + f" | {sparkline(geo)} | {change(geo)} | {_ci_cell(runs[-1].get('geomean_ci'))} |"
    )
    queries = sorted({q for r in runs for q in r["ratios"]}, key=lambda q: (len(q), q))
    for q in queries:
        vals = [r["ratios"].get(q) for r in runs]
        cells = " | ".join(
            _cell(v, r["indicative"]) if q in r["ratios"] else "" for v, r in zip(vals, runs, strict=True)
        )
        ci = _ci_cell(runs[-1].get("cis", {}).get(q))
        lines.append(f"| {q} | {cells} | {sparkline(vals)} | {change(vals)} | {ci} |")
    return "\n".join(lines)


def render(groups: dict[tuple[str, str], list[dict[str, Any]]], last: int = DEFAULT_LAST) -> str:
    head = [
        "# Krishiv / DuckDB ratio over time",
        "",
        "Median Krishiv time divided by median DuckDB time per query (interleaved ABBA runs, `krishiv-bench abba`); "
        "below 1 means Krishiv is faster. `x`: the query failed that run (error, timeout or wrong result). "
        "`*`: the machine load was above the bound, so the run is indicative only. Change: latest run against the "
        f"previous measured one (within {FLAT:.0%} is `=`). Latest 95% CI: the bootstrap interval of the latest run's "
        "ratio (empty when that run failed the query or recorded none). It describes that run's ratio, not the change "
        "between runs.",
    ]
    if not groups:
        return "\n".join([*head, "", "No `abba` results found.", ""])
    body = [render_section(suite, sf, runs[-last:]) for (suite, sf), runs in sorted(groups.items())]
    return "\n\n".join(["\n".join(head), *body]) + "\n"


def run_trend(out: Path | None = None, last: int = DEFAULT_LAST, directory: Path | None = None) -> Path:
    """Write the trend page and return its path.

    # Errors
    ``SystemExit`` when ``last < 1``.
    """
    if last < 1:
        raise SystemExit("--last must be at least 1")
    directory = directory or results_dir()
    out = out or directory / "trend.md"
    text = render(load_runs(directory), last)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"trend: {out}")
    return out


__all__ = ["change", "load_runs", "render", "run_trend", "sparkline"]
