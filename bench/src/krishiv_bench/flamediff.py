"""Differential flamegraph of one query on two runner builds: ``krishiv-bench flamediff``.

Each side runs the query in a fresh runner under ``perf record`` (frame-pointer call graphs on the monotonic
clock, as in ``ledger``; build the runners with frame pointers, see ``ledger.py``): one untimed warm-up run, then
``--runs`` timed runs. Only the samples inside the timed runs are kept, so start-up and dataset registration do
not show. Samples are folded into ``frame;frame;leaf count`` stacks, the two sides are normalised to the same
total (so a faster build does not light up everything blue), and merged into the ``a b`` two-column format of
``inferno-diff-folded``. ``inferno-flamegraph`` (``cargo install inferno``) renders it to an SVG: red frames got
more samples in B, blue fewer.

Output in ``$KRISHIV_DATA/results/flamediff/<ts>_<suite>_<query>/``: ``a.folded``, ``b.folded``, ``diff.folded``,
``diff.svg`` (when inferno is installed) and ``summary.md``: the frames with the largest share change, by
self time and by inclusive time. Link ``summary.md`` and the SVG from a perf PR's A/B.
"""

from __future__ import annotations

import datetime as dt
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .abba import bench_lock, wait_idle
from .adapters.krishiv import KrishivEngine, RunnerCrashed, RunnerError
from .dataset import resolve_dataset
from .errors import describe_error
from .ledger import parse_perf_script, perf_launcher, perf_script
from .paths import results_dir
from .runner import load_queries

TOP = 15
FLAMEGRAPH = "inferno-flamegraph"


def fold(samples: list[tuple[float, list[str]]], t0: float, t1: float) -> Counter[str]:
    """Folded stacks (``root;...;leaf`` -> samples) of the samples with ``t0 <= time < t1``.

    ``perf script`` lists frames leaf first; a folded stack runs root first, and ``;`` in a symbol is replaced
    because it is the separator. A sample without frames is dropped.
    """
    out: Counter[str] = Counter()
    for ts, frames in samples:
        if t0 <= ts < t1 and frames:
            out[";".join(f.replace(";", ":") for f in reversed(frames))] += 1
    return out


def write_folded(path: Path, stacks: dict[str, int]) -> None:
    path.write_text("".join(f"{s} {n}\n" for s, n in sorted(stacks.items())), encoding="utf-8")


def normalise(stacks: Counter[str], total: int) -> dict[str, int]:
    """``stacks`` scaled so the counts sum to about ``total``, as integers (inferno reads whole counts only).

    A stack that rounds to zero is dropped; empty input stays empty.
    """
    have = sum(stacks.values())
    if have == 0:
        return {}
    scaled = {s: round(n * total / have) for s, n in stacks.items()}
    return {s: n for s, n in scaled.items() if n > 0}


def diff_folded(a: dict[str, int], b: dict[str, int]) -> list[tuple[str, int, int]]:
    """``(stack, a, b)`` for every stack of either side (the input of ``inferno-flamegraph`` differential mode)."""
    return [(s, a.get(s, 0), b.get(s, 0)) for s in sorted(a.keys() | b.keys())]


def shares(stacks: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    """(self share, inclusive share) per frame, as fractions of all samples."""
    total = sum(stacks.values())
    own: defaultdict[str, float] = defaultdict(float)
    incl: defaultdict[str, float] = defaultdict(float)
    if total == 0:
        return {}, {}
    for stack, n in stacks.items():
        frames = stack.split(";")
        own[frames[-1]] += n
        for f in set(frames):  # once per stack: recursion must not count a frame twice
            incl[f] += n
    return {f: n / total for f, n in own.items()}, {f: n / total for f, n in incl.items()}


def top_changes(
    a: Mapping[str, float], b: Mapping[str, float], n: int = TOP
) -> dict[str, list[tuple[str, float, float]]]:
    """Largest share changes (percentage points, B minus A) by self and inclusive time: ``(frame, a, b)``."""
    (sa, ia), (sb, ib) = shares(a), shares(b)
    out = {}
    for key, (x, y) in {"self": (sa, sb), "inclusive": (ia, ib)}.items():
        frames = sorted(x.keys() | y.keys(), key=lambda f: (-abs(y.get(f, 0.0) - x.get(f, 0.0)), f))
        out[key] = [(f, x.get(f, 0.0), y.get(f, 0.0)) for f in frames[:n]]
    return out


def render_summary(
    suite: str, query: str, a_runner: str, b_runner: str, a_samples: float, b_samples: float, changes: dict, svg: bool
) -> str:
    lines = [
        f"# Differential flamegraph: {suite} {query}",
        "",
        f"A = `{a_runner}` ({a_samples:g} samples), B = `{b_runner}` ({b_samples:g} samples). Shares are of each "
        "side's own samples; a positive change means B spends a larger share there.",
        "",
        "Flamegraph: `diff.svg` (red: larger share in B, blue: smaller)." if svg else
        f"No `diff.svg`: `{FLAMEGRAPH}` is not installed (`cargo install inferno`); render `diff.folded` with it.",
    ]  # fmt: skip
    for key in ("self", "inclusive"):
        lines += ["", f"## Largest changes, {key} time", "", "| frame | A | B | change |", "|---|---:|---:|---:|"]
        for frame, x, y in changes[key]:
            lines.append(f"| `{frame.replace('|', '/')}` | {x:.1%} | {y:.1%} | {100 * (y - x):+.2f} pp |")
    return "\n".join(lines) + "\n"


def sample_side(runner: str | None, ds, query, runs: int, threads: int, memory: str, timeout_s: float) -> Counter[str]:
    """Folded stacks of ``runs`` warm runs of ``query`` on ``runner`` under perf."""
    tmp = Path(tempfile.mkdtemp(prefix="flamediff-"))
    data = tmp / "perf.data"
    eng = KrishivEngine(threads=threads, memory_limit=memory, binary=runner, launcher=perf_launcher(data))
    eng.timeout_s = timeout_s or None
    try:
        eng.setup(ds)
        eng.profile(query)  # warm-up
        t0 = time.clock_gettime(time.CLOCK_MONOTONIC)
        for _ in range(runs):
            eng.profile(query)
        t1 = time.clock_gettime(time.CLOCK_MONOTONIC)
        eng.teardown()  # QUIT lets perf finalise its file
        return fold(parse_perf_script(perf_script(data)), t0, t1)
    finally:
        eng.cancel()
        shutil.rmtree(tmp, ignore_errors=True)


def render_svg(diff: Path, svg: Path, title: str) -> bool:
    """Render ``diff`` with inferno; False when inferno is not installed or fails."""
    svg.unlink(missing_ok=True)  # never leave the chart of an earlier run beside this run's summary
    exe = shutil.which(FLAMEGRAPH)
    if exe is None:
        return False
    with diff.open() as src, svg.open("w") as dst:
        done = subprocess.run([exe, "--title", title], stdin=src, stdout=dst, stderr=subprocess.PIPE, text=True)
    if done.returncode != 0:
        print(f"{FLAMEGRAPH} failed: {done.stderr.strip()[:200]}")
        svg.unlink(missing_ok=True)
        return False
    return True


def run_flamediff(
    suite: str,
    query_id: str,
    a_runner: str,
    b_runner: str,
    sf: float = 10,
    dataset: str | None = None,
    runs: int = 3,
    threads: int = 12,
    memory_limit: str = "24g",
    timeout_s: float = 600,
    out: Path | None = None,
    sampler: Any = None,
) -> Path:
    """Profile both builds and write the differential flamegraph; returns the output directory.

    # Errors
    ``SystemExit`` for a missing dataset, an unknown query, ``runs < 1`` or a side without samples.
    """
    if runs < 1:
        raise SystemExit("--runs must be at least 1")
    ds = resolve_dataset(suite, sf, dataset)
    queries = load_queries(suite, "krishiv", [query_id])
    if len(queries) != 1:
        raise SystemExit(f"unknown query {query_id} in suite {suite}")
    sampler = sampler or sample_side
    with bench_lock():
        wait_idle()
        stacks = []
        for side, runner in (("A", a_runner), ("B", b_runner)):
            try:
                stacks.append(sampler(runner, ds, queries[0], runs, threads, memory_limit.lower(), timeout_s))
            except (RunnerError, RunnerCrashed, subprocess.SubprocessError) as e:
                raise SystemExit(f"side {side} ({runner}): {describe_error(e)}") from e
    for side, st in zip("AB", stacks, strict=True):
        if not st:
            raise SystemExit(
                f"no perf samples for side {side}: is perf permitted (perf_event_paranoid) "
                "and the runner built with frame pointers?"
            )
    a_total, b_total = (sum(s.values()) for s in stacks)
    a, b = normalise(stacks[0], a_total), normalise(stacks[1], a_total)  # B is scaled to A's sample count
    started = dt.datetime.now(dt.UTC)
    out = out or results_dir() / "flamediff" / f"{started.strftime('%Y%m%dT%H%M%SZ')}_{suite}_{query_id}"
    out.mkdir(parents=True, exist_ok=True)
    write_folded(out / "a.folded", a)
    write_folded(out / "b.folded", b)
    rows = diff_folded(a, b)
    (out / "diff.folded").write_text("".join(f"{s} {x} {y}\n" for s, x, y in rows), encoding="utf-8")
    svg = render_svg(out / "diff.folded", out / "diff.svg", f"{suite} {query_id}: B vs A")
    summary = render_summary(suite, query_id, a_runner, b_runner, a_total, b_total, top_changes(a, b), svg)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"result: {out}")
    return out


__all__ = ["diff_folded", "fold", "normalise", "run_flamediff", "shares", "top_changes"]
