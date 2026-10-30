"""Find the commit that broke a query: ``krishiv-bench bisect --suite S --query qN --good SHA --bad SHA``.

The commits between ``--good`` (exclusive) and ``--bad`` (inclusive) are listed along the first-parent line of
``--bad`` (merge-queue history: one commit per PR) and searched by bisection. Every probed commit is built into
its own runner binary (``cargo build --profile dist -p krishiv-bench-runner`` in a scratch worktree, one shared
target directory so builds are incremental) and judged by one of two checks:

* ``--correctness`` (default): the runner's rows for the query must equal DuckDB's (``compare.rows_match``);
  an engine error, a crash or a timeout counts as bad.
* ``--time``: a paired ABBA comparison (``ab.one_query``) of the ``--good`` build (A) against the probed build
  (B). The commit is bad when B / A is at least ``--threshold`` and the lower bound of the bootstrap 95% CI is
  above 1. A failure of B (wrong rows, error, timeout) is bad too.

A commit whose build fails cannot be judged: it is skipped (like ``git bisect skip``). When the search ends with
skipped commits next to the first bad one, the result names the whole range instead of guessing.

The run holds ``$KRISHIV_DATA/.bench.lock`` and, for timing, waits for the load to decay before each probe.
Output: ``$KRISHIV_DATA/results/<ts>_bisect_<suite>_<query>.json`` and a Markdown summary on stdout.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from . import compare
from .ab import one_query
from .abba import bench_lock, wait_idle
from .adapters.base import Dataset, check_query
from .adapters.duckdb_engine import SPARK_NULL_ORDER, DuckDBEngine
from .adapters.krishiv import KrishivEngine, RunnerCrashed, RunnerError
from .dataset import resolve_dataset
from .errors import describe_error
from .paths import bench_root, data_root, results_dir
from .runner import load_queries

DEFAULT_THRESHOLD = 1.10
BUILD_TIMEOUT_S = 3600.0
RUNNER_CRATE = "krishiv-bench-runner"
MODES = ("correctness", "time")

# A probe answers True (bad), False (good) or None (cannot be judged, e.g. the build fails).
Probe = Callable[[str], bool | None]


class BisectError(Exception):
    """The inputs rule out a meaningful bisection (e.g. the ``--bad`` end is not bad)."""


def first_bad(commits: Sequence[str], probe: Probe) -> tuple[list[str], dict[str, bool | None]]:
    """Bisect ``commits`` (oldest first, the last one known bad) for the first bad commit.

    Returns ``(culprits, verdicts)``. ``culprits`` is one commit when the search is exact; when commits next to
    the answer could not be judged it is the contiguous range that contains the first bad commit (oldest first,
    ending with the first commit proven bad). ``verdicts`` holds every probe made.

    The search assumes the usual bisection monotonicity: good commits precede bad ones.
    """
    if not commits:
        raise BisectError("no commits between --good and --bad")
    verdicts: dict[str, bool | None] = {}

    def judge(i: int) -> bool | None:
        sha = commits[i]
        if sha not in verdicts:
            verdicts[sha] = probe(sha)
        return verdicts[sha]

    lo, hi = 0, len(commits) - 1  # the first bad commit is in [lo, hi]; commits[hi] is known bad
    while lo < hi:
        mid = (lo + hi) // 2
        # An unjudgeable commit is replaced by the nearest judgeable one in [lo, hi), preferring the lower side.
        order = sorted(range(lo, hi), key=lambda i: (abs(i - mid), i))
        for i in order:
            if judge(i) is not None:
                mid = i
                break
        else:
            break  # everything left is unjudgeable
        if verdicts[commits[mid]]:
            hi = mid
        else:
            lo = mid + 1
    # [lo, hi] now holds only unjudgeable commits followed by the bad one at hi.
    return list(commits[lo : hi + 1]), verdicts


def resolve(repo: Path, rev: str) -> str:
    """Full SHA of ``rev``; later git calls take only SHAs, so a revision cannot be read as an option.

    # Errors
    ``BisectError`` when ``rev`` is not a commit.
    """
    done = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "-q", "--end-of-options", f"{rev}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise BisectError(f"unknown revision {rev!r}")
    return done.stdout.strip()


def commits_between(repo: Path, good: str, bad: str) -> list[str]:
    """Full SHAs after ``good`` up to and including ``bad``, first-parent, oldest first.

    # Errors
    ``BisectError`` if git rejects either revision or ``good`` is not an ancestor of ``bad``.
    """
    good_sha, bad_sha = resolve(repo, good), resolve(repo, bad)
    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", good_sha, bad_sha], capture_output=True
    )
    if ancestor.returncode != 0:
        raise BisectError(f"--good {good} is not an ancestor of --bad {bad}")
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-list", "--reverse", "--first-parent", f"{good_sha}..{bad_sha}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()


def describe_commit(repo: Path, sha: str) -> dict[str, str]:
    out = subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--format=%H%x00%an%x00%cI%x00%s", sha],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.rstrip("\n")
    full, author, date, subject = out.split("\0", 3)
    return {"sha": full, "author": author, "date": date, "subject": subject}


class RunnerSource(Protocol):
    """What the probes need of a builder: the runner of a commit, or None when it does not build."""

    def runner(self, sha: str) -> Path | None: ...

    def close(self) -> None: ...


class Builder:
    """Builds the bench runner of a commit in a scratch worktree; binaries are cached by SHA for this run only."""

    def __init__(
        self,
        repo: Path,
        work: Path,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        cargo: Sequence[str] = ("cargo",),
    ):
        self.repo, self.work, self._run, self.cargo = repo, work, run, list(cargo)
        self.tree = work / "tree"
        self.target = work / "target"
        self.bin = work / "bin"

    def runner(self, sha: str) -> Path | None:
        """Path of the runner built from ``sha``, or None when the build fails."""
        dest = self.bin / sha / RUNNER_CRATE
        if dest.is_file():
            return dest
        try:
            if not self.tree.exists():
                self._git("worktree", "add", "-q", "--detach", str(self.tree), sha)
            else:
                self._git("-C", str(self.tree), "checkout", "-q", "--force", "--detach", sha)
            build = self._run(
                [
                    *self.cargo,
                    *("build", "-q", "--locked", "--profile", "dist", "-p", RUNNER_CRATE),
                    *("--target-dir", str(self.target)),
                ],
                cwd=self.tree,
                capture_output=True,
                text=True,
                timeout=BUILD_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            print(f"  {sha[:10]}: build timed out after {BUILD_TIMEOUT_S:.0f}s")
            return None
        except subprocess.CalledProcessError as e:
            print(f"  {sha[:10]}: checkout failed: {(e.stderr or '').strip()[-200:]}")
            return None
        built = self.target / "dist" / RUNNER_CRATE
        if build.returncode != 0 or not built.is_file():
            tail = (build.stderr or "").strip().splitlines()[-1:]
            print(f"  {sha[:10]}: build failed: {tail[0] if tail else 'no output'}")
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built, dest)
        return dest

    def _git(self, *args: str) -> None:
        cmd = ["git", *args] if args[0] == "-C" else ["git", "-C", str(self.repo), *args]
        self._run(cmd, check=True, capture_output=True, text=True)

    def close(self) -> None:
        if self.tree.exists():
            self._run(
                ["git", "-C", str(self.repo), "worktree", "remove", "--force", str(self.tree)],
                capture_output=True,
                text=True,
            )
        self._run(["git", "-C", str(self.repo), "worktree", "prune"], capture_output=True, text=True)
        shutil.rmtree(self.target, ignore_errors=True)
        shutil.rmtree(self.bin, ignore_errors=True)


def correctness_probe(
    builder: RunnerSource, ds: Dataset, query, reference_rows: list[tuple], threads: int, memory: str, timeout_s: float
) -> Probe:
    """Bad when the build's rows differ from ``reference_rows`` or the engine fails."""
    ordered = compare.has_top_level_order_by(query.sql)

    def probe(sha: str) -> bool | None:
        binary = builder.runner(sha)
        if binary is None:
            return None
        eng = KrishivEngine(threads=threads, memory_limit=memory, binary=binary)
        eng.timeout_s = timeout_s or None
        try:
            try:
                eng.setup(ds)
            except (RunnerError, RunnerCrashed) as e:  # e.g. a runner protocol older than this harness
                print(f"  {sha[:10]}: skipped, runner did not start ({describe_error(e)})")
                return None
            try:
                rows = eng.run(check_query(query))
            except (RunnerError, RunnerCrashed) as e:
                print(f"  {sha[:10]}: bad ({describe_error(e)})")
                return True
        finally:
            eng.teardown()
        ok, msg = compare.rows_match(rows, reference_rows, ordered, sql=query.sql)
        print(f"  {sha[:10]}: {'good' if ok else 'bad (' + msg[:120] + ')'}")
        return not ok

    return probe


def time_probe(
    builder: RunnerSource,
    ds: Dataset,
    query,
    good_runner: Path,
    threads: int,
    memory: str,
    timeout_s: float,
    reps: int,
    threshold: float,
    settle: Callable[[], None] | None = None,
) -> Probe:
    """Bad when B (the probed build) is at least ``threshold`` x slower than A (the good build), CI above 1."""

    def probe(sha: str) -> bool | None:
        binary = builder.runner(sha)
        if binary is None:
            return None
        engines = []
        for path in (good_runner, binary):
            eng = KrishivEngine(threads=threads, memory_limit=memory, binary=path)
            eng.timeout_s = timeout_s or None
            engines.append(eng)
        a, b = engines
        (settle or wait_idle)()
        entry: dict[str, Any]
        try:
            try:
                a.setup(ds)
                b.setup(ds)
            except (RunnerError, RunnerCrashed) as e:  # a runner that does not start cannot be judged
                print(f"  {sha[:10]}: skipped, runner did not start ({describe_error(e)})")
                return None
            entry = one_query(query, a, b, reps)
        finally:
            a.teardown()
            b.teardown()
        if entry["status"] != "ok":
            if entry["reason"].startswith("a "):  # the good build failed: not evidence against this commit
                print(f"  {sha[:10]}: skipped, the good build failed ({entry['reason'][:100]})")
                return None
            print(f"  {sha[:10]}: bad ({entry['reason'][:120]})")
            return True
        bad = entry["ratio"] >= threshold and entry["ratio_ci95"][0] > 1.0
        print(f"  {sha[:10]}: {'bad' if bad else 'good'} (B/A x{entry['ratio']:.2f})")
        return bad

    return probe


def render(record: dict[str, Any]) -> str:
    lines = [
        f"bisect {record['suite']} {record['query']} ({record['mode']}): {record['good']} .. {record['bad']}",
        f"{record['commits']} commits, {record['probes']} probed.",
    ]
    culprits = record["culprits"]
    if len(culprits) == 1:
        c = culprits[0]
        lines.append(f"**First bad commit: {c['sha'][:12]}** {c['subject']} ({c['author']}, {c['date']})")
    else:
        lines.append(f"**First bad commit is one of these {len(culprits)} (the earlier ones could not be built):**")
        lines.extend(f"- {c['sha'][:12]} {c['subject']} ({c['author']})" for c in culprits)
    return "\n".join(lines)


def run_bisect(
    suite: str,
    query_id: str,
    good: str,
    bad: str,
    mode: str = "correctness",
    sf: float = 10,
    dataset: str | None = None,
    threads: int = 12,
    memory_limit: str = "24g",
    timeout_s: float = 600,
    reps: int = 3,
    threshold: float = DEFAULT_THRESHOLD,
    repo: Path | None = None,
    work: Path | None = None,
    builder: RunnerSource | None = None,
) -> Path:
    """Bisect ``good..bad`` for ``query_id``; see the module docstring.

    # Errors
    ``SystemExit`` for a missing dataset or unknown query; ``BisectError`` (the CLI exits with its message)
    when the revisions are unusable or the ``--bad`` end does not reproduce the problem.
    """
    if mode not in MODES:
        raise BisectError(f"mode must be one of {MODES}, got {mode!r}")
    if reps < 1:
        raise BisectError("--reps must be at least 1")
    if threshold <= 0:
        raise BisectError("--threshold must be positive")
    repo = repo or bench_root().parent
    ds = resolve_dataset(suite, sf, dataset)
    queries = load_queries(suite, "krishiv", [query_id])
    if len(queries) != 1:
        raise SystemExit(f"unknown query {query_id} in suite {suite}")
    query = queries[0]
    shas = commits_between(repo, good, bad)
    if not shas:
        raise BisectError(f"no commits between --good {good} and --bad {bad}")
    started = dt.datetime.now(dt.UTC)
    work = work or data_root() / "bisect" / started.strftime("%Y%m%dT%H%M%SZ")
    work.mkdir(parents=True, exist_ok=True)
    builder = builder or Builder(repo, work)
    memory = memory_limit.lower()
    good_sha = resolve(repo, good)
    # `timeout` (the nightly script) stops us with SIGTERM: unwind so the scratch worktree is removed.
    previous_handler = signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        with bench_lock():
            if mode == "correctness":
                duck = DuckDBEngine(threads=threads, memory_limit=memory_limit.upper(), null_order=SPARK_NULL_ORDER)
                duck_q = next(iter(load_queries(suite, "duckdb", [query_id])), query)
                duck.setup(ds)
                try:
                    reference = duck.run(duck_q)
                finally:
                    duck.teardown()
                probe = correctness_probe(builder, ds, query, reference, threads, memory, timeout_s)
            else:
                good_runner = builder.runner(good_sha)
                if good_runner is None:
                    raise BisectError(f"the --good commit {good} does not build")
                probe = time_probe(builder, ds, query, good_runner, threads, memory, timeout_s, reps, threshold)
            if probe(shas[-1]) is not True:
                raise BisectError(f"the --bad commit {bad} does not show the problem ({mode}); nothing to bisect")
            culprits, verdicts = first_bad(shas, lambda s: True if s == shas[-1] else probe(s))
            verdicts[shas[-1]] = True  # probed above
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        builder.close()
    record = {
        "schema": 1,
        "kind": "bisect",
        "suite": suite,
        "query": query_id,
        "mode": mode,
        "threshold": threshold if mode == "time" else None,
        "started_utc": started.isoformat(),
        "good": good,
        "bad": bad,
        "commits": len(shas),
        "probes": len(verdicts),
        "verdicts": {sha: v for sha, v in verdicts.items()},
        "culprits": [describe_commit(repo, c) for c in culprits],
    }
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{started.strftime('%Y%m%dT%H%M%SZ')}_bisect_{suite}_{query_id}.json"
    f.write_text(json.dumps(record, indent=2))
    print(render(record))
    print(f"result: {f}")
    return f


__all__ = ["BisectError", "Builder", "commits_between", "first_bad", "render", "run_bisect"]
