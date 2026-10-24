from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import h2o
from .adapters import ENGINES
from .paths import data_root
from .report import write_report
from .runner import run_suite

ABBA_DATASET = {"tpch": "tpch-duckdb", "tpcds": "tpcds", "clickbench": "clickbench"}


def _add_memcap_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rows", default="10m", help="memcap: sales rows, a whole number of millions (10m, 50m, 150m)")
    p.add_argument("--cap", default="4.5g", help="memcap: hard memory cap of the process tree (cgroup MemoryMax)")
    p.add_argument("--engine-memory", default="3g", help="memcap: the engine's own memory setting (3g, or 1g to spill)")


def _memcap_engines(ap: argparse.ArgumentParser, names: list[str]) -> list[str]:
    from .memcap import ENGINES as MEMCAP_ENGINES

    out = ["spark" if n == "spark-local" else n for n in names]
    bad = [n for n in out if n not in MEMCAP_ENGINES]
    if bad:
        ap.error(f"memcap engines are {', '.join(MEMCAP_ENGINES)} (got {', '.join(bad)})")
    return out


def _run_memcap(ap: argparse.ArgumentParser, a, engines: list[str]) -> None:
    from .memcap import resolve_queries, run_memcap

    try:
        qs = resolve_queries(a.queries.split(",")) if a.queries else None
        run_memcap(a.rows, a.cap, _memcap_engines(ap, engines), a.engine_memory, a.reps, qs, a.timeout or None)
    except (ValueError, FileNotFoundError) as e:
        ap.error(str(e))


RUNNER_HELP = "krishiv-bench-runner binary (wins over $KRISHIV_BENCH_RUNNER)"


def _set_runner(path: str | None) -> None:
    """Makes ``--runner`` win everywhere the adapter resolves the binary (worker processes inherit the env)."""
    if path:
        os.environ["KRISHIV_BENCH_RUNNER"] = path


H2O_SUITES = list(h2o.SUITES)


def add_h2o_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--size", choices=list(h2o.SIZES), help="db-benchmark data size (h2o-* suites): 1e7 / 1e8 / 1e9 rows"
    )
    p.add_argument("--h2o-version", default=h2o.h2o_data.GENERATOR_VERSION, help="generator version directory")
    p.add_argument("--na", type=int, default=0, help="h2o-*: percent of NAs in the dataset variant")
    p.add_argument("--sorted", action="store_true", help="h2o-*: the sorted dataset variant")
    p.add_argument("--k", type=int, default=h2o.DEFAULT_KEYS, help="h2o-groupby: number of id1/id2 groups (K)")


def h2o_target(ap: argparse.ArgumentParser, a: argparse.Namespace) -> tuple[Path, str, float]:
    """(directory, dataset path relative to $KRISHIV_DATA, size in GB) of an h2o-* run."""
    if not a.size:
        ap.error(f"--suite {a.suite} needs --size {{{','.join(h2o.SIZES)}}}")
    try:
        root = h2o.dataset_root(a.suite, a.size, a.h2o_version, a.k, a.na, a.sorted)
    except ValueError as e:
        ap.error(str(e))
    if getattr(a, "dataset", None):
        root = root.parent / a.dataset
    return root, str(root.relative_to(data_root())), h2o.SIZE_GB[a.size]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="krishiv-bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a suite on an engine, write JSON to $KRISHIV_DATA/results")
    r.add_argument("--engine", required=True, choices=[*ENGINES, "spark"])
    r.add_argument("--suite", default="tpch", choices=["tpch", "tpcds", "memcap", *H2O_SUITES])
    add_h2o_args(r)
    r.add_argument("--sf", type=float, default=1)
    r.add_argument(
        "--dataset", help="dataset dir name under $KRISHIV_DATA (default per suite; tpch-dbgen = official-answer data)"
    )
    r.add_argument("--iterations", type=int, help="timed runs per query (default 5; 3 for h2o-* and memcap)")
    r.add_argument("--warmup", type=int, help="discarded warm-up runs per query (default 2; 1 for h2o-*)")
    r.add_argument("--queries", help="comma list, e.g. q01,q05")
    r.add_argument("--spark-remote", default="sc://127.0.0.1:15002")
    r.add_argument("--duckdb-threads", type=int)
    r.add_argument("--duckdb-memory-limit", default="12GB")
    r.add_argument("--timeout", type=float, default=600, help="per-query timeout seconds (0 = none)")
    r.add_argument("--runner", help=RUNNER_HELP)
    _add_memcap_args(r)
    ab = sub.add_parser(
        "abba", help="interleaved ABBA Krishiv vs DuckDB (P2-T142); needs the dist-profile runner binary"
    )
    ab.add_argument("--suite", default="tpch", choices=["tpch", "tpcds", "clickbench"])
    ab.add_argument("--sf", type=float, default=10)
    ab.add_argument("--dataset", default=None, help="dataset dir under $KRISHIV_DATA (default per suite)")
    ab.add_argument("--reps", type=int, default=3, help="ABBA blocks per query (2 samples per engine each)")
    ab.add_argument("--queries", help="comma list, e.g. q01,q05")
    ab.add_argument("--threads", type=int, default=8, help="threads for both engines")
    ab.add_argument("--memory-limit", default="12g", help="memory limit for both engines")
    ab.add_argument("--timeout", type=float, default=600, help="per-run timeout seconds")
    ab.add_argument("--runner", help=RUNNER_HELP)
    xb = sub.add_parser(
        "ab",
        help="build-vs-build A/B: two runner binaries, or one binary with two --conf sets, interleaved ABBA",
    )
    xb.add_argument("--suite", default="tpch", choices=["tpch", "tpcds", "clickbench", *H2O_SUITES])
    add_h2o_args(xb)
    xb.add_argument("--sf", type=float, default=10)
    xb.add_argument("--dataset", default=None, help="dataset dir under $KRISHIV_DATA (default per suite)")
    xb.add_argument("--a-runner", required=True, metavar="PATH", help="runner binary of side A (the baseline)")
    xb.add_argument("--b-runner", required=True, metavar="PATH", help="runner binary of side B (ratio = B / A)")
    xb.add_argument("--a-conf", action="append", default=[], metavar="K=V", help="session setting for A (repeatable)")
    xb.add_argument("--b-conf", action="append", default=[], metavar="K=V", help="session setting for B (repeatable)")
    xb.add_argument("--a-sha", help="git SHA of A's build, recorded in the result (e.g. $(git rev-parse main))")
    xb.add_argument("--b-sha", help="git SHA of B's build, recorded in the result")
    xb.add_argument("--reps", type=int, default=3, help="ABBA blocks per query (2 samples per side each)")
    xb.add_argument("--queries", help="comma list, e.g. q01,q05")
    xb.add_argument("--threads", type=int, default=8, help="threads for both sides")
    xb.add_argument("--memory-limit", default="12g", help="memory limit for both sides")
    xb.add_argument("--timeout", type=float, default=600, help="per-run timeout seconds")
    rr = sub.add_parser(
        "reproduce",
        help="reference engines side by side: correctness vs DuckDB, then ours (interleaved) and/or theirs (best of N)",
    )
    rr.add_argument("--suite", default="tpch", choices=["tpch", "tpcds", "clickbench", "memcap", *H2O_SUITES])
    add_h2o_args(rr)
    rr.add_argument("--sf", type=float, default=10)
    rr.add_argument("--dataset", help="dataset dir under $KRISHIV_DATA (default per suite)")
    rr.add_argument(
        "--engines",
        help="comma list (default: krishiv,duckdb,polars,datafusion,sail,spark-local; clickhouse is opt-in)",
    )
    rr.add_argument("--methods", default="ours,theirs", help="comma list of: ours, theirs")
    rr.add_argument("--reps", type=int, default=3, help="interleaved blocks per query (2 samples per engine each)")
    rr.add_argument("--queries", help="comma list, e.g. q01,q05")
    rr.add_argument("--threads", type=int, default=12)
    rr.add_argument("--memory-gb", type=int, default=32)
    rr.add_argument("--timeout", type=float, default=600, help="per-run timeout for the ours method (seconds)")
    rr.add_argument("--runs", type=int, help="hot runs per process for theirs (default: the published source's)")
    rr.add_argument("--their-timeout", type=float, help="per-run timeout for theirs (default: the source's)")
    rr.add_argument("--runner", help=RUNNER_HELP)
    rr.add_argument("--profile", action="store_true", help="also record Krishiv's per-phase wall times (ours method)")
    _add_memcap_args(rr)
    rp = sub.add_parser("report", help="Markdown comparison of the newest result per engine")
    rp.add_argument("--suite", default="tpch")
    add_h2o_args(rp)
    rp.add_argument("--sf", type=float, default=1)
    rp.add_argument("--dataset")
    rp.add_argument("--rows", default="10m", help="memcap: sales rows of the results to report")
    rp.add_argument("--baseline", default="duckdb")
    rp.add_argument("files", nargs="*", type=Path, help="explicit result JSONs (default: newest per engine)")
    a = ap.parse_args(argv)
    _set_runner(getattr(a, "runner", None))
    if a.cmd in ("reproduce", "ab") and h2o.is_h2o(a.suite):
        _, a.dataset, a.sf = h2o_target(ap, a)
    if a.cmd == "reproduce" and a.suite == "memcap":
        _run_memcap(ap, a, a.engines.split(",") if a.engines else ["krishiv", "duckdb", "polars", "spark"])
    elif a.cmd == "reproduce":
        from .reproduce import run_reproduce

        methods = tuple(m.strip() for m in a.methods.split(",") if m.strip())
        if not methods or set(methods) - {"ours", "theirs"}:
            ap.error(f"--methods must be a comma list of: ours, theirs (got {a.methods!r})")
        run_reproduce(
            a.suite,
            a.sf,
            a.dataset,
            a.engines.split(",") if a.engines else None,
            methods,
            a.reps,
            a.queries.split(",") if a.queries else None,
            a.threads,
            a.memory_gb,
            a.timeout,
            a.runs,
            a.their_timeout,
            a.profile,
        )
    elif a.cmd == "abba":
        from .abba import run_abba

        run_abba(
            a.sf,
            a.dataset or ABBA_DATASET[a.suite],
            a.reps,
            a.queries.split(",") if a.queries else None,
            a.threads,
            a.memory_limit,
            a.timeout,
            a.suite,
            a.runner,
        )
    elif a.cmd == "ab":
        from .ab import parse_conf, run_ab

        try:
            a_conf, b_conf = parse_conf(a.a_conf), parse_conf(a.b_conf)
        except ValueError as e:
            ap.error(f"--a-conf/--b-conf: {e}")
        run_ab(
            a.sf,
            a.dataset or ABBA_DATASET[a.suite],
            a.a_runner,
            a.b_runner,
            a_conf,
            b_conf,
            a.reps,
            a.queries.split(",") if a.queries else None,
            a.threads,
            a.memory_limit,
            a.timeout,
            a.suite,
            a.a_sha,
            a.b_sha,
        )
    elif a.cmd == "run" and a.suite == "memcap":
        a.reps = a.iterations if a.iterations is not None else 3
        _run_memcap(ap, a, [a.engine])
    elif a.cmd == "run":
        if a.engine == "spark":
            ap.error("--engine spark is a memcap engine; use spark-connect or spark-local elsewhere")
        is_h2o = h2o.is_h2o(a.suite)
        iterations = a.iterations or (3 if is_h2o else 5)
        warmup = a.warmup if a.warmup is not None else (1 if is_h2o else 2)
        kw = {}
        if a.engine == "spark-connect":
            kw["remote"] = a.spark_remote
        if a.engine == "krishiv":
            kw["threads"] = a.duckdb_threads
        if a.engine == "duckdb":
            kw["memory_limit"] = a.duckdb_memory_limit
            if a.duckdb_threads:
                kw["threads"] = a.duckdb_threads
        if is_h2o:
            root, _, gb = h2o_target(ap, a)
            f, failed = h2o.run_h2o(
                a.engine,
                a.suite,
                root,
                gb,
                iterations,
                warmup,
                a.queries.split(",") if a.queries else None,
                kw,
                a.timeout or None,
            )
            print(f"result: {f}")
            if failed:
                sys.exit(f"{failed} question(s) failed the answer-checksum comparison or errored: see {f}")
            return
        f = run_suite(
            a.engine,
            a.suite,
            a.sf,
            iterations,
            warmup,
            a.queries.split(",") if a.queries else None,
            kw,
            a.timeout or None,
            a.dataset,
        )
        print(f"result: {f}")
    elif a.suite == "memcap":
        from .memcap import write_memcap_report

        print(f"report: {write_memcap_report(a.rows)}")
    else:
        if h2o.is_h2o(a.suite):
            root, _, a.sf = h2o_target(ap, a)
            a.dataset = a.dataset or root.name  # the variant (K, NA %, sorted) is part of the dataset name
        print(f"report: {write_report(a.suite, a.sf, a.baseline, a.files or None, a.dataset)}")


if __name__ == "__main__":
    main()
