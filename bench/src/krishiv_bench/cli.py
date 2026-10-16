from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import ENGINES
from .report import write_report
from .runner import run_suite


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="krishiv-bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a suite on an engine, write JSON to $KRISHIV_DATA/results")
    r.add_argument("--engine", required=True, choices=ENGINES)
    r.add_argument("--suite", default="tpch", choices=["tpch", "tpcds"])
    r.add_argument("--sf", type=float, default=1)
    r.add_argument("--iterations", type=int, default=5)
    r.add_argument("--warmup", type=int, default=1)
    r.add_argument("--queries", help="comma list, e.g. q01,q05")
    r.add_argument("--spark-remote", default="sc://127.0.0.1:15002")
    r.add_argument("--duckdb-threads", type=int)
    rp = sub.add_parser("report", help="Markdown comparison of the newest result per engine")
    rp.add_argument("--suite", default="tpch")
    rp.add_argument("--sf", type=float, default=1)
    rp.add_argument("--baseline", default="duckdb")
    rp.add_argument("files", nargs="*", type=Path, help="explicit result JSONs (default: newest per engine)")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        kw = {}
        if a.engine == "spark-connect":
            kw["remote"] = a.spark_remote
        if a.engine == "duckdb" and a.duckdb_threads:
            kw["threads"] = a.duckdb_threads
        f = run_suite(a.engine, a.suite, a.sf, a.iterations, a.warmup,
                      a.queries.split(",") if a.queries else None, kw)
        print(f"result: {f}")
    else:
        print(f"report: {write_report(a.suite, a.sf, a.baseline, a.files or None)}")


if __name__ == "__main__":
    main()
