# SPDX-License-Identifier: Apache-2.0
"""Embedded Python UDF throughput (design P2-T059 §5 and §6.3): an arrow-cheap Int64 ``x + 1`` UDF over N rows.

Each pool size runs in a fresh interpreter, since the UDF pool is process-wide and sized once. A child times the same
query with and without the UDF (``sum(py_plus(id))`` against ``sum(id + 1)``), in 8,192-row batches as E04 used, and
reports rows/s of the whole query and of the UDF's share (N over the time difference). Results must be equal for every
pool size. Run it with the interpreter under test (3.14t with ``PYTHON_GIL=0``, or a GIL build), holding the bench lock:

    python bench/scripts/python_udf_throughput.py --threads 1 8 --rows 10000000 --reps 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import textwrap

CHILD = textwrap.dedent(
    """
    import json, sys, time
    import pyarrow.compute as pc
    from krishiv import _native
    from krishiv.sql.session import SparkSession

    threads, rows, reps, partitions = (int(a) for a in sys.argv[1:5])
    conf = [("krishiv.python.udf.threads", str(threads))]
    spark = SparkSession(_native.Session(conf), conf)
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "8192")
    spark._native.register_python_udf(
        "py_plus", 101, "bigint", True, True, lambda x: pc.call_function("add", [x, 1])
    )
    def timed(sql):
        out = []
        for _ in range(reps + 1):  # the first run warms up
            t0 = time.perf_counter()
            value = spark.sql(sql).collect()[0][0]
            out.append(time.perf_counter() - t0)
        return value, out[1:]
    source = f"range(0, {rows}, 1, {partitions})"
    base_value, base = timed(f"SELECT sum(id + 1) FROM {source}")
    udf_value, udf = timed(f"SELECT sum(py_plus(id)) FROM {source}")
    gil = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
    print(json.dumps({"base": base, "udf": udf, "base_value": base_value, "udf_value": udf_value, "gil": gil}))
    """
)


def run(threads: int, rows: int, reps: int, partitions: int) -> dict[str, object]:
    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(threads), str(rows), str(reps), str(partitions)],
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )
    if done.returncode != 0:
        raise SystemExit(f"threads={threads}: exit {done.returncode}\n{done.stderr[-3000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--partitions", type=int, default=12)
    args = parser.parse_args()
    expected = sum(range(1, args.rows + 1))
    report = {"python": sys.version, "rows": args.rows, "reps": args.reps, "partitions": args.partitions, "runs": []}
    for threads in args.threads:
        r = run(threads, args.rows, args.reps, args.partitions)
        if r["udf_value"] != expected or r["base_value"] != expected:
            raise SystemExit(f"threads={threads}: wrong sum {r['udf_value']} / {r['base_value']}, expected {expected}")
        base, udf = statistics.median(r["base"]), statistics.median(r["udf"])  # type: ignore[arg-type]
        net = udf - base
        report["runs"].append(  # type: ignore[union-attr]
            {
                "threads": threads,
                "gil": r["gil"],
                "base_s": base,
                "udf_s": udf,
                "query_rows_per_s": args.rows / udf,
                "udf_rows_per_s": args.rows / net if net > 0 else None,
                "udf_samples_s": r["udf"],
            }
        )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
