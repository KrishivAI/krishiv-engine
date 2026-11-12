# SPDX-License-Identifier: Apache-2.0
"""E6 spike (P3, branch only): Python UDFs in-process (#660, thread pool) against a process pool.

In-process side: the engine's native `register_python_udf` over `range(N)`. Process-pool side: the same
batches handed to a `ProcessPoolExecutor` as Arrow IPC bytes, no engine in the loop, so it is the *floor* of
a process pool's cost (the engine <-> worker transport of a real pool would add to it).
Modes: `speed` (light and heavy UDF, 1/2/4/8 workers), `isolation` (crash, hang, memory growth).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

N = int(os.environ.get("E6_ROWS", "2000000"))
BATCH = 65536
ARROW_BATCHED = 101


def _load() -> str:
    return Path("/proc/loadavg").read_text().strip()


def light(x: pa.Array) -> pa.Array:
    return pc.call_function("add", [x, 1])


def heavy(x: pa.Array) -> pa.Array:
    # Pure-Python per-row work that holds the GIL: ~20 us a row.
    out = []
    for v in x.to_pylist():
        h = v
        for _ in range(40):
            h = (h * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(h)
    return pa.array(out, type=pa.int64())


FUNCS = {"light": light, "heavy": heavy}


def _pool_task(args: tuple[str, bytes]) -> int:
    name, payload = args
    batch = pa.ipc.open_stream(payload).read_all()
    out = FUNCS[name](batch.column(0).combine_chunks())
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, pa.schema([("r", out.type)])) as w:
        w.write_table(pa.table({"r": out}))
    return len(sink.getvalue())


def process_pool(name: str, workers: int, rows: int) -> float:
    batches = []
    for start in range(0, rows, BATCH):
        t = pa.table({"x": pa.array(range(start, min(start + BATCH, rows)), type=pa.int64())})
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, t.schema) as w:
            w.write_table(t)
        batches.append((name, sink.getvalue().to_pybytes()))
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(_pool_task, batches[:workers]))  # warm the workers
        t0 = time.perf_counter()
        list(pool.map(_pool_task, batches))
        return time.perf_counter() - t0


def in_process(name: str, threads: int, rows: int) -> float:
    from krishiv import _native
    from krishiv.sql.session import SparkSession

    conf = [("krishiv.python.udf.threads", str(threads))]
    spark = SparkSession(_native.Session(conf), conf)
    spark._native.register_python_udf("e6", ARROW_BATCHED, "bigint", True, True, FUNCS[name])
    spark.sql("SELECT e6(id) FROM range(1000)").collect()
    t0 = time.perf_counter()
    n = spark.sql(f"SELECT count(e6(id)) FROM range({rows})").collect()[0][0]
    dt = time.perf_counter() - t0
    assert n == rows, n
    return dt


def speed() -> None:
    print(f"gil_enabled={getattr(sys, '_is_gil_enabled', lambda: True)()} cpus={os.cpu_count()} rows={N}")
    print(f"load before: {_load()}")
    for name in ("light", "heavy"):
        rows = N * 10 if name == "light" else N // 10
        for w in (1, 2, 4, 8):
            # The in-process pool is process-wide and sized by the first registration: one process per size.
            r = subprocess.run(
                [sys.executable, __file__, "one", name, str(w), str(rows)], capture_output=True, text=True
            )
            inproc = float(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else float("nan")
            pp = process_pool(name, w, rows)
            print(
                f"{name:5} workers={w} rows={rows}: in-process {inproc:.2f}s ({rows / inproc / 1e6:.2f} Mrows/s) | "
                f"process pool {pp:.2f}s ({rows / pp / 1e6:.2f} Mrows/s)"
            )
    print(f"load after: {_load()}")


def isolation() -> None:
    # A UDF that kills its interpreter: in-process it takes the engine down with it.
    code = (
        "import os, pyarrow as pa\nfrom krishiv import _native\nfrom krishiv.sql.session import SparkSession\n"
        "def die(x):\n    os._exit(139)\n"
        "c=[('krishiv.python.udf.threads','2')]\ns=SparkSession(_native.Session(c),c)\n"
        "s._native.register_python_udf('die',101,'bigint',True,True,die)\n"
        "try:\n    s.sql('SELECT die(id) FROM range(10)').collect()\n"
        "except Exception as e:\n    print('RAISED', type(e).__name__)\n"
        "print('ENGINE ALIVE')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(f"in-process crash UDF: exit code {r.returncode}, 'ENGINE ALIVE' printed: {'ENGINE ALIVE' in r.stdout}")
    with ProcessPoolExecutor(2) as pool:
        fut = pool.submit(os._exit, 139)
        try:
            fut.result(timeout=30)
        except Exception as e:
            print(f"process pool crash UDF: parent alive, task raised {type(e).__name__}")
    # A hang: a process can be killed on a deadline, a thread cannot.
    print(
        "hang: a pool worker process can be killed on a deadline; an in-process UDF thread cannot be "
        "interrupted from Rust (it holds the GIL or a Python frame)"
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "speed"
    if mode == "one":
        print(in_process(sys.argv[2], int(sys.argv[3]), int(sys.argv[4])))
    elif mode == "speed":
        speed()
    else:
        isolation()
