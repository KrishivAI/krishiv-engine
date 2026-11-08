# SPDX-License-Identifier: Apache-2.0
"""Conformance of the embedded Python UDF runtime (design P2-T059 unit 3a): the `thread_safe` counter test
(section 6.4), the lifecycle tests (6.6) and the Spark semantics (6.5).

"Cancelling a query mid-UDF" is not here: execute() returns once the UDF phase is done (the sink and source do not
overlap yet; ops/needs-owner.md, 2026-11-07), and the Python API has no cancel call, so a Python test cannot
cancel a running UDF. The Rust test `a_cancel_while_calls_are_in_flight_releases_everything` covers it.

Every test that can hang or take the interpreter down runs in a child process with a timeout. The pool is
process-wide and sized by the first session, so the 8-thread test needs a process of its own.
Not here: the cancel interrupt, TSan/ASan, the 3.14t benchmark, registry drop order, REE and nested outputs
(`PyThreadState_SetAsyncExc` and the sanitizers are unit 3b); a panicking job cannot be provoked from Python
without a test hook, so the lock-poisoning case stays a Rust test.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import textwrap
import time
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")

from krishiv.sql import SparkSession  # noqa: E402
from krishiv.sql import functions as F  # noqa: E402
from krishiv.sql.types import LongType  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

#: What every child script starts with: a session with `threads` pool threads and a UDF registrar.
PRELUDE = """
import json, os, sys, threading, time
import pyarrow as pa
import pyarrow.compute as pc
from krishiv import _native
from krishiv.sql.session import SparkSession

def session(threads, *more):
    conf = [("krishiv.python.udf.threads", str(threads)), *more]
    return SparkSession(_native.Session(conf), conf)

def plus_one(x):
    return pc.call_function("add", [x, 1])

def register(spark, name, fn, thread_safe=True):
    spark._native.register_python_udf(name, 101, "bigint", True, thread_safe, fn)
"""


def run_child(body: str, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    """Runs `PRELUDE + body` in a fresh interpreter; a hang is a failure with the child's output."""
    script = PRELUDE + textwrap.dedent(body)
    try:
        return subprocess.run(
            [sys.executable, "-W", "ignore", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        pytest.fail(f"the child did not finish within {timeout} s (hang?)\nstdout: {out}\nstderr: {err}")


def ok(result: subprocess.CompletedProcess[str]) -> str:
    assert result.returncode == 0, f"exit {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    return result.stdout


# --- 6.4: the thread_safe counter test ---

COUNTER = """
spark = session(8)
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "10")
state = {"active": 0, "peak": 0, "calls": 0, "violations": 0}
lock = threading.Lock()

def counted(x):
    # asserts nothing is active, increments, sleeps 1 ms (the GIL is released), decrements
    with lock:
        if state["active"] != 0:
            state["violations"] += 1
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        state["calls"] += 1
    time.sleep(0.001)
    with lock:
        state["active"] -= 1
    return plus_one(x)

register(spark, "py_counted", counted, thread_safe=%s)
# 10,000 rows of 10 = 1,000 batches over 8 partitions
rows = spark.sql("SELECT py_counted(id) FROM range(0, 10000, 1, 8)").collect()
print(json.dumps({"rows": len(rows), **state}))
"""


def test_thread_safe_false_never_overlaps_over_8_threads_and_1000_batches() -> None:
    out = json.loads(ok(run_child(COUNTER % "False")).strip().splitlines()[-1])
    assert out["rows"] == 10000
    assert out["calls"] == 1000, out
    assert out["violations"] == 0, out
    assert out["peak"] == 1, out


def test_thread_safe_true_does_overlap_so_the_counter_test_can_fail() -> None:
    out = json.loads(ok(run_child(COUNTER % "True")).strip().splitlines()[-1])
    assert out["calls"] == 1000, out
    assert out["peak"] > 1, out
    assert out["violations"] > 0, out


# --- 6.6: lifecycle (each in a child with a timeout) ---

EXIT_DURING = """
spark = session(4)
def slow(x):
    time.sleep(0.3)
    return plus_one(x)
register(spark, "py_slow", slow)
def query():
    try:
        spark.sql("SELECT py_slow(id) FROM range(1000)").collect()
    except BaseException:
        pass  # the query may fail as the interpreter exits; printing from a daemon thread then aborts Python

threading.Thread(target=query, daemon=True).start()
time.sleep(0.1)
sys.exit(0)
"""

EXIT_AFTER = """
spark = session(4)
register(spark, "py_plus", plus_one)
assert [r[0] for r in spark.sql("SELECT py_plus(id) FROM range(5) ORDER BY 1").collect()] == [1, 2, 3, 4, 5]
sys.exit(0)
"""

EXIT_RUNS = 100  # 100 children per kind, as the design asks; about 0.5 s each


@pytest.mark.parametrize("script", [EXIT_DURING, EXIT_AFTER], ids=["during_a_query", "after_a_query"])
def test_interpreter_exit_during_and_after_a_query_is_clean_100_times(script: str) -> None:
    start = time.monotonic()
    for i in range(EXIT_RUNS):
        r = run_child(script, timeout=60)
        assert r.returncode == 0, f"run {i}: exit {r.returncode}\nstdout: {r.stdout}\nstderr: {r.stderr}"
    assert time.monotonic() - start < 600


DROP_ORDER = """
import gc
freed = []
class PlusOne:
    def __call__(self, x):
        return plus_one(x)
    def __del__(self):
        freed.append(1)
spark = session(4)
register(spark, "py_plus", PlusOne())
assert spark.sql("SELECT count(py_plus(id)) FROM range(100000)").collect()[0][0] == 100000
del spark
gc.collect()
# A pyo3 handle dropped on an unattached thread is decref'd at the next attach: any native call flushes it.
_native.Session([])
gc.collect()
print(json.dumps({"freed": len(freed)}))
class AtExit:
    def __call__(self, x):
        return plus_one(x)
    def __del__(self):
        print("released at exit", flush=True)
kept = session(4)
register(kept, "py_kept", AtExit())
assert kept.sql("SELECT count(py_kept(id)) FROM range(100000)").collect()[0][0] == 100000
"""


def test_a_dropped_session_releases_its_udf_callables_and_one_alive_at_exit_is_safe() -> None:
    # Design P2-T059 §7.6: a UDF callable's `Py` handle is dropped attached or deferred by pyo3 and flushed before
    # finalization. A dropped session's callable is released (after the next attach); a session still alive at
    # interpreter exit, its callable last run on the pool, does not crash the exit.
    out = ok(run_child(DROP_ORDER))
    first = json.loads(out.splitlines()[0])
    assert first == {"freed": 1}, out


FAILING_BATCH = """
spark = session(4, ("krishiv.memory.limit", "512m"))
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
def flaky(x):
    if pc.any(pc.equal(x, 5000)).as_py():
        raise ValueError("one batch fails")
    time.sleep(0.05)
    return plus_one(x)
register(spark, "py_flaky", flaky)
failures = 0
for i in range(20):
    try:
        spark.sql("SELECT py_flaky(id) FROM range(0, 400000, 1, 4)").collect()
    except Exception as e:
        assert "one batch fails" in str(e), str(e)
        failures += 1
register(spark, "py_plus", plus_one)
n = spark.sql("SELECT count(py_plus(id)) FROM range(200000)").collect()[0][0]
print(json.dumps({"failures": failures, "count": n}))
"""


def test_one_failing_batch_among_those_in_flight_releases_every_reservation() -> None:
    out = json.loads(ok(run_child(FAILING_BATCH, timeout=180)).strip().splitlines()[-1])
    assert out == {"failures": 20, "count": 200000}


CONSUMERS = """
spark = session(4)
register(spark, "py_plus", plus_one)
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "100")
def df():
    return spark.sql("SELECT py_plus(id) AS v FROM range(0, 5000, 1, 4)")
out = {}
out["pyarrow"] = pa.RecordBatchReader.from_stream(df()).read_all().num_rows
import polars as pl
out["polars"] = pl.DataFrame(df()).height
import duckdb
out["duckdb"] = duckdb.from_arrow(df()).count("*").fetchall()[0][0]
print(json.dumps(out))
"""


def test_pyarrow_polars_and_duckdb_consumers_of_a_udf_plan_do_not_deadlock() -> None:
    pytest.importorskip("polars")
    pytest.importorskip("duckdb")
    out = json.loads(ok(run_child(CONSUMERS, timeout=180)).strip().splitlines()[-1])
    assert out == {"pyarrow": 5000, "polars": 5000, "duckdb": 5000}


FORK = """
spark = session(4)
register(spark, "py_plus", plus_one)
assert [r[0] for r in spark.sql("SELECT py_plus(id) FROM range(3) ORDER BY 1").collect()] == [1, 2, 3]
pid = os.fork()
if pid == 0:
    code = 1
    try:
        child = session(4)
        register(child, "py_plus", plus_one)
        got = [r[0] for r in child.sql("SELECT py_plus(id) FROM range(3) ORDER BY 1").collect()]
        code = 0 if got == [1, 2, 3] else 2
    finally:
        os._exit(code)
_, status = os.waitpid(pid, 0)
print(os.waitstatus_to_exitcode(status))
"""


def test_os_fork_after_a_query_runs_udfs_in_a_child_with_a_new_session() -> None:
    assert ok(run_child(FORK, timeout=90)).strip().splitlines()[-1] == "0"


# --- 6.5: Spark semantics ---


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


def test_a_udf_raising_on_rows_of_an_untaken_case_branch_still_raises(spark: SparkSession) -> None:
    # Spark evaluates a Python UDF on every row of its input, before the CASE (the `udf` docstring on evaluation:
    # "no guarantee of the order of evaluation of subexpressions"; `ExtractPythonUDFs`).
    def boom(x: int) -> int:
        if x < 0:
            raise ValueError("negative")
        return x

    f = F.udf(boom, LongType())
    spark.udf.register("boom_case", boom, LongType())
    # The call runs on every row, so it raises although `id >= 0` never takes the ELSE branch.
    with pytest.raises(Exception, match="negative"):
        spark.range(3).selectExpr("CASE WHEN id >= 0 THEN id ELSE boom_case(-1) END AS v").collect()
    with pytest.raises(Exception, match="negative"):
        spark.range(3).select(F.when(F.col("id") >= 0, F.col("id")).otherwise(f(F.lit(-1))).alias("v")).collect()


def test_row_order_and_alignment_hold_across_slicing(spark: SparkSession) -> None:
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "7")
    try:
        twice = F.udf(lambda x: x * 2, LongType())
        rows = spark.range(0, 1000, 1, 3).select("id", twice("id").alias("t"), F.col("id").alias("again")).collect()
        assert len(rows) == 1000
        assert sorted(r[0] for r in rows) == list(range(1000))
        assert all(r[1] == r[0] * 2 and r[2] == r[0] for r in rows)
        single = spark.range(0, 100).select("id", twice("id").alias("t")).collect()
        assert sorted((r[0], r[1]) for r in single) == [(i, i * 2) for i in range(100)]
    finally:
        spark.conf.unset("spark.sql.execution.arrow.maxRecordsPerBatch")


def _explain(df: Any) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df.explain(True)
    return buffer.getvalue()


def test_as_nondeterministic_blocks_filter_pushdown_through_the_udf(spark: SparkSession) -> None:
    deterministic = F.udf(lambda x: x + 1, LongType())
    nondeterministic = F.udf(lambda x: x + 1, LongType()).asNondeterministic()

    def physical(f: Any) -> list[str]:
        df = spark.range(10).select("id", f("id").alias("v")).filter("id > 5")
        return _explain(df).split("== Physical Plan ==")[1].strip().splitlines()

    pushed = physical(deterministic)
    assert any("PythonEval" in line for line in pushed)
    assert any("PushedFilters" in line for line in pushed), pushed  # the filter reached the scan, below the UDF
    assert not pushed[0].startswith("Filter"), pushed

    blocked = physical(nondeterministic)
    assert blocked[0].startswith("Filter"), blocked  # the filter stays above the UDF
    assert not any("PushedFilters" in line for line in blocked), blocked
    assert [r[0] for r in spark.range(10).select(nondeterministic("id")).filter("`<lambda>(id)` > 6").collect()]
