# SPDX-License-Identifier: Apache-2.0
"""The embedded Python UDF runtime (design P2-T059 unit 2b), driven through the native
`Session.register_python_udf` with plain pyarrow wrappers (the `udf`/`pandas_udf` facades are unit 2c).

The pool is process-wide and sized by the first registration, so every session here asks for 4 threads:
that is what makes the `thread_safe=False` test able to fail.
"""

from __future__ import annotations

import threading
import time

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from krishiv import _native
from krishiv.sql.session import SparkSession

ARROW_BATCHED = 101


def _spark() -> SparkSession:
    config = [("krishiv.python.udf.threads", "4")]
    return SparkSession(_native.Session(config), config)


def _plus_one(x: pa.Array) -> pa.Array:
    return pc.call_function("add", [x, 1])


def _ids(rows) -> list:
    return [r[0] for r in rows]


def test_a_udf_runs_over_every_row_by_name_and_version() -> None:
    spark = _spark()
    version = spark._native.register_python_udf("py_plus", ARROW_BATCHED, "bigint", True, True, _plus_one)
    assert version == 1
    rows = spark.sql("SELECT py_plus(id) FROM range(5) ORDER BY 1").collect()
    assert _ids(rows) == [1, 2, 3, 4, 5]


def test_a_raising_udf_fails_with_python_exception_and_its_traceback() -> None:
    # Spark 4.2.0: PYTHON_EXCEPTION, SQLSTATE 38000, the traceback ending with the exception line
    # (research/spark-notes/python-udf-errors.md).
    def boom(x: pa.Array) -> pa.Array:
        raise ValueError("boom")

    spark = _spark()
    spark._native.register_python_udf("py_boom", ARROW_BATCHED, "bigint", True, True, boom)
    with pytest.raises(Exception, match="PYTHON_EXCEPTION") as caught:
        spark.sql("SELECT py_boom(id) FROM range(3)").collect()
    text = str(caught.value)
    assert "PYTHON_EXCEPTION" in text, text
    assert "An exception was thrown from the Python worker" in text, text
    assert "ValueError: boom" in text, text


@pytest.mark.parametrize("thread_safe", [False, True])
def test_thread_safe_false_never_runs_two_calls_at_once(thread_safe: bool) -> None:
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}

    def counted(x: pa.Array) -> pa.Array:
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.02)
        with lock:
            state["active"] -= 1
        return pc.call_function("add", [x, 1])

    spark = _spark()
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "10")
    spark._native.register_python_udf("py_counted", ARROW_BATCHED, "bigint", True, thread_safe, counted)
    # Run on a daemon thread, bounded: a lock waited for while holding the GIL would hang (review B1).
    out: dict = {}
    runner = threading.Thread(
        target=lambda: out.setdefault("rows", spark.sql("SELECT py_counted(id) FROM range(0, 400, 1, 8)").collect()),
        daemon=True,
    )
    runner.start()
    runner.join(60)
    assert not runner.is_alive(), "the query did not finish within 60 s (deadlock?)"
    assert sorted(_ids(out["rows"])) == list(range(1, 401))
    if not thread_safe:
        assert state["peak"] == 1, state
    else:
        # The control: with the lock off, a 4-thread pool does overlap calls (on a GIL build the
        # sleep releases the GIL), so the assertion above can fail.
        assert state["peak"] > 1, state


def test_a_result_of_another_type_is_refused() -> None:
    spark = _spark()
    spark._native.register_python_udf(
        "py_wrong", ARROW_BATCHED, "bigint", True, True, lambda x: pa.array(["a"] * len(x))
    )
    with pytest.raises(Exception, match="was declared"):
        spark.sql("SELECT py_wrong(id) FROM range(3)").collect()


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_forked_child_runs_udfs_on_a_pool_of_its_own() -> None:
    # The child's copy of the parent's pool has no threads (review B2): its call must not hang. The child
    # uses a session of its own: a parent's session does not run queries after a fork even without UDFs
    # (its executor's threads stayed in the parent).
    import os

    spark = _spark()
    spark._native.register_python_udf("py_plus_fork", ARROW_BATCHED, "bigint", True, True, _plus_one)
    assert _ids(spark.sql("SELECT py_plus_fork(id) FROM range(2) ORDER BY 1").collect()) == [1, 2]
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            child = _spark()
            child._native.register_python_udf("py_plus_fork", ARROW_BATCHED, "bigint", True, True, _plus_one)
            got = _ids(child.sql("SELECT py_plus_fork(id) FROM range(3) ORDER BY 1").collect())
            code = 0 if got == [1, 2, 3] else 2
        finally:
            os._exit(code)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            assert os.waitstatus_to_exitcode(status) == 0
            return
        time.sleep(0.05)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    pytest.fail("the forked child's UDF call did not finish within 60 s")


def test_a_wrapper_returning_no_pyarrow_array_is_refused() -> None:
    spark = _spark()
    spark._native.register_python_udf("py_list", ARROW_BATCHED, "bigint", True, True, lambda x: list(range(len(x))))
    with pytest.raises(Exception, match=r"pyarrow\.Array"):
        spark.sql("SELECT py_list(id) FROM range(3)").collect()


def test_a_udf_that_runs_a_query_fails_instead_of_waiting_for_itself() -> None:
    # T098b B1: the call back into the engine would wait for the pool thread the UDF holds. PySpark raises
    # CONTEXT_ONLY_VALID_ON_DRIVER for a session used in worker code (pyspark/core/context.py 4.2.0
    # lines 2636-2647); the query then fails with PYTHON_EXCEPTION carrying it.
    spark = _spark()

    def nested(x: pa.Array) -> pa.Array:
        spark.sql("SELECT 1").collect()
        return pc.call_function("add", [x, 1])

    spark._native.register_python_udf("py_nested", ARROW_BATCHED, "bigint", True, True, nested)
    out: dict = {}

    def run() -> None:
        try:
            spark.sql("SELECT py_nested(id) FROM range(3)").collect()
        except Exception as e:
            out["error"] = str(e)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    runner.join(60)
    assert not runner.is_alive(), "a UDF running a query hung (no reentrancy refusal)"
    text = out.get("error", "")
    assert "PYTHON_EXCEPTION" in text, text
    assert "CONTEXT_ONLY_VALID_ON_DRIVER" in text, text


def test_a_running_call_of_a_failed_query_is_interrupted() -> None:
    # Design §2.5 "Cancel" (T098b I2): when the query fails, another call still running in Python gets
    # an asynchronous KeyboardInterrupt instead of holding its pool thread until it ends. Two one-row
    # batches on two pool threads: one loops (pure Python, so the interrupt is raised at its next
    # bytecode), the other fails after 0.5 s.
    seen: dict = {}

    def stuck_or_boom(x: pa.Array) -> pa.Array:
        if x[0].as_py() == 1:
            time.sleep(0.5)
            raise ValueError("boom")
        started = time.monotonic()
        seen["looping"] = started
        try:
            while time.monotonic() - started < 60:
                pass
        except KeyboardInterrupt:
            seen["interrupted"] = time.monotonic() - started
            raise
        seen["ran_out"] = True
        return x

    spark = _spark()
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1")
    spark._native.register_python_udf("py_stuck", ARROW_BATCHED, "bigint", True, True, stuck_or_boom)
    t0 = time.monotonic()
    with pytest.raises(Exception, match="ValueError: boom"):
        spark.sql("SELECT py_stuck(id) FROM range(0, 2, 1, 2)").collect()
    assert time.monotonic() - t0 < 30, "the failed query waited for the looping call"
    # The interrupt lands on the looping call within a few seconds of the failure.
    deadline = time.monotonic() + 15
    while "interrupted" not in seen and time.monotonic() < deadline:
        time.sleep(0.05)
    assert "looping" in seen, seen
    assert "interrupted" in seen, seen
    assert "ran_out" not in seen, seen
    # The pool thread serves later calls normally: no interrupt is left pending for them.
    spark._native.register_python_udf("py_plus_after", ARROW_BATCHED, "bigint", True, True, _plus_one)
    for _ in range(20):
        rows = spark.sql("SELECT py_plus_after(id) FROM range(0, 8, 1, 4) ORDER BY 1").collect()
        assert _ids(rows) == list(range(1, 9))
