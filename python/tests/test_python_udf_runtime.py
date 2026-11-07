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
    return pc.add(x, 1)


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
        return pc.add(x, 1)

    spark = _spark()
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "10")
    spark._native.register_python_udf("py_counted", ARROW_BATCHED, "bigint", True, thread_safe, counted)
    rows = spark.sql("SELECT py_counted(id) FROM range(0, 400, 1, 8)").collect()
    assert sorted(_ids(rows)) == list(range(1, 401))
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


def test_a_wrapper_returning_no_pyarrow_array_is_refused() -> None:
    spark = _spark()
    spark._native.register_python_udf("py_list", ARROW_BATCHED, "bigint", True, True, lambda x: list(range(len(x))))
    with pytest.raises(Exception, match=r"pyarrow\.Array"):
        spark.sql("SELECT py_list(id) FROM range(3)").collect()
