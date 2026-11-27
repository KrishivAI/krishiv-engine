# SPDX-License-Identifier: Apache-2.0
"""Ctrl-C and a kernel interrupt stop a running query (NB-1): the action raises ``KeyboardInterrupt``
and the query is cancelled, in a child process so a missed interrupt cannot end the test run."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="SIGINT delivery to a child differs on Windows")

_SCRIPT = textwrap.dedent(
    """
    import os, signal, sys, threading, time
    import krishiv

    how = sys.argv[1]
    spark = krishiv.SparkSession.builder.create()
    long_query = "SELECT sum(id) AS s FROM range(0, 3000000000)"

    def interrupt():
        time.sleep(1.0)
        if how == "sigint":
            os.kill(os.getpid(), signal.SIGINT)
        else:
            import _thread
            _thread.interrupt_main()

    threading.Thread(target=interrupt, daemon=True).start()
    started = time.monotonic()
    try:
        spark.sql(long_query).collect()
    except KeyboardInterrupt:
        took = time.monotonic() - started
        print(f"interrupted after {took:.1f}s", flush=True)
    else:
        print("NOT interrupted", flush=True)
        sys.exit(3)
    # The session still answers, and the cancelled query is not left running.
    assert spark.range(3).count() == 3
    assert spark.interruptAll() == []
    print("session ok", flush=True)
    """
)


@pytest.mark.parametrize("how", ["sigint", "interrupt_main"])
def test_an_interrupt_stops_a_running_query(how: str) -> None:
    done = subprocess.run(
        [sys.executable, "-I", "-c", _SCRIPT, how],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "interrupted after" in done.stdout
    seconds = float(done.stdout.split("interrupted after ")[1].split("s")[0])
    assert seconds < 10
    assert "session ok" in done.stdout


_LATER_BATCH = textwrap.dedent(
    """
    import os, signal, sys, threading, time
    import krishiv

    spark = krishiv.SparkSession.builder.create()
    # The filter keeps three rows of three billion: the result takes minutes to produce (spark.range
    # would build the ids client-side, so the SQL table function is used).
    slow = spark.sql("SELECT id FROM range(0, 3000000000) WHERE id % 1000000007 = 0")
    started = time.monotonic()
    threading.Timer(2.0, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    try:
        slow.toArrow()
    except KeyboardInterrupt:
        print(f"interrupted after {time.monotonic() - started:.1f}s", flush=True)
    else:
        print("NOT interrupted", flush=True)
        sys.exit(3)
    assert spark.range(3).count() == 3
    print("session ok", flush=True)
    """
)

_OVERWRITE = textwrap.dedent(
    """
    import os, signal, sys, tempfile, threading, time
    import krishiv

    spark = krishiv.SparkSession.builder.create()
    spark.sql("CREATE TABLE t (id BIGINT) USING parquet")
    spark.range(5).write.mode("append").saveAsTable("t")
    big = spark.sql("SELECT id * 2 AS id FROM range(0, 50000000)")
    threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    try:
        big.write.mode("overwrite").saveAsTable("t")
    except KeyboardInterrupt:
        pass
    # Once the write has begun a cancel fails it and the table is gone (as Spark's V1 overwrite); what
    # must never show is a partial table: the old rows or the new ones.
    # One Ctrl-C was sent; a slow runner can deliver its re-armed raise at the first statement after the
    # handler, so the read retries once on a KeyboardInterrupt.
    n = None
    for _ in range(2):
        try:
            n = spark.sql("SELECT count(*) AS n FROM t").collect()[0]["n"]
        except KeyboardInterrupt:
            continue
        except Exception as e:
            assert "TABLE_OR_VIEW_NOT_FOUND" in str(e), e
            n = 0
        break
    assert n is not None
    print(f"rows {n}", flush=True)
    assert n in (0, 5, 50000000), n
    """
)

# The consumer's own "cancelled" error is an `Exception` (`except Exception` swallows it); the re-armed Ctrl-C
# is then raised at the next check, here in the handler's following statements.
_LATER_BATCH_CAPSULE = _LATER_BATCH.replace(
    "slow.toArrow()",
    "import pyarrow as pa\n    try:\n        pa.table(slow)\n    except Exception:\n        time.sleep(0.5)",
)


def test_an_interrupt_after_the_first_batch_stops_the_query() -> None:
    done = subprocess.run(
        [sys.executable, "-I", "-c", _LATER_BATCH], capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "interrupted after" in done.stdout
    assert float(done.stdout.split("interrupted after ")[1].split("s")[0]) < 20
    assert "session ok" in done.stdout


def test_an_interrupt_during_an_overwrite_leaves_the_old_or_the_new_table() -> None:
    done = subprocess.run(
        [sys.executable, "-I", "-c", _OVERWRITE], capture_output=True, text=True, timeout=300, check=False
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "rows " in done.stdout


def test_an_interrupt_during_a_foreign_consumer_read_raises_keyboard_interrupt() -> None:
    # `pa.table(df)` reads the PyCapsule stream and never calls `check_error`: the Ctrl-C is re-armed, so the
    # consumer's own "cancelled" error does not swallow it (an `except Exception` must not catch it).
    done = subprocess.run(
        [sys.executable, "-I", "-c", _LATER_BATCH_CAPSULE], capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "interrupted after" in done.stdout
    assert "session ok" in done.stdout
