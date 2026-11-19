# SPDX-License-Identifier: Apache-2.0
"""Ctrl-C and a kernel interrupt stop a running query (NB-1): the action raises ``KeyboardInterrupt``
and the query is cancelled, in a child process so a missed interrupt cannot end the test run."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

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
    # The first batches arrive at once; the rest of the result takes minutes.
    slow = spark.range(0, 3000000000).filter("id % 1000000007 = 0")
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
    big = spark.range(0, 50000000).selectExpr("id * 2 AS id")
    threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    try:
        big.write.mode("overwrite").saveAsTable("t")
    except KeyboardInterrupt:
        pass
    # Either the old rows or the new ones, never a dropped or emptied table.
    n = spark.sql("SELECT count(*) AS n FROM t").collect()[0]["n"]
    print(f"rows {n}", flush=True)
    assert n in (5, 50000000), n
    """
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
