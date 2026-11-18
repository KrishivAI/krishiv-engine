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
