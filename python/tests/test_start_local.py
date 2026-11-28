# SPDX-License-Identifier: Apache-2.0
"""``krishiv.start_local()`` (P3-T016): the in-process Connect server for stock PySpark."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import time

import pytest

import krishiv
import krishiv.sql.functions


def _bindable(port: int) -> bool:
    # SO_REUSEADDR: a closed connection's TIME_WAIT entry (macOS keeps the server side of the
    # test's own connection there) must not make a freed port look taken; a live listener still fails.
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


def _threads() -> int:
    return len(os.listdir("/proc/self/task"))


def _quiesced_threads() -> int:
    """The thread count once it has not changed for 300 ms (engine pool threads leave just after stop)."""
    last, since = _threads(), time.monotonic()
    while time.monotonic() - since < 0.3:
        time.sleep(0.02)
        now = _threads()
        if now != last:
            last, since = now, time.monotonic()
    return last


def _wait_threads(expected: int) -> None:
    """Wait for the count to fall to ``expected`` and settle there.

    A leaked server thread leaves the count above ``expected`` and fails. A count below it is not a leak: a
    thread that was alive at the baseline (an engine pool thread of an earlier test) can exit on its own.
    """
    deadline = time.monotonic() + 10
    while _threads() > expected and time.monotonic() < deadline:
        time.sleep(0.01)
    settled = _quiesced_threads()
    assert settled <= expected, f"{settled - expected} thread(s) outlived the stopped servers"


def test_url_names_a_loopback_port_and_a_token() -> None:
    with krishiv.start_local() as server:
        assert server.running
        assert server.url == f"sc://localhost:{server.port}/;token={server.token}"
        assert len(server.token) == 64
        # Loopback only: the listener is on 127.0.0.1 and not on any other address.
        with socket.create_connection(("127.0.0.1", server.port), timeout=5):
            pass
        assert not _bindable(server.port)
    assert not server.running
    assert _bindable(server.port)


def test_every_start_has_its_own_token_and_port() -> None:
    with krishiv.start_local() as a, krishiv.start_local() as b:
        assert a.token != b.token
        assert a.port != b.port


@pytest.mark.skipif(not os.path.isdir("/proc/self/task"), reason="counts threads through /proc")
def test_start_stop_100_times() -> None:
    krishiv.start_local().stop()  # lazily created process-wide threads exist after one start
    before = _quiesced_threads()
    ports = []
    for _ in range(100):
        server = krishiv.start_local()
        ports.append(server.port)
        server.stop()
        server.stop()  # a second stop does nothing
        assert not server.running
    assert all(_bindable(p) for p in ports)
    _wait_threads(before)


def test_interpreter_exit_stops_a_running_server() -> None:
    # The probe's atexit handler is registered before ``krishiv`` is imported, so it runs after
    # krishiv's own (LIFO): the server must already be gone, port and thread, while the process lives.
    code = textwrap.dedent(
        """
        import atexit, os, socket, time

        state = {}

        def probe():
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                s = socket.socket()
                try:
                    s.bind(("127.0.0.1", state["port"]))
                    print("port free", flush=True)
                    return
                except OSError:
                    time.sleep(0.01)
                finally:
                    s.close()
            print("port still bound", flush=True)

        atexit.register(probe)
        import krishiv

        server = krishiv.start_local()
        state["port"] = server.port
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.strip() == "port free"


def test_str_and_repr_do_not_show_the_token() -> None:
    with krishiv.start_local() as server:
        assert server.token not in str(server)
        assert server.token not in repr(server)
