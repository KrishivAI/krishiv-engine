# SPDX-License-Identifier: Apache-2.0
"""``krishiv.ui()``: the web console of an embedded session (ADR 0024)."""

from __future__ import annotations

import gc
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import pytest

import krishiv
from krishiv import SparkSession, _native
from krishiv.errors import UnsupportedOperationException, unsupported_feature

pytestmark = pytest.mark.skipif(not _native.__ui__, reason="a build without the `ui` feature")


def _bindable(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


def _get(console: krishiv.Console, path: str, token: str | None = None) -> tuple[int, bytes]:
    request = urllib.request.Request(f"http://127.0.0.1:{console.port}{path}")
    if token is not None:
        request.add_header("authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _json(console: krishiv.Console, path: str) -> Any:
    status, body = _get(console, path, console.token)
    assert status == 200, body
    return json.loads(body)


@pytest.fixture
def spark() -> Any:
    session = SparkSession.builder.create()
    yield session
    session.stop()


def test_ui_starts_the_console_and_prints_the_url(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    console = krishiv.ui(spark)
    assert isinstance(console, krishiv.Console)
    assert console.running
    assert console.url == f"http://127.0.0.1:{console.port}/?token={console.token}"
    assert len(console.token) == 64
    assert capsys.readouterr().out == f"Krishiv console: {console.url}\n"
    # The page is the console's, and it is on the loopback interface.
    status, body = _get(console, "/")
    assert status == 200
    assert b"<script" in body
    assert not _bindable(console.port)


def test_requests_need_the_token(spark: SparkSession) -> None:
    console = krishiv.ui(spark)
    assert _get(console, "/api/internal/health")[0] == 401
    assert _get(console, "/api/internal/health", "0" * 64)[0] == 401
    health = _json(console, "/api/internal/health")
    assert health["mode"] == "embedded"
    assert health["session_id"] == console.session_id


def test_the_console_shows_its_own_session_only(spark: SparkSession) -> None:
    other = SparkSession.builder.create()
    try:
        other.sql("SELECT 1 AS from_the_other_session").collect()
        spark.sql("SELECT 1 AS from_my_session").collect()
        console = krishiv.ui(spark)
        texts = [q["query_text"] for q in _json(console, "/api/internal/history")["queries"]]
        assert any("from_my_session" in t for t in texts)
        assert not any("from_the_other_session" in t for t in texts)
    finally:
        other.stop()


def test_a_second_call_returns_the_running_console(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    first = krishiv.ui(spark)
    capsys.readouterr()
    assert krishiv.ui(spark) is first
    assert krishiv.ui(spark, port=first.port) is first
    assert capsys.readouterr().out == ""  # the URL is printed when it starts, not again


def test_a_named_port_and_a_restart_on_another(spark: SparkSession) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        # While the port is still held: the first console's own free port can then never be this one.
        first = krishiv.ui(spark)
    second = krishiv.ui(spark, port=port)
    assert second.port == port
    assert second.running
    assert not first.running
    assert second.token != first.token
    # A port that is taken is an error naming it, and the running console stays.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        taken = s.getsockname()[1]
        with pytest.raises(RuntimeError, match=f"cannot start the console: bind 127.0.0.1:{taken}"):
            krishiv.ui(spark, port=taken)
    assert second.running
    assert krishiv.ui(spark) is second


def test_an_engine_error_at_start_keeps_its_class(spark: SparkSession, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object) -> None:
        raise unsupported_feature("a console here")

    monkeypatch.setattr(_native, "Console", refuse)
    with pytest.raises(UnsupportedOperationException) as raised:
        krishiv.ui(spark)
    assert raised.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
    assert raised.value.getSqlState() == "0A000"
    assert spark._console is None


def test_stop_waits_for_a_start_in_progress() -> None:
    session = SparkSession.builder.create()
    console = krishiv.ui(session)
    stopping = threading.Thread(target=session.stop)
    # A start holds the session's console lock from reading the running console to storing the new one.
    with session._console_lock:
        stopping.start()
        stopping.join(0.2)
        assert stopping.is_alive()
        assert console.running
    stopping.join(30)
    assert not stopping.is_alive()
    assert not console.running
    assert session._console is None


def test_session_stop_stops_the_console() -> None:
    session = SparkSession.builder.create()
    console = krishiv.ui(session)
    port = console.port
    session.stop()
    assert not console.running
    assert _bindable(port)
    console.stop()  # stopping a stopped console does nothing


def test_stop_and_with_block(spark: SparkSession) -> None:
    with krishiv.ui(spark) as console:
        port = console.port
        assert console.running
    assert not console.running
    assert _bindable(port)
    # The session is still usable, and a new call starts a new console.
    assert spark.sql("SELECT 1").collect()[0][0] == 1
    again = krishiv.ui(spark)
    assert again is not console
    assert again.running


def test_a_dropped_session_takes_its_console_with_it() -> None:
    session = SparkSession.builder.create()
    port = krishiv.ui(session).port
    del session
    gc.collect()
    deadline = time.monotonic() + 10
    while not _bindable(port) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _bindable(port)


def test_ui_without_an_argument_uses_the_active_session() -> None:
    session = SparkSession.builder.getOrCreate()
    try:
        console = krishiv.ui()
        assert console is session._console
        assert console.session_id == _json(console, "/api/internal/health")["session_id"]
    finally:
        session.stop()


def test_the_editor_runs_in_the_session(spark: SparkSession) -> None:
    spark.sql("CREATE TEMPORARY VIEW console_view AS SELECT 41 + 1 AS answer")
    console = krishiv.ui(spark)
    request = urllib.request.Request(
        f"http://127.0.0.1:{console.port}/api/internal/sql",
        data=json.dumps({"sql": "SELECT answer FROM console_view"}).encode(),
        headers={"authorization": f"Bearer {console.token}", "content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        run = json.loads(response.read())
    deadline = time.monotonic() + 30
    status: Any = None
    while time.monotonic() < deadline:
        status = _json(console, f"/api/internal/sql/{run['run_id']}")
        if status.get("state") != "running":
            break
        time.sleep(0.05)
    assert status["state"] == "finished", status
    assert status["rows"] == [[42]]


def test_repr_hides_the_token_and_html_links_the_url(spark: SparkSession) -> None:
    console = krishiv.ui(spark)
    assert console.token not in repr(console)
    assert f"port={console.port}" in repr(console)
    html = console._repr_html_()
    assert f'href="http://127.0.0.1:{console.port}/?token={console.token}"' in html
    console.stop()
    assert "stopped" in console._repr_html_()
    assert "running=False" in repr(console)


def test_path_is_refused_for_the_console(spark: SparkSession, tmp_path: Any) -> None:
    untyped: Any = krishiv.ui  # no overload takes a session and a path
    with pytest.raises(TypeError, match="path belongs to a profile view"):
        untyped(spark, path=tmp_path / "x.html")
    assert spark._console is None


def test_a_profile_source_still_gives_the_offline_view(spark: SparkSession) -> None:
    view = krishiv.ui(spark.sql("SELECT 1"))
    assert isinstance(view, krishiv.ProfileView)
    assert spark._console is None


def test_port_is_refused_for_a_profile_view() -> None:
    untyped: Any = krishiv.ui  # no overload takes a profile and a port
    with pytest.raises(TypeError, match="port belongs to the console"):
        untyped({"operators": []}, port=4041)
