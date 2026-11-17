# SPDX-License-Identifier: Apache-2.0
"""The pool worker's program (``krishiv._udf.pool_worker``) and the worker-only branches of ``_host`` and
``types`` (design ``P3-T014-connect-udfs-sandbox.md`` §3.1, §3.3), run in this process.

The server runs the same code in a separate interpreter (``krishiv-udf-pool``'s ``tests/pool.rs`` starts real
workers with ``pyspark-client``). Here the worker is driven over a real ``AF_UNIX``/``SOCK_SEQPACKET`` socket
pair with real memfds, as the server drives it; the wheel's test environment has no ``pyspark``, so the two
``pyspark`` names the worker imports (``cloudpickle`` and ``pyspark.errors``) are stand-ins: ``pickle`` and the
``krishiv.errors`` classes, which take the same arguments.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import socket
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")

import krishiv.errors as kerrors  # noqa: E402
from krishiv._udf import _host, pool_worker  # noqa: E402
from krishiv._udf import types as udf_types  # noqa: E402
from krishiv.sql.types import LongType, StructField, StructType  # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="memfds are Linux only")

ARROW_BATCHED = 101
PYTHON_VER = f"{sys.version_info[0]}.{sys.version_info[1]}"


def _plus_one(x: Any) -> Any:
    return None if x is None else x + 1


def _boom(x: Any) -> Any:
    raise ValueError("界" * (48 * 1024))


@pytest.fixture
def fake_pyspark(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """``pyspark`` with the names the worker imports: ``cloudpickle`` (``pickle`` here) and the errors."""
    pyspark = types.ModuleType("pyspark")
    pyspark.__version__ = "4.2.1"  # ty: ignore[unresolved-attribute]
    pyspark.cloudpickle = pickle  # ty: ignore[unresolved-attribute]
    errors = types.ModuleType("pyspark.errors")
    for name in (
        "PySparkImportError",
        "PySparkNotImplementedError",
        "PySparkRuntimeError",
        "PySparkTypeError",
        "PySparkValueError",
        "UnsupportedOperationException",
    ):
        setattr(errors, name, getattr(kerrors, name))
    pyspark.errors = errors  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "pyspark", pyspark)
    monkeypatch.setitem(sys.modules, "pyspark.errors", errors)
    return pyspark


@pytest.fixture
def pair() -> Iterator[tuple[socket.socket, socket.socket]]:
    """``(server end, worker end)`` of a control socket."""
    server, worker = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    yield server, worker
    server.close()
    worker.close()


def _memfd(data: bytes) -> int:
    fd = os.memfd_create("test-udf", os.MFD_CLOEXEC)
    os.write(fd, data)
    return fd


def _send(sock: socket.socket, msg: dict[str, Any], data: bytes | None = None) -> None:
    raw = json.dumps(msg).encode()
    if data is None:
        sock.send(raw)
        return
    fd = _memfd(data)
    try:
        socket.send_fds(sock, [raw], [fd])
    finally:
        os.close(fd)


def _recv(sock: socket.socket) -> tuple[dict[str, Any], list[int]]:
    data, fds, _flags, _addr = socket.recv_fds(sock, 1 << 20, 4)
    return json.loads(data), fds


def _udf(func: Any, python_ver: str = PYTHON_VER) -> tuple[dict[str, Any], bytes]:
    command = pickle.dumps((func, LongType()))
    return {"t": "udf", "id": 7, "eval_type": ARROW_BATCHED, "python_ver": python_ver, "len": len(command)}, command


def _batch(values: list[int | None]) -> tuple[dict[str, Any], bytes]:
    batch = pa.RecordBatch.from_arrays([pa.array(values, pa.int64())], names=["a"])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    data = sink.getvalue().to_pybytes()
    return {"t": "batch", "udf": 7, "len": len(data), "conf": {}}, data


def _result(sock: socket.socket) -> list[Any]:
    msg, fds = _recv(sock)
    assert msg["t"] == "result", msg
    (fd,) = fds
    try:
        data = os.pread(fd, msg["len"], 0)
    finally:
        os.close(fd)
    return pa.ipc.open_stream(data).read_all().column(0).to_pylist()


def test_main_greets_loads_runs_and_ends_at_eof(
    fake_pyspark: types.ModuleType, pair: tuple[socket.socket, socket.socket]
) -> None:
    server, worker = pair
    _send(server, *_udf(_plus_one))
    _send(server, *_batch([1, None, 41]))
    server.shutdown(socket.SHUT_WR)
    pool_worker.main(worker)
    hello, _ = _recv(server)
    assert hello["t"] == "hello"
    assert hello["wire"] == pool_worker.WIRE
    assert hello["pyspark"] == "4.2.1"
    assert hello["python"].startswith(PYTHON_VER)
    assert _recv(server)[0] == {"t": "ready"}
    assert _result(server) == [2, None, 42]


def test_main_reports_a_missing_package_and_exits(
    monkeypatch: pytest.MonkeyPatch, pair: tuple[socket.socket, socket.socket]
) -> None:
    server, worker = pair
    monkeypatch.setitem(sys.modules, "pyspark", None)
    with pytest.raises(SystemExit) as exit_info:
        pool_worker.main(worker)
    assert exit_info.value.code == 1
    msg, _ = _recv(server)
    assert msg["t"] == "error"
    assert msg["kind"] == "python"
    assert "pyspark" in msg["message"]


def test_python_errors_keep_the_worker(
    fake_pyspark: types.ModuleType, pair: tuple[socket.socket, socket.socket]
) -> None:
    server, worker = pair
    header, command = _udf(_plus_one, python_ver="3.9")
    _send(server, header, command)  # the client's Python differs: Spark's PYTHON_VERSION_MISMATCH
    _send(server, {**header, "len": 0}, b"")  # an empty command cannot be unpickled
    _send(server, *_udf(_plus_one))
    _send(server, {**_batch([1])[0], "udf": 99}, _batch([1])[1])  # an id never loaded
    _send(server, *_batch([5]))
    server.shutdown(socket.SHUT_WR)
    pool_worker._Worker(worker).serve()
    mismatch, _ = _recv(server)
    assert mismatch["kind"] == "python"
    assert "PYTHON_VERSION_MISMATCH" in mismatch["message"]
    empty, _ = _recv(server)
    assert empty["kind"] == "python"
    assert "Error" in empty["message"]
    assert _recv(server)[0] == {"t": "ready"}
    unknown, _ = _recv(server)
    assert unknown["kind"] == "python"
    assert "KeyError: 99" in unknown["message"]
    assert _result(server) == [6]


def test_a_huge_non_ascii_traceback_fits_one_message(
    fake_pyspark: types.ModuleType, pair: tuple[socket.socket, socket.socket]
) -> None:
    # 48 Ki characters, each escaped by JSON as six bytes: more than one SOCK_SEQPACKET message can hold.
    server, worker = pair
    _send(server, *_udf(_boom))
    _send(server, *_batch([1]))
    _send(server, *_batch([2]))
    server.shutdown(socket.SHUT_WR)
    pool_worker._Worker(worker).serve()
    assert _recv(server)[0] == {"t": "ready"}
    for _ in range(2):  # the worker answered the first error and served the next batch
        data, fds, _flags, _addr = socket.recv_fds(server, 1 << 20, 4)
        assert not fds
        assert len(data) <= pool_worker.MAX_MESSAGE
        msg = json.loads(data)
        assert msg["kind"] == "python"
        assert msg["message"].startswith("...\n")
        assert msg["message"].endswith("界" * 100)


def test_a_huge_error_message_is_cut_to_the_message_budget(pair: tuple[socket.socket, socket.socket]) -> None:
    server, worker = pair
    for text in ("\x00" * (200 * 1024), "a" * (200 * 1024) + "end", "short"):
        pool_worker._error(worker, "protocol", text)
        data = server.recv(1 << 20)
        assert len(data) <= pool_worker.MAX_MESSAGE
        message = json.loads(data)["message"]
        assert text.endswith(message.removeprefix("...\n"))
        assert (message == text) == (text == "short")


@pytest.mark.parametrize(
    ("raw", "fds", "text"),
    [
        (b"{not json", 1, "unreadable message"),
        (json.dumps({"no": "kind"}).encode(), 1, "unreadable message"),
        (json.dumps({"t": "what"}).encode(), 1, "unexpected message 'what' with 1 descriptors"),
        (json.dumps({"t": "batch"}).encode(), 0, "unexpected message 'batch' with 0 descriptors"),
        (json.dumps({"t": "batch"}).encode(), 2, "unexpected message 'batch' with 2 descriptors"),
        (json.dumps({"t": "task"}).encode(), 1, "unexpected message 'task' with 1 descriptors"),
        (b"x" * (pool_worker.MAX_MESSAGE + 1), 0, "truncated"),
        (json.dumps({"t": "batch"}).encode(), 5, "truncated"),
    ],
)
def test_unreadable_messages_end_the_worker(
    pair: tuple[socket.socket, socket.socket], raw: bytes, fds: int, text: str
) -> None:
    server, worker = pair
    attached = [_memfd(b"") for _ in range(fds)]
    try:
        socket.send_fds(server, [raw], attached) if attached else server.send(raw)
    finally:
        for fd in attached:
            os.close(fd)
    _send(server, *_batch([1]))  # never read: the worker stops at the bad message
    pool_worker._Worker(worker).serve()
    msg, _ = _recv(server)
    assert msg["t"] == "error"
    assert msg["kind"] == "protocol"
    assert text in msg["message"]


def test_a_failed_result_write_closes_its_memfd(
    fake_pyspark: types.ModuleType, monkeypatch: pytest.MonkeyPatch, pair: tuple[socket.socket, socket.socket]
) -> None:
    _server, worker = pair
    w = pool_worker._Worker(worker)
    header, command = _udf(_plus_one)
    fd = _memfd(command)
    try:
        w.load_udf(header, fd)
    finally:
        os.close(fd)
    closed: list[int] = []
    real_close = os.close

    def failing_write(_fd: int, _data: Any) -> int:
        raise OSError(28, "No space left on device")

    def recording_close(fd: int) -> None:
        closed.append(fd)
        real_close(fd)

    header, data = _batch([1])
    fd = _memfd(data)
    try:
        monkeypatch.setattr(pool_worker.os, "write", failing_write)
        monkeypatch.setattr(pool_worker.os, "close", recording_close)
        with pytest.raises(OSError, match="No space"):
            w.run_batch(header, fd)
    finally:
        monkeypatch.undo()
        os.close(fd)
    assert len(closed) == 1


def _load_as_worker(monkeypatch: pytest.MonkeyPatch, module: types.ModuleType, name: str) -> types.ModuleType:
    """A fresh copy of ``module`` imported as a pool worker imports it (under the stub ``krishiv`` package)."""
    stub = types.ModuleType("krishiv")
    stub._UDF_WORKER = True  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "krishiv", stub)
    path = module.__file__
    assert path is not None
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    copy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copy)
    return copy


def test_the_worker_host_takes_its_names_from_pyspark(
    fake_pyspark: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = _load_as_worker(monkeypatch, _host, "_worker_host")
    assert host.IN_WORKER
    assert host.PySparkRuntimeError is kerrors.PySparkRuntimeError
    err = host.not_implemented("relation UDFs")
    assert isinstance(err, kerrors.PySparkNotImplementedError)
    assert err.getCondition() == "NOT_IMPLEMENTED"
    assert err.getMessageParameters() == {"feature": "relation UDFs"}
    host.set_runner_conf({"spark.sql.session.timeZone": "UTC"})
    assert host.runner_conf() == {"spark.sql.session.timeZone": "UTC"}
    # The worker's types are pyspark's own (here: those the in-process module has under the same names).
    monkeypatch.setitem(sys.modules, "krishiv._udf._host", host)
    pyspark_globals = types.ModuleType("pyspark._globals")
    pyspark_globals._NoValue = udf_types._NoValue  # ty: ignore[unresolved-attribute]
    pyspark_sql = types.ModuleType("pyspark.sql")
    pyspark_types = types.ModuleType("pyspark.sql.types")
    pyspark_types.__getattr__ = lambda name: getattr(udf_types, name)  # ty: ignore[invalid-assignment]
    for name, mod in (
        ("pyspark._globals", pyspark_globals),
        ("pyspark.sql", pyspark_sql),
        ("pyspark.sql.types", pyspark_types),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    worker_types = _load_as_worker(monkeypatch, udf_types, "_worker_types")
    assert worker_types.LongType is udf_types.LongType
    assert worker_types.TimeType is udf_types.TimeType
    assert worker_types._NoValue is udf_types._NoValue


def test_the_in_process_host_keeps_a_set_conf_for_workers_only() -> None:
    assert not _host.IN_WORKER
    _host.set_runner_conf({"a": "b"})
    try:
        assert "a" not in _host.runner_conf()  # in-process, the conf is the native call's
    finally:
        _host.set_runner_conf({})


def test_in_process_stand_ins() -> None:
    assert repr(udf_types._NoValue) == "<no value>"
    assert udf_types.TimeType(3).simpleString() == "time(3)"
    assert udf_types.TimeType().precision == 6
    udt = udf_types.UserDefinedType()
    for call in (udt.sqlType, lambda: udt.serialize(1), lambda: udt.deserialize(1)):
        with pytest.raises(NotImplementedError):
            call()
    geometry = udf_types.GeometryType(4326)
    assert geometry.srid == 4326
    assert udf_types.GeographyType("ANY").srid == "ANY"
    with pytest.raises(NotImplementedError):
        geometry.toInternal(b"")
    for cls in (udf_types.Geometry, udf_types.Geography):
        with pytest.raises(NotImplementedError, match="spatial values"):
            cls.fromWKB(b"", 0)


def test_an_unknown_eval_type_is_not_implemented() -> None:
    from krishiv._udf.worker import make_wrapper

    with pytest.raises(kerrors.PySparkNotImplementedError) as err:
        make_wrapper(_plus_one, 999, LongType())
    assert err.value.getMessageParameters() == {"feature": "Python UDFs of eval type 999"}


# --- relation tasks (design ``P3-T085-embedded-python-udf-kinds.md`` §3.2) ---
MAP_ARROW = 207
GROUPED_AGG_PANDAS = 202
GROUPED_MAP_PANDAS = 201
V = StructType([StructField("v", LongType())])


def _double(batches: Any) -> Any:
    for b in batches:
        yield pa.record_batch([pa.compute.multiply(b.column(0), 2)], names=["v"])


def _first_batch_only(batches: Any) -> Any:
    yield next(batches)


def _explode(batches: Any) -> Any:
    raise ValueError("kaboom")


def _total(values: Any) -> int:
    return int(values.sum())


def _identity_group(df: Any) -> Any:
    return df


@pytest.fixture
def relation_conf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_host, "runner_conf", lambda: {"spark.sql.session.timeZone": "UTC"})


def _relation_udf(func: Any, eval_type: int) -> tuple[dict[str, Any], bytes]:
    command = pickle.dumps((func, V if eval_type != GROUPED_AGG_PANDAS else LongType()))
    return {"t": "udf", "id": 7, "eval_type": eval_type, "python_ver": PYTHON_VER, "len": len(command)}, command


def _task(key_count: int = 0, arity: int = 0, max_rows: int = 0, udf: int = 7) -> dict[str, Any]:
    return {"t": "task", "udf": udf, "key_count": key_count, "arity": arity, "max_rows": max_rows, "conf": {}}


def _frame(sock: socket.socket, values: list[int | None], side: int = 0) -> None:
    header, data = _batch(values)
    _send(sock, {"t": "frame", "side": side, "len": header["len"]}, data)


def _collect(sock: socket.socket) -> list[tuple[dict[str, Any], list[Any]]]:
    """Every message the worker sent, up to EOF, with the rows of each ``result``."""
    sock.setblocking(False)
    out: list[tuple[dict[str, Any], list[Any]]] = []
    while True:
        try:
            data, fds, _flags, _addr = socket.recv_fds(sock, 1 << 20, 4)
        except BlockingIOError:
            return out
        if not data:
            return out
        msg = json.loads(data)
        rows: list[Any] = []
        for fd in fds:
            try:
                rows = pa.ipc.open_stream(os.pread(fd, msg["len"], 0)).read_all().column(0).to_pylist()
            finally:
                os.close(fd)
        out.append((msg, rows))


def _serve(worker: socket.socket, server: socket.socket) -> list[tuple[dict[str, Any], list[Any]]]:
    server.shutdown(socket.SHUT_WR)
    pool_worker._Worker(worker).serve()
    return _collect(server)


def test_a_map_task_streams_its_results_as_frames_arrive(
    fake_pyspark: types.ModuleType, relation_conf: None, pair: tuple[socket.socket, socket.socket]
) -> None:
    server, worker = pair
    _send(server, *_relation_udf(_double, MAP_ARROW))
    _send(server, _task())
    _frame(server, [1, 2])
    _frame(server, [3])
    _send(server, {"t": "end"})
    sent = _serve(worker, server)
    kinds = [m["t"] for m, _ in sent]
    assert kinds == ["ready", "want", "result", "want", "result", "want", "done"]
    assert [rows for m, rows in sent if m["t"] == "result"] == [[2, 4], [6]]


def test_a_task_whose_udf_stops_reading_is_done_before_the_end(
    fake_pyspark: types.ModuleType, relation_conf: None, pair: tuple[socket.socket, socket.socket]
) -> None:
    server, worker = pair
    _send(server, *_relation_udf(_first_batch_only, MAP_ARROW))
    _send(server, _task())
    _frame(server, [1, 2])
    sent = _serve(worker, server)
    assert [m["t"] for m, _ in sent] == ["ready", "want", "result", "done"]
    assert sent[2][1] == [1, 2]


def test_an_aggregate_task_sends_a_group_value_per_want(
    fake_pyspark: types.ModuleType, relation_conf: None, pair: tuple[socket.socket, socket.socket]
) -> None:
    server, worker = pair
    _send(server, *_relation_udf(_total, GROUPED_AGG_PANDAS))
    _send(server, _task(arity=1))
    _frame(server, [1, 2])
    _send(server, {"t": "end_group"})
    _frame(server, [10])
    _send(server, {"t": "end_group"})
    _send(server, {"t": "end"})
    sent = _serve(worker, server)
    assert [rows for m, rows in sent if m["t"] == "result"] == [[3], [10]]
    assert sent[-1][0] == {"t": "done"}


def test_an_aggregate_task_with_a_row_cap_sends_each_value_at_once(
    fake_pyspark: types.ModuleType, relation_conf: None, pair: tuple[socket.socket, socket.socket]
) -> None:
    server, worker = pair
    _send(server, *_relation_udf(_total, GROUPED_AGG_PANDAS))
    _send(server, _task(arity=1, max_rows=1))
    _frame(server, [4])
    _send(server, {"t": "end_group"})
    _send(server, {"t": "end"})
    kinds = [m["t"] for m, _ in _serve(worker, server)]
    # the cap flushes the value as soon as the runner yields it, before the next want
    assert kinds == ["ready", "want", "want", "result", "want", "done"]


def test_results_wait_to_be_sent_together_up_to_the_row_cap(pair: tuple[socket.socket, socket.socket]) -> None:
    server, worker = pair
    results = pool_worker._Results(worker, 0, aggregate=True)
    assert results.max_rows == pool_worker.DEFAULT_MAX_ROWS
    results.flush()  # nothing pending: nothing sent
    results.add(pa.record_batch([pa.array([1])], names=["_0"]))
    results.add(pa.record_batch([pa.array([2, 3])], names=["_0"]))
    server.setblocking(False)
    with pytest.raises(BlockingIOError):
        server.recv(1)
    results.flush()
    assert [rows for _m, rows in _collect(server)] == [[1, 2, 3]]
    assert (results.pending, results.rows) == ([], 0)


def test_a_udf_error_in_a_task_keeps_the_worker(
    fake_pyspark: types.ModuleType, relation_conf: None, pair: tuple[socket.socket, socket.socket]
) -> None:
    server, worker = pair
    _send(server, *_relation_udf(_explode, MAP_ARROW))
    _send(server, _task())
    _send(server, _task(udf=99))  # an id never loaded
    sent = _serve(worker, server)
    assert [m["t"] for m, _ in sent] == ["ready", "error", "error"]
    assert sent[1][0]["kind"] == "python"
    assert "ValueError: kaboom" in sent[1][0]["message"]
    assert "KeyError: 99" in sent[2][0]["message"]


def test_a_task_for_a_scalar_udf_and_a_batch_for_a_relation_are_refused(
    fake_pyspark: types.ModuleType, relation_conf: None, pair: tuple[socket.socket, socket.socket]
) -> None:
    server, worker = pair
    _send(server, *_udf(_plus_one))
    _send(server, _task())
    sent = _serve(worker, server)
    assert [m["t"] for m, _ in sent] == ["ready", "error"]
    assert sent[1][0]["kind"] == "protocol"
    assert "a task for a scalar UDF" in sent[1][0]["message"]

    server2, worker2 = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        _send(server2, *_relation_udf(_double, MAP_ARROW))
        _send(server2, *_batch([1]))
        sent = _serve(worker2, server2)
    finally:
        server2.close()
        worker2.close()
    assert [m["t"] for m, _ in sent] == ["ready", "error"]
    assert sent[1][0]["kind"] == "python"
    assert "a batch for a relation UDF" in sent[1][0]["message"]


def test_a_frame_error_ends_the_worker_as_a_protocol_error(
    fake_pyspark: types.ModuleType, relation_conf: None, pair: tuple[socket.socket, socket.socket]
) -> None:
    server, worker = pair
    _send(server, *_relation_udf(_identity_group, GROUPED_MAP_PANDAS))
    _send(server, _task(key_count=1))
    _send(server, {"t": "end_group"})  # a group must start with a batch
    _send(server, *_udf(_plus_one))  # never read
    sent = _serve(worker, server)
    assert [m["t"] for m, _ in sent] == ["ready", "want", "error"]
    assert sent[2][0]["kind"] == "protocol"
    assert "a group starts with 'end_group'" in sent[2][0]["message"]


@pytest.mark.parametrize(
    ("replies", "text"),
    [
        ([b"{not json"], "unreadable frame"),
        ([json.dumps({"no": "kind"}).encode()], "unreadable frame"),
        ([json.dumps({"t": "frame", "side": 0, "len": 1}).encode()], "unexpected frame 'frame' with 0 descriptors"),
        ([json.dumps({"t": "nonsense"}).encode()], "unexpected frame 'nonsense' with 0 descriptors"),
        ([b""], "the server closed the socket inside a task"),
        ([b"x" * (pool_worker.MAX_MESSAGE + 1)], "truncated"),
    ],
)
def test_frames_the_worker_cannot_read_are_protocol_errors(
    pair: tuple[socket.socket, socket.socket], replies: list[bytes], text: str
) -> None:
    server, worker = pair
    results = pool_worker._Results(worker, 0, aggregate=False)
    frames = pool_worker._Frames(worker, results)
    if replies == [b""]:
        server.shutdown(socket.SHUT_WR)
    else:
        server.send(replies[0])
    with pytest.raises(pool_worker._ProtocolError, match=text):
        frames.next()
    assert json.loads(server.recv(100)) == {"t": "want"}


def test_a_frame_is_read_once_and_nothing_follows_the_end(pair: tuple[socket.socket, socket.socket]) -> None:
    server, worker = pair
    frames = pool_worker._Frames(worker, pool_worker._Results(worker, 0, aggregate=False))
    _frame(server, [5], side=1)
    _send(server, {"t": "end_group"})
    _send(server, {"t": "end"})
    kind, side, batch = frames.next()
    assert (kind, side, batch.column(0).to_pylist()) == ("batch", 1, [5])
    assert frames.next() == ("end_group",)
    assert frames.next() == ("end",)
    with pytest.raises(pool_worker._ProtocolError, match="asked for after the end"):
        frames.next()
