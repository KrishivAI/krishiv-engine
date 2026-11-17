# SPDX-License-Identifier: Apache-2.0
"""The pool worker's program (design ``P3-T014-connect-udfs-sandbox.md`` §3.1, §3.3): Krishiv's own worker,
not ``pyspark.worker``.

The server starts the interpreter (through the sandbox trampoline) with ``-I`` and a bootstrap (``-c``,
``krishiv-udf-pool``'s ``BOOTSTRAP``) that closes every inherited descriptor above 2 before anything else runs
(the sandbox crate's invariant): the ``close_range`` system call through ``ctypes``, and ``os.closerange`` up to
the server's descriptor limit where that call is missing (some CPython builds implement ``os.closerange`` as a
loop over every number, minutes for ``2**31``). Only then does it put ``DIR`` on ``sys.path`` and call
:func:`main`. ``DIR`` holds this package under a stub ``krishiv/__init__.py`` (``_host``).

Wire ``krishiv.udf.wire.v1`` over the control socket (``AF_UNIX``/``SOCK_SEQPACKET``, the worker's stdin): one
message is one JSON object, ``t`` naming its kind, with any descriptors attached as ``SCM_RIGHTS``. Data never
crosses the socket: an Arrow IPC stream in a memfd does.

- worker → server, once at start: ``hello`` ``{wire, python, pyspark}``, or ``error`` and exit.
- server → worker: ``udf`` ``{id, eval_type, python_ver, len}`` + a memfd holding Spark's ``command`` bytes
  (cloudpickled ``(func, return_type)``); answered by ``ready`` or ``error``.
- server → worker: ``batch`` ``{udf, len, conf}`` + a sealed memfd holding one record batch, one column per
  argument; answered by ``result`` ``{len}`` + a memfd holding one record batch with one column, or ``error``.
- server → worker: ``task`` ``{udf, key_count, arity, max_rows, conf}`` (no descriptor): a relation kind's
  task (design ``P3-T085-embedded-python-udf-kinds.md`` §3.2) over loaded UDF ``udf``. The worker then asks
  for each frame with ``want``, and the server answers with exactly one of ``frame`` ``{side, len}`` + a sealed
  memfd holding one record batch, ``end_group`` or ``end``. The worker sends ``result`` ``{len}`` + a memfd
  holding one record batch of the declared columns whenever it has rows, and ``done`` when the UDF finished
  (also before ``end``, when the UDF stopped reading: the server drops the frames left), or ``error``. The
  server writes only after a ``want``, so neither side ever blocks on a full socket while the other writes.
- EOF on the socket: the worker exits 0.

``error`` is ``{kind, message}``: ``kind`` ``python`` is an exception raised while loading or running the UDF
(the server raises ``PYTHON_EXCEPTION`` with ``message``, the traceback); ``protocol`` is a message the worker
cannot read (a server bug: the server discards the worker).
"""

from __future__ import annotations

import json
import os
import socket
import sys
import traceback
from typing import Any

#: The wire version; the server refuses a ``hello`` with another.
WIRE = 1
#: The largest control message either side sends (the server's receive buffer is larger).
MAX_MESSAGE = 1 << 17
#: A traceback longer than this is cut, keeping its end (the exception line).
MAX_TRACEBACK = 96 * 1024


def _send(sock: socket.socket, msg: dict[str, Any], fds: list[int] | None = None) -> None:
    data = json.dumps(msg, separators=(",", ":")).encode()
    if fds:
        socket.send_fds(sock, [data], fds)
    else:
        sock.send(data)


def _traceback(exc: BaseException) -> str:
    """The traceback as the in-process runtime gives it (``python_error`` in ``krishiv-python``): the formatted
    frames ending with the exception line, without the trailing newline."""
    text = "".join(traceback.format_exception(exc)).rstrip("\n")
    if len(text) > MAX_TRACEBACK:
        text = "...\n" + text[-MAX_TRACEBACK:]
    return text


def _error(sock: socket.socket, kind: str, message: str) -> None:
    """Sends an ``error``, its message cut (keeping its end) so the encoded packet fits :data:`MAX_MESSAGE`: a
    character count does not bound the bytes, since JSON escapes a character to up to six (twelve for a
    surrogate pair)."""
    msg = {"t": "error", "kind": kind, "message": message}
    data = json.dumps(msg, separators=(",", ":")).encode()
    if len(data) > MAX_MESSAGE:
        mark = "...\n"
        msg["message"] = mark
        budget = MAX_MESSAGE - len(json.dumps(msg, separators=(",", ":")).encode())
        start = len(message)
        while start > 0:
            cost = len(json.dumps(message[start - 1])) - 2  # the character as JSON escapes it
            if cost > budget:
                break
            budget -= cost
            start -= 1
        msg["message"] = mark + message[start:]
        data = json.dumps(msg, separators=(",", ":")).encode()
    sock.send(data)


def _read_memfd(fd: int, length: int) -> Any:
    """The first ``length`` bytes of ``fd`` as a read-only ``pyarrow`` buffer, mapped (no copy)."""
    import mmap

    import pyarrow as pa

    if length == 0:
        return pa.py_buffer(b"")
    mapped = mmap.mmap(fd, length, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
    return pa.py_buffer(mapped)


def _check_python_version(python_ver: str) -> None:
    """``worker_util.py:83-99`` ``check_python_version`` (v4.2.0): the client's ``major.minor`` must be this
    interpreter's. The error surfaces as ``PYTHON_EXCEPTION`` with Spark's text, as in Spark (note §3.1)."""
    from pyspark.errors import PySparkRuntimeError  # ty: ignore[unresolved-import]  # pool workers only

    worker_version = f"{sys.version_info[0]}.{sys.version_info[1]}"
    if python_ver != worker_version:
        raise PySparkRuntimeError(
            errorClass="PYTHON_VERSION_MISMATCH",
            messageParameters={"worker_version": worker_version, "driver_version": str(python_ver)},
        )


class _ProtocolError(Exception):
    """A message this worker cannot read in its state: reported as a ``protocol`` error, then the worker exits."""


def _read_batch(fd: int, length: int) -> Any:
    """The one record batch of the Arrow stream in memfd ``fd``."""
    import pyarrow as pa

    return pa.ipc.open_stream(_read_memfd(fd, length)).read_next_batch()


def _write_batch(batch: Any) -> tuple[int, int]:
    """``batch`` as an Arrow stream (uncompressed) in a new memfd, and its length."""
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    data = memoryview(sink.getvalue())
    result_fd = os.memfd_create("krishiv-udf-result", os.MFD_CLOEXEC)
    try:
        written = 0
        while written < len(data):
            written += os.write(result_fd, data[written:])
    except BaseException:
        os.close(result_fd)
        raise
    return result_fd, len(data)


class _Relation:
    """A loaded relation UDF: its runner (``relations.make_relation``) and whether it is an aggregate."""

    def __init__(self, runner: Any, aggregate: bool) -> None:
        self.runner = runner
        self.aggregate = aggregate


#: The most rows of an aggregate's values kept before they are sent, when the task gives no ``max_rows``
#: (Spark's default ``spark.sql.execution.arrow.maxRecordsPerBatch``).
DEFAULT_MAX_ROWS = 10_000


class _Results:
    """A task's results on their way to the server. A grouped aggregate yields one row per group, so its rows
    are kept and sent together before the worker next asks for a frame (or ends); other kinds' batches go at
    once, so a map's output streams while its input does.

    ``max_rows`` is only an upper bound: the values are flushed before every ``want`` anyway (one group's
    value per frame request, in practice), so it caps the rows of one result only when a frame yields
    more values than that."""

    def __init__(self, sock: socket.socket, max_rows: int, aggregate: bool) -> None:
        self.sock = sock
        self.max_rows = max_rows if max_rows > 0 else DEFAULT_MAX_ROWS
        self.aggregate = aggregate
        self.pending: list[Any] = []
        self.rows = 0

    def add(self, batch: Any) -> None:
        self.pending.append(batch)
        self.rows += batch.num_rows
        if not self.aggregate or self.rows >= self.max_rows:
            self.flush()

    def flush(self) -> None:
        import pyarrow as pa

        if not self.pending:
            return
        batch = self.pending[0] if len(self.pending) == 1 else pa.concat_batches(self.pending)
        self.pending = []
        self.rows = 0
        result_fd, length = _write_batch(batch)
        try:
            _send(self.sock, {"t": "result", "len": length}, [result_fd])
        finally:
            os.close(result_fd)


class _Frames:
    """A task's input, asked for frame by frame (``relations.FrameSource``)."""

    def __init__(self, sock: socket.socket, results: _Results) -> None:
        self.sock = sock
        self.results = results
        self.ended = False

    def next(self) -> tuple[Any, ...]:
        if self.ended:
            raise _ProtocolError("a frame asked for after the end")
        self.results.flush()
        _send(self.sock, {"t": "want"})
        data, fds, flags, _addr = socket.recv_fds(self.sock, MAX_MESSAGE, 2)
        try:
            if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                raise _ProtocolError("a frame or its descriptor was truncated")
            if not data:
                raise _ProtocolError("the server closed the socket inside a task")
            try:
                msg = json.loads(data)
                kind = msg["t"]
            except (ValueError, KeyError, TypeError) as e:
                raise _ProtocolError(f"unreadable frame: {e}") from None
            if kind == "frame" and len(fds) == 1:
                return ("batch", int(msg["side"]), _read_batch(fds[0], int(msg["len"])))
            if kind in ("end_group", "end") and not fds:
                self.ended = kind == "end"
                return (kind,)
            raise _ProtocolError(f"unexpected frame {kind!r} with {len(fds)} descriptors")
        finally:
            for fd in fds:
                os.close(fd)


class _Worker:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.udfs: dict[int, Any] = {}

    def load_udf(self, msg: dict[str, Any], fd: int) -> None:
        from pyspark import cloudpickle  # ty: ignore[unresolved-import]  # pool workers only

        from krishiv._udf.relations import AGGREGATE_EVAL_TYPES, RELATION_EVAL_TYPES, make_relation
        from krishiv._udf.worker import make_wrapper

        _check_python_version(str(msg["python_ver"]))
        command = _read_memfd(fd, int(msg["len"])).to_pybytes()
        func, return_type = cloudpickle.loads(command)
        eval_type = int(msg["eval_type"])
        if eval_type in RELATION_EVAL_TYPES:
            runner = make_relation(func, eval_type, return_type)
            self.udfs[int(msg["id"])] = _Relation(runner, eval_type in AGGREGATE_EVAL_TYPES)
        else:
            self.udfs[int(msg["id"])] = make_wrapper(func, eval_type, return_type)

    def run_batch(self, msg: dict[str, Any], fd: int) -> tuple[int, int]:
        """Runs one batch; returns a memfd holding the result stream and its length."""
        import pyarrow as pa

        from krishiv._udf._host import set_runner_conf

        wrapper = self.udfs[int(msg["udf"])]
        if isinstance(wrapper, _Relation):
            raise _ProtocolError("a batch for a relation UDF")
        batch = _read_batch(fd, int(msg["len"]))
        set_runner_conf(dict(msg.get("conf") or {}))
        result = wrapper(*batch.columns)
        return _write_batch(pa.RecordBatch.from_arrays([result], names=["_0"]))

    def run_task(self, msg: dict[str, Any]) -> None:
        """Runs one relation task to its ``done`` (see the module doc); a UDF error propagates, with the
        worker still in step (it sends nothing until the server's next message)."""
        from krishiv._udf._host import set_runner_conf

        relation = self.udfs[int(msg["udf"])]
        if not isinstance(relation, _Relation):
            raise _ProtocolError("a task for a scalar UDF")
        set_runner_conf(dict(msg.get("conf") or {}))
        results = _Results(self.sock, int(msg.get("max_rows") or 0), relation.aggregate)
        frames = _Frames(self.sock, results)
        for batch in relation.runner(frames, int(msg["key_count"]), int(msg["arity"])):
            results.add(batch)
        results.flush()
        _send(self.sock, {"t": "done"})

    def serve(self) -> None:
        while True:
            data, fds, flags, _addr = socket.recv_fds(self.sock, MAX_MESSAGE, 4)
            if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                for fd in fds:
                    os.close(fd)
                _error(self.sock, "protocol", "a message or its descriptors were truncated")
                return
            if not data:
                return  # EOF: the server is done with this worker
            try:
                msg = json.loads(data)
                kind = msg["t"]
                wanted = 0 if kind == "task" else 1
                if kind not in ("udf", "batch", "task") or len(fds) != wanted:
                    _error(self.sock, "protocol", f"unexpected message {kind!r} with {len(fds)} descriptors")
                    return
            except (ValueError, KeyError, TypeError) as e:
                _error(self.sock, "protocol", f"unreadable message: {e}")
                return
            finally:
                extra = fds[1:] if len(fds) > 1 else []
                for fd in extra:
                    os.close(fd)
            if kind == "task":
                try:
                    self.run_task(msg)
                except _ProtocolError as e:
                    _error(self.sock, "protocol", str(e))
                    return
                except Exception as e:
                    from krishiv._udf.relations import FrameError

                    if isinstance(e, FrameError):
                        _error(self.sock, "protocol", str(e))
                        return
                    _error(self.sock, "python", _traceback(e))
                continue
            fd = fds[0]
            try:
                if kind == "udf":
                    self.load_udf(msg, fd)
                    _send(self.sock, {"t": "ready"})
                else:
                    result_fd, length = self.run_batch(msg, fd)
                    try:
                        _send(self.sock, {"t": "result", "len": length}, [result_fd])
                    finally:
                        os.close(result_fd)
            except Exception as e:  # every UDF failure goes back to the server
                _error(self.sock, "python", _traceback(e))
            finally:
                os.close(fd)


def main(sock: socket.socket | None = None) -> None:
    """The worker's entry: greets the server over ``sock`` (the worker's stdin when not given), then serves
    ``udf`` and ``batch`` messages until EOF."""
    if sock is None:
        sock = socket.socket(fileno=0)
    try:
        import pyarrow
        import pyspark  # ty: ignore[unresolved-import]  # pool workers only

        import krishiv._udf.worker  # noqa: F401  # fail at start, not at the first batch
    except Exception as e:  # a missing package is reported to the server, then the worker exits
        _error(sock, "python", _traceback(e))
        sys.exit(1)
    _send(
        sock,
        {
            "t": "hello",
            "wire": WIRE,
            "python": ".".join(str(n) for n in sys.version_info[:3]),
            "pyspark": pyspark.__version__,
            "pyarrow": pyarrow.__version__,
        },
    )
    _Worker(sock).serve()
