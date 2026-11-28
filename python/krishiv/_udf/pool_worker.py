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
- server → worker: ``task`` ``{udf, eval_type, key_count, arity, max_rows, conf}`` (no descriptor): a relation
  kind's task (design ``P3-T085-embedded-python-udf-kinds.md`` §3.2) over loaded UDF ``udf``, run as
  ``eval_type`` (the UDF's own, or the window kind of a grouped-aggregate UDF: 202 as 203, 252 as 253, as Spark
  runs one over a window; a scalar iterator, 204 and 251, is a task too). The worker then asks
  for each frame with ``want``, and the server answers with exactly one of ``frame`` ``{side, len}`` + a sealed
  memfd holding one record batch, ``end_group`` or ``end``. The worker sends ``result`` ``{len}`` + a memfd
  holding one record batch of the declared columns whenever it has rows, and ``done`` when the UDF finished
  (also before ``end``, when the UDF stopped reading: the server drops the frames left), or ``error``. The
  server writes only after a ``want``, so neither side ever blocks on a full socket while the other writes.
- a UDTF's ``task`` (eval types 300, 301, 302) also carries ``udtf_len`` + a sealed memfd holding its call as
  JSON (``udtf.UdtfCall``: the argument offsets, the partition child indexes, the declared return type as Spark's
  JSON and ``analyze``'s pickled result as hex); its ``udf`` command is the cloudpickled UDTF class alone, as
  Spark Connect sends it.
- server → worker: ``analyze`` ``{udf, len, max}`` + a sealed memfd holding one record batch of one row: each
  constant argument's value as a column, the schema metadata ``krishiv.udtf.analyze`` the call's name and
  arguments as JSON; run loaded UDTF ``udf``'s static ``analyze`` (``udtf.analyze_udtf``); answered by
  ``analyzed`` ``{len}`` + a memfd holding the result as JSON (at most ``max`` bytes, else a Python error), or
  ``error``.
- EOF on the socket: the worker exits 0.

``error`` is ``{kind, message}``: ``kind`` ``python`` is an exception raised while loading or running the UDF
(the server raises ``PYTHON_EXCEPTION`` with ``message``, the traceback); ``spark`` is an error Spark's JVM raises
converting a 300 UDTF's row, ``message`` the JSON ``{class, parameters}`` (the server raises that class, for the
classes it knows); ``protocol`` is a message the worker cannot read (a server bug: the server discards the worker).
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
    from krishiv._udf._host import PySparkRuntimeError

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
    return _write_memfd(memoryview(sink.getvalue()))


def _write_memfd(data: memoryview) -> tuple[int, int]:
    """``data`` in a new memfd, and its length."""
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
    """A loaded relation UDF: the function, its return type and eval type, and its runners by the eval type a
    task runs it as (``relations.make_relation``)."""

    def __init__(self, func: Any, return_type: Any, eval_type: int) -> None:
        from krishiv._udf.relations import make_relation

        self.func = func
        self.return_type = return_type
        self.eval_type = eval_type
        self.runners = {eval_type: make_relation(func, eval_type, return_type)}

    def runner(self, eval_type: int) -> Any:
        """The runner of a task run as ``eval_type``: the UDF's own kind, or the window kind of a
        grouped-aggregate UDF."""
        from krishiv._udf.relations import WINDOW_OF, make_relation

        if eval_type not in self.runners:
            if WINDOW_OF.get(self.eval_type) != eval_type:
                raise _ProtocolError(f"a task of eval type {eval_type} for a UDF of eval type {self.eval_type}")
            self.runners[eval_type] = make_relation(self.func, eval_type, self.return_type)
        return self.runners[eval_type]


class _Udtf:
    """A loaded Python table function: its class and eval type (``udtf.make_udtf_runner`` per task). A command
    that does not unpickle is kept as its error, raised by :meth:`handler`: Spark reads the class inside the
    ``analyze`` or task program, so the failure is that program's (an ``analyze`` failure for ``analyze``)."""

    def __init__(self, command: bytes, eval_type: int) -> None:
        from pyspark import cloudpickle  # ty: ignore[unresolved-import]  # pool workers only

        self.eval_type = eval_type
        self._handler: Any = None
        self._error: BaseException | None = None
        try:
            self._handler = cloudpickle.loads(command)
        except BaseException as e:  # raised again by every use, as Spark would raise it
            self._error = e

    def handler(self) -> Any:
        if self._error is not None:
            raise self._error
        return self._handler


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

        from krishiv._udf.relations import RELATION_EVAL_TYPES
        from krishiv._udf.worker import make_wrapper

        _check_python_version(str(msg["python_ver"]))
        from krishiv._udf.udtf import UDTF_EVAL_TYPES

        command = _read_memfd(fd, int(msg["len"])).to_pybytes()
        eval_type = int(msg["eval_type"])
        if eval_type in UDTF_EVAL_TYPES:
            # A UDTF's command is its class alone (Spark Connect's `PythonUDTF.command`).
            self.udfs[int(msg["id"])] = _Udtf(command, eval_type)
            return
        func, return_type = cloudpickle.loads(command)
        if eval_type in RELATION_EVAL_TYPES:
            self.udfs[int(msg["id"])] = _Relation(func, return_type, eval_type)
        else:
            self.udfs[int(msg["id"])] = make_wrapper(func, eval_type, return_type)

    def run_batch(self, msg: dict[str, Any], fd: int) -> tuple[int, int]:
        """Runs one batch; returns a memfd holding the result stream and its length."""
        import pyarrow as pa

        from krishiv._udf._host import set_runner_conf

        wrapper = self.udfs[int(msg["udf"])]
        if isinstance(wrapper, (_Relation, _Udtf)):
            raise _ProtocolError("a batch for a relation UDF")
        batch = _read_batch(fd, int(msg["len"]))
        set_runner_conf(dict(msg.get("conf") or {}))
        result = wrapper(*batch.columns)
        return _write_batch(pa.RecordBatch.from_arrays([result], names=["_0"]))

    def run_task(self, msg: dict[str, Any], fd: int | None) -> None:
        """Runs one relation task to its ``done`` (see the module doc; ``fd`` holds a UDTF's call); a UDF error
        propagates, with the worker still in step (it sends nothing until the server's next message)."""
        from krishiv._udf._host import set_runner_conf
        from krishiv._udf.relations import AGGREGATE_EVAL_TYPES
        from krishiv._udf.udtf import UdtfCall, make_udtf_runner

        relation = self.udfs[int(msg["udf"])]
        if isinstance(relation, _Udtf):
            eval_type = int(msg.get("eval_type") or relation.eval_type)
            if eval_type != relation.eval_type or fd is None:
                raise _ProtocolError(f"a task of eval type {eval_type} for a UDTF of eval type {relation.eval_type}")
            try:
                call = json.loads(_read_memfd(fd, int(msg["udtf_len"])).to_pybytes())
            except ValueError as e:
                raise _ProtocolError(f"an unreadable UDTF call: {e}") from None
            if not isinstance(call, dict):
                raise _ProtocolError("a UDTF call that is not a JSON object")
            runner = make_udtf_runner(relation.handler(), eval_type, UdtfCall.of(call))
        elif isinstance(relation, _Relation):
            eval_type = int(msg.get("eval_type") or relation.eval_type)
            runner = relation.runner(eval_type)
        else:
            raise _ProtocolError("a task for a scalar UDF")
        if fd is not None and not isinstance(relation, _Udtf):
            raise _ProtocolError("a task with a UDTF call for a UDF that is not a table function")
        set_runner_conf(dict(msg.get("conf") or {}))
        results = _Results(self.sock, int(msg.get("max_rows") or 0), eval_type in AGGREGATE_EVAL_TYPES)
        frames = _Frames(self.sock, results)
        for batch in runner(frames, int(msg["key_count"]), int(msg["arity"])):
            results.add(batch)
        results.flush()
        _send(self.sock, {"t": "done"})

    def analyze(self, msg: dict[str, Any], fd: int) -> tuple[int, int]:
        """Runs a loaded UDTF's ``analyze`` (``udtf.analyze_udtf``) over the call in memfd ``fd``; returns a
        memfd holding the result as JSON and its length. A result over the server's ``max`` raises, as a
        Python error (the server's analyze class)."""
        from krishiv._udf.udtf import ANALYZE_METADATA, analyze_udtf

        udtf = self.udfs[int(msg["udf"])]
        if not isinstance(udtf, _Udtf):
            raise _ProtocolError("an analyze for a UDF that is not a table function")
        values = _read_batch(fd, int(msg["len"]))
        try:
            call = json.loads((values.schema.metadata or {})[ANALYZE_METADATA])
            name, args = str(call["name"]), list(call["args"])
        except (KeyError, TypeError, ValueError) as e:
            raise _ProtocolError(f"an unreadable analyze call: {e!r}") from None
        result = analyze_udtf(udtf.handler(), name, args, values)
        data = json.dumps(result, separators=(",", ":")).encode()
        limit = int(msg.get("max") or 0)
        if limit and len(data) > limit:
            from krishiv._udf._host import PySparkValueError

            raise PySparkValueError(f"the analyze result of {len(data)} bytes is over the limit of {limit} bytes")
        return _write_memfd(memoryview(data))

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
                wanted = int("udtf_len" in msg) if kind == "task" else 1
                if kind not in ("udf", "batch", "task", "analyze") or len(fds) != wanted:
                    _error(self.sock, "protocol", f"unexpected message {kind!r} with {len(fds)} descriptors")
                    return
            except (ValueError, KeyError, TypeError) as e:
                _error(self.sock, "protocol", f"unreadable message: {e}")
                return
            finally:
                extra = fds[1:] if len(fds) > 1 else []
                for fd in extra:
                    os.close(fd)
            if kind == "analyze":
                try:
                    result_fd, length = self.analyze(msg, fds[0])
                except _ProtocolError as e:
                    _error(self.sock, "protocol", str(e))
                    return
                except BaseException as e:  # Spark's `worker_run` reports every exception, `SystemExit` too
                    _error(self.sock, "python", _traceback(e))
                    continue
                finally:
                    os.close(fds[0])
                try:
                    _send(self.sock, {"t": "analyzed", "len": length}, [result_fd])
                finally:
                    os.close(result_fd)
                continue
            if kind == "task":
                try:
                    self.run_task(msg, fds[0] if fds else None)
                except _ProtocolError as e:
                    _error(self.sock, "protocol", str(e))
                    return
                except Exception as e:
                    from krishiv._udf.relations import FrameError
                    from krishiv._udf.udtf import SparkJvmError

                    if isinstance(e, FrameError):
                        _error(self.sock, "protocol", str(e))
                        return
                    if isinstance(e, SparkJvmError):
                        _error(self.sock, "spark", json.dumps({"class": e.condition, "parameters": e.parameters}))
                    else:
                        _error(self.sock, "python", _traceback(e))
                finally:
                    for fd in fds:
                        os.close(fd)
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


def _add_paths(paths: list[str]) -> None:
    """The session's artifact paths on ``sys.path``: each at index 1, in order, once (``worker_util.py:59-65``
    ``add_path`` and ``setup_spark_files``, v4.2.0), after the worker's own modules are imported, as Spark adds
    them after its worker started."""
    import importlib

    changed = False
    for path in paths:
        if path not in sys.path:
            sys.path.insert(1, path)
            changed = True
    if changed:
        importlib.invalidate_caches()


def main(sock: socket.socket | None = None, paths: list[str] | None = None) -> None:
    """The worker's entry: greets the server over ``sock`` (the worker's stdin when not given), then serves
    ``udf`` and ``batch`` messages until EOF. ``paths`` are the session's artifact paths."""
    if sock is None:
        sock = socket.socket(fileno=0)
    try:
        import pyarrow
        import pyspark  # ty: ignore[unresolved-import]  # pool workers only

        import krishiv._udf.relations  # fail at start, not at the first task
        import krishiv._udf.udtf  # fail at start, not at the first UDTF
        import krishiv._udf.worker  # noqa: F401  # fail at start, not at the first batch
    except Exception as e:  # a missing package is reported to the server, then the worker exits
        _error(sock, "python", _traceback(e))
        sys.exit(1)
    _add_paths(list(paths or []))
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
