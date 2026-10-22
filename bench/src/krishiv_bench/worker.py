"""Out-of-process engine host for ``reproduce``: one engine adapter per process, driven over a pickle protocol.

Running every engine in its own process session gives each query a hard timeout (the parent kills the whole
process group: JVMs and runner children included), keeps one engine's allocator and caches from touching another's,
and is the unit of "a process per query" for the published-method runs. The protocol is length-prefixed pickle frames
on private duplicates of stdin/stdout (fd 1 is redirected to stderr so engine noise cannot corrupt the stream);
both ends are this module, so pickle is a trusted format here.

Requests: ``{"op": "setup", "dataset": Dataset}``, ``{"op": "run"|"time"|"profile", "query": Query}``,
``{"op": "quit"}``.
Replies: ``{"ok": ...}`` or ``{"error": "Type: message"}``.
"""

from __future__ import annotations

import os
import pickle
import struct
import sys
from typing import IO, Any

from .errors import describe_error

_HEADER = struct.Struct("<Q")


def send(out: IO[bytes], obj: Any) -> None:
    blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    out.write(_HEADER.pack(len(blob)) + blob)
    out.flush()


def read_exact(inp: IO[bytes], n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = inp.read(n - len(buf))
        if not chunk:
            raise EOFError("peer closed the pipe")
        buf += chunk
    return bytes(buf)


def recv(inp: IO[bytes]) -> Any:
    (n,) = _HEADER.unpack(read_exact(inp, _HEADER.size))
    return pickle.loads(read_exact(inp, n))  # both ends are this module


def main(argv: list[str]) -> None:
    name, threads, memory_gb = argv
    os.environ["POLARS_MAX_THREADS"] = threads  # polars reads it at import
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    inp = sys.stdin.buffer

    from .adapters import make_engine

    mem = f"{memory_gb}GiB" if name == "duckdb" else f"{memory_gb}g"
    eng = make_engine(name, threads=int(threads), memory_limit=mem)
    while True:
        try:
            req = recv(inp)
        except EOFError:
            break
        op = req["op"]
        if op == "quit":
            break
        try:
            if op == "setup":
                eng.setup(req["dataset"])
                reply: Any = {"version": eng.version(), "config": eng.config(), "build": eng.build_info()}
            elif op == "run":
                reply = eng.run(req["query"])
            elif op == "time":
                reply = eng.timed_run(req["query"])
            elif op == "profile":
                reply = eng.profile(req["query"])
            else:
                raise ValueError(f"unknown op {op!r}")
            send(out, {"ok": reply})
        except Exception as e:
            send(out, {"error": describe_error(e)})
    try:
        eng.teardown()
    finally:
        out.close()


if __name__ == "__main__":
    main(sys.argv[1:])
