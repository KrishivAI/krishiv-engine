# SPDX-License-Identifier: Apache-2.0
"""``krishiv.start_local()``: an in-process Spark Connect server for stock PySpark (P3-T016)."""

from __future__ import annotations

import atexit
import contextlib
from typing import TYPE_CHECKING

from krishiv import _native

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import TracebackType

__all__ = ["LocalConnect", "start_local"]

# Servers not yet stopped are held here, so a dropped handle does not stop one a client still uses
# (``remote(krishiv.start_local().url)``); the exit hook stops them: no thread or port outlives
# the interpreter.
_RUNNING: set[LocalConnect] = set()


def _stop_all() -> None:
    for server in list(_RUNNING):
        # One failing stop must not leave the other servers running.
        with contextlib.suppress(Exception):
            server.stop()


atexit.register(_stop_all)


class LocalConnect:
    """A Spark Connect server running in this process, on a free loopback port.

    Connect any Spark Connect client to :attr:`url` (``sc://localhost:<port>/;token=<token>``); the
    token is generated for this server and every request must carry it. The server is stopped by
    :meth:`stop`, by leaving a ``with`` block, or at interpreter exit. Dropping the handle does
    not stop it: the server stays up until one of those.
    """

    def __init__(self, native: _native.LocalConnectServer) -> None:
        self._native = native
        _RUNNING.add(self)

    @property
    def url(self) -> str:
        """The connection string: ``sc://localhost:<port>/;token=<token>``."""
        return self._native.connection_string

    @property
    def port(self) -> int:
        """The loopback port the server listens on (``127.0.0.1`` and ``[::1]``, nothing else)."""
        return self._native.port

    @property
    def token(self) -> str:
        """The token a client must present (sent as ``authorization: Bearer <token>``)."""
        return self._native.token

    @property
    def running(self) -> bool:
        """Whether the server is still serving."""
        return self._native.is_running()

    def stop(self) -> None:
        """Stop the server and wait for its thread: the port is free when this returns.

        Stopping a stopped server does nothing.
        """
        try:
            self._native.stop()
        finally:
            # The native stop joins the thread even when it reports a failure: the server is gone.
            _RUNNING.discard(self)

    def __enter__(self) -> LocalConnect:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"LocalConnect(port={self.port}, running={self.running})"


def start_local(config: Mapping[str, str] | None = None) -> LocalConnect:
    """Start an in-process Spark Connect server and return its handle.

    The server listens on a free port of the loopback addresses ``127.0.0.1`` and ``[::1]``
    only. A stock ``pyspark-client`` connects with
    ``SparkSession.builder.remote(server.url).getOrCreate()``. ``config`` holds ``krishiv.*``
    settings of the server's sessions.

    Stop the client (``spark.stop()``) before the server: ``pyspark-client`` runs a cleanup command
    when the interpreter exits, and against a server that is already stopped it retries for minutes.

    This is not ``SparkSession.builder.remote("local")``, which starts a JVM and is not supported.

    Example::

        with krishiv.start_local() as server:
            spark = SparkSession.builder.remote(server.url).getOrCreate()  # pyspark-client
            spark.range(3).show()
            spark.stop()  # before the server stops: the client's exit cleanup needs it
    """
    pairs = [(str(k), str(v)) for k, v in (config or {}).items()]
    return LocalConnect(_native.LocalConnectServer(pairs or None))
