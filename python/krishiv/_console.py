# SPDX-License-Identifier: Apache-2.0
"""``krishiv.ui()``: the web console of a session, served from this process.

The console is the one ``krishiv server`` serves, limited to one session: its queries and their
profiles, a SQL editor that runs in the session itself, its catalog and settings, and this
process's memory and caches.
"""

from __future__ import annotations

import atexit
import contextlib
import weakref
from html import escape
from typing import TYPE_CHECKING, Any, overload

from krishiv import _native
from krishiv.errors import unsupported_feature
from krishiv.sql import profile_view
from krishiv.sql.profile_view import ProfileView

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from krishiv.sql.dataframe import DataFrame
    from krishiv.sql.session import SparkSession

__all__ = ["Console", "ui"]

# Consoles not yet stopped, so the exit hook stops them: no thread or port outlives the interpreter.
# Weak: a console belongs to its session and goes with it.
_RUNNING: weakref.WeakSet[Console] = weakref.WeakSet()


def _stop_all() -> None:
    for console in list(_RUNNING):
        # One failing stop must not leave the other consoles running.
        with contextlib.suppress(Exception):
            console.stop()


atexit.register(_stop_all)


class Console:
    """The running web console of one session.

    Open :attr:`url` in a browser on this machine. The console listens on ``127.0.0.1`` only, and
    every request must carry the token in the URL, which is generated for this start: treat the URL
    as a password for the session. It is stopped by :meth:`stop`, by leaving a ``with`` block, by
    the session's ``stop()`` and at interpreter exit; it also stops once neither the session nor
    this handle is referenced any more.
    """

    def __init__(self, native: _native.Console) -> None:
        self._native = native
        _RUNNING.add(self)

    @property
    def url(self) -> str:
        """The address to open: ``http://127.0.0.1:<port>/?token=<token>``."""
        return self._native.url

    @property
    def port(self) -> int:
        """The loopback port the console listens on."""
        return self._native.port

    @property
    def token(self) -> str:
        """The token every request of the console must carry."""
        return self._native.token

    @property
    def session_id(self) -> str:
        """The id of the session the console shows."""
        return self._native.session_id

    @property
    def running(self) -> bool:
        """Whether the console is still serving."""
        return self._native.is_running()

    def stop(self) -> None:
        """Stop the console and wait for its thread: the port is free when this returns.

        Stopping a stopped console does nothing.
        """
        self._native.stop()
        _RUNNING.discard(self)

    def __enter__(self) -> Console:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def __repr__(self) -> str:
        # Not the URL: it holds the token, and a repr ends up in logs and tracebacks.
        return f"Console(port={self.port}, session_id={self.session_id!r}, running={self.running})"

    def _repr_html_(self) -> str:
        """A link to the console, for notebooks."""
        if not self.running:
            return "<span>Krishiv console (stopped)</span>"
        return (
            f'<a href="{escape(self.url, quote=True)}" target="_blank" rel="noopener noreferrer">'
            f"Krishiv console</a> <span>(session {escape(self.session_id)}, port {self.port})</span>"
        )


def _start(session: SparkSession, port: int) -> Console:
    """The session's console: the running one, or a new one on ``port`` (0 picks a free port)."""
    if not _native.__ui__:
        raise unsupported_feature("krishiv.ui() (this build has no web console)")
    # The session's own lock, which its `stop()` takes too: a start is not cut in two by a stop, which
    # would otherwise forget the new console while stopping only the old one.
    with session._console_lock:
        current: Console | None = session._console
        if current is not None and current.running and port in (0, current.port):
            return current
        # The new one first: when it cannot start (the port is taken), the running one stays.
        # A set-up failure (the port is taken) is a `RuntimeError`, said here with what was being done. An
        # error of the engine is not one: it passes as it is, with its error class and SQLSTATE.
        try:
            console = Console(_native.Console(session._native, port))
        except RuntimeError as e:
            raise RuntimeError(f"cannot start the console: {e}") from e.__cause__
        session._console = console
        if current is not None:
            current.stop()
    # Printed once, when it starts: a script has nowhere else to learn the port and the token.
    print(f"Krishiv console: {console.url}", flush=True)
    return console


@overload
def ui(source: SparkSession | None = None, *, port: int = 0) -> Console: ...
@overload
def ui(source: ProfileView | DataFrame | dict[str, Any] | str, *, path: str | Path | None = None) -> ProfileView: ...
def ui(source: Any = None, *, path: str | Path | None = None, port: int = 0) -> Console | ProfileView:
    """The web console of a session, or the viewer of one query profile.

    ``ui()`` starts the console of the active session (making a session when there is none) and
    ``ui(spark)`` the console of ``spark``. It prints the URL to open and returns the
    :class:`Console`. The console listens on ``127.0.0.1`` only, on a free port unless ``port``
    names one, behind a token generated for this start. A second call returns the running console.
    It stops with the session (``spark.stop()``) or with :meth:`Console.stop`.

    With any other ``source`` this is the offline viewer of one profile: a profile document
    (``dict`` or JSON text), a :class:`ProfileView`, or a DataFrame (which runs
    ``EXPLAIN ANALYZE`` and views its profile). With ``path`` the page is also written there. In a
    notebook the returned view renders itself.
    """
    from krishiv.sql.session import SparkSession

    if source is None or isinstance(source, SparkSession):
        if path is not None:
            raise TypeError("path belongs to a profile view: the console is a server, not a page to write")
        return _start(SparkSession.builder.getOrCreate() if source is None else source, port)
    if port != 0:
        raise TypeError("port belongs to the console: a profile view is a page, not a server")
    return profile_view.ui(source, path=path)
