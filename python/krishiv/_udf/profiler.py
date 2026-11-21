# SPDX-License-Identifier: Apache-2.0
"""The Python UDF profiler (P3-T080): ``spark.sql.pyspark.udf.profiler`` and ``spark.profile``.

Spark 4.x profiles a UDF's calls when the session setting is ``perf`` (``cProfile``) or ``memory``
(``memory_profiler``, line by line). Krishiv follows the API (``spark.profile.show``, ``dump``,
``clear``) and the ``perf`` output. Two differences, both on purpose:

* ``memory`` uses ``tracemalloc``: per UDF it reports the calls, the peak and the net bytes the calls
  allocated, not lines. It needs no extra package.
* ``cProfile`` and ``tracemalloc`` are one per process, so a profiled UDF call takes one process-wide lock: with a
  profiler on, calls of profiled UDFs do not overlap (also on free-threaded Python). With it off, which is
  the default, nothing here runs: the UDF's wrapper is not even built.

It profiles the embedded runtime (UDFs that run in this process). The server's isolated workers
(``krishiv-udf-pool``) do not report profiles yet.
"""

from __future__ import annotations

import cProfile
import io
import itertools
import json
import os
import pstats
import threading
import tracemalloc
import weakref
from collections.abc import Callable
from typing import Any

#: The setting that turns the profiler on, and the values it takes.
CONF_KEY = "spark.sql.pyspark.udf.profiler"
MODES = ("perf", "memory")

_ids = itertools.count(1)
#: One lock for the process: `cProfile` and `tracemalloc` are one per process, not per session.
_PROCESS_LOCK = threading.RLock()


def next_udf_id() -> int:
    """The id a profiled UDF is shown under (``Profile of UDF<id=N>``)."""
    with _PROCESS_LOCK:
        return next(_ids)


def ensure_udf_id(udf: Any) -> int:
    """The profiler id of ``udf`` (its ``_profile_id``), assigned once even when two threads race."""
    with _PROCESS_LOCK:
        if udf._profile_id is None:
            udf._profile_id = next(_ids)
        return int(udf._profile_id)


class _Entry:
    """What one UDF collected."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.perf: cProfile.Profile | None = None
        self.calls = 0
        self.peak = 0
        self.net = 0


class Registry:
    """The profiles of one session, by UDF id."""

    def __init__(self) -> None:
        self.lock = _PROCESS_LOCK
        #: The profiled handles by UDF object and mode; they go with the session.
        self.handles: weakref.WeakKeyDictionary[Any, dict[str, Any]] = weakref.WeakKeyDictionary()
        self.perf: dict[int, _Entry] = {}
        self.memory: dict[int, _Entry] = {}

    def wrap(self, func: Callable[..., Any], udf_id: int, name: str, mode: str) -> Callable[..., Any]:
        """``func`` with its calls profiled under ``udf_id`` in ``mode``."""
        if mode == "perf":
            with self.lock:
                entry = self.perf.setdefault(udf_id, _Entry(name))
            return self._perf(func, entry)
        with self.lock:
            entry = self.memory.setdefault(udf_id, _Entry(name))
        return self._memory(func, entry)

    def _perf(self, func: Callable[..., Any], entry: _Entry) -> Callable[..., Any]:
        def run(*args: Any, **kwargs: Any) -> Any:
            with self.lock:
                # Read per call: `clear` swaps the profile for an empty one.
                profile = entry.perf
                if profile is None:
                    profile = entry.perf = cProfile.Profile()
                profile.enable()
                try:
                    return func(*args, **kwargs)
                finally:
                    profile.disable()

        return run

    def _memory(self, func: Callable[..., Any], entry: _Entry) -> Callable[..., Any]:
        def run(*args: Any, **kwargs: Any) -> Any:
            with self.lock:
                started = not tracemalloc.is_tracing()
                if started:
                    tracemalloc.start()
                before, _ = tracemalloc.get_traced_memory()
                tracemalloc.reset_peak()
                try:
                    return func(*args, **kwargs)
                finally:
                    after, peak = tracemalloc.get_traced_memory()
                    entry.calls += 1
                    entry.peak = max(entry.peak, peak - before)
                    entry.net += after - before
                    if started:
                        tracemalloc.stop()

        return run


def _kinds(type: str | None) -> tuple[str, ...]:
    if type is None:
        return MODES
    if type not in MODES:
        raise ValueError(f"type must be 'perf' or 'memory', got {type!r}")
    return (type,)


class Profile:
    """``spark.profile``: shows, dumps and clears what the profiler collected."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    def _table(self, kind: str) -> dict[int, _Entry]:
        return self._registry.perf if kind == "perf" else self._registry.memory

    def _selected(self, kind: str, id: int | None) -> list[tuple[int, _Entry]]:
        table = self._table(kind)
        with self._registry.lock:
            chosen = sorted(table.items()) if id is None else [(id, table[id])] if id in table else []
            # A cleared or never-called UDF has nothing to show.
            return [(i, e) for i, e in chosen if (e.perf is not None if kind == "perf" else e.calls > 0)]

    def show(self, id: int | None = None, *, type: str | None = None) -> None:
        """Prints the profiles of UDF ``id`` (all when ``None``) of ``type`` (``perf`` and ``memory`` when ``None``)."""
        for kind in _kinds(type):
            for udf_id, entry in self._selected(kind, id):
                print(self._render(kind, udf_id, entry), end="")  # the API prints, as Spark's

    def showPerfProfiles(self, id: int | None = None) -> None:  # Spark's name
        """``show(id, type="perf")``."""
        self.show(id, type="perf")

    def showMemoryProfiles(self, id: int | None = None) -> None:  # Spark's name
        """``show(id, type="memory")``."""
        self.show(id, type="memory")

    def dump(self, path: str, id: int | None = None, *, type: str | None = None) -> None:
        """Writes ``udf_<id>_perf.pstats`` and ``udf_<id>_memory.json`` files under ``path``."""
        os.makedirs(path, exist_ok=True)
        for kind in _kinds(type):
            for udf_id, entry in self._selected(kind, id):
                if kind == "perf":
                    with self._registry.lock:
                        # Read under the lock: `clear` may have swapped the profile since the listing.
                        if entry.perf is not None:
                            entry.perf.dump_stats(os.path.join(path, f"udf_{udf_id}_perf.pstats"))
                elif kind == "memory":
                    with self._registry.lock:  # one consistent snapshot of the counters
                        body = {
                            "name": entry.name,
                            "calls": entry.calls,
                            "peak_bytes": entry.peak,
                            "net_bytes": entry.net,
                        }
                    with open(os.path.join(path, f"udf_{udf_id}_memory.json"), "w", encoding="utf-8") as f:
                        json.dump(body, f)

    def clear(self, id: int | None = None, *, type: str | None = None) -> None:
        """Forgets what was collected for UDF ``id`` (all when ``None``) of ``type``."""
        for kind in _kinds(type):
            table = self._table(kind)
            with self._registry.lock:
                for udf_id in [id] if id is not None else list(table):
                    entry = table.get(udf_id)
                    if entry is None:
                        continue
                    if kind == "perf":
                        entry.perf = None
                    else:
                        entry.calls = entry.peak = entry.net = 0

    def _render(self, kind: str, udf_id: int, entry: _Entry) -> str:
        bar = "=" * 60
        head = f"{bar}\nProfile of UDF<id={udf_id}>\n{bar}\n"
        if kind == "perf":
            out = io.StringIO()
            if entry.perf is not None:
                with self._registry.lock:
                    stats = pstats.Stats(entry.perf, stream=out)
                    stats.sort_stats("time", "cumulative").print_stats()
            return head + out.getvalue()
        with self._registry.lock:
            calls, peak, net = entry.calls, entry.peak, entry.net
        return head + f"Memory of {entry.name}: {calls} call(s), peak {peak} bytes, net {net} bytes\n"
