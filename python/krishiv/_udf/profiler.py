# SPDX-License-Identifier: Apache-2.0
"""The Python UDF profiler (Spark 4.x ``spark.sql.pyspark.udf.profiler``, P3-T080), embedded host only.

``perf`` runs each UDF call under :mod:`cProfile` and sums the statistics per UDF; ``memory`` records, per UDF,
the peak :mod:`tracemalloc` memory of its calls (Spark's ``memory`` profiler reports line by line with
``memory_profiler``; the per-UDF peak is a documented divergence). Off by default: with the conf unset a call
costs one dictionary lookup. The results live in a process table read by ``spark.profile`` (``show``,
``dump``, ``clear``).
"""

from __future__ import annotations

import cProfile
import pstats
import threading
import tracemalloc
from collections.abc import Callable
from typing import Any

from krishiv._udf import _host

CONF_KEY = "spark.sql.pyspark.udf.profiler"

_lock = threading.Lock()
_next_id = 0
#: UDF id -> its summed cProfile statistics.
_perf: dict[int, pstats.Stats] = {}
#: UDF id -> [calls, peak bytes].
_memory: dict[int, list[int]] = {}
_names: dict[int, str] = {}


def new_udf_id(name: str) -> int:
    """A process-unique id for one UDF wrapper, shown in the reports (``UDF<id=N>``)."""
    global _next_id  # a process-wide counter, as Spark's `_next_id` of profile ids
    with _lock:
        _next_id += 1
        _names[_next_id] = name
        return _next_id


def profiled(udf_id: int, wrapper: Callable[..., Any]) -> Callable[..., Any]:
    """``wrapper`` that records its calls when the session's conf asks for it."""

    def call(*arrays: Any) -> Any:
        mode = _host.runner_conf().get(CONF_KEY)
        if mode == "perf":
            profile = cProfile.Profile()
            try:
                return profile.runcall(wrapper, *arrays)
            finally:
                _add_perf(udf_id, profile)
        if mode == "memory":
            return _run_memory(udf_id, wrapper, arrays)
        return wrapper(*arrays)

    return call


def _add_perf(udf_id: int, profile: cProfile.Profile) -> None:
    with _lock:
        stats = _perf.get(udf_id)
        if stats is None:
            _perf[udf_id] = pstats.Stats(profile)
        else:
            stats.add(profile)


def _run_memory(udf_id: int, wrapper: Callable[..., Any], arrays: tuple[Any, ...]) -> Any:
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    try:
        return wrapper(*arrays)
    finally:
        peak = max(tracemalloc.get_traced_memory()[1] - base, 0)
        if started:
            tracemalloc.stop()
        with _lock:
            entry = _memory.setdefault(udf_id, [0, 0])
            entry[0] += 1
            entry[1] = max(entry[1], peak)


def _selected(table: dict[int, Any], udf_id: int | None) -> list[int]:
    return sorted(table) if udf_id is None else [udf_id] if udf_id in table else []


def _kinds(type: str | None) -> tuple[bool, bool]:
    if type not in (None, "perf", "memory"):
        raise _host.PySparkValueError(
            errorClass="VALUE_NOT_ALLOWED",
            messageParameters={"arg_name": "type", "allowed_values": "perf, memory"},
        )
    return type in (None, "perf"), type in (None, "memory")


def show(udf_id: int | None = None, type: str | None = None) -> None:
    """Prints the collected profiles (``spark.profile.show``)."""
    print(render_text(udf_id, type), end="")


def render_text(udf_id: int | None = None, type: str | None = None) -> str:
    """The text ``show`` prints."""
    import io

    want_perf, want_memory = _kinds(type)
    out = io.StringIO()
    with _lock:
        if want_perf:
            for i in _selected(_perf, udf_id):
                out.write("============================================================\n")
                out.write(f"Profile of UDF<id={i}> ({_names.get(i, '')})\n")
                out.write("============================================================\n")
                stats = pstats.Stats(stream=out).add(_perf[i])
                stats.sort_stats("cumulative").print_stats()
        if want_memory:
            for i in _selected(_memory, udf_id):
                calls, peak = _memory[i]
                out.write("============================================================\n")
                out.write(f"Profile of UDF<id={i}> ({_names.get(i, '')})\n")
                out.write("============================================================\n")
                out.write(f"calls: {calls}\npeak memory: {peak / (1 << 20):.3f} MiB\n")
    return out.getvalue()


def dump(path: str, udf_id: int | None = None, type: str | None = None) -> None:
    """Writes the profiles under ``path`` (``spark.profile.dump``): ``udf_<id>_perf.pstats`` and
    ``udf_<id>_memory.txt``."""
    import os

    want_perf, want_memory = _kinds(type)
    os.makedirs(path, exist_ok=True)
    with _lock:
        if want_perf:
            for i in _selected(_perf, udf_id):
                _perf[i].dump_stats(os.path.join(path, f"udf_{i}_perf.pstats"))
        if want_memory:
            for i in _selected(_memory, udf_id):
                calls, peak = _memory[i]
                with open(os.path.join(path, f"udf_{i}_memory.txt"), "w", encoding="utf-8") as f:
                    f.write(f"calls: {calls}\npeak_bytes: {peak}\n")


def clear(udf_id: int | None = None, type: str | None = None) -> None:
    """Drops the collected profiles (``spark.profile.clear``)."""
    want_perf, want_memory = _kinds(type)
    with _lock:
        if want_perf:
            for i in _selected(_perf, udf_id):
                del _perf[i]
        if want_memory:
            for i in _selected(_memory, udf_id):
                del _memory[i]
