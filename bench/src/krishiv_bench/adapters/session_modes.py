# SPDX-License-Identifier: Apache-2.0
"""What the session-based Krishiv modes (``krishiv-python``, ``krishiv-connect``, ``krishiv-flightsql``) share with
the native runner (``krishiv``), so a mode comparison compares transports, not settings.

The runner always runs with ``krishiv.memory.limit`` = its ``--memory-limit`` (12g by default) and the session time
zone UTC (``crates/krishiv-bench-runner/src/lib.rs``). The session modes get the same two settings unless the caller
sets them (``--krishiv-conf``). Compute threads have no config key: a session uses every core the process may run on,
so a run that asks the runner for fewer threads is refused for these modes (``cli.py``).
"""

from __future__ import annotations

import os

RUNNER_MEMORY_LIMIT = "12g"
MEMORY_LIMIT = "krishiv.memory.limit"
TIME_ZONE = "krishiv.sql.session.timeZone"
TIME_ZONE_ALIAS = "spark.sql.session.timeZone"


def usable_cores() -> int:
    """The cores this process may run on (its affinity), which is what a session's executor uses."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover  # not Linux
        return os.cpu_count() or 1


def with_runner_defaults(conf: list[tuple[str, str]] | None) -> list[tuple[str, str]]:
    """``conf`` plus the runner's memory limit and time zone where ``conf`` does not set them (an explicit one wins)."""
    out = list(conf or [])
    given = {k for k, _ in out}
    if MEMORY_LIMIT not in given:
        out.append((MEMORY_LIMIT, RUNNER_MEMORY_LIMIT))
    if TIME_ZONE not in given and TIME_ZONE_ALIAS not in given:
        out.append((TIME_ZONE, "UTC"))
    return out


def effective(conf: list[tuple[str, str]]) -> dict:
    """The settings a result file shows for the mode: memory limit, time zone and compute threads."""
    d = dict(conf)
    return {
        "memory_limit": d.get(MEMORY_LIMIT),
        "time_zone": d.get(TIME_ZONE, d.get(TIME_ZONE_ALIAS)),
        "threads": usable_cores(),
    }
