# SPDX-License-Identifier: Apache-2.0
"""What the session-based Krishiv modes (``krishiv-python``, ``krishiv-connect``, ``krishiv-flightsql``) share with
the native runner (``krishiv``), so a mode comparison compares transports, not settings.

The runner always runs with ``krishiv.memory.limit`` = its ``--memory-limit`` (12g by default) and the session time
zone UTC (``crates/krishiv-bench-runner/src/lib.rs``). The session modes get the same two settings unless the caller
sets them (``--krishiv-conf``). Compute threads have no config key: a session uses every core the process may run on,
so a run that asks the runner for fewer threads is refused for these modes (``cli.py``).

All three register the dataset's tables with the same statement (``table_sql``), so they plan the same scans: a
catalog table over the Parquet file. A temporary view does not do on the server: the DataFrame form
(``createOrReplaceTempView``, Connect's ``CreateDataFrameViewCommand``) is not translated (``Missing`` in
``crates/krishiv-connect/src/coverage.rs``, P3-T012), and a SQL temporary view is created but a Spark Connect query
that reads it is refused (``KRISHIV_UNSUPPORTED_FEATURE`` "views", ``krishiv-plan/src/analyzer/ops.rs``) while Flight
SQL reads it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

RUNNER_MEMORY_LIMIT = "12g"
MEMORY_LIMIT = "krishiv.memory.limit"
TIME_ZONE = "krishiv.sql.session.timeZone"
TIME_ZONE_ALIAS = "spark.sql.session.timeZone"


def cgroup_cpu_limit(root: str = "/sys/fs/cgroup", self_cgroup: str = "/proc/self/cgroup") -> int | None:
    """The tightest cgroup v2 CPU quota on this process's cgroup and its parents, in whole cores (``quota // period``,
    at least 1), or None without one: what Rust's ``available_parallelism``, which sizes the engine's executor, also
    applies on top of the affinity."""
    try:
        rel = next(line[3:].strip() for line in Path(self_cgroup).read_text().splitlines() if line.startswith("0::"))
    except (OSError, StopIteration):
        return None
    best: int | None = None
    d = Path(root, rel.lstrip("/"))
    while True:
        try:
            quota, period = (d / "cpu.max").read_text().split()[:2]
            if quota != "max" and int(quota) > 0 and int(period) > 0:
                cores = max(1, int(quota) // int(period))
                best = cores if best is None else min(best, cores)
        except (OSError, ValueError):
            pass
        if d == Path(root) or d.parent == d:
            return best
        d = d.parent


def usable_cores() -> int:
    """The cores a session's executor uses: this process's affinity, capped by a cgroup CPU quota."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover  # not Linux
        cores = os.cpu_count() or 1
    limit = cgroup_cpu_limit()
    return min(cores, limit) if limit is not None else cores


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


def table_sql(table: str, path: str) -> str:
    """The statement that registers ``table`` over the Parquet at ``path``; ``IF NOT EXISTS``, so a reconnect to the
    same server (``krishiv-flightsql`` after a failed cancel) registers again without an error."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table) or "'" in path or "\\" in path:
        raise ValueError(f"cannot register {table!r} over {path!r}: name or path needs quoting")
    return f"CREATE TABLE IF NOT EXISTS {table} USING parquet LOCATION '{path}'"
