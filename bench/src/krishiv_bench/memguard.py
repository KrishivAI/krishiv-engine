"""Host-memory guard: wait for free memory before a timed run, and flag runs during which the host reclaimed it.

A balloon on the host can take most of RAM mid-run; a timing taken then measures the host, not the engine. The guard
reads ``MemAvailable`` from ``/proc/meminfo``: it waits until the value is at least a threshold before each timed
query, samples it while the query runs, and reports the lowest value seen so the runner can mark the sample invalid.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

MEMINFO = Path("/proc/meminfo")
DEFAULT_MIN_FRACTION = 0.75
DEFAULT_RETRIES = 2
SAMPLE_INTERVAL_S = 0.25
WAIT_POLL_S = 2.0
WAIT_LIMIT_S = 600.0
INVALID_REASON = "memory reclaimed"


def read_meminfo(path: Path = MEMINFO) -> tuple[int, int]:
    """``(MemTotal, MemAvailable)`` in bytes; raises ``OSError`` or ``ValueError`` if either line is missing."""
    found: dict[str, int] = {}
    for line in path.read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            found[key] = int(rest.split()[0]) * 1024  # the file reports kB
    try:
        return found["MemTotal"], found["MemAvailable"]
    except KeyError as e:
        raise ValueError(f"{path} has no {e.args[0]} line") from None


def threshold_bytes(fraction: float, path: Path = MEMINFO) -> int:
    """The minimum MemAvailable: ``fraction`` of MemTotal."""
    return int(read_meminfo(path)[0] * fraction)


def wait_for_memory(
    threshold: int,
    path: Path = MEMINFO,
    poll_s: float = WAIT_POLL_S,
    limit_s: float = WAIT_LIMIT_S,
    sleep=time.sleep,
    clock=time.monotonic,
) -> tuple[bool, float]:
    """Block until MemAvailable >= ``threshold``; ``(reached, seconds waited)``. Gives up after ``limit_s``."""
    start = clock()
    while read_meminfo(path)[1] < threshold:
        if clock() - start >= limit_s:
            return False, clock() - start
        sleep(poll_s)
    return True, clock() - start


class MemSampler:
    """Context manager: a thread that records the lowest MemAvailable seen while the block runs."""

    def __init__(self, path: Path = MEMINFO, interval_s: float = SAMPLE_INTERVAL_S):
        self.path = path
        self.interval_s = interval_s
        self.min_available: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> None:
        try:
            avail = read_meminfo(self.path)[1]
        except (OSError, ValueError):
            return
        if self.min_available is None or avail < self.min_available:
            self.min_available = avail

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def __enter__(self) -> MemSampler:
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()


@dataclass
class InvalidSample:
    """One timed run thrown away because the host reclaimed memory during it."""

    iteration: int
    attempt: int
    min_mem_avail_bytes: int
    threshold_bytes: int
    reason: str = INVALID_REASON

    def as_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "attempt": self.attempt,
            "reason": self.reason,
            "min_mem_avail_bytes": self.min_mem_avail_bytes,
            "threshold_bytes": self.threshold_bytes,
        }


def is_invalid(min_available: int | None, threshold: int) -> bool:
    """True when the sampler saw MemAvailable below the threshold during the run."""
    return min_available is not None and min_available < threshold
