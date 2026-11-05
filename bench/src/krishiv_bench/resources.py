"""Per-query resource fields for the result JSON, and the cold-cache (``--cold``) page-cache eviction."""

from __future__ import annotations

import contextlib
import os
import statistics
from pathlib import Path

from .adapters import Measure


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _max(xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def resource_fields(meas: list[Measure], times: list[float]) -> dict:
    """Resource columns of one query's timed samples (``meas[i]`` was taken around the run that took ``times[i]``).

    Medians over the samples, except peak RSS and spill (maximum). ``avg_cores`` is the median of per-sample CPU
    seconds over wall seconds; ``read_size_mean_bytes`` is rchar over read syscalls of the medians. A quantity the
    engine did not report is ``None``, never 0.
    """
    cores = [m.cpu_seconds / t for m, t in zip(meas, times, strict=True) if m.cpu_seconds is not None and t > 0]
    rchar, syscr = _med([m.rchar for m in meas]), _med([m.syscr for m in meas])
    return {
        "peak_rss_bytes": _max([m.peak_rss_bytes for m in meas]),
        "baseline_rss_bytes": _max([m.baseline_rss_bytes for m in meas]),
        "cpu_seconds_median": _med([m.cpu_seconds for m in meas]),
        "avg_cores": statistics.median(cores) if cores else None,
        "spill_bytes": _max([m.spill_bytes for m in meas]),
        "read_bytes_median": _med([m.read_bytes for m in meas]),
        "rchar_median": rchar,
        "syscr_median": syscr,
        "read_size_mean_bytes": (rchar / syscr) if rchar is not None and syscr else None,
    }


def evict_page_cache(root: Path) -> dict:
    """Ask the kernel to drop the cached pages of every file under ``root`` (``POSIX_FADV_DONTNEED``; no root needed).

    Returns ``{"files": n, "bytes": total size}`` of the files it advised. Dirty pages stay cached, so the files
    must have been written back (a generated dataset has been, long since). An unreadable file is skipped.
    """
    files = size = 0
    for dirpath, _, names in os.walk(root):
        for name in names:
            path = Path(dirpath, name)
            with contextlib.suppress(OSError):
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    size += os.fstat(fd).st_size
                    files += 1
                finally:
                    os.close(fd)
    return {"files": files, "bytes": size}
