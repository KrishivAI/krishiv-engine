"""Fresh-process runs under a hard cgroup memory cap, shared by the capped suites.

``memcap`` and ``remote-filter`` both run every query in a fresh child process inside a ``systemd-run``
scope with ``MemoryMax`` and no swap, then classify the run as ``finished`` | ``killed`` | ``timeout`` |
``failed`` and record wall time, the scope's peak memory and the engine scratch directory's peak size.
This module holds that machinery so both suites share one implementation; suite modules import and
re-export what their tests address (``memcap`` keeps its names working).
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

KILL_PEAK_FRACTION = 0.95  # a SIGKILL with the scope this full is the memory controller's, even if the event was missed
POLL_S = 0.01
SPILL_EVERY = 10  # scratch directory scan every Nth poll
CGROUP_ROOT = Path("/sys/fs/cgroup")
GIB = 1 << 30


# ---------------------------------------------------------------- classification
@dataclass
class Observation:
    returncode: int | None
    timed_out: bool
    peak_bytes: int  # the scope's memory.peak: what the cap is enforced on (page cache included)
    oom_kills: int
    spill_peak_bytes: int | None
    wall_s: float
    rss_peak_bytes: int = 0  # highest sampled resident set of the tree (anon + mapped file + shmem; no page cache)


def classify(obs: Observation, cap_bytes: int | None, got_result: bool) -> str:
    """``finished`` | ``killed`` | ``timeout`` | ``failed``.

    ``killed`` is the memory cap's doing: the scope's ``oom_kill`` counter moved, or the process died of SIGKILL
    (return code -9 or 137) with the scope's peak within 5% of the cap (the cgroup is removed with its last
    process, so the counter can be gone before the last poll). A run that exits 0 without its result line is
    ``failed``, like any other non-zero exit. With no cap (``cap_bytes`` None) the SIGKILL-near-cap rule cannot
    fire: only a counted ``oom_kill`` kills.
    """
    if obs.timed_out:
        return "timeout"
    if obs.oom_kills > 0:
        return "killed"
    if cap_bytes is not None and obs.returncode in (-9, 137) and obs.peak_bytes >= KILL_PEAK_FRACTION * cap_bytes:
        return "killed"
    if obs.returncode == 0 and got_result:
        return "finished"
    return "failed"


class Cgroup:
    """The scope's cgroup-v2 files. Reads return ``None`` once the scope is gone."""

    def __init__(self, path: Path):
        self.path = path

    def _int(self, name: str) -> int | None:
        try:
            return int((self.path / name).read_text().strip())
        except (OSError, ValueError):
            return None

    def peak(self) -> int | None:
        values = [v for v in (self._int("memory.peak"), self._int("memory.current")) if v is not None]
        return max(values) if values else None

    def rss(self) -> int | None:
        """Resident memory of the scope's processes: anonymous + mapped file + shared memory pages (no page cache)."""
        try:
            stat = dict(
                (k, int(v))
                for k, _, v in (ln.partition(" ") for ln in (self.path / "memory.stat").read_text().splitlines())
            )
        except (OSError, ValueError):
            return None
        return stat.get("anon", 0) + stat.get("file_mapped", 0) + stat.get("shmem", 0)

    def oom_kills(self) -> int | None:
        try:
            for line in (self.path / "memory.events").read_text().splitlines():
                key, _, value = line.partition(" ")
                if key == "oom_kill":
                    return int(value)
        except (OSError, ValueError):
            return None
        return None

    def kill_all(self) -> None:
        """SIGKILLs every process in the scope (the engine's JVM and helper children included)."""
        with contextlib.suppress(OSError):
            (self.path / "cgroup.kill").write_text("1")


def dir_bytes(path: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(dirpath, f)).st_size
    return total


def monitor(
    proc,
    cgroup,
    scratch: Path | None,
    timeout_s: float | None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    dir_size: Callable[[Path], int] = dir_bytes,
) -> Observation:
    """Polls ``proc`` (a ``Popen``-like: ``poll()``, ``returncode``, ``kill()``, ``wait()``) until it exits.

    Keeps the highest scope peak and ``oom_kill`` count seen and the scratch directory's peak size. On timeout the
    whole scope is killed. ``cgroup`` is ``None`` when the scope could not be located (peak is then 0).
    """
    t0 = clock()
    peak = oom = rss_peak = tick = 0
    spill: int | None = None
    timed_out = False
    while True:
        rc = proc.poll()
        if cgroup is not None:
            p, o = cgroup.peak(), cgroup.oom_kills()
            peak = max(peak, p or 0)
            oom = max(oom, o or 0)
            rss_peak = max(rss_peak, cgroup.rss() or 0)
        if scratch is not None and tick % SPILL_EVERY == 0:
            spill = max(spill or 0, dir_size(scratch))
        tick += 1
        if rc is not None:
            break
        if timeout_s is not None and clock() - t0 > timeout_s:
            timed_out = True
            if cgroup is not None:
                cgroup.kill_all()
            proc.kill()
            proc.wait()
            rc = proc.returncode
            break
        sleep(POLL_S)
    wall = clock() - t0
    if cgroup is not None:
        cgroup.kill_all()  # stragglers (a JVM that outlived its parent) must not leak into the next run
    return Observation(rc, timed_out, peak, oom, spill, wall, rss_peak)


def scope_command(cap_bytes: int | None, argv: list[str]) -> list[str]:
    """``systemd-run`` scope: hard cap and no swap, or ``MemoryMax=max`` (still a scope, for measurement)."""
    limit = "max" if cap_bytes is None else f"{-(-cap_bytes // (1 << 20))}M"
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "-p",
        f"MemoryMax={limit}",
        "-p",
        "MemorySwapMax=0",
        "--",
        *argv,
    ]


def cgroup_of(pid: int | str) -> str | None:
    """The cgroup-v2 path of a process (``/user.slice/.../run-1234.scope``), or None."""
    try:
        line = Path(f"/proc/{pid}/cgroup").read_text().strip().splitlines()[0]
    except (OSError, IndexError):
        return None
    return line.partition("::")[2] or None


def find_cgroup(pid: int, wait_s: float = 5.0, _resolve=None) -> Cgroup | None:
    """The scope's cgroup: ``systemd-run --scope`` moves its own pid into the scope, then execs the command.

    Our own cgroup (this harness may itself run in a scope) never counts, so it is never killed by mistake.
    ``_resolve`` names the ``/proc/<pid>/cgroup`` reader (a seam for tests).
    """
    resolve = _resolve or cgroup_of
    own = resolve("self")
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        rel = resolve(pid)
        if rel is None:
            return None
        if rel != own and rel.endswith(".scope"):
            return Cgroup(CGROUP_ROOT / rel.lstrip("/"))
        time.sleep(0.005)
    return None


# ---------------------------------------------------------------- one run
def read_result(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT "):
            with contextlib.suppress(ValueError):
                return json.loads(line[7:])
    return None


def run_scoped(
    argv: list[str],
    *,
    env_extra: dict[str, str] | None = None,
    cap_bytes: int | None,
    timeout_s: float | None,
    work: Path,
    scratch: Path | None = None,
) -> tuple[Observation, str, str]:
    """Runs ``argv`` once inside a scope: hard-capped, or ``MemoryMax=max`` when ``cap_bytes`` is None.

    The uncapped scope is still a scope (so peak memory is measured the same way); only the kill rule in
    ``classify`` treats it as unconstrained. Returns the observation, the child's full stdout and the tail
    of its stderr. The caller classifies the run (``classify``) and interprets the ``RESULT`` line
    (``read_result``). ``work`` holds the child's ``stdout``/``stderr`` logs.
    """
    work.mkdir(parents=True, exist_ok=True)
    stdout_f, stderr_f = work / "stdout", work / "stderr"
    env = dict(os.environ, **(env_extra or {}))
    cmd = scope_command(cap_bytes, argv)
    with stdout_f.open("wb") as so, stderr_f.open("wb") as se:
        proc = subprocess.Popen(cmd, stdout=so, stderr=se, env=env, stdin=subprocess.DEVNULL)
        cgroup = find_cgroup(proc.pid)
        obs = monitor(proc, cgroup, scratch, timeout_s)
    tail = "\n".join(stderr_f.read_text(errors="replace").strip().splitlines()[-5:])
    return obs, stdout_f.read_text(errors="replace"), tail


__all__ = [
    "CGROUP_ROOT",
    "GIB",
    "KILL_PEAK_FRACTION",
    "POLL_S",
    "SPILL_EVERY",
    "Cgroup",
    "Observation",
    "cgroup_of",
    "classify",
    "dir_bytes",
    "find_cgroup",
    "monitor",
    "read_result",
    "run_scoped",
    "scope_command",
]
