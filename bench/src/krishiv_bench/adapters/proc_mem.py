"""Peak-RSS measurement shared by all adapters: per-run VmHWM after resetting it with clear_refs=5.

Same method for in-process engines (read /proc/self) and containers (docker exec on the JVM pid).
Caveat (documented in README): a JVM retains its committed heap, so after a big query the next query's
baseline is already high; DuckDB's number is the whole harness process (Python + pyarrow baseline included).
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from .base import Measure


def _status_kb(text: str, key: str) -> int:
    for line in text.splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1])
    raise RuntimeError(f"{key} not found in /proc status")


@dataclass(frozen=True)
class IoCounters:
    """Cumulative ``/proc/<pid>/io`` counters: ``rchar`` bytes asked of read-like syscalls (page cache included),
    ``read_bytes`` bytes fetched from the storage layer (a page-cache hit adds nothing), ``syscr`` read syscalls."""

    rchar: int = 0
    read_bytes: int = 0
    syscr: int = 0

    def __add__(self, o: IoCounters) -> IoCounters:
        return IoCounters(self.rchar + o.rchar, self.read_bytes + o.read_bytes, self.syscr + o.syscr)

    def __sub__(self, o: IoCounters) -> IoCounters:
        return IoCounters(self.rchar - o.rchar, self.read_bytes - o.read_bytes, self.syscr - o.syscr)


def parse_io(text: str) -> IoCounters:
    """The counters in the text of ``/proc/<pid>/io`` (``key: value`` lines)."""
    vals = {}
    for line in text.splitlines():
        key, _, val = line.partition(":")
        if val.strip().isdigit():
            vals[key.strip()] = int(val)
    missing = {"rchar", "read_bytes", "syscr"} - vals.keys()
    if missing:
        raise RuntimeError(f"{sorted(missing)} not found in /proc io")
    return IoCounters(vals["rchar"], vals["read_bytes"], vals["syscr"])


class LocalProc:
    def __init__(self, pid: str = "self"):
        self.pid = pid

    def check(self) -> None:
        self.reset()  # raises if not permitted

    def reset(self) -> None:
        with open(f"/proc/{self.pid}/clear_refs", "w") as f:
            f.write("5")

    def read(self) -> tuple[int, int, float]:
        txt = Path(f"/proc/{self.pid}/status").read_text()
        st = Path(f"/proc/{self.pid}/stat").read_text().rsplit(")", 1)[1].split()
        cpu = (int(st[11]) + int(st[12])) / 100.0  # utime+stime (fields 14,15)
        return _status_kb(txt, "VmRSS") * 1024, _status_kb(txt, "VmHWM") * 1024, cpu

    def read_io(self) -> IoCounters:
        return parse_io(Path(f"/proc/{self.pid}/io").read_text())


class DockerProc:
    def __init__(self, container: str, process_name: str = "java"):
        self.container, self.pname, self.pid = container, process_name, None

    def _sh(self, script: str) -> str:
        r = subprocess.run(
            ["docker", "exec", self.container, "bash", "-c", script], capture_output=True, text=True, timeout=30
        )
        if r.returncode != 0:
            raise RuntimeError(f"docker exec failed (rc={r.returncode}): {script!r}: {r.stderr.strip()[:200]}")
        return r.stdout.strip()

    def check(self) -> None:
        # exact process-name match avoids matching the `bash -c` wrapper's own cmdline
        self.pid = self._sh(f"pgrep -x {self.pname} | head -1")
        if not self.pid.isdigit():
            raise RuntimeError(f"no '{self.pname}' process found in {self.container}")
        self.reset()

    def reset(self) -> None:
        self._sh(f"echo 5 > /proc/{self.pid}/clear_refs")

    def read(self) -> tuple[int, int, float]:
        out = self._sh(
            f"awk '/^VmRSS|^VmHWM/{{print $2}}' /proc/{self.pid}/status; cut -d' ' -f14,15 /proc/{self.pid}/stat"
        )
        rss, hwm, ticks = out.split("\n")
        u, s = ticks.split()
        return int(rss) * 1024, int(hwm) * 1024, (int(u) + int(s)) / 100.0

    def read_io(self) -> IoCounters:
        return parse_io(self._sh(f"cat /proc/{self.pid}/io"))


def descendants(root: int) -> list[int]:
    """``root`` and every live descendant pid (a JVM's helper processes, a runner child), via /proc ppid links."""
    children: dict[int, list[int]] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            # the comm field may contain spaces and parentheses: split after the last ')'
            ppid = int(Path(entry.path, "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue  # exited while we were scanning
        children.setdefault(ppid, []).append(int(entry.name))
    out, todo = [], [root]
    while todo:
        pid = todo.pop()
        out.append(pid)
        todo.extend(children.get(pid, []))
    return out


def tree_cpu_seconds(root: int) -> float:
    """User + system CPU seconds consumed so far by ``root`` and its live descendants (an exited one is not counted)."""
    total = 0.0
    for pid in descendants(root):
        with contextlib.suppress(OSError, RuntimeError, ValueError):
            total += LocalProc(str(pid)).read()[2]
    return total


class TreeProc:
    """Peak RSS, CPU time and I/O of a process and all its descendants over a window (``start`` .. ``stop``).

    Per-process VmHWM is reset with clear_refs=5 (``LocalProc``) at ``start``. Peak = the larger of
    (a) the highest sum of VmRSS over the whole tree seen by a sampler thread (processes peaking together add up)
    and (b) the sum of the VmHWM of the tree members alive at ``stop`` (catches spikes between samples; an upper
    bound when members peak at different moments). A descendant that exits inside the window is seen only by (a).

    CPU seconds and I/O counters are cumulative per process: the window's value is, per pid, the last reading the
    sampler (or ``stop``) took minus the reading at ``start`` (zero for a process born inside the window), summed.
    A process that exits between its last sample and ``stop`` loses its final slice (at most ``interval_s``).
    After ``stop``, ``cpu_seconds`` and ``io`` hold them; ``None`` where the platform gave no reading.
    """

    def __init__(self, pid: int, interval_s: float = 0.02):
        self.pid, self.interval_s = pid, interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak = 0
        self._cpu0: dict[int, float] = {}
        self._io0: dict[int, IoCounters] = {}
        self._cpu: dict[int, float] = {}
        self._io: dict[int, IoCounters] = {}
        self.cpu_seconds: float | None = None
        self.io: IoCounters | None = None

    def _sample(self) -> int:
        total = 0
        for pid in descendants(self.pid):
            p = LocalProc(str(pid))
            with contextlib.suppress(OSError, RuntimeError, ValueError):
                rss, _, cpu = p.read()
                total += rss
                self._cpu[pid] = cpu
            with contextlib.suppress(OSError, RuntimeError, ValueError):
                self._io[pid] = p.read_io()
        return total

    def _hwm_sum(self) -> int:
        total = 0
        for pid in descendants(self.pid):
            with contextlib.suppress(OSError, RuntimeError, ValueError):
                total += LocalProc(str(pid)).read()[1]
        return total

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._peak = max(self._peak, self._sample())
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        for pid in descendants(self.pid):
            with contextlib.suppress(OSError):
                LocalProc(str(pid)).reset()
        self._cpu, self._io = {}, {}
        self._peak = self._sample()
        self._cpu0, self._io0 = dict(self._cpu), dict(self._io)
        self.cpu_seconds, self.io = None, None
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        """Ends the window; returns the peak in bytes."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        peak = max(self._peak, self._sample(), self._hwm_sum())
        if self._cpu:
            self.cpu_seconds = max(0.0, sum(c - self._cpu0.get(pid, 0.0) for pid, c in self._cpu.items()))
        if self._io:
            zero = IoCounters()
            total = IoCounters()
            for pid, c in self._io.items():
                total = total + (c - self._io0.get(pid, zero))
            self.io = total
        return peak


def tree_measure(tree: TreeProc | None) -> Measure:
    """The ``Measure`` of a stopped window (peak RSS, CPU seconds, I/O); empty when there was no window."""
    if tree is None:
        return Measure()
    peak = tree.stop()
    io = tree.io
    return Measure(
        peak_rss_bytes=peak,
        cpu_seconds=tree.cpu_seconds,
        rchar=io.rchar if io else None,
        read_bytes=io.read_bytes if io else None,
        syscr=io.syscr if io else None,
    )


class SelfTreeMeasure:
    """``begin_measure`` / ``end_measure`` for an engine that runs in the harness process (and its children):
    Polars, DataFusion, Spark local mode. The numbers cover the whole harness process, Python baseline included."""

    _tree: TreeProc | None = None

    def begin_measure(self) -> None:
        self.end_measure()
        self._tree = TreeProc(os.getpid())
        self._tree.start()

    def end_measure(self) -> Measure:
        tree, self._tree = self._tree, None
        return tree_measure(tree)
