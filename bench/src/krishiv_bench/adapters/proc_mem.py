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
from pathlib import Path


def _status_kb(text: str, key: str) -> int:
    for line in text.splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1])
    raise RuntimeError(f"{key} not found in /proc status")


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
    """Peak RSS of a process and all its descendants over a window (``start`` .. ``stop``).

    Per-process VmHWM is reset with clear_refs=5 (``LocalProc``) at ``start``. Peak = the larger of
    (a) the highest sum of VmRSS over the whole tree seen by a sampler thread (processes peaking together add up)
    and (b) the sum of the VmHWM of the tree members alive at ``stop`` (catches spikes between samples; an upper
    bound when members peak at different moments). A descendant that exits inside the window is seen only by (a).
    """

    def __init__(self, pid: int, interval_s: float = 0.02):
        self.pid, self.interval_s = pid, interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak = 0

    def _sample(self) -> int:
        total = 0
        for pid in descendants(self.pid):
            with contextlib.suppress(OSError, RuntimeError, ValueError):
                total += LocalProc(str(pid)).read()[0]
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
        self._peak = self._sample()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        """Ends the window; returns the peak in bytes."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        return max(self._peak, self._sample(), self._hwm_sum())
