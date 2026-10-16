"""Peak-RSS measurement shared by all adapters: per-run VmHWM after resetting it with clear_refs=5.

Same method for in-process engines (read /proc/self) and containers (docker exec on the JVM pid).
Caveat (documented in README): a JVM retains its committed heap, so after a big query the next query's
baseline is already high; DuckDB's number is the whole harness process (Python + pyarrow baseline included).
"""
from __future__ import annotations

import subprocess


def _status_kb(text: str, key: str) -> int | None:
    for line in text.splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1])
    return None


class LocalProc:
    def __init__(self, pid: str = "self"):
        self.pid = pid

    def check(self) -> None:
        self.reset()  # raises if not permitted

    def reset(self) -> None:
        with open(f"/proc/{self.pid}/clear_refs", "w") as f:
            f.write("5")

    def read(self) -> tuple[int, int, float]:
        txt = open(f"/proc/{self.pid}/status").read()
        st = open(f"/proc/{self.pid}/stat").read().rsplit(")", 1)[1].split()
        cpu = (int(st[11]) + int(st[12])) / 100.0  # utime+stime (fields 14,15)
        return _status_kb(txt, "VmRSS") * 1024, _status_kb(txt, "VmHWM") * 1024, cpu


class DockerProc:
    def __init__(self, container: str, process_name: str = "java"):
        self.container, self.pname, self.pid = container, process_name, None

    def _sh(self, script: str) -> str:
        r = subprocess.run(["docker", "exec", self.container, "bash", "-c", script],
                           capture_output=True, text=True, timeout=30)
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
        out = self._sh(f"awk '/^VmRSS|^VmHWM/{{print $2}}' /proc/{self.pid}/status; cut -d' ' -f14,15 /proc/{self.pid}/stat")
        rss, hwm, ticks = out.split("\n")
        u, s = ticks.split()
        return int(rss) * 1024, int(hwm) * 1024, (int(u) + int(s)) / 100.0
