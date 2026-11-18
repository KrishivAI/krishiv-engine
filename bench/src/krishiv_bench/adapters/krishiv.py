"""Krishiv adapter: drives the bench-only ``krishiv-bench-runner`` binary (dist-profile build of ``krishiv-session``).

The Python facade cannot execute queries yet, so the adapter speaks the runner's line protocol
(see ``crates/krishiv-bench-runner/src/lib.rs``) over stdin/stdout: tables are registered once as Parquet
views at ``setup()`` (no pre-loading, for parity with the other adapters), then every query is one line.

* ``timed_run(query)`` -> ``(row_count, seconds)``: ``RUN``; the seconds are measured inside the runner
  around parse + analyse + plan + execute + materialising every batch.
* ``profile(query)`` -> dict: ``PROFILE``; as ``timed_run`` plus the wall time of the parse+analyse and the
  plan+execute phases (per-operator rows and times wait for the session to expose the executor's metrics).
* ``plan_profile(query)`` -> dict: ``PLAN``; the ``krishiv.query-profile`` v1 document of one run (physical plan
  tree and per-operator rows and times), plus ``rows`` and ``seconds``. Detailed metrics are on: not a timing.
* ``run(query)`` -> rows: ``DUMP`` (the result is written to Parquet outside the timed region, then read back),
  for correctness checks. The generic suite runner times this call, so use ``abba`` for Krishiv timings.

Build the runner first (the shipped build level, see bench/README.md):
``RUSTFLAGS='-C target-cpu=x86-64-v3 -C llvm-args=-x86-branches-within-32B-boundaries' cargo build --profile dist
-p krishiv-bench-runner`` (path override: ``--runner PATH`` on the CLI, which wins over ``$KRISHIV_BENCH_RUNNER``).
Extra session settings go through ``conf`` (``--conf key=value`` on the runner).
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import select
import shutil
import signal
import subprocess
import tempfile
import urllib.parse
from pathlib import Path

from .base import Dataset, Engine, Measure, Query
from .proc_mem import TreeProc, tree_measure


def _enc(path: Path) -> str:
    """Percent-encodes a path for the line protocol (no whitespace; see ``encode_path`` in the runner)."""
    return urllib.parse.quote(str(path), safe="/-_.")


class RunnerError(RuntimeError):
    """The runner answered ``ERR`` (the engine rejected or failed the query)."""


class RunnerCrashed(RuntimeError):
    """The runner died or timed out; it is restarted on the next command."""


class RunnerBuildMismatch(RuntimeError):
    """The runner is not built at the shipped CPU level, so its timings are not the product's (bench/README.md)."""


# The CPU level the shipped binaries and wheels are built for, per machine (`.github/workflows/binaries.yml`).
SHIPPED_TARGET_CPU = {"x86_64": "x86-64-v3", "amd64": "x86-64-v3", "aarch64": "neoverse-n1", "arm64": "neoverse-n1"}
# Set to 1 to run a runner of another CPU level anyway (a build-level experiment, never a Krishiv timing).
ANY_BUILD_ENV = "KRISHIV_BENCH_ANY_BUILD"
SHIPPED_BUILD_COMMAND = (
    "RUSTFLAGS='-C target-cpu=x86-64-v3 -C llvm-args=-x86-branches-within-32B-boundaries' "
    "cargo build --profile dist -p krishiv-bench-runner"
)


def shipped_rustflags(machine: str | None = None) -> str:
    """The ``RUSTFLAGS`` of a runner at the shipped level of ``machine`` (bench/README.md); empty where none is."""
    want = SHIPPED_TARGET_CPU.get((machine or platform.machine()).lower())
    if want is None:
        return ""
    align = " -C llvm-args=-x86-branches-within-32B-boundaries" if want.startswith("x86-64") else ""
    return f"-C target-cpu={want}{align}"


def build_level_problem(binary: Path, build: dict[str, str], machine: str | None = None) -> str | None:
    """Why ``build`` (a runner's ``--build-info`` fields) is not the shipped CPU level of ``machine``, or None.

    A runner that cannot report its level (``unknown``: too old, or a test double) and a machine with no shipped
    level are not judged. Study 28 measured a ``target-cpu=default`` runner 1.14x slower over TPC-H (q04 1.59x), so a
    timing of one says little about what users run.
    """
    want = SHIPPED_TARGET_CPU.get((machine or platform.machine()).lower())
    cpu = build.get("target_cpu", "unknown")
    if want is None or cpu in ("unknown", want):
        return None
    return (
        f"{binary} is built for target-cpu={cpu}, not the shipped {want}: rebuild it "
        f"(RUSTFLAGS='{shipped_rustflags(machine)}' cargo build --profile dist -p krishiv-bench-runner), "
        f"or set {ANY_BUILD_ENV}=1 for a run that compares build levels"
    )


def any_build_allowed() -> bool:
    """Whether ``$KRISHIV_BENCH_ANY_BUILD`` lets a runner of another CPU level run."""
    return os.environ.get(ANY_BUILD_ENV, "") not in ("", "0")


def read_build_info(binary: Path) -> dict[str, str]:
    """``binary``'s ``--build-info`` as ``build_profile``, ``target_cpu`` and (when reported) ``branch_align``;
    ``unknown`` fields when it cannot say (missing, too old, or run on a CPU below its build level)."""
    build = {"build_profile": "unknown", "target_cpu": "unknown"}
    try:
        out = subprocess.run(
            [str(binary), "--build-info"], capture_output=True, text=True, timeout=30, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return build
    fields = dict(kv.split("=", 1) for kv in out.split() if "=" in kv)
    build = {"build_profile": fields.get("profile", "unknown"), "target_cpu": fields.get("target-cpu", "unknown")}
    if "branch-align" in fields:
        build["branch_align"] = fields["branch-align"]
    return build


def require_shipped_build(binary: Path | str | None = None) -> None:
    """Exits with the reason when the runner (``runner_binary(binary)``) is not built at the shipped CPU level and
    ``$KRISHIV_BENCH_ANY_BUILD`` is unset: a check made before a long run starts, not after its first query."""
    path = runner_binary(binary)
    if any_build_allowed() or not path.exists():
        return
    problem = build_level_problem(path, read_build_info(path))
    if problem:
        raise SystemExit(problem)


def runner_binary(explicit: Path | str | None = None) -> Path:
    """The runner path: an explicit ``--runner`` wins over ``$KRISHIV_BENCH_RUNNER``, then the dist target."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("KRISHIV_BENCH_RUNNER")
    if env:
        return Path(env)
    target = Path(os.environ.get("CARGO_TARGET_DIR", Path(__file__).resolve().parents[4] / "target"))
    return target / "dist" / "krishiv-bench-runner"


class KrishivEngine(Engine):
    name = "krishiv"

    def __init__(
        self,
        threads: int | None = None,
        memory_limit: str = "12g",
        binary: Path | str | None = None,
        conf: list[tuple[str, str]] | None = None,
        launcher: list[str] | None = None,
        tables: dict[str, list[str]] | None = None,
        checksum_in_run: bool = False,
    ):
        self.threads = threads or os.cpu_count() or 1
        self.memory_limit = memory_limit
        # Time with ``RUN_CHECKED``: the runner also returns the result's checksum, taken after the seconds, so the
        # timed run itself is verified (see ``timed_run``). Off by default: it holds the result a little longer.
        self.checksum_in_run = checksum_in_run
        self._run_checksum: dict | None = None
        self.binary = runner_binary(binary)
        self.conf = list(conf or [])
        # Tables read from URLs (``--table NAME=URL``: s3://, https://, hf://); when given, the dataset
        # directory is not read (the runner is started without ``--data``).
        self.tables = {name: list(urls) for name, urls in (tables or {}).items()}
        self.launcher = list(launcher or [])  # command prefix (e.g. `perf record ... --`)
        self._proc: subprocess.Popen[str] | None = None
        self._dataset: Dataset | None = None
        self._version = "unknown"
        self._tmp = Path(tempfile.mkdtemp(prefix="krishiv-bench-"))
        self.timeout_s: float | None = None
        self._starts = 0
        self._build: dict[str, str] | None = None
        self._stderr_log: Path | None = None
        self._tree: TreeProc | None = None

    def version(self) -> str:
        return self._version

    def begin_measure(self) -> None:
        """Opens a resource window (peak RSS, CPU, I/O) over the runner process, starting it first if need be."""
        self.end_measure()
        if self._proc is None and self._dataset is not None:
            self._start()
        if self._proc is not None:
            self._tree = TreeProc(self._proc.pid)
            self._tree.start()

    def end_measure(self) -> Measure:
        tree, self._tree = self._tree, None
        return tree_measure(tree)

    def config(self) -> dict:
        return {
            "threads": self.threads,
            "memory_limit": self.memory_limit,
            "binary": self.binary.name,
            "conf": [f"{k}={v}" for k, v in self.conf],
            **self._runner_build(),
        }

    def build_info(self) -> dict[str, str]:
        build = self._runner_build()
        return {"allocator": "mimalloc", "runner": f"krishiv-bench-runner ({build['build_profile']})", **build}

    def _runner_build(self) -> dict[str, str]:
        """The runner's own ``--build-info`` (``profile=dist target-cpu=x86-64-v3 branch-align=32B``), so a result says
        what it measured. ``branch_align`` is present only when the runner reports it.

        ``unknown`` when the binary cannot say (missing, too old, or run on a CPU below its build level).
        """
        if self._build is None:
            self._build = read_build_info(self.binary)
        return self._build

    def runner_build_info(self) -> dict[str, str] | None:
        """The runner's ``--build-info`` fields, or None when the binary cannot report them (too old, missing)."""
        build = self._runner_build()
        return None if build["build_profile"] == "unknown" else dict(build)

    # ------------------------------------------------------------------ process
    def _start(self) -> None:
        assert self._dataset is not None, "setup() not called"
        if not self.binary.exists():
            raise FileNotFoundError(f"{self.binary} missing: {SHIPPED_BUILD_COMMAND}")
        cmd = [
            *self.launcher,
            str(self.binary),
            *(["--data", str(self._dataset.root)] if not self.tables else []),
            *[arg for name, urls in self.tables.items() for url in urls for arg in ("--table", f"{name}={url}")],
            "--threads",
            str(self.threads),
            "--memory-limit",
            self.memory_limit,
        ]
        for k, v in self.conf:
            cmd += ["--conf", f"{k}={v}"]
        self._starts += 1
        self._stderr_log = self._tmp / f"runner-{self._starts}.stderr"
        with self._stderr_log.open("wb") as err:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=err,
                text=True,
                bufsize=1,
                # A launcher (perf) is the direct child and the runner its child: own process group, so a
                # kill reaches both and never orphans the runner.
                start_new_session=bool(self.launcher),
            )
        ready = self._readline(120)
        parts = ready.split(" ", 2)
        if parts[0] != "READY" or len(parts) < 3:
            raise self._crashed(f"runner did not start: {ready!r}")
        self._version = parts[1]

    def _stderr_tail(self) -> str:
        """The last lines the runner wrote to stderr (startup errors, panics), for crash messages."""
        if self._stderr_log is None:
            return ""
        try:
            lines = self._stderr_log.read_text(errors="replace").strip().splitlines()[-5:]
        except OSError:
            return ""
        return " | ".join(lines)

    def _crashed(self, what: str) -> RunnerCrashed:
        tail = self._stderr_tail()
        return RunnerCrashed(f"{what}: {tail}" if tail else what)

    def _readline(self, timeout: float | None) -> str:
        proc = self._proc
        assert proc is not None
        assert proc.stdout is not None
        if timeout is not None:
            ready, _, _ = select.select([proc.stdout], [], [], timeout)
            if not ready:
                self.cancel()
                raise RunnerCrashed(f"no answer within {timeout:.0f}s")
        line = proc.stdout.readline()
        if not line:
            self._kill()
            raise self._crashed("runner exited")
        return line.rstrip("\n")

    def _command(self, text: str) -> str:
        if self._proc is None or self._proc.poll() is not None:
            self._kill()
            self._start()
        proc = self._proc
        assert proc is not None
        assert proc.stdin is not None
        try:
            proc.stdin.write(text + "\n")
            proc.stdin.flush()
        except BrokenPipeError:
            self._kill()
            raise self._crashed("runner exited") from None
        reply = self._readline(self.timeout_s)
        if reply.startswith("ERR "):
            raise RunnerError(reply[4:])
        if not reply.startswith("OK "):
            self._kill()  # later replies would be matched to the wrong commands
            raise RunnerCrashed(f"unexpected reply {reply!r}")
        return reply[3:]

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            if self.launcher:
                with contextlib.suppress(OSError):
                    os.killpg(proc.pid, signal.SIGKILL)
            proc.kill()
            proc.wait()
            for f in (proc.stdin, proc.stdout):
                with contextlib.suppress(OSError):
                    if f:
                        f.close()

    def _sql_file(self, query: Query) -> Path:
        f = self._tmp / f"{query.id}.sql"
        f.write_text(query.sql)
        return f

    # ------------------------------------------------------------------ Engine
    def setup(self, dataset: Dataset) -> None:
        self._tmp.mkdir(parents=True, exist_ok=True)  # `teardown()` removes it; setup may run again
        self._dataset = dataset
        self.check_build_level()
        self._start()

    def check_build_level(self) -> None:
        """Refuses (``RunnerBuildMismatch``) a runner not built at the shipped CPU level, unless
        ``$KRISHIV_BENCH_ANY_BUILD`` is set. A missing binary is left to ``_start`` to report."""
        if any_build_allowed() or not self.binary.exists():
            return
        problem = build_level_problem(self.binary, self._runner_build())
        if problem:
            raise RunnerBuildMismatch(problem)

    def timed_run(self, query: Query) -> tuple[int, float]:
        """Executes to completion; returns the row count and the runner-measured seconds.

        With ``checksum_in_run`` the runner's ``RUN_CHECKED`` hands back the checksum of the very result it timed
        (computed after the seconds are taken). A runner that predates it answers ``bad command``; the engine then
        falls back to ``RUN`` and the checksum of an untimed ``DUMP`` of the query (``result_checksum``).
        """
        self._run_checksum = None
        if self.checksum_in_run:
            try:
                rows, secs, detail = self._command(f"RUN_CHECKED {_enc(self._sql_file(query))}").split(" ", 2)
            except RunnerError as e:
                if not str(e).startswith("bad command"):
                    raise
                self.checksum_in_run = False  # an older runner
            else:
                self._run_checksum = json.loads(detail)
                self._pending = None
                return int(rows), float(secs)
        rows, secs = self._command(f"RUN {_enc(self._sql_file(query))}").split()
        self._pending = query
        return int(rows), float(secs)

    def result_checksum(self) -> dict | None:
        """The checksum of the last timed run: the runner's own (``RUN_CHECKED``) or, from a runner whose ``RUN``
        hands no rows back, that of an untimed ``DUMP`` of the query (marked ``"source": "dump"``)."""
        import pyarrow.parquet as pq

        from ..result_check import checksum

        if self._run_checksum is not None:
            done, self._run_checksum = self._run_checksum, None
            return done
        query, self._pending = self._pending, None
        if query is None:
            return None
        out = self._tmp / f"{query.id}.chk"
        shutil.rmtree(out, ignore_errors=True)
        try:
            self.dump(query, out)
            parts = sorted(out.glob("*.parquet")) if out.is_dir() else [out]
            import pyarrow as pa

            return {**checksum(pa.concat_tables([pq.read_table(p) for p in parts])), "source": "dump"}
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def profile(self, query: Query) -> dict:
        """One ``PROFILE`` run: row count, total seconds and the runner's per-phase wall times."""
        rows, secs, detail = self._command(f"PROFILE {_enc(self._sql_file(query))}").split(" ", 2)
        return {"rows": int(rows), "seconds": float(secs), **json.loads(detail)}

    def plan_profile(self, query: Query) -> dict:
        """One ``PLAN`` run: the profile document of the query's physical plan, its row count and seconds."""
        rows, secs, doc = self._command(f"PLAN {_enc(self._sql_file(query))}").split(" ", 2)
        return {"rows": int(rows), "seconds": float(secs), "profile": json.loads(doc)}

    def dump(self, query: Query, out: Path) -> tuple[int, float]:
        """Executes to completion and writes the result as Parquet to ``out``; returns the rows and runner seconds."""
        rows, secs = self._command(f"DUMP {_enc(self._sql_file(query))} {_enc(out)}").split()
        return int(rows), float(secs)

    def run(self, query: Query) -> list[tuple]:
        import pyarrow.parquet as pq

        out = self._tmp / f"{query.id}.out"
        shutil.rmtree(out, ignore_errors=True)
        self.dump(query, out)
        parts = sorted(out.glob("*.parquet")) if out.is_dir() else [out]
        rows: list[tuple] = []
        for part in parts:
            table = pq.read_table(part)
            cols = [c.to_pylist() for c in table.columns]
            rows.extend(zip(*cols, strict=True) if cols else [])
        shutil.rmtree(out, ignore_errors=True)
        return rows

    def cancel(self) -> None:
        self._kill()

    def teardown(self) -> None:
        proc = self._proc
        if proc is not None and proc.stdin is not None:
            with contextlib.suppress(OSError):
                proc.stdin.write("QUIT\n")
                proc.stdin.flush()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
        self._kill()
        shutil.rmtree(self._tmp, ignore_errors=True)
