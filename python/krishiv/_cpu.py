# SPDX-License-Identifier: Apache-2.0
"""CPU check run before the native extension is imported.

The shipped wheels are built for a CPU level above the architecture baseline
(``x86-64-v3`` on x86-64, ``neoverse-n1`` on Linux arm64). On an older CPU the
extension would die with SIGILL on its first such instruction, which no Python
code can catch. This module runs first, in pure Python, and raises a normal
``ImportError`` that names the missing features and the ``krishiv-compat``
package instead.

Which level a wheel was built for is recorded by the build in the data file
``_cpu_level`` next to this module (the release workflow writes it before it
builds the main wheels). Without that file (the ``krishiv-compat`` wheel, a
development install, macOS and Windows wheels) nothing is checked. The
features come from ``/proc/cpuinfo`` (or ``$KRISHIV_CPUINFO``), so the check runs on Linux only; if the
file cannot be read the check is skipped rather than blocking the import.

``krishiv`` and ``krishiv-compat`` are one package in two builds: both install the
folder ``krishiv``. pip has no way to refuse the second one, so the import warns
when the metadata of both is beside the package (:func:`warn_if_both_builds_installed`).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

COMPAT_PACKAGE = "krishiv-compat"

# Feature name shown to the user -> the flag the Linux kernel reports for it.
_X86_V3 = {
    "sse3": "pni",
    "ssse3": "ssse3",
    "sse4.1": "sse4_1",
    "sse4.2": "sse4_2",
    "popcnt": "popcnt",
    "cmpxchg16b": "cx16",
    "avx": "avx",
    "avx2": "avx2",
    "bmi1": "bmi1",
    "bmi2": "bmi2",
    "f16c": "f16c",
    "fma": "fma",
    "lzcnt": "abm",
    "movbe": "movbe",
    "xsave": "xsave",
}
_NEOVERSE_N1 = {
    "aes": "aes",
    "crc": "crc32",
    "dotprod": "asimddp",
    "fp16": "fphp",
    "lse": "atomics",
    "rcpc": "lrcpc",
    "rdm": "asimdrdm",
    "sha2": "sha2",
}
_LEVELS = {
    "x86-64-v3": ("flags", _X86_V3),
    "neoverse-n1": ("Features", _NEOVERSE_N1),
}


def missing_features(level: str, cpuinfo: str) -> list[str]:
    """The features of ``level`` that the ``/proc/cpuinfo`` text ``cpuinfo`` does not report.

    An unknown level, or a text without the flags line, reports nothing: the CPU cannot be told.
    """
    spec = _LEVELS.get(level)
    if spec is None:
        return []
    key, features = spec
    for line in cpuinfo.splitlines():
        name, sep, value = line.partition(":")
        if sep and name.strip() == key:
            flags = set(value.split())
            return [name for name, flag in features.items() if flag not in flags]
    return []


def failure_message(level: str, missing: list[str]) -> str:
    """The ``ImportError`` text."""
    head = f"this build of Krishiv requires a {level} CPU, but this CPU lacks: {', '.join(missing)}. "
    if level == "x86-64-v3":
        return head + (
            f"Use the compat build instead: `pip install {COMPAT_PACKAGE}` "
            "(uninstall krishiv first; it imports as krishiv)."
        )
    # The compat package is published for x86-64 only.
    return head + "No compat build is published for this architecture."


def ensure_supported(
    level_file: Path | None = None,
    cpuinfo_path: str | None = None,
    platform: str | None = None,
) -> None:
    """Raise ``ImportError`` when the running CPU lacks what this wheel was built for.

    ``cpuinfo_path`` defaults to ``$KRISHIV_CPUINFO`` and then ``/proc/cpuinfo``. The
    environment variable exists for the CI test that imports the wheel under an emulated
    pre-AVX2 CPU, where ``/proc/cpuinfo`` still describes the host.
    """
    if not (platform or sys.platform).startswith("linux"):
        return
    try:
        level = (level_file or Path(__file__).with_name("_cpu_level")).read_text().strip()
        cpuinfo = Path(cpuinfo_path or os.environ.get("KRISHIV_CPUINFO", "/proc/cpuinfo")).read_text()
    except OSError:
        return
    missing = missing_features(level, cpuinfo)
    if missing:
        raise ImportError(failure_message(level, missing))


def both_builds_installed(site_packages: Path) -> bool:
    """Whether ``site_packages`` holds the metadata of both ``krishiv`` and ``krishiv-compat``.

    Read from the ``*.dist-info`` folder names (``<name>-<version>.dist-info``), which costs one
    directory listing; ``importlib.metadata`` would add its own import time to every ``import krishiv``.
    A folder that cannot be listed counts as "not both".
    """
    try:
        entries = [entry.name for entry in site_packages.glob("[Kk]rishiv*.dist-info")]
    except OSError:
        return False
    names = set()
    for entry in entries:
        name = entry.removesuffix(".dist-info").rpartition("-")[0]
        names.add(name.lower().replace("-", "_").replace(".", "_"))
    return {"krishiv", COMPAT_PACKAGE.replace("-", "_")} <= names


def warn_if_both_builds_installed(site_packages: Path | None = None) -> None:
    """Warn when ``krishiv`` and ``krishiv-compat`` are installed in the same environment.

    The later install overwrote the files of the earlier one, and ``pip uninstall`` of either
    removes the files of both while pip still lists the other.
    """
    site = site_packages or Path(__file__).resolve().parent.parent
    if both_builds_installed(site):
        warnings.warn(
            f"krishiv and {COMPAT_PACKAGE} are both installed in {site}. They are one package in two builds "
            "and share their files: the one installed last is the one that runs, and uninstalling either "
            f"removes the files of both. Run `pip uninstall krishiv {COMPAT_PACKAGE}`, then install one of them.",
            RuntimeWarning,
            stacklevel=2,
        )
