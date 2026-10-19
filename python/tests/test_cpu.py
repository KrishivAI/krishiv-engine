# SPDX-License-Identifier: Apache-2.0
"""The pure-Python CPU check that runs before the native extension is imported."""

from pathlib import Path

import pytest

from krishiv import _cpu

NEHALEM = "processor : 0\nflags\t\t: fpu sse sse2 pni ssse3 cx16 sse4_1 sse4_2 popcnt lahf_lm\n"
HASWELL = (
    "processor : 0\nflags\t\t: fpu sse sse2 pni ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt xsave avx "
    "f16c bmi1 avx2 bmi2 abm\n"
)
A72 = "processor : 0\nFeatures\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 cpuid\n"
N1 = "Features\t: fp asimd aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc asimddp\n"


def test_old_x86_cpu_lacks_the_avx_family():
    assert _cpu.missing_features("x86-64-v3", NEHALEM) == [
        "avx",
        "avx2",
        "bmi1",
        "bmi2",
        "f16c",
        "fma",
        "lzcnt",
        "movbe",
        "xsave",
    ]


def test_haswell_class_x86_cpu_is_complete():
    assert _cpu.missing_features("x86-64-v3", HASWELL) == []


def test_arm_features_follow_the_kernel_flag_names():
    assert _cpu.missing_features("neoverse-n1", A72) == ["dotprod", "fp16", "lse", "rcpc", "rdm"]
    assert _cpu.missing_features("neoverse-n1", N1) == []


@pytest.mark.parametrize("level", ["", "unknown", "x86-64"])
def test_unknown_level_checks_nothing(level):
    assert _cpu.missing_features(level, NEHALEM) == []


def test_text_without_a_flags_line_checks_nothing():
    assert _cpu.missing_features("x86-64-v3", "processor : 0\nmodel name : x\n") == []


def _files(tmp_path: Path, level: str | None, cpuinfo: str) -> tuple[Path, str]:
    level_file = tmp_path / "_cpu_level"
    if level is not None:
        level_file.write_text(level + "\n")
    info = tmp_path / "cpuinfo"
    info.write_text(cpuinfo)
    return level_file, str(info)


def test_unsupported_cpu_raises_a_clean_import_error(tmp_path):
    level_file, info = _files(tmp_path, "x86-64-v3", NEHALEM)
    with pytest.raises(ImportError) as err:
        _cpu.ensure_supported(level_file, info, "linux")
    message = str(err.value)
    assert "avx2" in message
    assert "x86-64-v3" in message
    assert "krishiv-compat" in message


def test_supported_cpu_passes(tmp_path):
    level_file, info = _files(tmp_path, "x86-64-v3", HASWELL)
    _cpu.ensure_supported(level_file, info, "linux")


def test_wheel_without_a_level_file_is_not_checked(tmp_path):
    level_file, info = _files(tmp_path, None, NEHALEM)
    _cpu.ensure_supported(level_file, info, "linux")


def test_other_platforms_are_not_checked(tmp_path):
    level_file, info = _files(tmp_path, "x86-64-v3", NEHALEM)
    _cpu.ensure_supported(level_file, info, "darwin")
    _cpu.ensure_supported(level_file, info, "win32")


def test_unreadable_cpuinfo_does_not_block_the_import(tmp_path):
    level_file, _ = _files(tmp_path, "x86-64-v3", NEHALEM)
    _cpu.ensure_supported(level_file, str(tmp_path / "missing"), "linux")


def test_cpuinfo_can_be_pointed_at_by_the_environment(tmp_path, monkeypatch):
    level_file, info = _files(tmp_path, "x86-64-v3", NEHALEM)
    monkeypatch.setenv("KRISHIV_CPUINFO", info)
    with pytest.raises(ImportError):
        _cpu.ensure_supported(level_file, platform="linux")
