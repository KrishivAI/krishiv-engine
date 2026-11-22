# SPDX-License-Identifier: Apache-2.0
import krishiv


def test_version_is_str():
    assert isinstance(krishiv.__version__, str)
    assert krishiv.__version__


def test_build_info_reports_mimalloc():
    # Every shipped wheel (abi3 and free-threaded) links mimalloc.
    info = krishiv.build_info()
    assert info["version"] == krishiv.__version__
    assert info["allocator"] == "mimalloc"
    assert set(info) == {"version", "allocator", "commit"}


def test_the_module_installs_the_allocator_policy():
    # The module owns its allocator, so import installs the release hook and the resident-set
    # probe at every memory limit (acceptance SQL-33, SQL-34).
    from krishiv import _native

    assert _native.__allocator_policy__ == "release-hook,resident-probe"


def test_the_pressure_epoch_is_readable():
    # A count for measurements: how often the guard asked threads to return freed memory.
    from krishiv import _native

    epoch = _native._pressure_epoch()
    assert isinstance(epoch, int) and epoch >= 0


def test_build_info_commit_is_a_sha_or_unknown():
    commit = krishiv.build_info()["commit"]
    assert commit == "unknown" or (len(commit) >= 7 and all(c in "0123456789abcdef" for c in commit))
