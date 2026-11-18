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


def test_build_info_commit_is_a_sha_or_unknown():
    commit = krishiv.build_info()["commit"]
    assert commit == "unknown" or (len(commit) >= 7 and all(c in "0123456789abcdef" for c in commit))
