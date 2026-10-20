# SPDX-License-Identifier: Apache-2.0
import krishiv


def test_version_is_str():
    assert isinstance(krishiv.__version__, str)
    assert krishiv.__version__


def test_build_info_reports_mimalloc():
    # Every shipped wheel (abi3 and free-threaded) links mimalloc.
    assert krishiv.build_info() == {"version": krishiv.__version__, "allocator": "mimalloc"}
