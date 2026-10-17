# SPDX-License-Identifier: Apache-2.0
import krishiv


def test_version_is_str():
    assert isinstance(krishiv.__version__, str)
    assert krishiv.__version__
