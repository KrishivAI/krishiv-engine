# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import check_proto_manifest as m


def test_manifest_matches_vendored_files():
    assert m.main([]) == 0


def test_version_delta_is_exactly_this():
    assert m.delta_problems() == []


def test_base_proto_unchanged_between_4_2_and_4_3():
    a = m.collect()
    assert a["4.2"]["files"]["base.proto"] == a["4.3"]["files"]["base.proto"]
    assert a["4.2"]["files"] == a["4.2.1"]["files"]
