# SPDX-License-Identifier: Apache-2.0
"""Unit tests for pyspark_suite (P3-T030).

Run: (cd tests/scripts && uv run --project ../../python pytest test_pyspark_suite.py)
"""

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from pyspark_suite import manifest as mf
from pyspark_suite.denominator import compute
from pyspark_suite.junit import parse
from pyspark_suite.report import dashboard, ratchet_regressions, summarize
from pyspark_suite.static_count import census


def _entry(target, cls=mf.ReasonClass.STREAMING_P4, versions=()):
    return mf.Entry(target, cls, "n", versions)


def test_denominator_parts_and_zero_pass_rate():
    collected = {f"a.py::T::t{i}" for i in range(10)} | {"streaming/s.py::T::t0", "b.py::T::skipme", "c.py::T::env"}
    den, remaining = compute(
        "4.3.0", collected, {"b.py::T::skipme"}, {"c.py::T::env"}, [_entry("streaming/", versions=("4.3.0",))]
    )
    assert (den.collected, den.spark_skips, den.oracle_env, den.manifest, den.denominator) == (13, 1, 1, 1, 10)
    assert den.cap == int(0.2 * 12)
    s = summarize(den, remaining, [], source="static")
    assert s["pass_rate"] == 0.0
    assert s["passed"] == 0
    assert s["not_run"] == 10


def test_manifest_entry_scoped_to_version():
    ids = {"x.py::T::t"}
    den, _ = compute("4.2.1", ids, set(), set(), [_entry("x.py", versions=("4.3.0",))])
    assert den.manifest == 0
    assert den.denominator == 1


def test_cap_is_checked():
    ids = {f"f{i}.py::T::t" for i in range(10)}
    den, _ = compute("4.3.0", ids, set(), set(), [_entry("f0.py"), _entry("f1.py"), _entry("f2.py")])
    assert den.manifest == 3
    assert not den.within_cap


def test_unknown_reason_class_is_rejected(tmp_path):
    p = tmp_path / "m.toml"
    p.write_text('[[exclude]]\ntarget = "a.py"\nreason_class = "BECAUSE"\nnote = "x"\n')
    with pytest.raises(mf.ManifestError):
        mf.load(p)


def test_draft_manifest_loads():
    entries = mf.load(Path(__file__).parents[1] / "pyspark" / "manifest.toml")
    assert any(e.reason_class is mf.ReasonClass.UNVERIFIED for e in entries)
    assert all(e.note for e in entries)


def test_census_resolves_mixins_across_files_and_skips(tmp_path):
    tests = tmp_path / "sql" / "tests"
    (tests / "connect").mkdir(parents=True)
    (tests / "test_basic.py").write_text(
        textwrap.dedent(
            """
            import unittest
            class BasicMixin:
                def test_a(self): ...
                @unittest.skip("no")
                def test_b(self): ...
                def helper(self): ...
            """
        )
    )
    (tests / "connect" / "test_parity_basic.py").write_text(
        "class BasicParityTests(BasicMixin, ReusedConnectTestCase):\n    def test_c(self): ...\n"
    )
    collected, skipped = census(tests / "connect")
    assert collected == {f"test_parity_basic.py::BasicParityTests::{m}" for m in ("test_a", "test_b", "test_c")}
    assert skipped == {"test_parity_basic.py::BasicParityTests::test_b"}


def test_junit_statuses(tmp_path):
    C = "pyspark.sql.tests.connect.test_f.FTests"
    x = tmp_path / "r.xml"
    x.write_text(
        "<testsuites><testsuite>"
        '<testcase classname="pyspark.sql.tests.connect.test_f.FTests" name="test_ok" time="0.5"/>'
        f'<testcase classname="{C}" name="test_bad" time="1"><failure message="x" type="AssertionError"/></testcase>'
        f'<testcase classname="{C}" name="test_err" time="1"><error message="x" type="grpc.RpcError"/></testcase>'
        '<testcase classname="pyspark.sql.tests.connect.test_f.FTests" name="test_sk" time="0"><skipped/></testcase>'
        "</testsuite></testsuites>"
    )
    recs = parse(x, "4.3.0", "test_f.py")
    assert [r.status for r in recs] == ["pass", "fail", "error", "skip"]
    assert recs[0].test_id == "test_f.py::FTests::test_ok"
    assert recs[0].ms == 500


def test_pass_rates_ratchet_and_dashboard(tmp_path):
    ids = {"test_connect_plan.py::T::a", "d.py::T::b", "d.py::T::c", "d.py::T::d"}
    den, remaining = compute("4.3.0", ids, set(), set(), [])
    x = tmp_path / "r.xml"
    x.write_text('<testsuite><testcase classname="m.T" name="a"/><testcase classname="m.T" name="b"/></testsuite>')
    recs = parse(x, "4.3.0", "test_connect_plan.py")[:1] + parse(x, "4.3.0", "d.py")[1:]
    s = summarize(den, remaining, recs, source="run")
    assert s["pass_rate"] == 50.0
    assert s["client_only"] == 1
    assert s["pass_rate_without_client_only"] == 33.33
    worse = dict(s, passed_ids=[])
    assert ratchet_regressions(s, worse) == sorted(s["passed_ids"])
    assert "| 4.3.0 | run |" in dashboard([s])
