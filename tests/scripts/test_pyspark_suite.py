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
from pyspark_suite.junit import oracle_skips, parse
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


def test_file_level_failure_is_reported() -> None:
    from pyspark_suite.denominator import Denominator
    from pyspark_suite.junit import Record

    den = Denominator("4.3.0", 1, 0, 0, 0, 1, 0, {})
    rec = Record("4.3.0", "a.py", "<file>", "<file>", "timeout", 1)
    out = summarize(den, {"a.py::A::t"}, [rec], source="run")
    assert out["file_failures"] == ["a.py"]
    assert out["not_run"] == 1


def test_server_refuses_stale_listener() -> None:
    import socket

    from pyspark_suite.runner import server

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        with pytest.raises(RuntimeError, match="already has a listener"), server(["true"], s.getsockname()[1]):
            pass


def test_server_fails_fast_when_process_exits() -> None:
    from pyspark_suite.runner import server

    with pytest.raises(RuntimeError, match="exited early"), server(["true"], 1):
        pass


def test_static_refuses_missing_checkout(tmp_path: Path) -> None:
    import argparse

    from pyspark_suite.__main__ import cmd_static

    with pytest.raises(SystemExit):
        cmd_static(argparse.Namespace(version="9.9.9", checkout=tmp_path / "nope"))


def test_collection_error_is_a_file_failure() -> None:
    from pyspark_suite.denominator import Denominator
    from pyspark_suite.junit import Record

    den = Denominator("4.3.0", 1, 0, 0, 0, 1, 0, {})
    rec = Record("4.3.0", "a.py", "", "a", "error", 0)
    out = summarize(den, {"a.py::A::t"}, [rec], source="run")
    assert out["file_failures"] == ["a.py"]


def test_conditional_skips_leave_the_denominator_outside_the_cap(tmp_path):
    # The runner names a file's XML `<rel with / as __>.xml`.
    G = "pyspark.sql.tests.connect.sub.test_g.GTests"
    (tmp_path / "sub__test_g.py.xml").write_text(
        "<testsuites><testsuite>"
        f'<testcase classname="{G}" name="test_jvm" time="0"><skipped message="Requires JVM access"/></testcase>'
        f'<testcase classname="{G}" name="test_np" time="0"><skipped message="numpy must be installed"/></testcase>'
        f'<testcase classname="{G}" name="test_bad" time="0"><failure message="x"/></testcase>'
        "</testsuite></testsuites>"
    )
    cond = oracle_skips(tmp_path, "4.3.0")
    assert cond == {"sub/test_g.py::GTests::test_jvm"}  # the numpy skip is an oracle-environment gap, not conditional
    collected = {f"a.py::T::t{i}" for i in range(8)} | cond | {"sub/test_g.py::GTests::test_bad", "b.py::T::skipme"}
    den, remaining = compute("4.3.0", collected, {"b.py::T::skipme"}, set(), [], cond)
    assert (den.collected, den.spark_skips, den.conditional_skips, den.denominator) == (11, 1, 1, 9)
    assert den.cap == int(0.2 * 9)
    assert den.as_dict()["conditional_skips"] == 1
    assert not (remaining & cond)


def test_no_oracle_run_means_no_conditional_skips():
    den, _ = compute("4.3.0", {"a.py::T::t"}, set(), set(), [])
    assert den.conditional_skips == 0


def test_ratchet_cli_fails_on_lost_pass_or_low_rate(tmp_path, monkeypatch):
    import argparse
    import json

    from pyspark_suite import __main__ as cli

    (tmp_path / "summary").mkdir()
    base = {"version": "4.0.5", "source": "run", "passed": 2, "pass_rate": 80.0, "passed_ids": ["a", "b"]}
    (tmp_path / "summary" / "4.0.5.json").write_text(json.dumps(base))
    monkeypatch.setattr(cli, "DATA", tmp_path)

    def run(cur, **kw):
        f = tmp_path / "cur.json"
        f.write_text(json.dumps(dict(base, **cur)))
        a = argparse.Namespace(version="4.0.5", current=f, min_rate=kw.get("min_rate"), report_only=kw.get("ro", False))
        cli.cmd_ratchet(a)

    run({})  # same list: ok
    run({"passed_ids": ["a", "b", "c"]})  # gained a pass: ok
    with pytest.raises(SystemExit):
        run({"passed_ids": ["a"]})  # lost b
    run({"passed_ids": ["a"]}, ro=True)  # report-only never fails
    with pytest.raises(SystemExit):
        run({}, min_rate=90.0)  # under the floor
    with pytest.raises(SystemExit):
        run({"source": "static"})  # not a run


def test_ratchet_with_no_committed_summary_and_report_only_with_min_rate(tmp_path, monkeypatch):
    import argparse
    import json

    from pyspark_suite import __main__ as cli

    monkeypatch.setattr(cli, "DATA", tmp_path)  # no summary/ dir: 3.5.7 before its first commit
    cur = {"version": "3.5.7", "source": "run", "passed": 1, "pass_rate": 10.0, "passed_ids": ["a"]}
    f = tmp_path / "cur.json"
    f.write_text(json.dumps(cur))
    ns = argparse.Namespace(version="3.5.7", current=f, min_rate=None, report_only=False)
    cli.cmd_ratchet(ns)  # an empty baseline loses nothing
    ns.min_rate = 50.0
    with pytest.raises(SystemExit):
        cli.cmd_ratchet(ns)
    ns.report_only = True
    cli.cmd_ratchet(ns)  # report-only silences the floor too
    ns.version = "4.0.5"
    with pytest.raises(SystemExit):
        cli.cmd_ratchet(ns)  # the file is for another version


def test_run_fails_when_every_file_fails_to_run(tmp_path, monkeypatch):
    import argparse

    from pyspark_suite import __main__ as cli
    from pyspark_suite.junit import Record

    conn = tmp_path / "co" / "python" / "pyspark" / "sql" / "tests" / "connect"
    conn.mkdir(parents=True)
    (conn / "test_a.py").write_text("import unittest\nclass FooTests(unittest.TestCase):\n    def test_x(self): pass\n")
    monkeypatch.setattr(cli, "DATA", tmp_path / "data")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "manifest.toml").write_text("")
    monkeypatch.setattr(cli, "check_pytest_timeout", lambda: None)
    monkeypatch.setattr(cli, "stage_tests", lambda _c: conn)
    monkeypatch.setattr(cli, "run_file", lambda v, _t, rel, *_a: [Record(v, rel, "<file>", "<file>", "error", 0)])
    a = argparse.Namespace(
        version="4.0.5",
        checkout=tmp_path / "co",
        port=1,
        oracle_junit=None,
        only=None,
        out=tmp_path / "out",
        server_cmd=["x"],
    )
    with pytest.raises(SystemExit, match="every selected file failed"):
        cli.cmd_run(a)


def test_ratchet_refuses_min_rate_while_conditional_skips_are_in_the_denominator(tmp_path, monkeypatch):
    import argparse
    import json

    from pyspark_suite import __main__ as cli

    monkeypatch.setattr(cli, "DATA", tmp_path)
    cur = {"version": "4.0.5", "source": "run", "passed": 2, "pass_rate": 99.0, "passed_ids": [], "counts": {"skip": 3}}
    f = tmp_path / "cur.json"
    f.write_text(json.dumps(cur))
    a = argparse.Namespace(version="4.0.5", current=f, min_rate=75.0, report_only=False)
    with pytest.raises(SystemExit, match="conditional"):
        cli.cmd_ratchet(a)
    cur["denominator"] = {"conditional_skips": 3}
    f.write_text(json.dumps(cur))
    cli.cmd_ratchet(a)  # excluded from the denominator: the floor applies normally
