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


def _scratch(tmp_path, monkeypatch):
    import shutil

    for v in ("4.2.1", "4.3"):
        shutil.copytree(m.PROTO / v, tmp_path / v)
    monkeypatch.setattr(m, "PROTO", tmp_path)
    return tmp_path / "4.3" / "spark" / "connect"


def test_delta_rejects_an_added_field(tmp_path, monkeypatch):
    d = _scratch(tmp_path, monkeypatch)
    f = d / "relations.proto"
    f.write_text(f.read_text().replace("  Relation right = 2;\n", "  Relation right = 2;\n  Relation extra = 3;\n"))
    assert any("Relation extra = 3;" in p for p in m.delta_problems())


def test_delta_rejects_an_added_literal_variant(tmp_path, monkeypatch):
    d = _scratch(tmp_path, monkeypatch)
    f = d / "expressions.proto"
    f.write_text(
        f.read_text().replace(
            "TimestampLTZNanos timestamp_ltz_nanos = 30;\n",
            "TimestampLTZNanos timestamp_ltz_nanos = 30;\n      Bogus bogus = 31;\n",
            1,
        )
    )
    assert any("Bogus bogus = 31;" in p for p in m.delta_problems())


def test_delta_rejects_a_removed_line(tmp_path, monkeypatch):
    d = _scratch(tmp_path, monkeypatch)
    g = d / "pipelines.proto"
    old = (m.PROTO / "4.2.1" / "spark" / "connect" / "pipelines.proto").read_text().splitlines(keepends=True)
    victim = next(i for i, ln in enumerate(old) if " = 1;" in ln)
    cur = g.read_text().splitlines(keepends=True)
    cur = [ln for ln in cur if ln != old[victim]]
    g.write_text("".join(cur))
    assert any("removed or changed" in p for p in m.delta_problems())
