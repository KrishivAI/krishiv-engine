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


def _edit_once(path, old, new):
    text = path.read_text()
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new))


def test_delta_rejects_an_added_field(tmp_path, monkeypatch):
    d = _scratch(tmp_path, monkeypatch)
    _edit_once(d / "relations.proto", "message Zip {\n", "message Zip {\n  Relation extra = 3;\n")
    assert m.delta_problems() == ["relations.proto: unexpected added line 'Relation extra = 3;' (x1)"]


def test_delta_rejects_an_added_literal_variant(tmp_path, monkeypatch):
    d = _scratch(tmp_path, monkeypatch)
    _edit_once(
        d / "expressions.proto",
        "TimestampLTZNanos timestamp_ltz_nanos = 30;\n",
        "TimestampLTZNanos timestamp_ltz_nanos = 30;\n      Bogus bogus = 31;\n",
    )
    assert m.delta_problems() == ["expressions.proto: unexpected added line 'Bogus bogus = 31;' (x1)"]


def test_delta_rejects_a_removed_line(tmp_path, monkeypatch):
    d = _scratch(tmp_path, monkeypatch)
    base = m.PROTO / "4.2.1" / "spark" / "connect" / "pipelines.proto"
    victim = next(ln for ln in base.read_text().splitlines() if ln.strip().endswith(" = 1;"))
    _edit_once(d / "pipelines.proto", victim + "\n", "")
    assert m.delta_problems() == [f"pipelines.proto: removed or changed line {victim.strip()!r}"]


def test_delta_rejects_a_missing_addition(tmp_path, monkeypatch):
    d = _scratch(tmp_path, monkeypatch)
    # The 4.3 relations.proto reverted to 4.2.1: the file is identical to the baseline, so the Zip addition is missing.
    (d / "relations.proto").write_text((m.PROTO / "4.2.1" / "spark" / "connect" / "relations.proto").read_text())
    assert m.delta_problems() == [
        "relations.proto: missing added line 'Relation left = 1;' (x1)",
        "relations.proto: missing added line 'Relation right = 2;' (x1)",
        "relations.proto: missing added line 'Zip zip = 48;' (x1)",
        "relations.proto: missing added line 'message Zip {' (x1)",
    ]


def test_upstream_check_compares_each_file_with_its_tag():
    seen = []

    def serve(url):
        seen.append(url)
        # Serve the vendored file itself, except for one that is "changed upstream".
        name = url.rsplit("/", 1)[1]
        tag = url.split("/spark/")[1].split("/")[0]
        ver = next(v for v, t in m.TAGS.items() if t == tag)
        body = (m.PROTO / ver / "spark" / "connect" / name).read_bytes()
        return body + b"//changed\n" if (tag, name) == ("v4.3.0", "relations.proto") else body

    assert m.upstream_problems(serve) == ["4.3/relations.proto: differs from apache/spark v4.3.0"]
    assert len(seen) == sum(len(v["files"]) for v in m.collect().values())

    def down(url):
        raise OSError("offline")

    assert all("cannot fetch" in p for p in m.upstream_problems(down))
