# SPDX-License-Identifier: Apache-2.0
"""Unit tests for .github/scripts/third_party_notices.py (the third-party notices file).

Run: (cd python && uv run pytest ../tests/scripts/test_third_party_notices.py)
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / ".github" / "scripts" / "third_party_notices.py"
_spec = importlib.util.spec_from_file_location("third_party_notices", SCRIPT)
assert _spec is not None
assert _spec.loader is not None
tpn = importlib.util.module_from_spec(_spec)
sys.modules["third_party_notices"] = tpn  # dataclasses look their module up
_spec.loader.exec_module(tpn)


def _pkg(pid: str, name: str, root: Path, source: str | None = "registry", **extra: str) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    return {
        "id": pid,
        "name": name,
        "version": "1.0.0",
        "source": source,
        "license": extra.get("license", "MIT"),
        "license_file": extra.get("license_file"),
        "repository": extra.get("repository", f"https://example.org/{name}"),
        "manifest_path": str(root / "Cargo.toml"),
    }


def _dep(pkg: str, kind: str | None) -> dict:
    return {"pkg": pkg, "dep_kinds": [{"kind": kind, "target": None}]}


@pytest.fixture
def metadata(tmp_path: Path) -> dict:
    """app -> a (normal), b (dev), c (build); a -> d (normal); app and w are workspace crates."""
    pkgs = [
        _pkg("app", "app", tmp_path / "ws" / "app", source=None),
        _pkg("w", "w", tmp_path / "ws" / "w", source=None),
        _pkg("a", "a", tmp_path / "reg" / "a"),
        _pkg("b", "b", tmp_path / "reg" / "b"),
        _pkg("c", "c", tmp_path / "reg" / "c"),
        _pkg("d", "d", tmp_path / "reg" / "d", license="Apache-2.0"),
    ]
    nodes = [
        {"id": "app", "deps": [_dep("a", None), _dep("b", "dev"), _dep("c", "build"), _dep("w", None)]},
        {"id": "w", "deps": []},
        {"id": "a", "deps": [_dep("d", None)]},
        {"id": "b", "deps": []},
        {"id": "c", "deps": []},
        {"id": "d", "deps": []},
    ]
    return {"packages": pkgs, "resolve": {"nodes": nodes}}


def test_only_normal_third_party_dependencies_are_linked(metadata: dict) -> None:
    names = sorted(c.name for c in tpn.linked_crates(metadata, "app"))
    assert names == ["a", "d"]


def test_an_unknown_package_is_an_error(metadata: dict) -> None:
    with pytest.raises(SystemExit):
        tpn.linked_crates(metadata, "nope")


def test_notice_names() -> None:
    for yes in ("LICENSE", "LICENSE-MIT", "license.txt", "COPYING", "NOTICE", "LICENCE_APACHE", "UNLICENSE"):
        assert tpn.is_notice_name(yes), yes
    for no in ("README.md", "licensed.rs", "Cargo.toml", "NOTICES_x"):
        assert not tpn.is_notice_name(no), no


def test_nested_and_named_files_are_found_and_test_dirs_skipped(tmp_path: Path) -> None:
    root = tmp_path / "zstd-sys"
    (root / "zstd").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "LICENSES").mkdir()
    (root / "LICENSE-MIT").write_text("mit")
    (root / "zstd" / "COPYING").write_text("gpl or bsd")
    (root / "tests" / "LICENSE").write_text("not linked")
    (root / "LICENSES" / "BSD-3-Clause.txt").write_text("bsd")
    (root / "legal.txt").write_text("named by license-file")
    crate = tpn.Crate("zstd-sys", "1", "MIT", "", root, license_file="legal.txt")
    rel = [str(p.relative_to(root)) for p in tpn.notice_files(crate)]
    assert rel == ["LICENSE-MIT", "legal.txt", "LICENSES/BSD-3-Clause.txt", "zstd/COPYING"]


def test_depth_is_bounded(tmp_path: Path) -> None:
    deep = tmp_path / "x" / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    (deep / "LICENSE").write_text("too deep")
    (tmp_path / "x" / "a" / "b" / "c" / "LICENSE").write_text("depth 3")
    crate = tpn.Crate("x", "1", "MIT", "", tmp_path / "x")
    assert [str(p.relative_to(tmp_path / "x")) for p in tpn.notice_files(crate)] == ["a/b/c/LICENSE"]


def test_feature_flags_are_read_from_a_build_flag_string() -> None:
    args = "--profile dist --locked --no-default-features --features mimalloc,cloud,ui --interpreter python3.14t"
    assert tpn.parse_cargo_args(args) == ["--no-default-features", "--features", "mimalloc,cloud,ui"]
    assert tpn.parse_cargo_args("-F a --features=b --all-features") == [
        "--features",
        "a",
        "--features",
        "b",
        "--all-features",
    ]
    assert tpn.parse_cargo_args("--profile dist --locked") == []


def test_identical_texts_are_written_once_and_missing_files_named(tmp_path: Path) -> None:
    crates = []
    for name in ("one", "two"):
        root = tmp_path / name
        root.mkdir()
        (root / "LICENSE-APACHE").write_text("Apache License\r\nVersion 2.0  \n")
        crates.append(tpn.Crate(name, "1.0", "Apache-2.0", f"https://x/{name}", root))
    bare = tmp_path / "bare"
    bare.mkdir()
    crates.append(tpn.Crate("bare", "0.1", "MIT", "", bare))
    out = tpn.render("the test artifact", [*crates, crates[0]])  # a duplicate from a second target
    assert out.crates == 3
    assert out.without_files == ["bare 0.1"]
    assert out.text.count("Apache License\nVersion 2.0\n") == 1
    assert "  one 1.0: LICENSE-APACHE\n  two 1.0: LICENSE-APACHE" in out.text
    assert "- bare 0.1, MIT, (the package carries no license file)" in out.text
    assert out.text.startswith("Third-party software in the test artifact\n")
    assert "The test artifact also contains the 3 third-party Rust" in out.text


def test_the_workspace_cli_resolves(tmp_path: Path) -> None:
    """End to end on this repository: needs cargo and the registry sources.

    Skipped without cargo or with TPN_NO_CARGO set (the python-quality job: no toolchain there; the
    wheels workflow's notices job runs the script on the workspace).
    """
    import os
    import shutil

    if shutil.which("cargo") is None or os.environ.get("TPN_NO_CARGO"):
        pytest.skip("no cargo here")
    out = tmp_path / "n.txt"
    root = Path(__file__).parents[2]
    args = ["--package", "krishiv-cli", "--target", "x86_64-unknown-linux-gnu", "--title", "the CLI"]
    rc = tpn.main([*args, "--out", str(out), "--root", str(root)])
    assert rc == 0
    text = out.read_text()
    assert "- arrow-array " in text
    assert "- krishiv-" not in text  # workspace crates are Krishiv's own
