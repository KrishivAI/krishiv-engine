# SPDX-License-Identifier: Apache-2.0
"""The unpacking of a session's archive artifacts for its pool workers (``krishiv._udf.unpack``, P3-T014
PR-26c-3), run in this process.

The server runs the same program in the worker interpreter (``krishiv-udf-pool``'s ``tests/relations.rs``
imports from unpacked zip and tar.gz archives with real workers); that interpreter is not measured, so each
format and each refusal is run here.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path
from typing import Literal

import pytest

from krishiv._udf import unpack


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


Mode = Literal["w", "w:gz"]


def _tar(path: Path, members: dict[str, bytes], mode: Mode = "w:gz") -> Path:
    with tarfile.open(path, mode) as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


@pytest.mark.parametrize("name", ["deps.zip", "deps.JAR"])
def test_a_zip_or_jar_is_unpacked_within_its_budget(tmp_path: Path, name: str) -> None:
    archive = _zip(tmp_path / name, {"pkg/__init__.py": b"X = 1\n", "pkg/data.txt": b"abc"})
    dest = tmp_path / "out"
    dest.mkdir()
    assert unpack.unpack(str(archive), str(dest), 100) == 9
    assert (dest / "pkg" / "__init__.py").read_bytes() == b"X = 1\n"


@pytest.mark.parametrize(("name", "mode"), [("deps.tar", "w"), ("deps.tar.gz", "w:gz"), ("deps.tgz", "w:gz")])
def test_a_tar_is_unpacked_within_its_budget(tmp_path: Path, name: str, mode: Mode) -> None:
    archive = _tar(tmp_path / name, {"pkg/__init__.py": b"X = 1\n"}, mode)
    dest = tmp_path / "out"
    dest.mkdir()
    assert unpack.unpack(str(archive), str(dest), 100) == 6
    assert (dest / "pkg" / "__init__.py").read_bytes() == b"X = 1\n"


def test_an_archive_over_its_budget_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    dest.mkdir()
    big = {"a": b"x" * 60, "b": b"y" * 60}
    with pytest.raises(unpack.Refused, match="unpacks to 120 bytes, over the 100 bytes left"):
        unpack.unpack(str(_zip(tmp_path / "a.zip", big)), str(dest), 100)
    with pytest.raises(unpack.Refused, match="more than the 100 bytes left"):
        unpack.unpack(str(_tar(tmp_path / "a.tgz", big)), str(dest), 100)
    assert list(dest.iterdir()) == []


def test_an_archive_with_too_many_members_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(unpack, "MAX_MEMBERS", 1)
    members = {"a": b"1", "b": b"2"}
    with pytest.raises(unpack.Refused, match="more than 1 members"):
        unpack.unpack(str(_zip(tmp_path / "a.zip", members)), str(tmp_path), 100)
    with pytest.raises(unpack.Refused, match="more than 1 members"):
        unpack.unpack(str(_tar(tmp_path / "a.tar.gz", members)), str(tmp_path), 100)


def test_a_tar_member_leaving_the_directory_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tf:
        link = tarfile.TarInfo("escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        tf.addfile(link)
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(unpack.Refused, match="leaves the archive's directory"):
        unpack.unpack(str(archive), str(dest), 100)
    assert not (tmp_path / "outside").exists()


def test_a_tar_without_tarfile_s_data_filter_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(tarfile, "data_filter")
    with pytest.raises(unpack.Refused, match="no data filter"):
        unpack.unpack(str(_tar(tmp_path / "a.tgz", {"a": b"1"})), str(tmp_path), 100)


def test_another_format_is_refused(tmp_path: Path) -> None:
    with pytest.raises(unpack.Refused, match=r"not a zip, jar, tar, tar\.gz or tgz archive"):
        unpack.unpack(str(tmp_path / "a.rar"), str(tmp_path), 100)


def test_main_prints_the_bytes_or_the_reason(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dest = tmp_path / "out"
    dest.mkdir()
    unpack.main(str(_zip(tmp_path / "a.zip", {"a": b"123"})), str(dest), 100)
    assert capsys.readouterr().out == "3\n"

    with pytest.raises(SystemExit) as exit_info:
        unpack.main(str(tmp_path / "a.rar"), str(dest), 100)
    assert exit_info.value.code == 2
    assert capsys.readouterr().err == "it is not a zip, jar, tar, tar.gz or tgz archive\n"

    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a zip")
    with pytest.raises(SystemExit) as exit_info:
        unpack.main(str(corrupt), str(dest), 100)
    assert exit_info.value.code == 2
    assert capsys.readouterr().err.startswith("it cannot be unpacked: BadZipFile: ")


def test_a_tar_hard_link_counts_as_its_target(tmp_path: Path) -> None:
    def archive(links: list[tuple[str, str]]) -> Path:
        path = tmp_path / f"links{len(links)}.tar"
        with tarfile.open(path, "w") as tf:
            data = b"x" * 60
            info = tarfile.TarInfo("pkg/a")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
            for name, target in links:
                link = tarfile.TarInfo(name)
                link.type = tarfile.LNKTYPE
                link.linkname = target
                tf.addfile(link)
        return path

    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(unpack.Refused, match="more than the 100 bytes left"):
        unpack.unpack(str(archive([("pkg/b", "./pkg/a")])), str(dest), 100)
    assert list(dest.iterdir()) == []
    assert unpack.unpack(str(archive([("pkg/b", "./pkg/a")])), str(dest), 120) == 120
    assert (dest / "pkg" / "b").read_bytes() == b"x" * 60
    # A link to no earlier member counts nothing (tarfile cannot extract it).
    assert unpack._tar_sizes([_link("missing")]) == [0]


def _link(target: str) -> tarfile.TarInfo:
    link = tarfile.TarInfo("l")
    link.type = tarfile.LNKTYPE
    link.linkname = target
    return link


def _symlink(name: str, target: str) -> tarfile.TarInfo:
    link = tarfile.TarInfo(name)
    link.type = tarfile.SYMTYPE
    link.linkname = target
    return link


def test_a_tar_symbolic_link_counts_as_its_target(tmp_path: Path) -> None:
    # tarfile copies a symbolic link's target when the link cannot be made (the sandbox off, a filesystem
    # without links), so a link counts as its target: resolved from the link's directory, over the whole
    # archive (a later member too), the last member of that name, through links to links.
    path = tmp_path / "syms.tar"
    with tarfile.open(path, "w") as tf:
        tf.addfile(_symlink("pkg/early", "a"))
        data = b"x" * 60
        info = tarfile.TarInfo("pkg/a")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
        tf.addfile(_symlink("pkg/chain", "early"))
        tf.addfile(_symlink("top", "pkg/a"))
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(unpack.Refused, match="more than the 239 bytes left"):
        unpack.unpack(str(path), str(dest), 239)
    assert list(dest.iterdir()) == []
    assert unpack.unpack(str(path), str(dest), 240) == 240
    assert (dest / "top").read_bytes() == data


def test_a_tar_link_loop_or_dangling_link_counts_nothing(tmp_path: Path) -> None:
    path = tmp_path / "loop.tar"
    with tarfile.open(path, "w") as tf:
        tf.addfile(_symlink("a", "b"))
        tf.addfile(_symlink("b", "a"))
        tf.addfile(_symlink("c", "missing"))
    with tarfile.open(path) as tf:
        assert unpack._tar_sizes(tf.getmembers()) == [0, 0, 0]
