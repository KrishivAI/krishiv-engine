# SPDX-License-Identifier: Apache-2.0
"""Unpacks one archive artifact for a session's pool workers (P3-T014 PR-26c-3; ``krishiv-udf-pool``'s
``artifacts.rs``).

The server runs it with the worker interpreter, in the sandbox when it is on (it reads the archive and writes only
the destination directory), before any worker of the session's artifact set starts. Spark's executor unpacks
``.zip``, ``.jar``, ``.tar``, ``.tar.gz`` and ``.tgz`` archives (``Utils.unpack``); the same formats are read here.

The archive comes from a client, so: the members' sizes together must stay within ``budget`` bytes (checked
before anything is written; a member never writes past its declared size; a tar's hard or symbolic link counts as
its target, which ``tarfile`` may copy), there are at most ``MAX_MEMBERS`` members, and a tar archive is extracted with
``tarfile``'s ``data`` filter (no absolute paths, no ``..``, no link leaving the directory, no device files). A
zip's members cannot be links, and ``zipfile`` drops absolute paths and ``..``. A refusal is one line on stderr and
exit status 2; the server raises it as ``KRISHIV_PYTHON_ARTIFACT_REFUSED``. On success the unpacked byte count is
the one line on stdout.
"""

from __future__ import annotations

import bisect
import os
import sys
import tarfile

#: The most members one archive may have.
MAX_MEMBERS = 100_000


class Refused(Exception):
    """The archive is refused (its message is the reason)."""


def _unpack_zip(archive: str, dest: str, budget: int) -> int:
    import zipfile

    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_MEMBERS:
            raise Refused(f"it has more than {MAX_MEMBERS} members")
        total = sum(i.file_size for i in infos)
        if total > budget:
            raise Refused(f"it unpacks to {total} bytes, over the {budget} bytes left of the artifact limit")
        zf.extractall(dest)
    return total


#: The most links followed to find what one link counts as (``tarfile`` itself would recurse without end).
MAX_LINK_DEPTH = 40


def _tar_sizes(members: list[tarfile.TarInfo]) -> list[int]:
    """The bytes each of ``members`` may write: a file's size, and a link's target's.

    When a link cannot be made, ``tarfile`` writes a full copy of its target instead (``makelink_with_filter``
    falls back to ``_find_link_target``), so a link counts as its target, found as ``tarfile`` finds it: a hard
    link's among the members before it, a symbolic link's in the whole archive and relative to the link's own
    directory, the last member of that normalized name either way, through links to links. The ``data`` filter
    keeps such links when they stay inside the directory, and the fallback is reachable with the sandbox off on a
    filesystem without links. A link to no member, or a loop, counts nothing (``tarfile`` cannot extract it).
    """
    by_name: dict[str, list[int]] = {}
    for i, member in enumerate(members):
        by_name.setdefault(os.path.normpath(member.name), []).append(i)

    def target(i: int) -> int | None:
        member = members[i]
        if member.issym():
            name = "/".join(filter(None, (os.path.dirname(member.name), member.linkname)))
            found = by_name.get(os.path.normpath(name), [])
        else:
            # The members of a name in order: the last one before `i`, by a binary search (no scan, so many
            # hard links to one name stay linear-logarithmic).
            found = by_name.get(os.path.normpath(member.linkname), [])
            before = bisect.bisect_left(found, i)
            return found[before - 1] if before else None
        return found[-1] if found else None

    def size(i: int) -> int:
        for _ in range(MAX_LINK_DEPTH):
            member = members[i]
            if member.isfile():
                return member.size
            if not (member.islnk() or member.issym()):
                return 0
            j = target(i)
            if j is None:
                return 0
            i = j
        return 0

    return [size(i) for i in range(len(members))]


def _unpack_tar(archive: str, dest: str, budget: int) -> int:
    if not hasattr(tarfile, "data_filter"):
        raise Refused("this interpreter's tarfile has no data filter (Python 3.11.4 or later is needed)")
    with tarfile.open(archive, "r:*") as tf:
        members = []
        for member in tf:
            members.append(member)
            if len(members) > MAX_MEMBERS:
                raise Refused(f"it has more than {MAX_MEMBERS} members")
        total = sum(_tar_sizes(members))
        if total > budget:
            raise Refused(f"it unpacks to more than the {budget} bytes left of the artifact limit")
        try:
            tf.extractall(dest, members=members, filter="data")
        except tarfile.FilterError as e:
            raise Refused(f"a member leaves the archive's directory or is not a plain file: {e}") from None
    return total


def unpack(archive: str, dest: str, budget: int) -> int:
    """Unpacks ``archive`` into ``dest`` (an empty directory) within ``budget`` bytes; the bytes unpacked."""
    name = archive.lower()
    if name.endswith((".zip", ".jar")):
        return _unpack_zip(archive, dest, budget)
    if name.endswith((".tar", ".tar.gz", ".tgz")):
        return _unpack_tar(archive, dest, budget)
    raise Refused("it is not a zip, jar, tar, tar.gz or tgz archive")


def main(archive: str, dest: str, budget: int) -> None:
    """The program's entry (see the module doc)."""
    try:
        total = unpack(archive, dest, budget)
    except Refused as e:
        sys.stderr.write(f"{e}\n")
        sys.exit(2)
    except Exception as e:  # a corrupt archive: the reader's own error names it
        sys.stderr.write(f"it cannot be unpacked: {type(e).__name__}: {e}\n")
        sys.exit(2)
    sys.stdout.write(f"{total}\n")
