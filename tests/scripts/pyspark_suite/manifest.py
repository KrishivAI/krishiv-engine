# SPDX-License-Identifier: Apache-2.0
"""The exclusion manifest: reason classes are an enum, entries match a file or one test id."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# The owner-frozen cap on excluded tests, as a share of (collected - Spark's own skips).
EXCLUSION_CAP = 0.20


class ReasonClass(StrEnum):
    STREAMING_P4 = "STREAMING_P4"
    LOCAL_SERVER = "LOCAL_SERVER"
    DEFERRED_TYPE = "DEFERRED_TYPE"
    OPTIONAL_DEP = "OPTIONAL_DEP"
    ORACLE_ENV = "ORACLE_ENV"
    UNVERIFIED = "UNVERIFIED"


@dataclass(frozen=True)
class Entry:
    """One exclusion: `target` is a file path prefix (ends with `.py` or `/`) or a full test id."""

    target: str
    reason_class: ReasonClass
    note: str
    versions: tuple[str, ...]  # empty = every version

    def applies(self, version: str) -> bool:
        return not self.versions or version in self.versions

    def matches(self, test_id: str) -> bool:
        """Test ids are `<file>::<Class>::<test>`; a file target matches by path prefix."""
        if "::" in self.target:
            return test_id == self.target
        file = test_id.split("::", 1)[0]
        return file == self.target or (self.target.endswith("/") and file.startswith(self.target))


class ManifestError(ValueError):
    pass


def load(path: Path) -> list[Entry]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    entries = []
    for i, raw in enumerate(data.get("exclude", [])):
        try:
            reason = ReasonClass(raw["reason_class"])
        except (KeyError, ValueError) as e:
            raise ManifestError(f"{path}: entry {i} has a missing or unknown reason_class: {e}") from e
        if "target" not in raw or not raw.get("note"):
            raise ManifestError(f"{path}: entry {i} needs `target` and `note`")
        entries.append(Entry(raw["target"], reason, raw["note"], tuple(raw.get("versions", ()))))
    return entries
