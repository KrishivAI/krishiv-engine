# SPDX-License-Identifier: Apache-2.0
"""JUnit XML (pytest `--junitxml`) to per-test records and back to ids."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

STATUSES = ("pass", "fail", "error", "skip", "timeout")


@dataclass(frozen=True)
class Record:
    version: str
    file: str
    cls: str
    test: str
    status: str
    ms: int
    reason_class: str | None = None

    @property
    def test_id(self) -> str:
        return f"{self.file}::{self.cls}::{self.test}"

    def as_dict(self) -> dict:
        d = {
            "version": self.version,
            "file": self.file,
            "class": self.cls,
            "test": self.test,
            "status": self.status,
            "ms": self.ms,
        }
        if self.reason_class:
            d["reason_class"] = self.reason_class
        return d


def parse(xml_path: Path, version: str, file: str) -> list[Record]:
    """`file` is the path relative to the connect tests dir; pytest's classname is `<mod path>.<Class>`."""
    out = []
    for case in ET.parse(xml_path).getroot().iter("testcase"):
        cls = (case.get("classname") or "").rsplit(".", 1)[-1]
        ms = int(float(case.get("time") or 0) * 1000)
        status = "pass"
        reason = None
        for child in case:
            if child.tag == "skipped":
                status = "skip"
            elif child.tag == "failure":
                status = "timeout" if "Timeout" in (child.get("message") or "") else "fail"
            elif child.tag == "error":
                status = "error"
            if status != "pass":
                reason = (child.get("type") or None) if child.tag != "skipped" else None
                break
        out.append(Record(version, file, cls, case.get("name") or "", status, ms, reason))
    return out
