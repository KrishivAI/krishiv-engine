# SPDX-License-Identifier: Apache-2.0
"""Tests for report.py (run: cd tests/connect-clients && python3 -m pytest)."""

from __future__ import annotations

import json
from pathlib import Path

import report


def doc(client: str, statuses: dict[str, str]) -> dict:
    """A results document as the clients write it."""
    return {
        "client": client,
        "version": f"{client} 1",
        "remote": "sc://x:1",
        "results": [
            {"name": n, "status": s, "detail": "boom" if s != "pass" else "", "ms": 1}
            for n, s in statuses.items()
        ],
    }


def write(directory: Path, client: str, statuses: dict[str, str]) -> None:
    """Write one client's results file."""
    (directory / f"{client}.json").write_text(json.dumps(doc(client, statuses)))


def test_all_pass_needs_both_clients(tmp_path: Path) -> None:
    write(tmp_path, "go", {"a": "pass"})
    assert not report.all_pass(report.load(tmp_path))
    write(tmp_path, "scala", {"a": "pass"})
    assert report.all_pass(report.load(tmp_path))


def test_summary_marks_failures_and_lists_details(tmp_path: Path) -> None:
    write(tmp_path, "go", {"a": "pass", "b": "fail"})
    write(tmp_path, "scala", {"a": "pass", "b": "error"})
    text = report.render(report.load(tmp_path))
    assert "**go** (go 1): 1/2 pass" in text
    assert "| b | FAIL | ERROR |" in text
    assert "`go` `b`: boom" in text


def test_missing_client_is_reported(tmp_path: Path) -> None:
    write(tmp_path, "go", {"a": "pass"})
    text = report.render(report.load(tmp_path))
    assert "**scala**: no results file" in text


def test_check_missing_from_one_client_shows_a_dash(tmp_path: Path) -> None:
    write(tmp_path, "go", {"a": "pass", "only_go": "pass"})
    write(tmp_path, "scala", {"a": "pass"})
    assert "| only_go | pass | - |" in report.render(report.load(tmp_path))


def test_report_only_unless_strict(tmp_path: Path) -> None:
    write(tmp_path, "go", {"a": "fail"})
    assert report.main([str(tmp_path)]) == 0
    assert (tmp_path / "summary.md").is_file()
    assert report.main([str(tmp_path), "--strict"]) == 1


def test_queries_file_is_well_formed() -> None:
    queries = json.loads((Path(__file__).parent / "queries.json").read_text())
    names = [q["name"] for q in queries]
    assert len(names) == len(set(names))
    for q in queries:
        assert q["sql"]
        assert ("rows" in q) != ("error" in q), q["name"]
