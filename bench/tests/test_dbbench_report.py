"""db-benchmark time.csv comparison (P3-T094)."""

from __future__ import annotations

import csv
from pathlib import Path

from krishiv_bench import dbbench_report as dr

HEADER = ["batch", "timestamp", "task", "data", "question", "solution", "run", "time_sec", "on_disk"]


def write(path: Path, rows: list[list[object]]) -> Path:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    return path


def test_headline_is_the_second_run_and_the_newest_row(tmp_path: Path) -> None:
    f = write(
        tmp_path / "t.csv",
        [
            ["b1", 1, "groupby", "G1_1e7", "q1", "krishiv", 1, 9.0, "FALSE"],
            ["b1", 2, "groupby", "G1_1e7", "q1", "krishiv", 2, 1.5, "FALSE"],
            ["b2", 3, "groupby", "G1_1e7", "q1", "krishiv", 2, 1.0, "TRUE"],
        ],
    )
    assert dr.read_times(f) == {("groupby", "G1_1e7", "q1", "krishiv"): (1.0, True)}
    assert dr.read_times(f, batch="b1") == {("groupby", "G1_1e7", "q1", "krishiv"): (1.5, False)}


def test_table_ratio_fail_and_spill_marks(tmp_path: Path) -> None:
    pub = dr.read_times(
        write(
            tmp_path / "p.csv",
            [
                ["b", 1, "groupby", "G1_1e7", "q1", "duckdb", 2, 0.5, "FALSE"],
                ["b", 1, "groupby", "G1_1e7", "q2", "duckdb", 2, 2.0, "FALSE"],
            ],
        )
    )
    ours = dr.read_times(write(tmp_path / "o.csv", [["b", 1, "groupby", "G1_1e7", "q1", "krishiv", 2, 1.0, "TRUE"]]))
    rows = dr.build_table(ours, pub)
    assert [r["question"] for r in rows] == ["q1", "q2"]
    assert rows[0]["duckdb_ratio"] == 2.0
    assert rows[1][dr.OURS] is None
    md = dr.to_markdown(rows)
    assert "1.00*" in md
    assert "FAIL" in md
    assert "1 without a Krishiv row" in md


def test_data_filter(tmp_path: Path) -> None:
    ours = dr.read_times(
        write(
            tmp_path / "o.csv",
            [
                ["b", 1, "groupby", "G1_1e7", "q1", "krishiv", 2, 1.0, "FALSE"],
                ["b", 1, "groupby", "G1_1e8", "q1", "krishiv", 2, 9.0, "FALSE"],
            ],
        )
    )
    assert len(dr.build_table(ours, {}, data="G1_1e8")) == 1
