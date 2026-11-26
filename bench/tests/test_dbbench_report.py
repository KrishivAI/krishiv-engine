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


CHECK_HEADER = ["batch", "task", "data", "question", "solution", "run", "out_rows", "out_cols", "chk"]


def write_check(path: Path, rows: list[list[object]]) -> Path:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CHECK_HEADER)
        w.writerows(rows)
    return path


def test_check_passes_on_equal_answers_within_tolerance(tmp_path: Path) -> None:
    f = write_check(
        tmp_path / "t.csv",
        [
            ["b1", "groupby", "G1", "q1", "duckdb", 1, 100, 2, "3000.0;5.0000001"],
            ["b1", "groupby", "G1", "q1", "krishiv", 1, 100, 2, "3000;5.0"],
            ["b0", "groupby", "G1", "q1", "krishiv", 2, 7, 2, "1;1"],
        ],
    )
    assert dr.check_answers(f, "duckdb", "b1") == (1, [])
    assert dr.main(["--ours", str(f), "--check", "duckdb", "--batch", "b1"]) == 0


def test_check_reports_a_different_or_missing_answer(tmp_path: Path) -> None:
    f = write_check(
        tmp_path / "t.csv",
        [
            ["b1", "groupby", "G1", "q1", "duckdb", 1, 100, 2, "3000;5"],
            ["b1", "groupby", "G1", "q1", "krishiv", 1, 100, 2, "3001;5"],
            ["b1", "groupby", "G1", "q2", "duckdb", 1, 100, 2, "1;2"],
            ["b1", "groupby", "G1", "q2", "krishiv", 1, 99, 2, "1;2"],
            ["b1", "groupby", "G1", "q3", "duckdb", 1, 1, 1, "1"],
            ["b1", "join", "J1", "q1", "krishiv", 2, 1, 1, "1"],
        ],
    )
    compared, problems = dr.check_answers(f, "duckdb")
    assert compared == 4
    assert [p.split(":")[0] for p in problems] == [
        "groupby G1 q1 1",
        "groupby G1 q2 1",
        "groupby G1 q3 1",
        "join J1 q1 2",
    ]
    assert problems[2].endswith("no row from krishiv")
    assert problems[3].endswith("no row from duckdb")
    assert dr.main(["--ours", str(f), "--check", "duckdb"]) == 1
    # --data narrows the check to one data set: G1's three problems, without J1's.
    assert dr.check_answers(f, "duckdb", data="G1")[0] == 3
    assert dr.check_answers(f, "duckdb", data="J1") == (1, ["join J1 q1 2: no row from duckdb"])


def test_check_of_one_data_set_ignores_the_others(tmp_path: Path) -> None:
    f = write_check(
        tmp_path / "t.csv",
        [
            ["b1", "join", "J1", "q1", "duckdb", 1, 5, 1, "1"],
            ["b1", "join", "J1", "q1", "krishiv", 1, 5, 1, "1"],
            ["b1", "join", "J2", "q1", "duckdb", 1, 5, 1, "1"],
        ],
    )
    assert dr.main(["--ours", str(f), "--check", "duckdb"]) == 1
    assert dr.main(["--ours", str(f), "--check", "duckdb", "--data", "J1"]) == 0


def test_check_of_an_empty_file_fails(tmp_path: Path) -> None:
    f = write_check(tmp_path / "t.csv", [])
    assert dr.main(["--ours", str(f), "--check", "duckdb"]) == 1
