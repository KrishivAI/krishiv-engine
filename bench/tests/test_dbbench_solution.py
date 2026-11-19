"""The krishiv solution directory for the db-benchmark repository (P3-T094)."""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

from krishiv_bench import dbbench_solution as ds
from krishiv_bench import h2o


def test_every_h2o_question_has_its_benchmark_text() -> None:
    for task, suite in ds.SUITE.items():
        qs = ds.questions(task)
        assert len(qs) == len(list(__import__("krishiv_bench.paths", fromlist=["x"]).queries_dir(suite).glob("q*.sql")))
        assert [q["id"] for q in qs] == sorted(h2o.CHK_COLUMNS[suite])
        assert all(q["sql"] and not q["sql"].endswith(";") for q in qs)


def test_unknown_task_is_refused() -> None:
    with pytest.raises(ValueError, match="groupby or join"):
        ds.questions("window")


@pytest.mark.parametrize("task", ["groupby", "join"])
def test_task_script_compiles(task: str, tmp_path: Path) -> None:
    f = tmp_path / f"{task}-krishiv.py"
    f.write_text(ds.render_task_script(task))
    py_compile.compile(str(f), doraise=True)
    text = f.read_text()
    assert f'task = "{task}"' in text
    assert "write_log(" in text
    assert "SRC_DATANAME" in text
    assert ("join_to_tbls" in text) == (task == "join")


def test_write_solution_lays_out_the_benchmark_files(tmp_path: Path) -> None:
    out = tmp_path / "krishiv"
    names = {p.name for p in ds.write_solution(out)}
    assert names == {
        "groupby-krishiv.py",
        "join-krishiv.py",
        "setup-krishiv.sh",
        "upg-krishiv.sh",
        "ver-krishiv.sh",
    }
    assert all((out / n).stat().st_mode & 0o100 for n in names)
    assert (out / "setup-krishiv.sh").read_text().startswith("#!/bin/bash")
