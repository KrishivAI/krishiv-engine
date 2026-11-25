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


def fake_checkout(root: Path) -> Path:
    """The lines of a db-benchmark checkout that ``register`` places its edits next to."""
    files = {
        "_launcher/launcher.R": 'x = switch(s, "datafusion"=, "polars"="py", "duckdb"="R")\n',
        "_launcher/solution.R": 'x = switch(s, "datafusion"=, "polars"="py", "duckdb"="R")\n',
        "run.sh": 'if [[ "$RUN_SOLUTIONS" =~ "datafusion" ]]; then ./datafusion/ver-datafusion.sh; fi;\nRscript x\n',
        "_report/report.R": 'c("duckdb", "chdb", "haskell")\n',
        ".github/workflows/regression.yml": "solution: [duckdb, chdb, haskell, spark]\n",
        "README.md": "  - [x] [(haskell)dataframe](https://github.com/mchav/dataframe)\n",
        "_control/solutions.csv": "solution,task\nduckdb,groupby",
        "_benchplot/benchplot-dict.R": "".join(f'{name} = list(\n  "haskell" = list()\n)\n' for name in "abcdefg"),
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def test_register_names_the_solution_once(tmp_path: Path) -> None:
    root = fake_checkout(tmp_path)
    assert len(ds.register(root)) == 8
    assert '"datafusion"=, "krishiv"=, "polars"="py"' in (root / "_launcher/launcher.R").read_text()
    run = (root / "run.sh").read_text()
    order = [run.index(s) for s in ("ver-datafusion.sh", "upg-krishiv.sh", "ver-krishiv.sh", "Rscript")]
    assert order == sorted(order)
    assert (root / "_control/solutions.csv").read_text().endswith("duckdb,groupby\nkrishiv,groupby\nkrishiv,join\n")
    plot = (root / "_benchplot/benchplot-dict.R").read_text()
    assert plot.count('"krishiv" =') == 7
    assert plot.count('"sum v1 by id1" = ') == 1
    assert plot.count('"small inner on int" = ') == 1
    assert plot.index('"krishiv" =') < plot.index('"haskell" =')
    assert ds.register(root) == []


def test_register_refuses_a_checkout_that_changed(tmp_path: Path) -> None:
    root = fake_checkout(tmp_path / "first")
    (root / "_report/report.R").write_text("c()\n")
    with pytest.raises(ds.RegisterError, match=r"report\.R"):
        ds.register(root)
    root = fake_checkout(tmp_path / "second")
    (root / "_benchplot/benchplot-dict.R").write_text('  "haskell" = list()\n')
    with pytest.raises(ds.RegisterError, match="7 haskell entries, found 1"):
        ds.register(root)


def test_main_writes_and_registers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = fake_checkout(tmp_path / "checkout")
    assert ds.main(["--checkout", str(root)]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 13
    assert (root / "krishiv" / "join-krishiv.py").is_file()
    assert ds.main(["--out", str(tmp_path / "only")]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 5


def test_the_answer_is_held_or_written_to_disk() -> None:
    text = ds.render_task_script("groupby")
    assert "read_all()" in text
    assert "KRISHIV_ON_DISK" in text
    assert "os.chmod(spill_dir, 0o700)" in text
    assert "krishiv.build_info()" in text
    assert 'on_disk="TRUE" if on_disk else "FALSE"' in text
