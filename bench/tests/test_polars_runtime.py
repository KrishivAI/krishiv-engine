"""The bench pins Polars' 64-bit-index runtime: TPC-H SF1000 lineitem is one 6.0 B-row Parquet file, past the default
runtime's 2^32-1 rows per frame."""

import os

from krishiv_bench.adapters.polars_engine import PolarsEngine, polars_runtime


def test_the_64_bit_runtime_is_the_one_that_loads():
    assert polars_runtime() == "64"


def test_the_runtime_is_recorded_in_the_engine_config():
    assert PolarsEngine(threads=1, streaming=True).config()["runtime"] == "64"


def test_the_engine_memory_is_the_out_of_core_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    # The constructor sets these with os.environ.setdefault. setenv first, so monkeypatch records each variable and
    # undoes what the constructor sets (a bare delenv of an unset variable records nothing to undo).
    for name in ("POLARS_OOC_MEMORY_BUDGET_MB", "POLARS_OOC_SPILL_DIR", "POLARS_MAX_THREADS"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    cfg = PolarsEngine(threads=1, memory_limit="20g").config()
    assert cfg["ooc_memory_budget_mb"] == str(20 * 1024)
    assert cfg["ooc_spill_dir"] == str(tmp_path / "tmp" / f"polars-{os.getpid()}")
    assert not os.path.exists(cfg["ooc_spill_dir"])  # made by setup(), not by the constructor


def test_teardown_removes_an_empty_spill_directory(monkeypatch, tmp_path):
    for name in ("POLARS_OOC_MEMORY_BUDGET_MB", "POLARS_OOC_SPILL_DIR", "POLARS_MAX_THREADS"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    spill = tmp_path / "spill"
    monkeypatch.setenv("POLARS_OOC_SPILL_DIR", str(spill))
    eng = PolarsEngine(threads=1)
    spill.mkdir()
    eng.teardown()
    assert not spill.exists()
    spill.mkdir()
    (spill / "left").write_text("x")
    eng.teardown()  # a directory still holding files stays
    assert (spill / "left").exists()
