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
    monkeypatch.delenv("POLARS_OOC_MEMORY_BUDGET_MB", raising=False)
    monkeypatch.delenv("POLARS_OOC_SPILL_DIR", raising=False)
    cfg = PolarsEngine(threads=1, memory_limit="20g").config()
    assert cfg["ooc_memory_budget_mb"] == str(20 * 1024)
    assert cfg["ooc_spill_dir"] == str(tmp_path / "tmp" / f"polars-{os.getpid()}")
    assert os.path.isdir(cfg["ooc_spill_dir"])
