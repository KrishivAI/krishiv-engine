"""The bench pins Polars' 64-bit-index runtime: TPC-H SF1000 lineitem is one 6.0 B-row Parquet file, past the default
runtime's 2^32-1 rows per frame."""

from krishiv_bench.adapters.polars_engine import PolarsEngine, polars_runtime


def test_the_64_bit_runtime_is_the_one_that_loads():
    assert polars_runtime() == "64"


def test_the_runtime_is_recorded_in_the_engine_config():
    assert PolarsEngine(threads=1, streaming=True).config()["runtime"] == "64"
