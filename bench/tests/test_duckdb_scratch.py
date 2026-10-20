import os

import duckdb
import pytest

from krishiv_bench.adapters.base import Dataset
from krishiv_bench.adapters.duckdb_engine import DuckDBEngine


def _scratch(root):
    return root / "tmp" / f"duckdb-{os.getpid()}"


def test_scratch_under_data_root_with_quote_and_removed_on_teardown(tmp_path, monkeypatch):
    data = tmp_path / "user's data"
    monkeypatch.setenv("KRISHIV_DATA", str(data))
    eng = DuckDBEngine(threads=1, memory_limit="1GB")
    eng.setup(Dataset(name="t", sf=1, root=tmp_path, tables=[]))
    assert _scratch(data).is_dir()
    assert eng.con is not None
    row = eng.con.execute("SELECT current_setting('temp_directory')").fetchone()
    assert row is not None
    assert row[0] == str(_scratch(data))
    eng.teardown()
    assert not _scratch(data).exists()


def test_scratch_removed_when_setup_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path / "data"))
    eng = DuckDBEngine(threads=1, memory_limit="1GB")
    with pytest.raises(duckdb.Error):
        eng.setup(Dataset(name="t", sf=1, root=tmp_path, tables=["missing"]))
    assert not _scratch(tmp_path / "data").exists()
