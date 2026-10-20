import json

from krishiv_bench.report import latest_results


def test_multi_engine_results_are_skipped(tmp_path):
    ds = {"name": "tpch-dbgen", "scale_factor": 1}
    run = {"suite": "tpch", "dataset": ds, "engine": {"name": "krishiv"}, "queries": {}}
    abba = {"suite": "tpch", "dataset": ds, "queries": {}}
    (tmp_path / "a_run.json").write_text(json.dumps(run))
    (tmp_path / "b_abba.json").write_text(json.dumps(abba))
    got = latest_results("tpch", 1, "tpch-dbgen", directory=tmp_path)
    assert list(got) == ["krishiv"]
    assert got["krishiv"]["_file"] == "a_run.json"
