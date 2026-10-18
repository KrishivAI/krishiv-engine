"""Unit tests for mutation_score.py.

Run: (cd bench && uv run --group dev pytest ../tests/scripts/test_mutation_score.py)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mutation_score as m


def _outcomes(path: Path, **n: int) -> Path:
    rows: list[dict] = [{"scenario": "Baseline", "summary": "Success"}]
    for summary, k in n.items():
        rows += [{"scenario": {"Mutant": {}}, "summary": summary} for _ in range(k)]
    path.write_text(json.dumps({"outcomes": rows}))
    return path


def test_score_excludes_baseline_and_unviable(tmp_path):
    p = _outcomes(tmp_path / "o.json", CaughtMutant=6, MissedMutant=2, Timeout=2, Unviable=5)
    areas = m.merge([("a", p)])
    assert m.score(areas["a"]) == 60.0
    assert areas["a"]["Unviable"] == 5


def test_shards_of_one_area_are_summed(tmp_path):
    p1 = _outcomes(tmp_path / "1.json", CaughtMutant=3, MissedMutant=1)
    p2 = _outcomes(tmp_path / "2.json", CaughtMutant=4, MissedMutant=2)
    areas = m.merge([("a", p1), ("a", p2)])
    assert areas["a"]["CaughtMutant"] == 7
    assert areas["a"]["MissedMutant"] == 3


def test_no_tested_mutants_has_no_score(tmp_path):
    p = _outcomes(tmp_path / "o.json", Unviable=3)
    assert m.score(m.merge([("a", p)])["a"]) is None


def test_only_enforced_areas_fail(tmp_path):
    low = _outcomes(tmp_path / "low.json", CaughtMutant=1, MissedMutant=1)
    empty = _outcomes(tmp_path / "empty.json")
    areas = m.merge([("low", low), ("empty", empty), ("report", low)])
    floors = {
        "target": 70,
        "areas": {"low": {"floor": 70, "enforce": True}, "empty": {"enforce": True}, "report": {"floor": 90}},
    }
    rows, failures = m.evaluate(areas, floors)
    assert len(failures) == 2
    assert "low: 50.0%" in failures[1]
    assert "empty: no mutants tested" in failures[0]
    assert {r["area"]: r["enforce"] for r in rows} == {"empty": True, "low": True, "report": False}


def test_main_writes_json_and_returns_status(tmp_path, capsys):
    p = _outcomes(tmp_path / "o.json", CaughtMutant=9, MissedMutant=1)
    floors = tmp_path / "f.json"
    floors.write_text(json.dumps({"target": 70, "areas": {"a": {"floor": 95, "enforce": True}}}))
    out = tmp_path / "out.json"
    assert m.main(["--floors", str(floors), "--out", str(out), f"a={p}"]) == 1
    assert json.loads(out.read_text())["areas"][0]["score"] == 90.0
    assert "| a | 90.0% |" in capsys.readouterr().out


def test_enforced_area_without_input_fails():
    rows, failures = m.evaluate({}, {"areas": {"gone": {"floor": 70, "enforce": True}, "soft": {}}})
    assert [r["area"] for r in rows] == ["gone", "soft"]
    assert failures == ["gone: no mutants tested is below the floor 70%"]
