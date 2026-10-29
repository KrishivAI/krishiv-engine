"""Unit tests for .github/scripts/codspeed-plan.py (the CodSpeed shard planner).

Run: (cd bench && uv run --group dev pytest ../tests/scripts/test_codspeed_plan.py)
"""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / ".github" / "scripts" / "codspeed-plan.py"
_spec = importlib.util.spec_from_file_location("codspeed_plan", SCRIPT)
assert _spec is not None
assert _spec.loader is not None
planner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(planner)


def test_every_listed_target_lands_in_its_shard():
    matrix, errors = planner.plan({"a": ["x", "y"], "b": ["z"]}, {"x", "y", "z"})
    assert errors == []
    assert matrix == [
        {"shard": "a", "bench_flags": "--bench x --bench y"},
        {"shard": "b", "bench_flags": "--bench z"},
    ]


def test_unlisted_targets_run_in_a_catch_all_shard():
    matrix, errors = planner.plan({"a": ["x"]}, {"x", "new_b", "new_a"})
    assert errors == []
    assert matrix[-1] == {"shard": "catch-all", "bench_flags": "--bench new_a --bench new_b"}


def test_a_target_in_two_shards_or_in_none_of_the_workspace_fails():
    _, errors = planner.plan({"a": ["x"], "b": ["x", "gone"]}, {"x"})
    assert any("more than one shard: x" in e for e in errors)
    assert any("not a bench target: gone" in e for e in errors)


def test_an_empty_shard_fails_instead_of_running_every_benchmark():
    matrix, errors = planner.plan({"a": ["x"], "empty": []}, {"x"})
    assert any("without a benchmark target: empty" in e for e in errors)
    assert [m["shard"] for m in matrix] == ["a"]
