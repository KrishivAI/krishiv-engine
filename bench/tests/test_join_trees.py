"""join-trees (P2-T113): arm parsing, EXPLAIN VERBOSE tree notes, and the summary across arms."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from krishiv_bench import cli, join_trees
from krishiv_bench.adapters.base import Query
from krishiv_bench.adapters.krishiv import RunnerCrashed, RunnerError

NOTE = (
    "tree inputs=12 edges=11 hyperedges=1 enumerator=dpsub fallback=inputs pairs=0 considered=0 measure=rows "
    "cost=1.250000e6 shape=((0 1) (2 3))"
)


def test_arms_parse() -> None:
    name, conf = join_trees.parse_arm("dpsub12=krishiv.optimizer.joinEnumerator=dpsub,krishiv.optimizer.joinDpLimit=12")
    assert name == "dpsub12"
    assert conf == [("krishiv.optimizer.joinEnumerator", "dpsub"), ("krishiv.optimizer.joinDpLimit", "12")]
    for bad in ("nosettings", "=a=b", "x=novalue"):
        with pytest.raises(SystemExit):
            join_trees.parse_arm(bad)


def test_a_tree_note_parses() -> None:
    t = join_trees.parse_tree(NOTE)
    assert t == {
        "inputs": 12,
        "edges": 11,
        "hyperedges": 1,
        "enumerator": "dpsub",
        "fallback": "inputs",
        "pairs": 0,
        "considered": 0,
        "measure": "rows",
        "cost": 1.25e6,
        "shape": "((0 1) (2 3))",
    }


def step(note: str, batch: str = "JoinOrder") -> str:
    return f"== Optimizer: {batch} round 1, JoinOrder: {note} ==\n\n"


def test_the_last_ordering_block_is_the_plans() -> None:
    first = NOTE.replace("cost=1.250000e6", "cost=9.000000e6")
    text = (
        "== Analyzed Logical Plan ==\nJoin\n"
        + step(first)
        + step(first)
        + "== Optimizer: EagerAggregate round 1, EagerAggregate: moved ==\n- a\n+ b\n"
        + step(NOTE)
        + step(NOTE.replace("inputs=12", "inputs=3"))
        + "== Optimizer: Prune round 1, Prune ==\n- x\n"
        + "== Optimized Logical Plan ==\nJoin\n"
    )
    trees = join_trees.trees_of(text)
    assert [(t["inputs"], t["cost"], t["shape"]) for t in trees] == [
        (12, 1.25e6, "((0 1) (2 3))"),
        (3, 1.25e6, "((0 1) (2 3))"),
    ]
    assert join_trees.trees_of("no joins") == []


def tree(inputs: int, shape: str, cost: float, fallback: str = "none", hyperedges: int = 0) -> dict:
    return {
        "inputs": inputs,
        "edges": inputs - 1,
        "hyperedges": hyperedges,
        "enumerator": "x",
        "fallback": fallback,
        "pairs": 10,
        "considered": 20,
        "cost": cost,
        "shape": shape,
    }


def test_the_summary_counts_trees_and_differing_plans() -> None:
    record = {
        "arms": {
            "dpsub": {
                "queries": {
                    "q1": {"trees": [tree(12, "(a b)", 100.0, "inputs")], "plan_p99": 0.004},
                    "q2": {"trees": [tree(3, "(c d)", 5.0, hyperedges=1)], "plan_p99": 0.002},
                    "q3": {"error": "unsupported"},
                }
            },
            "dphyp": {
                "queries": {
                    "q1": {"trees": [tree(12, "(b a)", 80.0)], "plan_p99": 0.001},
                    "q2": {"trees": [tree(3, "(c d)", 5.0, hyperedges=1)], "plan_p99": 0.002},
                    "q3": {"error": "unsupported"},
                }
            },
        }
    }
    s = join_trees.summarize(record)
    assert s["base"] == "dpsub"
    base, hyp = s["arms"]["dpsub"], s["arms"]["dphyp"]
    assert base["trees_by_inputs"] == {3: 1, 12: 1}
    assert base["fallbacks"] == {"inputs": 1, "none": 1}
    assert base["large_or_hyperedge_trees"] == 2
    assert base["errors"] == ["q3"]
    assert base["worst_plan_p99"] == {"query": "q1", "seconds": 0.004}
    assert base["plans_differing_from_base"] == []
    assert hyp["plans_differing_from_base"] == [{"query": "q1", "cost_ratio": 0.8}]


def test_the_cli_takes_join_trees(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(join_trees, "run_join_trees", lambda *a: seen.update(args=a))
    cli.main(["join-trees", "--suite", "dsb", "--sf", "10", "--arm", "a=k=v", "--queries", "q1,q2"])
    suite, sf, dataset, arms, _runner, only, reps, *_ = seen["args"]
    assert (suite, sf, dataset, arms, only, reps) == ("dsb", 10.0, "dsb@v1", ["a=k=v"], ["q1", "q2"], 50)
    cli.main(["join-trees", "--suite", "tpch"])
    assert seen["args"][3] == join_trees.DEFAULT_ARMS


def test_percentile_is_nearest_rank() -> None:
    xs = [0.5, 0.1, 0.4, 0.2, 0.3]
    assert join_trees.percentile(xs, 0.5) == 0.3
    assert join_trees.percentile(xs, 0.99) == 0.5
    assert join_trees.percentile(xs, 0.0) == 0.1
    assert join_trees.percentile([], 0.5) != join_trees.percentile([], 0.5)  # NaN


class FakeEngine:
    """Answers EXPLAIN VERBOSE with one tree note, times EXPLAIN, and fails the query named ``bad``."""

    def __init__(self, bad: str | None = None) -> None:
        self.bad = bad
        self.events: list[str] = []
        self.kwargs: dict[str, Any] = {}

    def run(self, query):
        if query.id == self.bad:
            raise RunnerError("PLAN_ERROR: cannot plan\nsecond line")
        assert query.sql.startswith("EXPLAIN VERBOSE select")
        assert not query.sql.endswith(";")
        return [(step(NOTE) + "== Optimized Logical Plan ==\nJoin",)]

    def timed_run(self, query):
        assert query.sql.startswith("EXPLAIN select")
        return 0, 0.004

    def setup(self, dataset) -> None:
        self.events.append("setup")

    def teardown(self) -> None:
        self.events.append("teardown")

    def config(self) -> dict:
        return {"binary": "fake"}


def test_plan_arm_records_trees_timing_and_errors() -> None:
    queries = [Query("q1", "select 1;", "tpch"), Query("q2", "select 2", "tpch"), Query("q3", "select 3", "tpch")]
    out = join_trees.plan_arm(cast(Any, FakeEngine(bad="q2")), queries, reps=3)
    assert [t["inputs"] for t in out["q1"]["trees"]] == [12]
    assert out["q1"]["plan_seconds"] == [0.004] * 3
    assert out["q1"]["plan_p50"] == 0.004
    assert out["q1"]["plan_p99"] == 0.004
    assert out["q2"] == {"error": "PLAN_ERROR: cannot plan"}
    assert "trees" in out["q3"]
    crashed = join_trees.plan_arm(cast(Any, CrashingEngine()), [Query("q9", "select 9", "tpch")], reps=1)
    assert crashed["q9"] == {"error": "RunnerCrashed"}
    no_reps = join_trees.plan_arm(cast(Any, FakeEngine()), [Query("q1", "select 1", "tpch")], reps=0)
    assert no_reps["q1"]["plan_p50"] is None
    assert no_reps["q1"]["plan_p99"] is None


class CrashingEngine:
    def run(self, query):
        raise RunnerCrashed

    def timed_run(self, query):
        raise RunnerCrashed


def summary_record() -> dict:
    def arm(shape: str, cost: float) -> dict:
        return {
            "queries": {
                "q1": {"trees": [tree(12, shape, cost, "inputs")], "plan_p99": 0.004},
                "q2": {"error": "unsupported"},
            }
        }

    record = {
        "suite": "tpch",
        "dataset": {"scale_factor": 10.0},
        "arms": {"dpsub": arm("(a b)", 100.0), "greedy": arm("(b a)", 80.0)},
    }
    record["summary"] = join_trees.summarize(record)
    return record


def test_markdown_names_differing_plans_and_errors() -> None:
    text = join_trees.markdown(summary_record())
    assert text.startswith("# join-trees tpch sf10 (base arm: dpsub)")
    assert "| dpsub | 1 | 1 | inputs 1 | 10 | 20 | q1 4.00 ms | 0 |" in text
    assert "| greedy | 1 | 1 | inputs 1 | 10 | 20 | q1 4.00 ms | 1 |" in text
    assert "trees by inputs (dpsub): {12: 1}" in text
    assert "greedy differs on: q1 (0.800x)" in text
    assert "dpsub errors: q2" in text
    # An arm with no plan time and no errors prints a dash, and a plan without a cost ratio just its name.
    record = summary_record()
    record["summary"]["arms"]["greedy"]["worst_plan_p99"] = None
    record["summary"]["arms"]["greedy"]["errors"] = []
    record["summary"]["arms"]["greedy"]["plans_differing_from_base"] = [{"query": "q7", "cost_ratio": None}]
    text = join_trees.markdown(record)
    assert "| - |" in text
    assert "greedy differs on: q7" in text
    assert "greedy errors" not in text


def test_run_join_trees_writes_one_record_and_prints_the_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    engines: list[FakeEngine] = []

    def make_engine(**kwargs) -> FakeEngine:
        engines.append(FakeEngine())
        engines[-1].kwargs = kwargs
        return engines[-1]

    monkeypatch.setattr(join_trees, "dataset_dir", lambda name, sf: tmp_path / name)
    monkeypatch.setattr(join_trees, "discover_tables", lambda root: ["t"])
    monkeypatch.setattr(join_trees, "table_paths", lambda root, names: {"t": ["p"]})
    monkeypatch.setattr(
        join_trees, "load_queries", lambda suite, engine, only, dataset_root=None: [Query("q1", "select 1", suite)]
    )
    monkeypatch.setattr(join_trees, "KrishivEngine", make_engine)
    monkeypatch.setattr(join_trees, "machine_info", lambda: {"cpu": "x"})
    monkeypatch.setattr(join_trees, "git_info", lambda: {"sha": "y"})
    out = join_trees.run_join_trees(
        "tpch", 10.0, "tpch-duckdb", ["a=k=v", "b=k=w"], reps=2, threads=2, memory_limit="1g", out_dir=tmp_path / "out"
    )
    assert out.parent == tmp_path / "out"
    assert out.name.endswith("_join-trees_tpch_sf10.json")
    record = json.loads(out.read_text())
    assert list(record["arms"]) == ["a", "b"]
    assert record["plan_reps"] == 2
    assert record["arms"]["a"]["conf"] == ["k=v"]
    assert record["arms"]["a"]["runner"] == {"binary": "fake"}
    assert record["summary"]["base"] == "a"
    assert [e.events for e in engines] == [["setup", "teardown"]] * 2
    assert engines[0].kwargs["conf"] == [("k", "v")]
    assert engines[0].kwargs["threads"] == 2
    printed = capsys.readouterr().out
    assert "# join-trees tpch sf10" in printed
    assert f"result: {out}" in printed
    # No dataset, no run.
    monkeypatch.setattr(join_trees, "discover_tables", lambda root: [])
    with pytest.raises(SystemExit, match="no dataset at"):
        join_trees.run_join_trees("tpch", 10.0, "tpch-duckdb", ["a=k=v"])
