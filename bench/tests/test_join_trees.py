"""join-trees (P2-T113): arm parsing, EXPLAIN VERBOSE tree notes, and the summary across arms."""

from __future__ import annotations

import pytest

from krishiv_bench import cli, join_trees

NOTE = (
    "tree inputs=12 edges=11 hyperedges=1 enumerator=dpsub fallback=inputs pairs=0 considered=0 "
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
