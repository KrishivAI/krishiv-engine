// SPDX-License-Identifier: Apache-2.0
use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_common::error::ErrorClass;
use krishiv_exec::physical::{
    ExchangeId, ExchangePlan, ExprEnv, HashScheme, Partitioning, PhysNode, PhysicalPlan, RangeNode,
    SharedId, SharedPlan,
};
use krishiv_exec::planner::ExplainNode;
use krishiv_plan::AttrId;

use crate::{Stage, fragment};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]))
}

fn env() -> ExprEnv {
    ExprEnv {
        ansi: true,
        legacy_statistical_aggregate: false,
        zone: Some(0),
        zone_name: None,
        query_start_micros: 0,
        user: Arc::from("u"),
        map_key_dedup_last_win: false,
        legacy_java_charsets: false,
        legacy_coding_error_action: false,
        legacy_time_parser: false,
    }
}

fn range() -> PhysNode {
    PhysNode::Range(RangeNode {
        start: 0,
        end: 10,
        step: 1,
        num_slices: None,
        output: AttrId(1),
    })
}

fn exchange(id: u64, input: PhysNode, partitioning: Partitioning) -> PhysNode {
    PhysNode::Exchange(ExchangePlan {
        id: ExchangeId(id),
        input: Box::new(input),
        partitioning,
    })
}

fn hash() -> Partitioning {
    Partitioning::Hash {
        keys: vec![0],
        scheme: HashScheme::KeyGroupV1,
    }
}

fn plan(root: PhysNode) -> PhysicalPlan {
    PhysicalPlan::new(root, schema(), ExplainNode::leaf("Root", ""), env())
}

fn schemas(ids: &[u64]) -> BTreeMap<ExchangeId, SchemaRef> {
    ids.iter().map(|i| (ExchangeId(*i), schema())).collect()
}

/// Every exchange id in `n`, and every exchange source id.
fn ids(n: &PhysNode) -> (Vec<u64>, Vec<u64>) {
    let (mut exchanges, mut sources) = (Vec::new(), Vec::new());
    let mut stack = vec![n];
    while let Some(m) = stack.pop() {
        match m {
            PhysNode::Exchange(e) => exchanges.push(e.id.0),
            PhysNode::ExchangeSource { id, .. } => sources.push(id.0),
            _ => {}
        }
        stack.extend(m.children());
    }
    exchanges.sort_unstable();
    sources.sort_unstable();
    (exchanges, sources)
}

fn output(s: &Stage) -> Option<u64> {
    s.output.as_ref().map(|(id, _)| id.0)
}

#[test]
fn every_exchange_is_cut_and_stages_come_before_their_readers() {
    // Union(Exchange 2 (Hash, Exchange 1 (Broadcast, range)), Exchange 3 (Single, range))
    let root = PhysNode::Union {
        inputs: vec![
            exchange(2, exchange(1, range(), Partitioning::Broadcast), hash()),
            exchange(3, range(), Partitioning::Single),
        ],
    };
    let stages = fragment(&plan(root), &schemas(&[1, 2, 3])).unwrap();
    assert_eq!(
        stages.iter().map(output).collect::<Vec<_>>(),
        vec![Some(1), Some(2), Some(3), None]
    );
    for (i, s) in stages.iter().enumerate() {
        assert_eq!(s.id, i);
        for input in &s.inputs {
            assert!(input.stage < s.id);
            assert_eq!(output(&stages[input.stage]), Some(input.exchange.0));
        }
        // A map stage's fragment holds exactly its own exchange.
        let (exchanges, _) = ids(s.plan.root());
        assert_eq!(exchanges, output(s).into_iter().collect::<Vec<_>>());
    }
    assert_eq!(ids(stages[1].plan.root()).1, vec![1]);
    assert_eq!(ids(stages[3].plan.root()), (vec![], vec![2, 3]));
    // The range stages slice a range and the root reads a gather by its
    // rows: each runs whole. Stage 1 reads only a broadcast.
    assert_eq!(
        stages.iter().map(|s| s.single).collect::<Vec<_>>(),
        vec![true, false, true, true]
    );
    assert!(stages.iter().all(|s| !s.scans));
}

#[test]
fn a_hash_stage_without_whole_inputs_may_run_as_many_tasks() {
    let root = exchange(
        3,
        PhysNode::Union {
            inputs: vec![exchange(1, range(), hash()), exchange(2, range(), hash())],
        },
        Partitioning::Single,
    );
    let stages = fragment(&plan(root), &schemas(&[1, 2, 3])).unwrap();
    // Stage 2 writes the gather; it reads two hash exchanges and holds
    // nothing that must run whole.
    assert_eq!(stages.len(), 4);
    assert!(!stages[2].single);
    assert_eq!(stages[2].inputs.len(), 2);
    assert!(stages[2].inputs.iter().all(|i| !i.subquery));
    // The root only reads the gather: one task.
    assert!(stages[3].single);
}

#[test]
fn a_one_row_branch_beside_a_hash_read_runs_as_one_task() {
    // `SELECT k FROM t GROUP BY k UNION ALL SELECT 0 ORDER BY k`: the
    // constant branch would come once per task of the stage writing the
    // range exchange.
    let root = exchange(
        2,
        PhysNode::Union {
            inputs: vec![exchange(1, range(), hash()), PhysNode::OneRow],
        },
        Partitioning::Single,
    );
    let stages = fragment(&plan(root), &schemas(&[1, 2])).unwrap();
    assert_eq!(stages.len(), 3);
    assert!(stages[1].single);
}

#[test]
fn a_fragment_is_not_cut_again() {
    let source = PhysNode::ExchangeSource {
        id: ExchangeId(9),
        partitioning: hash(),
        schema: schema(),
    };
    let e = fragment(&plan(exchange(1, source, hash())), &schemas(&[1])).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
}

#[test]
fn unpinned_hash_exchanges_and_missing_schemas_are_refused() {
    let unpinned = exchange(
        1,
        range(),
        Partitioning::Hash {
            keys: vec![0],
            scheme: HashScheme::Unpinned,
        },
    );
    let e = fragment(&plan(unpinned), &schemas(&[1])).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    let e = fragment(&plan(exchange(1, range(), hash())), &schemas(&[])).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
}

fn shared(id: u64, deterministic: bool) -> SharedPlan {
    SharedPlan {
        id: SharedId(id),
        plan: exchange(5, range(), hash()),
        explain: ExplainNode::leaf("Shared", ""),
        deterministic,
        ordered: false,
    }
}

fn read(id: u64) -> PhysNode {
    PhysNode::SharedRead {
        id: SharedId(id),
        layout: vec![AttrId(1)],
    }
}

#[test]
fn shared_plans_go_whole_with_each_stage_that_reads_them() {
    // A shared plan read on both sides of an exchange.
    let root = PhysNode::Union {
        inputs: vec![read(4), exchange(1, read(4), hash())],
    };
    let p = plan(root).with_shared(vec![shared(4, true)]).unwrap();
    let stages = fragment(&p, &schemas(&[1, 5])).unwrap();
    assert_eq!(stages.len(), 2);
    for s in &stages {
        assert_eq!(s.plan.shared().len(), 1);
        // The body's exchange stays: it is lowered where it runs.
        assert_eq!(ids(&s.plan.shared()[0].plan).0, vec![5]);
        assert!(s.single);
    }
    // Two stages would compute a nondeterministic body twice.
    let p = plan(PhysNode::Union {
        inputs: vec![read(4), exchange(1, read(4), hash())],
    })
    .with_shared(vec![shared(4, false)])
    .unwrap();
    let e = fragment(&p, &schemas(&[1, 5])).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    // Read by one stage only, it is fine.
    let p = plan(exchange(1, read(4), hash()))
        .with_shared(vec![shared(4, false)])
        .unwrap();
    let stages = fragment(&p, &schemas(&[1, 5])).unwrap();
    assert!(stages[1].plan.shared().is_empty());
    assert_eq!(stages[0].plan.shared().len(), 1);
}

#[test]
fn fragments_round_trip_through_the_plan_codec() {
    let root = PhysNode::Union {
        inputs: vec![
            exchange(2, exchange(1, range(), Partitioning::Broadcast), hash()),
            exchange(3, range(), Partitioning::Single),
        ],
    };
    for s in fragment(&plan(root), &schemas(&[1, 2, 3])).unwrap() {
        let bytes = krishiv_plan_codec::encode(&s.plan).unwrap();
        let back = krishiv_plan_codec::decode(&bytes).unwrap();
        assert_eq!(back.root(), s.plan.root());
    }
}
