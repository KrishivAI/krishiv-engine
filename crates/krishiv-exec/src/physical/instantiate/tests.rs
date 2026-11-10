// SPDX-License-Identifier: Apache-2.0
//! Unit tests of instantiation.

use std::sync::Arc;

use arrow_schema::SchemaRef;

use super::Unbuilt;
use crate::pipeline::{ExecNode as _, PipelineBuilder};

/// The pipelines `root` instantiates to, one `(preserve_order)` per pipeline.
fn driver_orders(root: crate::physical::PhysNode) -> Vec<bool> {
    use arrow_schema::{DataType, Field, Schema};

    let plan = crate::physical::PhysicalPlan::new(
        root,
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
        crate::planner::ExplainNode::leaf("Range", ""),
        crate::physical::ExprEnv {
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
            binary_output_style: None,
        },
    );
    let worker = super::WorkerContext::new(Arc::new(crate::planner::MemoryTables::new()));
    let node = super::instantiate(&plan, &worker).unwrap();
    let mut b = PipelineBuilder::new(Arc::new(crate::testing::Count::default()));
    node.build_pipelines(&mut b).unwrap();
    let (pipelines, _) = b.finish().unwrap();
    pipelines.iter().map(|p| p.preserve_order).collect()
}

fn range(num_slices: u32) -> crate::physical::PhysNode {
    crate::physical::PhysNode::Range(crate::physical::RangeNode {
        start: 0,
        end: 64,
        step: 1,
        num_slices: Some(num_slices),
        output: krishiv_plan::AttrId(0),
    })
}

/// A multi-slice range streamed to the result runs on one driver, so its
/// batches reach the result in slice order whatever the scheduling.
#[test]
fn a_range_streamed_to_the_result_keeps_one_driver() {
    assert_eq!(driver_orders(range(8)), [true]);
    let limited = crate::physical::PhysNode::Limit {
        input: Box::new(range(8)),
        offset: 0,
        limit: 5,
    };
    assert_eq!(driver_orders(limited), [true]);
}

/// An exchange's planned columns must be its built input's, on one
/// worker too (the fragmenter hands the planned ones to its readers).
#[test]
fn an_exchange_planned_with_other_columns_fails_to_instantiate() {
    use arrow_schema::{DataType, Field, Schema};

    use crate::physical::{ExchangeId, ExchangePlan, Partitioning, PhysNode, PhysicalPlan};

    let id = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let plan = |schema| {
        let root = PhysNode::Exchange(ExchangePlan {
            id: ExchangeId(0),
            input: Box::new(range(1)),
            partitioning: Partitioning::Single,
            filters: Vec::new(),
            schema,
        });
        PhysicalPlan::new(
            root,
            Arc::clone(&id),
            crate::planner::ExplainNode::leaf("Range", ""),
            crate::physical::ExprEnv {
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
                binary_output_style: None,
            },
        )
    };
    let worker = super::WorkerContext::new(Arc::new(crate::planner::MemoryTables::new()));
    let built = super::exchange_schemas(&plan(None), &worker).unwrap();
    let fine = plan(Some(SchemaRef::clone(&built[&ExchangeId(0)])));
    assert!(super::instantiate(&fine, &worker).is_ok());
    assert_eq!(super::planned_exchange_schemas(&fine), Some(built));
    assert_eq!(super::planned_exchange_schemas(&plan(None)), None);
    let other = Arc::new(Schema::new(vec![Field::new("x", DataType::Utf8, true)]));
    let Err(e) = super::instantiate(&plan(Some(other)), &worker) else {
        panic!("a wrong planned schema instantiated");
    };
    assert!(e.to_string().contains("exchange 0 planned"), "{e}");
}

#[test]
fn a_validation_input_never_runs() {
    let mut b = PipelineBuilder::new(Arc::new(crate::testing::Count::default()));
    let e = Arc::new(Unbuilt).build_pipelines(&mut b).unwrap_err();
    assert!(e.to_string().contains("validation-only"), "{e}");
}

/// `rows`, under the phases [`super::distinct_phases`] plans for `single`,
/// around a hash exchange.
fn written(single: &super::AggregatePlan) -> (crate::physical::PhysNode, super::AggregatePlan) {
    use crate::physical::{ExchangeId, ExchangePlan, Partitioning, PhysNode, RangeNode};

    let [first, merge, last] = super::distinct_phases(single).unwrap();
    let rows = PhysNode::Range(RangeNode {
        start: 0,
        end: 10,
        step: 1,
        num_slices: None,
        output: krishiv_plan::AttrId(0),
    });
    let first = PhysNode::Aggregate {
        input: Box::new(rows),
        aggregate: first,
    };
    let moved = PhysNode::Exchange(ExchangePlan {
        id: ExchangeId(0),
        input: Box::new(first),
        partitioning: Partitioning::Single,
        filters: Vec::new(),
        schema: None,
    });
    let merge = PhysNode::Aggregate {
        input: Box::new(moved),
        aggregate: merge,
    };
    (merge, last)
}

/// `count(DISTINCT c1), sum(c2), max(c1) GROUP BY c0`.
fn single() -> super::AggregatePlan {
    use crate::aggregate::{AggregateExpr, AggregateFunction as F, AggregateOutput};
    use arrow_schema::DataType;

    super::AggregatePlan {
        group_by: vec![0],
        aggregates: vec![
            AggregateExpr::new(F::Count, 1, DataType::Int64, "c"),
            AggregateExpr::new(F::Sum, 2, DataType::Int64, "s"),
            AggregateExpr::new(F::Max, 1, DataType::Int64, "m"),
        ],
        distinct: vec![true, false, false],
        output: AggregateOutput::Final,
    }
}

#[test]
fn written_distinct_phases_fuse_back_into_their_aggregation() {
    let single = single();
    let (merge, last) = written(&single);
    let (partial, below, fused) = super::fused_distinct(&last, &merge).unwrap();
    assert_eq!(fused, single);
    assert!(matches!(below, crate::physical::PhysNode::Range(_)));
    assert!(matches!(
        partial,
        crate::physical::PhysNode::Aggregate { .. }
    ));
    let [first, merged, final_] = super::distinct_phases(&single).unwrap();
    // Keys, the distinct argument, then the other aggregates' states.
    assert_eq!(first.group_by, [0, 1]);
    assert_eq!(first.aggregates.len(), 2);
    assert_eq!(merged.group_by, [0, 1]);
    assert_eq!(final_.group_by, [0]);
    assert_eq!(final_.aggregates[0].args, [1]);
}

/// Any change to one phase is a plan instantiation runs as written.
#[test]
fn a_near_miss_of_the_distinct_phases_does_not_fuse() {
    let single = single();
    let (merge, mut last) = written(&single);
    last.aggregates.swap(1, 2);
    assert!(super::fused_distinct(&last, &merge).is_none());
    let (mut merge, last) = written(&single);
    if let crate::physical::PhysNode::Aggregate { aggregate, .. } = &mut merge {
        aggregate.group_by = vec![0];
    }
    assert!(super::fused_distinct(&last, &merge).is_none());
    // Distinct aggregates over different arguments, or one with a
    // filter, have no phases.
    let mut two = single.clone();
    two.distinct = vec![true, false, true];
    two.aggregates[2].args = vec![2];
    assert!(super::distinct_phases(&two).is_none());
    let mut masked = single;
    masked.aggregates[0].mask = Some(3);
    assert!(super::distinct_phases(&masked).is_none());
}
