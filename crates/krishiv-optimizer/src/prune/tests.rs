// SPDX-License-Identifier: Apache-2.0
//! Hand-built plans given to `prune` directly: shapes the analyzer or the
//! earlier batches remove before pruning runs on a SQL query.

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_plan::{
    AttrId, BoundColumn, Grouping, LogicalPlan, Origin, PlanId, PlanNode, ReadSource, Scan,
    SetOperation, SourceId, Span,
};
use krishiv_types::DataType;

use super::prune;
use crate::expr::column;

fn int(id: u64, name: &str) -> BoundColumn {
    BoundColumn {
        id: AttrId(id),
        name: name.to_owned(),
        data_type: DataType::Integer,
        nullable: true,
    }
}

fn scan(table: &str, output: Vec<BoundColumn>) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(PlanNode::Scan(Scan {
        source: ReadSource::Table {
            name: vec![table.to_owned()],
            options: BTreeMap::new(),
        },
        output,
        version: None,
    })))
}

fn plan(node: PlanNode) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(node))
}

/// The scan's output columns anywhere in `p` (the first scan met).
fn scan_width(p: &LogicalPlan) -> Option<usize> {
    if let PlanNode::Scan(s) = &p.node {
        return Some(s.output.len());
    }
    let mut out = None;
    p.for_each_child(&mut |c| out = out.or_else(|| scan_width(c)));
    out
}

/// `Project [a]` over `node(Scan [a, b])`: the node reads its input whole.
fn reads_whole(node: impl FnOnce(Arc<LogicalPlan>) -> PlanNode) {
    let (a, b) = (int(1, "a"), int(2, "b"));
    let input = scan("t", vec![a.clone(), b]);
    let root = plan(PlanNode::Project {
        input: plan(node(input)),
        exprs: vec![column(&a)],
    });
    let out = prune(&root, None);
    assert_eq!(scan_width(&out), Some(2), "{out:?}");
}

#[test]
fn a_distinct_reads_its_input_whole() {
    reads_whole(|input| PlanNode::Distinct { input });
}

#[test]
fn a_set_operation_reads_both_inputs_whole() {
    let (c, d) = (int(3, "c"), int(4, "d"));
    let right = scan("u", vec![c, d]);
    reads_whole(|left| PlanNode::SetOp {
        left,
        right,
        op: SetOperation::Intersect,
        all: false,
    });
}

#[test]
fn a_hint_and_a_pipe_operator_read_their_input_whole() {
    reads_whole(|input| PlanNode::Hint {
        input,
        name: "broadcast".to_owned(),
        params: Vec::new(),
    });
    reads_whole(|input| PlanNode::PipeOperator { input });
}

#[test]
fn a_with_columns_and_a_pivot_read_their_input_whole() {
    reads_whole(|input| PlanNode::WithColumns {
        input,
        columns: Vec::new(),
    });
    reads_whole(|input| PlanNode::Pivot {
        input,
        pivot_columns: Vec::new(),
        values: Vec::new(),
        aggregates: Vec::new(),
    });
}

#[test]
fn a_projection_read_for_its_rows_alone_keeps_its_origin() {
    // Nothing reads the projection (a global aggregate with no call), and the
    // union below keeps its columns (it is read by position), so the
    // projection is rebuilt over the union's first column.
    let (a, b, c, d) = (int(1, "a"), int(2, "b"), int(3, "c"), int(4, "d"));
    let union = plan(PlanNode::Union {
        inputs: vec![scan("t", vec![a.clone(), b.clone()]), scan("u", vec![c, d])],
        by_name: false,
        allow_missing_columns: false,
    });
    let origin = Origin::sql(SourceId(0), Span::new(3, 9)).with_plan_id(PlanId(7));
    let placeholder = Arc::new(LogicalPlan {
        node: PlanNode::Project {
            input: union,
            exprs: vec![column(&b)],
        },
        origin,
    });
    let root = plan(PlanNode::Aggregate {
        input: placeholder,
        grouping: Grouping::GroupBy(Vec::new()),
        aggregates: Vec::new(),
    });
    let out = prune(&root, None);
    let PlanNode::Aggregate { input, .. } = &out.node else {
        panic!("{out:?}");
    };
    let PlanNode::Project { exprs, .. } = &input.node else {
        panic!("{input:?}");
    };
    assert_eq!(exprs, &vec![column(&a)]);
    assert_eq!(input.origin, origin);
}
