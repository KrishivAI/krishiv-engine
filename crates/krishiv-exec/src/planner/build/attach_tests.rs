// SPDX-License-Identifier: Apache-2.0
//! Tests that runtime filters reach the scans of every root of a plan.

use std::collections::BTreeMap;

use arrow_schema::DataType;
use krishiv_io::{ColumnPath, FilterId};
use krishiv_plan::{ReadSource, Scan};

use super::*;
use crate::physical::ScanNode;

fn scan(id: u64) -> PhysNode {
    PhysNode::Scan(ScanNode {
        id: ScanId(id),
        scan: Scan {
            source: ReadSource::Table {
                name: vec!["t".to_owned()],
                options: BTreeMap::new(),
            },
            output: Vec::new(),
            version: None,
        },
        projection: Vec::new(),
        filter: None,
        runtime_filters: Vec::new(),
        row_stage: crate::operators::RowStageSettings::default(),
        row_id: false,
    })
}

fn filter_of(node: &PhysNode) -> Vec<FilterId> {
    match node {
        PhysNode::Scan(s) => s.runtime_filters.iter().map(|f| f.id).collect(),
        _ => Vec::new(),
    }
}

/// A filter reaches its scan in whichever tree holds it.
#[test]
fn filters_reach_the_scans_of_every_root() {
    let filter = |id| ScanRuntimeFilter {
        id: FilterId(id),
        column: ColumnPath::new("k"),
        key_type: DataType::Int64,
    };
    let mut filters = HashMap::from([(ScanId(0), vec![filter(10)]), (ScanId(1), vec![filter(11)])]);
    let (mut a, mut b) = (scan(0), scan(1));
    attach_runtime_filters(
        &mut filters,
        Fuse {
            probe: true,
            filter: true,
        },
        [&mut a, &mut b],
    );
    assert_eq!(filter_of(&a), [FilterId(10)]);
    assert_eq!(filter_of(&b), [FilterId(11)]);
    assert!(filters.is_empty());
}

/// A `Filter` over a runtime-filtered scan runs its row stage (design P2-E3
/// U1), each filter at its column in the filter's input; not with the
/// setting off, nor over a predicate that is nondeterministic or a filter
/// target that the scan does not output.
#[test]
fn a_filter_over_a_scan_takes_its_row_stage() {
    use crate::physical::{ProbeFilterColumn, ProbeRowStage};
    use krishiv_plan::{
        Builtin, Expr, ExprNode, FunctionId, FunctionKind, Literal, ResolvedExpr, ResolvedFunction,
    };
    let runtime = |id, column: &str| ScanRuntimeFilter {
        id: FilterId(id),
        column: ColumnPath::new(column),
        key_type: DataType::Int64,
    };
    let leaf = || {
        let PhysNode::Scan(mut s) = scan(0) else {
            unreachable!()
        };
        s.projection = vec![ColumnPath::new("a"), ColumnPath::new("k")];
        PhysNode::Scan(s)
    };
    let filter = |predicate: Expr| PhysNode::Filter {
        input: Box::new(leaf()),
        layout: Vec::new(),
        predicate: Arc::new(predicate),
        subqueries: Vec::new(),
        row_stage: None,
    };
    let stage_of = |n: &PhysNode| match n {
        PhysNode::Filter { row_stage, .. } => row_stage.clone(),
        _ => None,
    };
    let fuse = |filter_on, node: &mut PhysNode, column: &str| {
        let mut filters = HashMap::from([(ScanId(0), vec![runtime(5, column)])]);
        attach_runtime_filters(
            &mut filters,
            Fuse {
                probe: true,
                filter: filter_on,
            },
            [node],
        );
    };
    let constant = || Expr::new(ExprNode::Literal(Literal::Boolean(true)));
    let mut on = filter(constant());
    fuse(true, &mut on, "k");
    assert_eq!(
        stage_of(&on),
        Some(ProbeRowStage {
            scan: ScanId(0),
            columns: vec![ProbeFilterColumn {
                id: FilterId(5),
                column: 1,
            }],
        })
    );
    let PhysNode::Filter { input, .. } = &on else {
        unreachable!()
    };
    assert_eq!(filter_of(input), [FilterId(5)], "the scan still prunes");
    let mut off = filter(constant());
    fuse(false, &mut off, "k");
    assert_eq!(stage_of(&off), None);
    let mut elsewhere = filter(constant());
    fuse(true, &mut elsewhere, "z");
    assert_eq!(stage_of(&elsewhere), None);
    let random = Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id: FunctionId::Builtin(Builtin::Gt),
            kind: FunctionKind::Scalar,
            args: Vec::new(),
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type: krishiv_types::DataType::Boolean,
            nullable: false,
            deterministic: false,
        },
    )));
    let mut nondeterministic = filter(random);
    fuse(true, &mut nondeterministic, "k");
    assert_eq!(stage_of(&nondeterministic), None);
}
