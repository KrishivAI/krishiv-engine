// SPDX-License-Identifier: Apache-2.0
//! The refusals [`merge`] rests on, over hand-built cores: each refused case
//! sits beside the firing one it differs from by one defect, so removing the
//! gate makes the refused case fire.

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, FunctionKind, Grouping,
    JoinCondition, JoinKind, LogicalPlan, PlanNode, ReadSource, ResolvedExpr, ResolvedFunction,
    Scan,
};
use krishiv_types::DataType;

use super::merge;
use crate::expr::{call, column};

fn int(id: u64, name: &str) -> BoundColumn {
    BoundColumn {
        id: AttrId(id),
        name: name.to_owned(),
        data_type: DataType::Integer,
        nullable: true,
    }
}

/// `f(fk#0, fv#1)`, `d(dk#2, dv#3)`.
fn fk() -> BoundColumn {
    int(0, "fk")
}
fn fv() -> BoundColumn {
    int(1, "fv")
}
fn dk() -> BoundColumn {
    int(2, "dk")
}
fn dv() -> BoundColumn {
    int(3, "dv")
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

fn f() -> Arc<LogicalPlan> {
    scan("f", vec![fk(), fv()])
}

fn d() -> Arc<LogicalPlan> {
    scan("d", vec![dk(), dv()])
}

fn test(b: Builtin, c: &BoundColumn) -> Arc<Expr> {
    call(b, vec![column(c)], DataType::Boolean)
}

fn not_null(c: &BoundColumn) -> Arc<Expr> {
    test(Builtin::IsNotNull, c)
}

fn filter(input: &Arc<LogicalPlan>, predicate: Arc<Expr>) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(PlanNode::Filter {
        input: Arc::clone(input),
        predicate,
    }))
}

fn join(l: &Arc<LogicalPlan>, r: &Arc<LogicalPlan>, kind: JoinKind) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(PlanNode::Join {
        left: Arc::clone(l),
        right: Arc::clone(r),
        kind,
        condition: JoinCondition::On(call(
            Builtin::Eq,
            vec![column(&fk()), column(&dk())],
            DataType::Boolean,
        )),
    }))
}

/// `<name>(arg) AS out#id`.
fn agg_call(name: &str, arg: &BoundColumn, id: u64) -> Arc<Expr> {
    let f = ResolvedFunction {
        id: FunctionId::Registry {
            name: name.to_owned(),
            id: 0,
        },
        kind: FunctionKind::Aggregate,
        args: vec![column(arg)],
        distinct: false,
        filter: None,
        within_group: Vec::new(),
        data_type: DataType::Long,
        nullable: true,
        deterministic: true,
    };
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
        expr: Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f)))),
        output: BoundColumn {
            id: AttrId(id),
            name: format!("{name}{id}"),
            data_type: DataType::Long,
            nullable: true,
        },
    })))
}

fn sum_fv() -> Arc<Expr> {
    agg_call("sum", &fv(), 20)
}

fn aggregate(
    input: &Arc<LogicalPlan>,
    keys: &[BoundColumn],
    outputs: Vec<Arc<Expr>>,
) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(PlanNode::Aggregate {
        input: Arc::clone(input),
        grouping: Grouping::GroupBy(keys.iter().map(column).collect()),
        aggregates: outputs,
    }))
}

/// `Aggregate[key](input)` listing the key and `sum(fv)`.
fn by(input: &Arc<LogicalPlan>, key: &BoundColumn) -> Arc<LogicalPlan> {
    aggregate(
        input,
        std::slice::from_ref(key),
        vec![column(key), sum_fv()],
    )
}

fn merges(a: &Arc<LogicalPlan>, b: &Arc<LogicalPlan>) -> bool {
    merge(a, b).is_some()
}

/// The firing base: a null test on the key below one copy. The unfiltered
/// copy is the definition (in either argument order), and the filtered one
/// tests the definition's output of the key.
#[test]
fn the_unfiltered_copy_is_shared_and_the_filtered_one_tests_its_key() {
    let bare = by(&f(), &fk());
    let tested = by(&filter(&f(), not_null(&fk())), &fk());
    let m = merge(&tested, &bare).expect("merges");
    assert!(Arc::ptr_eq(&m.def, &bare), "the unfiltered aggregate");
    assert_eq!(m.tests_a, vec![not_null(&fk())]);
    assert!(m.tests_b.is_empty());
    let m = merge(&bare, &tested).expect("merges");
    assert!(Arc::ptr_eq(&m.def, &bare), "the unfiltered aggregate");
    assert!(m.tests_a.is_empty());
    assert_eq!(m.tests_b, vec![not_null(&fk())]);
}

/// A null test on the preserved side of an outer join commutes with it; on
/// the NULL-supplying side (and on either side of a FULL OUTER join) the
/// join could turn the key NULL, so the copies stay apart.
#[test]
fn a_null_test_on_a_null_supplying_side_is_not_stripped() {
    let firing = [
        (JoinKind::LeftOuter, 0),
        (JoinKind::RightOuter, 1),
        (JoinKind::Inner, 0),
        (JoinKind::Inner, 1),
    ];
    let refused = [
        (JoinKind::LeftOuter, 1),
        (JoinKind::RightOuter, 0),
        (JoinKind::FullOuter, 0),
        (JoinKind::FullOuter, 1),
    ];
    for (kind, side, fires) in firing
        .iter()
        .map(|&(k, s)| (k, s, true))
        .chain(refused.iter().map(|&(k, s)| (k, s, false)))
    {
        // The key is the filtered side's column.
        let key = if side == 0 { fk() } else { dk() };
        let (l, r) = if side == 0 {
            (filter(&f(), not_null(&key)), d())
        } else {
            (f(), filter(&d(), not_null(&key)))
        };
        let tested = by(&join(&l, &r, kind), &key);
        let bare = by(&join(&f(), &d(), kind), &key);
        assert_eq!(merges(&tested, &bare), fires, "{kind:?} side {side}");
    }
}

/// The filtered copy may add only bare group keys to the shared aggregate: a
/// call it lists that the unfiltered copy does not would run on the NULL
/// group the query never evaluated it on. The unfiltered copy may add calls.
#[test]
fn a_filtered_copy_adds_no_call() {
    let count = agg_call("count", &fv(), 21);
    let tested_wide = aggregate(
        &filter(&f(), not_null(&fk())),
        &[fk()],
        vec![column(&fk()), sum_fv(), Arc::clone(&count)],
    );
    let bare = by(&f(), &fk());
    assert!(!merges(&tested_wide, &bare));
    assert!(!merges(&bare, &tested_wide));

    let tested = by(&filter(&f(), not_null(&fk())), &fk());
    let bare_wide = aggregate(
        &f(),
        &[fk()],
        vec![column(&fk()), sum_fv(), Arc::clone(&count)],
    );
    let m = merge(&tested, &bare_wide).expect("the unfiltered copy adds a call");
    assert!(Arc::ptr_eq(&m.def, &bare_wide));

    // A bare key the filtered copy lists (pruned from the other) is added.
    let keys = [fk(), dk()];
    let input = join(&f(), &d(), JoinKind::Inner);
    let tested_keys = aggregate(
        &filter(&input, not_null(&fk())),
        &keys,
        vec![column(&fk()), column(&dk()), sum_fv()],
    );
    let bare_pruned = aggregate(&input, &keys, vec![column(&dk()), sum_fv()]);
    let m = merge(&tested_keys, &bare_pruned).expect("a bare key is added");
    let PlanNode::Aggregate { aggregates, .. } = &m.def.node else {
        panic!("an aggregate");
    };
    assert_eq!(aggregates.len(), 3);
    assert_eq!(m.tests_a, vec![not_null(&fk())]);
}

/// `isnull(k)` keeps the NULL group only and `isnotnull(k)` drops it: neither
/// set holds the other, so neither copy is the other's definition.
#[test]
fn isnull_and_isnotnull_copies_stay_apart() {
    let is_null = by(&filter(&f(), test(Builtin::IsNull, &fk())), &fk());
    let not = by(&filter(&f(), not_null(&fk())), &fk());
    assert!(!merges(&is_null, &not));
    assert!(!merges(&not, &is_null));
}

/// An `isnull(k)` copy shares the unfiltered aggregate and keeps its own
/// rows: it tests `isnull` of the definition's output of the key.
#[test]
fn an_isnull_copy_tests_isnull_over_the_shared_groups() {
    let is_null = by(&filter(&f(), test(Builtin::IsNull, &fk())), &fk());
    let bare = by(&f(), &fk());
    let m = merge(&is_null, &bare).expect("merges");
    assert!(Arc::ptr_eq(&m.def, &bare));
    assert_eq!(m.tests_a, vec![test(Builtin::IsNull, &fk())]);
    assert!(m.tests_b.is_empty());
}

/// A global aggregate has one row even over no input, so a filter below it
/// does not commute with it: `count(*)` over the NULL rows only is 0, not
/// "no row". Without keys there is nothing to share.
#[test]
fn a_global_aggregate_is_refused() {
    let tested = aggregate(&filter(&f(), not_null(&fk())), &[], vec![sum_fv()]);
    let bare = aggregate(&f(), &[], vec![sum_fv()]);
    assert!(!merges(&tested, &bare));
    assert!(!merges(&bare, &tested));
    // With the key, the same pair fires.
    assert!(merges(
        &by(&filter(&f(), not_null(&fk())), &fk()),
        &by(&f(), &fk())
    ));
}
