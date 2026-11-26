// SPDX-License-Identifier: Apache-2.0
//! The exactness guards of the lineage walk and the rollback, on resolved
//! (unoptimized) plans: outer-join elimination and pushdown turn most of
//! these shapes into others before the rule runs, so they are built here
//! directly (without the analyzer's `SubqueryAlias` nodes, which the
//! optimizer drops first). Literals are written `1L` (the columns are `BIGINT`): no
//! folding runs, so an `INT` literal would stay a cast.

use std::sync::Arc;

use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_plan::{
    AttrId, BoundColumn, Cte, CteRef, Expr, ExprNode, Grouping, LogicalPlan, PlanNode,
    ResolvedExpr, SourceId,
};
use krishiv_types::DataType;

use super::{Lift, Mode, all_shared, canonical_states, item_key, reader_exprs, reads_any, renamed};
use crate::expr::{column, strip_alias};

fn columns(names: &[&str]) -> Vec<TableColumn> {
    names
        .iter()
        .map(|n| TableColumn {
            name: (*n).to_owned(),
            data_type: DataType::Long,
            nullable: true,
            char_varchar: None,
        })
        .collect()
}

fn resolve(sql: &str) -> Arc<LogicalPlan> {
    let catalog = MemoryCatalog::new("spark_catalog")
        .with_table(&["default"], "t", columns(&["g", "x"]), None)
        .with_table(&["default"], "u", columns(&["k", "v"]), None);
    let ctx = context(catalog, ResolutionConf::default());
    let plan =
        krishiv_sql::parse_and_lower(sql, &krishiv_sql::ParserOptions::default(), SourceId(0))
            .unwrap_or_else(|e| panic!("{sql}: {e:?}"));
    let root = Arc::clone(
        block_on(DefaultAnalyzer.analyze(&plan, &ctx))
            .unwrap_or_else(|e| panic!("{sql}: {e:?}"))
            .plan
            .root(),
    );
    without_aliases(root)
}

/// `plan` without the `SubqueryAlias` nodes the optimizer removes before the
/// rule runs.
fn without_aliases(plan: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    let rebuilt = plan
        .map_children(|c| Ok::<_, ()>(without_aliases(c)))
        .unwrap_or_else(|()| unreachable!());
    crate::project::remove_alias(rebuilt)
}

/// The first aggregate below the root and its first group key.
fn aggregate(plan: &Arc<LogicalPlan>) -> (Arc<LogicalPlan>, AttrId) {
    let mut p = Arc::clone(plan);
    loop {
        if let PlanNode::Aggregate {
            grouping: Grouping::GroupBy(keys),
            ..
        } = &p.node
        {
            let ExprNode::Resolved(ResolvedExpr::Column(c)) = &keys[0].node else {
                panic!("an expression key: {p}");
            };
            return (Arc::clone(&p), c.id);
        }
        let mut next = None;
        p.for_each_child(&mut |c| {
            next.get_or_insert_with(|| Arc::clone(c));
        });
        p = next.unwrap_or_else(|| panic!("no aggregate in {plan}"));
    }
}

/// The literal the walk finds on the key's lineage (as text), or `None`
/// when it refuses.
fn lifted(sql: &str) -> Option<String> {
    let plan = resolve(sql);
    let (top, g) = aggregate(&plan);
    let mut lift = Lift::new(Mode::Mark);
    lift.core(&top, g)?;
    lift.single().map(|(l, _)| format!("{l:?}"))
}

#[test]
fn a_filter_on_the_key_is_followed() {
    let l = lifted("SELECT g, sum(x) FROM t WHERE g = 1L GROUP BY g");
    assert!(l.is_some_and(|l| l.contains('1')));
}

#[test]
fn the_preserved_side_of_an_outer_join_is_followed() {
    assert!(
        lifted(
            "SELECT t.g, sum(u.v) FROM (SELECT * FROM t WHERE g = 1L) t \
             LEFT JOIN u ON t.g = u.k GROUP BY t.g"
        )
        .is_some()
    );
    assert!(
        lifted(
            "SELECT u.k, sum(t.x) FROM t \
             RIGHT JOIN (SELECT * FROM u WHERE k = 1L) u ON t.g = u.k GROUP BY u.k"
        )
        .is_some()
    );
}

/// A key from the null-producing side: `g = 2L`'s rows of one instance are
/// not `g = 2L`'s rows of `g IN (1, 2)`, which also holds the NULL-extended
/// rows the other literal's join produced.
#[test]
fn the_null_producing_side_of_an_outer_join_is_refused() {
    assert_eq!(
        lifted(
            "SELECT u.k, sum(t.x) FROM t \
             LEFT JOIN (SELECT * FROM u WHERE k = 1L) u ON t.g = u.k GROUP BY u.k"
        ),
        None
    );
    assert_eq!(
        lifted(
            "SELECT t.g, sum(u.v) FROM (SELECT * FROM t WHERE g = 1L) t \
             RIGHT JOIN u ON t.g = u.k GROUP BY t.g"
        ),
        None
    );
}

#[test]
fn a_full_outer_join_is_refused() {
    assert_eq!(
        lifted(
            "SELECT t.g, sum(u.v) FROM (SELECT * FROM t WHERE g = 1L) t \
             FULL JOIN u ON t.g = u.k GROUP BY t.g"
        ),
        None
    );
}

#[test]
fn a_union_is_followed_only_when_every_branch_holds_the_conjunct() {
    assert!(
        lifted(
            "SELECT g, sum(x) FROM (SELECT g, x FROM t WHERE g = 1L \
             UNION ALL SELECT k, v FROM u WHERE k = 1L) GROUP BY g"
        )
        .is_some()
    );
    assert_eq!(
        lifted(
            "SELECT g, sum(x) FROM (SELECT g, x FROM t WHERE g = 1L \
             UNION ALL SELECT k, v FROM u) GROUP BY g"
        ),
        None
    );
}

#[test]
fn two_literals_in_one_instance_are_refused() {
    assert_eq!(
        lifted(
            "SELECT g, sum(x) FROM (SELECT g, x FROM t WHERE g = 1L \
             UNION ALL SELECT k, v FROM u WHERE k = 2L) GROUP BY g"
        ),
        None
    );
}

/// The rollback test: a definition is kept only when the shared plan
/// references it and no longer computes it in the main query.
#[test]
fn the_rewrite_is_kept_only_when_every_definition_is_shared() {
    let def = aggregate(&resolve("SELECT g, sum(x) FROM t GROUP BY g")).0;
    let out = def.resolved_output().unwrap();
    let ids: Vec<AttrId> = out.iter().map(|c| c.id).collect();
    let expected = vec![(ids.clone(), Arc::clone(&def))];
    let reference = Arc::new(LogicalPlan::new(PlanNode::CteRef(CteRef {
        name: "d".to_owned(),
        source: ids,
        output: out,
    })));
    let with = |input: Arc<LogicalPlan>| {
        Arc::new(LogicalPlan::new(PlanNode::WithCte {
            input,
            ctes: vec![Cte {
                name: "d".to_owned(),
                plan: Arc::clone(&def),
            }],
        }))
    };
    // Not shared: still computed where it was, no reference.
    assert!(!all_shared(&expected, &def));
    // Shared: one definition, read through a reference.
    assert!(all_shared(&expected, &with(Arc::clone(&reference))));
    // Referenced, but one instance still computes it in the main query.
    let both = Arc::new(LogicalPlan::new(PlanNode::Union {
        inputs: vec![reference, Arc::clone(&def)],
        by_name: false,
        allow_missing_columns: false,
    }));
    assert!(!all_shared(&expected, &with(both)));
}

fn long(id: u64, name: &str) -> BoundColumn {
    BoundColumn {
        id: AttrId(id),
        name: name.to_owned(),
        data_type: DataType::Long,
        nullable: true,
    }
}

/// `SELECT g, sum(x) FROM t WHERE g = 1L GROUP BY g` as eager aggregation
/// leaves it (K5): a `PartialAggregate` grouped by `g` whose `sum` state has
/// id `state`, the aggregate above outputting the state as `out`. The group
/// key's id is returned too.
fn eager(state: u64, out: u64) -> (Arc<LogicalPlan>, AttrId) {
    let (top, g) = aggregate(&resolve("SELECT g, sum(x) FROM t WHERE g = 1L GROUP BY g"));
    let PlanNode::Aggregate {
        input,
        grouping: Grouping::GroupBy(keys),
        aggregates,
    } = &top.node
    else {
        panic!("not a grouped aggregate: {top}");
    };
    let s = long(state, "sum_state");
    let partial = Arc::new(LogicalPlan::new(PlanNode::PartialAggregate {
        input: Arc::clone(input),
        keys: keys.clone(),
        aggregates: vec![Arc::clone(strip_alias(&aggregates[1]))],
        states: vec![s.clone()],
    }));
    let merged = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
        expr: column(&s),
        output: long(out, "total"),
    })));
    let plan = Arc::new(LogicalPlan::new(PlanNode::Aggregate {
        input: partial,
        grouping: Grouping::GroupBy(keys.clone()),
        aggregates: vec![Arc::clone(&aggregates[0]), merged],
    }));
    (plan, g)
}

/// The walk passes a `PartialAggregate` grouped by the key, as a grouped
/// aggregate (F-A L1b).
#[test]
fn a_partial_aggregate_on_the_key_is_followed() {
    let (top, g) = eager(1_000, 2_000);
    let mut lift = Lift::new(Mode::Mark);
    assert!(lift.core(&top, g).is_some());
    assert!(
        lift.single()
            .is_some_and(|(l, _)| format!("{l:?}").contains('1'))
    );
}

/// Two instances that differ only in eager aggregation's fresh state ids
/// compare equal once the states are renamed canonically, and their items
/// match by what they compute.
#[test]
fn instances_differing_in_state_ids_are_one_computation() {
    let (a, _) = eager(1_000, 2_000);
    let (b, _) = eager(1_001, 2_000);
    assert_ne!(a, b);
    let (ca, ma) = canonical_states(&a);
    let (cb, mb) = canonical_states(&b);
    assert_eq!(ca, cb);
    assert_eq!(ma.len(), 1);
    assert_eq!(ma.get(&AttrId(1_000)), mb.get(&AttrId(1_001)));

    let item = |p: &Arc<LogicalPlan>| match &p.node {
        PlanNode::Aggregate { aggregates, .. } => Arc::clone(&aggregates[1]),
        _ => panic!("not an aggregate: {p}"),
    };
    let (ia, ib) = (item(&a), item(&b));
    assert!(reads_any(&ia, &ma));
    assert!(!reads_any(&ia, &mb));
    assert!(!reads_any(&ia, &std::collections::HashMap::new()));
    assert_ne!(
        item_key(&ia, &ma),
        item_key(&ib, &std::collections::HashMap::new())
    );
    assert_eq!(item_key(&ia, &ma), item_key(&ib, &mb));
    // A column the map does not hold is kept.
    let g = column(&long(7, "g"));
    assert!(Arc::ptr_eq(&renamed(&g, &ma), &g));
}

/// A reader of a definition with other state ids reads the definition's item
/// that computes the same, under the instance's own id.
#[test]
fn a_reader_matches_items_across_state_ids() {
    let (def, _) = eager(1_000, 2_000);
    let (own, _) = eager(1_001, 2_001);
    let def_out = def.resolved_output().unwrap();
    let out = own.resolved_output().unwrap();
    let exprs = reader_exprs(&own, &out, &def, &def_out).unwrap();
    assert_eq!(exprs.len(), 2);
    // The key keeps its id: a bare column.
    assert!(matches!(
        &exprs[0].node,
        ExprNode::Resolved(ResolvedExpr::Column(c)) if c.id == out[0].id
    ));
    // The total: the definition's column under the instance's id.
    let ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) = &exprs[1].node else {
        panic!("not an alias: {:?}", exprs[1]);
    };
    assert_eq!(output.id, AttrId(2_001));
    assert!(matches!(
        &expr.node,
        ExprNode::Resolved(ResolvedExpr::Column(c)) if c.id == AttrId(2_000)
    ));
    // Same ids: read bare.
    let same = reader_exprs(&def, &def_out, &def, &def_out).unwrap();
    assert!(matches!(
        &same[1].node,
        ExprNode::Resolved(ResolvedExpr::Column(c)) if c.id == AttrId(2_000)
    ));
}
