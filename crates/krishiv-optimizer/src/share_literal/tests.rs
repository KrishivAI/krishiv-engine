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
    AttrId, Cte, CteRef, ExprNode, Grouping, LogicalPlan, PlanNode, ResolvedExpr, SourceId,
};
use krishiv_types::DataType;

use super::{Lift, Mode, all_shared};

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
