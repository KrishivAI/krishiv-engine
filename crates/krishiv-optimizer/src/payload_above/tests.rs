// SPDX-License-Identifier: Apache-2.0
//! The gates no SQL query reaches once the batches before the rule ran
//! (pushdown moves a filter on the dimension into it, projections collapse):
//! hand-built plans given to the rule directly. Each refusal case is the
//! firing base plan with one defect, so removing the gate makes it fire.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, FunctionKind, Grouping,
    JoinCondition, JoinKind, LogicalPlan, PlanNode, ReadSource, ResolvedExpr, ResolvedFunction,
    Scan,
};
use krishiv_types::DataType;

use super::rewrite;
use crate::expr::{call, column};
use crate::stats::{ColumnStats, Statistics, TableStats};

fn col(id: u64, name: &str, t: DataType) -> BoundColumn {
    BoundColumn {
        id: AttrId(id),
        name: name.to_owned(),
        data_type: t,
        nullable: true,
    }
}

fn int(id: u64, name: &str) -> BoundColumn {
    col(id, name, DataType::Integer)
}

fn string(id: u64, name: &str) -> BoundColumn {
    col(id, name, DataType::String)
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

fn eq(a: &BoundColumn, b: &BoundColumn) -> Arc<Expr> {
    call(Builtin::Eq, vec![column(a), column(b)], DataType::Boolean)
}

fn join(
    l: &Arc<LogicalPlan>,
    r: &Arc<LogicalPlan>,
    kind: JoinKind,
    on: Arc<Expr>,
) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(PlanNode::Join {
        left: Arc::clone(l),
        right: Arc::clone(r),
        kind,
        condition: JoinCondition::On(on),
    }))
}

fn project(input: &Arc<LogicalPlan>, exprs: Vec<Arc<Expr>>) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(PlanNode::Project {
        input: Arc::clone(input),
        exprs,
    }))
}

fn alias(e: Arc<Expr>, out: BoundColumn) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
        expr: e,
        output: out,
    })))
}

/// `count(fk) AS n#20`, the registry's `count` unless `id` says otherwise.
fn count_of(arg: &BoundColumn, id: FunctionId) -> Arc<Expr> {
    let f = ResolvedFunction {
        id,
        kind: FunctionKind::Aggregate,
        args: vec![column(arg)],
        distinct: false,
        filter: None,
        within_group: Vec::new(),
        data_type: DataType::Long,
        nullable: false,
        deterministic: true,
    };
    alias(
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f)))),
        col(20, "n", DataType::Long),
    )
}

fn registry_count() -> FunctionId {
    FunctionId::Registry {
        name: "count".to_owned(),
        id: 0,
    }
}

/// `Aggregate[dk, dname](input)` with `count(fk)`.
fn aggregate(input: &Arc<LogicalPlan>, id: FunctionId) -> Arc<LogicalPlan> {
    let (dk, dname) = (int(3, "dk"), string(4, "dname"));
    Arc::new(LogicalPlan::new(PlanNode::Aggregate {
        input: Arc::clone(input),
        grouping: Grouping::GroupBy(vec![column(&dk), column(&dname)]),
        aggregates: vec![column(&dk), column(&dname), count_of(&int(0, "fk"), id)],
    }))
}

/// Row counts for `f` (1M) and `d` (1,000), `e` (10); `d` unique on `dk`,
/// `e` on `ek`; `g` has no statistics.
#[derive(Debug)]
struct Stats;

impl Statistics for Stats {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        let rows = match name.last()?.as_str() {
            "f" => 1_000_000,
            "d" => 1_000,
            "e" => 10,
            _ => return None,
        };
        let columns: HashMap<String, ColumnStats> = scan
            .output
            .iter()
            .map(|c| {
                let distinct = match c.name.as_str() {
                    "fk" | "dk" | "dname" => 1_000,
                    "dek" | "ek" | "ename" => 10,
                    _ => rows,
                };
                let stats = ColumnStats {
                    distinct: Some(distinct),
                    nulls: Some(0),
                    ..ColumnStats::default()
                };
                (c.name.clone(), stats)
            })
            .collect();
        Some(TableStats { rows, columns })
    }

    fn unique_keys(&self, scan: &Scan) -> Vec<Vec<String>> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return Vec::new();
        };
        match name.last().map(String::as_str) {
            Some("d") => vec![vec!["dk".to_owned()]],
            Some("e") => vec![vec!["ek".to_owned()]],
            _ => Vec::new(),
        }
    }
}

fn f() -> Arc<LogicalPlan> {
    scan("f", vec![int(0, "fk"), string(1, "fs")])
}

fn d() -> Arc<LogicalPlan> {
    scan("d", vec![int(3, "dk"), string(4, "dname"), int(5, "dek")])
}

fn e() -> Arc<LogicalPlan> {
    scan("e", vec![int(6, "ek"), string(7, "ename")])
}

fn base_join() -> Arc<LogicalPlan> {
    join(
        &f(),
        &d(),
        JoinKind::Inner,
        eq(&int(0, "fk"), &int(3, "dk")),
    )
}

fn notes_of(root: Arc<LogicalPlan>) -> Vec<String> {
    rewrite(root, &Stats, true).unwrap().1
}

fn moved(notes: &[String]) -> bool {
    notes.iter().any(|n| n.starts_with("moved"))
}

fn refused(root: Arc<LogicalPlan>, why: &str) {
    let notes = notes_of(root);
    assert!(!moved(&notes), "{notes:?}");
    assert!(
        notes.iter().any(|n| n.contains(why)),
        "no {why:?} in {notes:?}"
    );
}

#[test]
fn the_base_plan_fires() {
    let notes = notes_of(aggregate(&base_join(), registry_count()));
    assert!(moved(&notes), "{notes:?}");
}

#[test]
fn a_filter_on_the_path_reading_the_dimension_refuses() {
    let pred = call(
        Builtin::Eq,
        vec![column(&string(4, "dname")), column(&string(1, "fs"))],
        DataType::Boolean,
    );
    let filtered = Arc::new(LogicalPlan::new(PlanNode::Filter {
        input: base_join(),
        predicate: pred,
    }));
    refused(
        aggregate(&filtered, registry_count()),
        "a filter on the path reads the dimension",
    );
}

#[test]
fn a_projection_on_the_path_computing_from_or_renaming_the_dimension_refuses() {
    let (fk, dk, dname) = (int(0, "fk"), int(3, "dk"), string(4, "dname"));
    let twice = call(
        Builtin::Add,
        vec![column(&dk), column(&dk)],
        DataType::Integer,
    );
    for item in [
        alias(twice, int(30, "dk2")),
        alias(column(&dname), string(31, "dname2")),
    ] {
        let p = project(
            &base_join(),
            vec![column(&fk), column(&dk), column(&dname), item],
        );
        refused(
            aggregate(&p, registry_count()),
            "a projection on the path computes from the dimension",
        );
    }
}

#[test]
fn a_session_aggregate_named_count_refuses() {
    let udaf = FunctionId::Udf {
        name: "count".to_owned(),
        version: 1,
    };
    refused(aggregate(&base_join(), udaf), "not a built-in aggregate");
}

/// `(f ⋈ d) ⋈ e` on `dek = ek`, grouped by `dk, dname, ename`.
fn absorbing(e: &Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    let joined = join(
        &base_join(),
        e,
        JoinKind::Inner,
        eq(&int(5, "dek"), &int(6, "ek")),
    );
    let (dk, dname, ename) = (int(3, "dk"), string(4, "dname"), string(7, "ename"));
    Arc::new(LogicalPlan::new(PlanNode::Aggregate {
        input: joined,
        grouping: Grouping::GroupBy(vec![column(&dk), column(&dname), column(&ename)]),
        aggregates: vec![
            column(&dk),
            column(&dname),
            column(&ename),
            count_of(&int(0, "fk"), registry_count()),
        ],
    }))
}

#[test]
fn an_absorbed_input_must_be_unique_and_not_shared() {
    let notes = notes_of(absorbing(&e()));
    assert!(
        notes.iter().any(|n| n.contains("absorbed [e]")),
        "{notes:?}"
    );
    // `e` without its key: `f` declares none, so its key is counted first
    // (a request, design P2b A5 (d)).
    let not_unique = scan("f", vec![int(6, "ek"), string(7, "ename")]);
    refused(absorbing(&not_unique), "awaiting a key count of f [ek]");
    // `e` read by another parent too.
    let shared = e();
    let root = Arc::new(LogicalPlan::new(PlanNode::Union {
        inputs: vec![
            absorbing(&shared),
            project(&shared, vec![column(&int(6, "ek"))]),
        ],
        by_name: false,
        allow_missing_columns: false,
    }));
    refused(root, "an absorbed input is read by another parent");
}

#[test]
fn a_non_inner_join_on_the_path_reading_the_dimension_refuses() {
    let outer = join(
        &base_join(),
        &e(),
        JoinKind::LeftOuter,
        eq(&int(5, "dek"), &int(6, "ek")),
    );
    refused(
        aggregate(&outer, registry_count()),
        "join on the path reads the dimension",
    );
}

/// A kept join with a table without statistics: the overflow proof's rows
/// would be a default count (review m1).
#[test]
fn default_rows_on_the_path_refuse() {
    let g = scan("g", vec![string(8, "gs")]);
    let kept = join(
        &base_join(),
        &g,
        JoinKind::Inner,
        eq(&string(1, "fs"), &string(8, "gs")),
    );
    refused(
        aggregate(&kept, registry_count()),
        "rest on default row counts",
    );
}

/// The proof through a `Shift`: the ids below are the ids above minus the offset.
#[test]
fn uniqueness_is_proven_through_a_shift() {
    let below = scan(
        "d",
        vec![int(103, "dk"), string(104, "dname"), int(105, "dek")],
    );
    let shifted = Arc::new(LogicalPlan::new(PlanNode::Shift {
        input: below,
        offset: 100,
    }));
    // The shifted instance's ids are dk#203 etc.: rebuild the join on them.
    let (fk, dk) = (int(0, "fk"), int(203, "dk"));
    let j = join(&f(), &shifted, JoinKind::Inner, eq(&fk, &dk));
    let dname = string(204, "dname");
    let agg = Arc::new(LogicalPlan::new(PlanNode::Aggregate {
        input: j,
        grouping: Grouping::GroupBy(vec![column(&dk), column(&dname)]),
        aggregates: vec![column(&dk), column(&dname), count_of(&fk, registry_count())],
    }));
    let notes = notes_of(agg);
    assert!(moved(&notes), "{notes:?}");
}
