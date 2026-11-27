// SPDX-License-Identifier: Apache-2.0
//! `ShareFilteredScan`: a table scan read only by filters, two or more, is
//! read once under the disjunction of their predicates
//! (`krishiv.optimizer.shareFilteredScans`, TPC-DS q28).

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_optimizer::{ColumnStats, NoStatistics, Statistics, TableStats, optimize_with_udfs};
use krishiv_plan::{Literal, LogicalPlan, Plan, PlanNode, ReadSource, Scan};
use krishiv_udf_sdk::UdfLibrary;

use proptest::prelude::*;

use crate::props::Gen;
use crate::support::{data, eval};
use crate::{support, tpcds};

/// `store_sales` at TPC-DS SF10 with `ss_quantity` in 1..=100, as the
/// Parquet footers report it.
#[derive(Debug)]
struct StoreSales;

impl Statistics for StoreSales {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        (name.last()? == "store_sales").then(|| TableStats {
            rows: 28_800_991,
            columns: HashMap::from([(
                "ss_quantity".to_owned(),
                ColumnStats {
                    distinct: Some(100),
                    nulls: Some(1_296_044),
                    min: Some(Literal::Integer(1)),
                    max: Some(Literal::Integer(100)),
                    ..ColumnStats::default()
                },
            )]),
        })
    }
}

fn config(on: bool) -> Config {
    let mut c = Config::new();
    c.set(
        "krishiv.optimizer.shareFilteredScans",
        if on { "true" } else { "false" },
    )
    .unwrap();
    c
}

fn optimize(sql: &str, stats: &dyn Statistics, on: bool) -> Plan {
    let plan = support::resolve(sql, &support::ctx_over(tpcds::catalog()));
    let out = optimize_with_udfs(&plan, stats, &UdfLibrary::new(), &config(on)).unwrap();
    assert_eq!(
        out.root().resolved_output(),
        plan.root().resolved_output(),
        "{sql}"
    );
    out
}

/// How many times `store_sales` is scanned when the plan runs: paths from the
/// root, each `WITH` definition counted once (a `CteRef` computes nothing).
fn scans(plan: &Arc<LogicalPlan>) -> usize {
    let mut n = usize::from(matches!(&plan.node, PlanNode::Scan(_)
        if plan.to_string().starts_with("Scan spark_catalog.default.store_sales ")));
    plan.for_each_child(&mut |c| n += scans(c));
    n
}

fn shared(plan: &Plan) -> usize {
    fn refs(p: &Arc<LogicalPlan>) -> usize {
        let mut n = usize::from(
            matches!(&p.node, PlanNode::CteRef(r) if r.name.starts_with("__shared_scan_")),
        );
        p.for_each_child(&mut |c| n += refs(c));
        n
    }
    refs(plan.root())
}

/// q28's six filtered aggregates read one `store_sales` scan: the scan runs
/// once, under the disjunction of the six predicates, and each filter reads the
/// stored rows. Off, the plan scans the table six times. Both are fixed points.
#[test]
fn q28_scans_store_sales_once() {
    let sql = tpcds::sql(28);
    let off = optimize(&sql, &StoreSales, false);
    assert_eq!(scans(off.root()), 6, "{}", off.root());
    assert_eq!(shared(&off), 0, "{}", off.root());

    let on = optimize(&sql, &StoreSales, true);
    let text = on.root().to_string();
    assert_eq!(scans(on.root()), 1, "{text}");
    assert_eq!(shared(&on), 6, "{text}");
    Plan::try_new(Arc::clone(on.root())).unwrap();
    let again = optimize_with_udfs(&on, &StoreSales, &UdfLibrary::new(), &config(true)).unwrap();
    assert!(again.structurally_eq(&on), "{}", again.root());
    insta::assert_snapshot!("share_scan_q28", text);
}

/// Without statistics the rule cannot tell that the filters keep few rows.
#[test]
fn no_statistics_no_sharing() {
    let on = optimize(&tpcds::sql(28), &NoStatistics, true);
    assert_eq!(shared(&on), 0, "{}", on.root());
}

const TWO_FILTERS: &str = "SELECT * FROM \
     (SELECT count(*) AS a FROM store_sales WHERE ss_quantity BETWEEN 1 AND 5) x, \
     (SELECT count(*) AS b FROM store_sales WHERE ss_quantity BETWEEN 96 AND 100) y";

/// Two filters are enough.
#[test]
fn two_filters_share_the_scan() {
    let on = optimize(TWO_FILTERS, &StoreSales, true);
    assert_eq!(scans(on.root()), 1, "{}", on.root());
    assert_eq!(shared(&on), 2, "{}", on.root());
}

/// The filters together keep more than a quarter of the table: rescanning is
/// cheaper than storing the rows.
#[test]
fn filters_that_keep_most_rows_do_not_share() {
    let sql = "SELECT * FROM \
         (SELECT count(*) AS a FROM store_sales WHERE ss_quantity < 40) x, \
         (SELECT count(*) AS b FROM store_sales WHERE ss_quantity > 60) y";
    let on = optimize(sql, &StoreSales, true);
    assert_eq!(shared(&on), 0, "{}", on.root());
}

/// A reader that is not a filter reads every row: nothing is shared.
#[test]
fn an_unfiltered_reader_keeps_the_scans() {
    let sql = "SELECT * FROM \
         (SELECT count(*) AS a FROM store_sales WHERE ss_quantity BETWEEN 1 AND 5) x, \
         (SELECT count(ss_quantity) AS b FROM store_sales) y";
    let on = optimize(sql, &StoreSales, true);
    assert_eq!(shared(&on), 0, "{}", on.root());
}

/// A predicate that may raise (ANSI division) stays on its own scan: the
/// disjunction would evaluate it on rows the query as written may never reach.
#[test]
fn a_raising_predicate_keeps_the_scans() {
    let sql = "SELECT * FROM \
         (SELECT count(*) AS a FROM store_sales WHERE ss_quantity BETWEEN 1 AND 5) x, \
         (SELECT count(*) AS b FROM store_sales \
          WHERE ss_quantity BETWEEN 96 AND 100 AND ss_ticket_number / ss_quantity > 3) y";
    let on = optimize(sql, &StoreSales, true);
    assert_eq!(shared(&on), 0, "{}", on.root());
}

/// Readers with equal predicates are one chain read twice: rescanning serves
/// as well, so nothing is stored (`ShareCte`'s rule for chains without a
/// pipeline breaker).
#[test]
fn equal_predicates_keep_the_scans() {
    let sql = "SELECT * FROM \
         (SELECT count(*) AS a FROM store_sales WHERE ss_quantity BETWEEN 1 AND 5) x, \
         (SELECT sum(ss_quantity) AS b FROM store_sales WHERE ss_quantity BETWEEN 1 AND 5) y";
    let on = optimize(sql, &StoreSales, true);
    assert_eq!(shared(&on), 0, "{}", on.root());
}

/// Readers inside scalar subqueries are redirected too (`Sharer` rewrites
/// subquery plans), and the plan is a fixed point.
#[test]
fn readers_in_subqueries_share_the_scan() {
    let sql = "SELECT \
         (SELECT count(*) FROM store_sales WHERE ss_quantity BETWEEN 1 AND 5) AS a, \
         (SELECT count(*) FROM store_sales WHERE ss_quantity BETWEEN 96 AND 100) AS b";
    let off = optimize(sql, &StoreSales, false);
    let on = optimize(sql, &StoreSales, true);
    let refs = |p: &Plan| p.root().to_string().matches("__shared_scan_0").count();
    if refs(&off) == 0 && on.root().to_string().contains("WithCte [__shared_scan_0]") {
        Plan::try_new(Arc::clone(on.root())).unwrap();
        let again =
            optimize_with_udfs(&on, &StoreSales, &UdfLibrary::new(), &config(true)).unwrap();
        assert!(again.structurally_eq(&on), "{}", again.root());
    } else {
        panic!("not shared:\n{}", on.root());
    }
}

/// Three filters over one `store_sales` scan, the ones whose predicate text holds a string of
/// `guard` reading the scan through a security barrier (policy `isnotnull(ss_quantity)`, which
/// is leakproof, so each filter's own predicate moves in beside it).
fn guarded(guard: &[&str]) -> Plan {
    use krishiv_common::{Principal, PrincipalKey, PrincipalKind};
    use krishiv_plan::analyzer::{Securable, TableIdent};
    use krishiv_plan::{
        Builtin, Expr, ExprNode, FunctionId, FunctionKind, PolicyRef, ResolvedExpr,
        ResolvedFunction,
    };
    let sql = "SELECT * FROM \
         (SELECT count(*) AS a FROM store_sales WHERE ss_quantity BETWEEN 1 AND 5) x, \
         (SELECT count(*) AS b FROM store_sales WHERE ss_quantity BETWEEN 96 AND 100) y, \
         (SELECT count(*) AS c FROM store_sales WHERE ss_quantity BETWEEN 50 AND 52) z";
    let plan = support::resolve(sql, &support::ctx_over(tpcds::catalog()));
    let policy = PolicyRef::new(
        Securable::Table(TableIdent {
            catalog: "spark_catalog".to_owned(),
            namespace: vec!["default".to_owned()],
            name: "store_sales".to_owned(),
        }),
        PrincipalKey::of(&Principal::new("ana", PrincipalKind::User)),
        1,
    );
    let root = Arc::clone(plan.root())
        .transform_up(&mut |p| {
            let PlanNode::Filter { input, predicate } = &p.node else {
                return Ok::<_, std::convert::Infallible>(p);
            };
            let PlanNode::Scan(scan) = &input.node else {
                return Ok(p);
            };
            let text = predicate.to_string();
            if !guard.iter().any(|g| text.contains(g)) {
                return Ok(p);
            }
            let Some(q) = scan.output.iter().find(|c| c.name == "ss_quantity") else {
                return Ok(p);
            };
            let policy_expr = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                ResolvedFunction {
                    id: FunctionId::Builtin(Builtin::IsNotNull),
                    kind: FunctionKind::Scalar,
                    args: vec![Arc::new(Expr::new(ExprNode::Resolved(
                        ResolvedExpr::Column(q.clone()),
                    )))],
                    distinct: false,
                    filter: None,
                    within_group: Vec::new(),
                    data_type: krishiv_types::DataType::Boolean,
                    nullable: false,
                    deterministic: true,
                },
            ))));
            // Over the same scan node the other filters read.
            let barrier = LogicalPlan::new(PlanNode::Filter {
                input: Arc::clone(input),
                predicate: policy_expr,
            })
            .security_barrier(policy.clone());
            Ok(Arc::new(barrier.filter(Arc::clone(predicate))))
        })
        .unwrap_or_else(|e| match e {});
    let plan = Plan::try_new(root).unwrap();
    let out = optimize_with_udfs(&plan, &StoreSales, &UdfLibrary::new(), &config(true)).unwrap();
    let again = optimize_with_udfs(&out, &StoreSales, &UdfLibrary::new(), &config(true)).unwrap();
    assert!(
        again.structurally_eq(&out),
        "{}\n{}",
        out.root(),
        again.root()
    );
    out
}

/// Whether anything under `p` (`p` included) is a `CteRef` or a shared-scan definition.
fn reads_a_share(p: &Arc<LogicalPlan>) -> bool {
    let mut found = match &p.node {
        PlanNode::CteRef(_) => true,
        PlanNode::WithCte { ctes, .. } => ctes.iter().any(|c| c.name.starts_with("__shared_scan_")),
        _ => false,
    };
    p.for_each_child(&mut |c| found |= reads_a_share(c));
    found
}

/// The scans under `p`'s barriers, each inside one.
fn guarded_scans(p: &Arc<LogicalPlan>, inside: bool, out: &mut usize) {
    if inside && matches!(p.node, PlanNode::Scan(_)) {
        *out += 1;
    }
    let inside = inside || matches!(p.node, PlanNode::SecurityBarrier { .. });
    p.for_each_child(&mut |c| guarded_scans(c, inside, out));
}

/// G2 (design P3-E14 §5): a scan read inside a security barrier is never shared, neither with a
/// reader outside it nor with another barrier: the guarded scan stays inside, under its policy.
#[test]
fn a_scan_read_inside_a_barrier_is_not_shared() {
    for guard in [&[">= 1)"][..], &[">= 1)", ">= 96)"][..]] {
        let out = guarded(guard);
        let text = out.root().to_string();
        let mut inside = Vec::new();
        fn barriers(p: &Arc<LogicalPlan>, out: &mut Vec<Arc<LogicalPlan>>) {
            if let PlanNode::SecurityBarrier { input, .. } = &p.node {
                out.push(Arc::clone(input));
            }
            p.for_each_child(&mut |c| barriers(c, out));
        }
        barriers(out.root(), &mut inside);
        assert_eq!(inside.len(), guard.len(), "{guard:?}\n{text}");
        for i in &inside {
            assert!(
                !reads_a_share(i),
                "{guard:?}: a share inside a barrier\n{text}"
            );
        }
        let mut n = 0;
        guarded_scans(out.root(), false, &mut n);
        assert_eq!(n, guard.len(), "{guard:?}\n{text}");
        // The guarded scan is in no shared definition.
        assert_eq!(shared(&out), 0, "{guard:?}\n{text}");
    }
    // Control: the same three filters with no barrier share one scan.
    assert_eq!(shared(&guarded(&[])), 3);
}

/// Every table ten million rows with every column from statistics (`u` of the
/// generated-query tests at scale).
#[derive(Debug)]
struct Large;

impl Statistics for Large {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { .. } = &scan.source else {
            return None;
        };
        Some(TableStats {
            rows: 10_000_000,
            columns: scan
                .output
                .iter()
                .map(|c| {
                    (
                        c.name.clone(),
                        ColumnStats {
                            distinct: Some(1_000_000),
                            nulls: Some(100),
                            min: Some(Literal::Integer(-1_000_000)),
                            max: Some(Literal::Integer(1_000_000)),
                            ..ColumnStats::default()
                        },
                    )
                })
                .collect(),
        })
    }
}

/// Seed 49 of perf-4's generated-query test (D1a): not a fixed point.
#[test]
fn a_generated_four_way_self_join_is_a_fixed_point() {
    let sql = "SELECT x1.d AS c0, x1.a + 1 AS c1 FROM u AS x1 \
         JOIN u AS x2 ON x1.a = x2.a AND x1.a IS NULL \
         JOIN u AS x3 ON x2.a = x3.a AND coalesce(x3.a, -2) = -2 \
         JOIN u AS x4 ON x1.a = x4.a AND NOT (x1.a IS NOT NULL) WHERE x1.a IN (2, 2)";
    let plan = support::resolve(sql, &support::ctx());
    let once = optimize_with_udfs(&plan, &Large, &UdfLibrary::new(), &config(true)).unwrap();
    let again = optimize_with_udfs(&once, &Large, &UdfLibrary::new(), &config(true)).unwrap();
    assert!(
        again.structurally_eq(&once),
        "not a fixed point:\n{}\n{}",
        once.root(),
        again.root()
    );
    assert!(shared(&once) > 0, "{}", once.root());
    let db = data::small(49);
    assert_eq!(eval::rows(&db, once.root()), eval::rows(&db, plan.root()));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Generated queries over large tables, the rule on: a fixed point, the same rows.
    #[test]
    fn generated_queries_with_shared_scans_are_fixed_points(seed in any::<u64>()) {
        let sql = Gen::new(seed).top();
        let plan = support::resolve(&sql, &support::ctx());
        let once = optimize_with_udfs(&plan, &Large, &UdfLibrary::new(), &config(true)).unwrap();
        let again = optimize_with_udfs(&once, &Large, &UdfLibrary::new(), &config(true)).unwrap();
        prop_assert!(
            again.structurally_eq(&once),
            "{sql}: not a fixed point:\n{}\n{}",
            once.root(),
            again.root()
        );
        let db = data::small(seed);
        prop_assert_eq!(eval::rows(&db, once.root()), eval::rows(&db, plan.root()), "{}", sql);
    }
}

/// The generator gives the proptest above queries the rule shares: without
/// them it would test nothing. About 3% do (2 of seeds 0..64), so its 256
/// cases meet several.
#[test]
fn generated_queries_share_scans() {
    let fired = (0..64_u64)
        .filter(|&seed| {
            let plan = support::resolve(&Gen::new(seed).top(), &support::ctx());
            let once =
                optimize_with_udfs(&plan, &Large, &UdfLibrary::new(), &config(true)).unwrap();
            shared(&once) > 0
        })
        .count();
    assert!(
        fired >= 2,
        "only {fired} of 64 generated queries share a scan"
    );
}
