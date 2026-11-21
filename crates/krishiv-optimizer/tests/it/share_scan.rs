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
