// SPDX-License-Identifier: Apache-2.0
//! `ShareFilteredScan` (TPC-DS q28's shape): a table read only by filters is
//! read once under the disjunction of their predicates, with the same rows as
//! one scan per filter.

use std::collections::HashMap;

use krishiv_common::config::Config;
use krishiv_optimizer::{ColumnStats, Statistics, TableStats};
use krishiv_plan::{Literal, ReadSource, Scan};

use super::*;

/// `t` as a 2M-row table with `b` in 0..=1990 (statistics choose the plan,
/// never its rows: the table holds 200).
#[derive(Debug)]
struct T;

impl Statistics for T {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        (name.last()? == "t").then(|| TableStats {
            rows: 2_000_000,
            columns: HashMap::from([(
                "b".to_owned(),
                ColumnStats {
                    distinct: Some(1991),
                    nulls: Some(0),
                    min: Some(Literal::Long(0)),
                    max: Some(Literal::Long(1990)),
                    ..ColumnStats::default()
                },
            )]),
        })
    }
}

/// Three global aggregates over disjoint ranges of `b`, one with a test on
/// the nullable `a`, cross-joined as q28 does.
const Q: &str = "SELECT * FROM \
     (SELECT count(*) AS n1, min(b) AS b1, count(DISTINCT a) AS d1, sum(a) AS s1 \
      FROM t WHERE b BETWEEN 0 AND 100) x, \
     (SELECT count(*) AS n2, max(b) AS b2, count(DISTINCT a) AS d2, sum(a) AS s2 \
      FROM t WHERE b BETWEEN 500 AND 600 AND (a IS NULL OR a < 150)) y, \
     (SELECT count(*) AS n3, sum(b) AS b3, count(DISTINCT s) AS d3, max(s) AS m3 \
      FROM t WHERE b > 1900) z";

fn rows_under(env: &Env, on: bool) -> (Vec<String>, usize) {
    let mut config = Config::new();
    config
        .set(
            "krishiv.optimizer.shareFilteredScans",
            if on { "true" } else { "false" },
        )
        .unwrap();
    let analyzed = env.analyze(Q);
    let plan = krishiv_optimizer::optimize_with_udfs(
        &analyzed,
        &T,
        &krishiv_udf_sdk::UdfLibrary::new(),
        &config,
    )
    .unwrap();
    fn refs(p: &Arc<krishiv_plan::LogicalPlan>) -> usize {
        let mut n = usize::from(matches!(&p.node,
            krishiv_plan::PlanNode::CteRef(r) if r.name.starts_with("__shared_scan_")));
        p.for_each_child(&mut |c| n += refs(c));
        n
    }
    assert_eq!(refs(plan.root()), if on { 3 } else { 0 }, "{}", plan.root());
    let cx = PlanningContext::new(Arc::clone(&env.tables) as _, Arc::new(config));
    let runnable = env.runnable(DefaultPlanner.plan(&plan, &cx).unwrap());
    let shared = runnable.plan.shared().len();
    let batches = run(&runnable).unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b.columns().iter().map(|c| cell(c.as_ref(), r)).collect();
            rows.push(cells.join("|"));
        }
    }
    (rows, shared)
}

/// On or off, the rows are the same; on, the filtered scan is one shared body.
#[test]
fn sharing_the_filtered_scan_keeps_the_rows() {
    let env = env_of(200);
    let (off, shared_off) = rows_under(&env, false);
    let (on, shared_on) = rows_under(&env, true);
    assert_eq!(off, on);
    assert_eq!(off.len(), 1);
    assert_eq!(shared_off, 0);
    assert_eq!(shared_on, 1);
}
