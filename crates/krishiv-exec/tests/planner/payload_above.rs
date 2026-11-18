// SPDX-License-Identifier: Apache-2.0
//! The payload-above-aggregate rule (design P2b A5, optimizer `payload_above`):
//! where the dimension's key is declared unique and the rule moves the
//! dimension above the aggregate, the executed plan returns the rows of the
//! same plan with the rule off.
//!
//! The data set is the committed TPC-H SF0.01 Parquet fixture, whose
//! `c_custkey`, `n_nationkey` and `o_orderkey` are unique.

use std::path::Path;
use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_exec::planner::ProviderStatistics;
use krishiv_optimizer::{RuleStep, Statistics, TableStats};
use krishiv_plan::analyzer::testing::block_on;
use krishiv_plan::analyzer::{Analyzer as _, DefaultAnalyzer};
use krishiv_plan::{Plan, ReadSource, Scan, SourceId};

use super::partial_aggregate::{physical, runtime, sorted};
use super::run;
use super::tpch::{Tpch, parquet_tables, sql};

/// The footers' statistics, with TPC-H's primary keys declared unique.
#[derive(Debug)]
struct Keys(ProviderStatistics);

impl Statistics for Keys {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        self.0.table(scan)
    }

    fn unique_keys(&self, scan: &Scan) -> Vec<Vec<String>> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return Vec::new();
        };
        let key = match name.last().map(String::as_str) {
            Some("customer") => "c_custkey",
            Some("nation") => "n_nationkey",
            Some("orders") => "o_orderkey",
            Some("supplier") => "s_suppkey",
            _ => return Vec::new(),
        };
        vec![vec![key.to_owned()]]
    }
}

/// `text` optimized with the keys declared and the rule `on` or off.
fn optimized(t: &Tpch, text: &str, on: bool) -> (Plan, Vec<RuleStep>) {
    let plan =
        krishiv_sql::parse_and_lower(text, &krishiv_sql::ParserOptions::default(), SourceId(0))
            .unwrap();
    let plan: Plan = block_on(DefaultAnalyzer.analyze(&plan, &t.ctx))
        .unwrap()
        .plan;
    let stats = Keys(ProviderStatistics(Arc::clone(&t.tables) as _));
    let mut config = Config::new();
    config
        .set(
            "krishiv.optimizer.payloadAboveAggregate",
            if on { "true" } else { "false" },
        )
        .unwrap();
    krishiv_optimizer::optimize_traced_with_udfs(
        &plan,
        &stats,
        &krishiv_udf_sdk::UdfLibrary::new(),
        &config,
    )
    .unwrap()
}

/// TPC-H q10, other grouping keys of the fact side, and a dimension grouped
/// by a key it holds once per nation: with the rule on and off.
#[test]
fn moved_payloads_return_the_rows_of_the_rule_off() {
    let rt = runtime();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    let t = parquet_tables(&dir, &rt);
    let queries = [
        sql(10),
        "SELECT c_custkey, c_name, o_orderpriority, sum(l_quantity), count(*) \
         FROM customer, orders, lineitem \
         WHERE c_custkey = o_custkey AND o_orderkey = l_orderkey \
         GROUP BY c_custkey, c_name, o_orderpriority"
            .to_owned(),
        "SELECT n_nationkey, n_name, count(*), sum(c_acctbal), min(c_phone) \
         FROM customer, nation WHERE c_nationkey = n_nationkey GROUP BY n_nationkey, n_name"
            .to_owned(),
    ];
    for q in &queries {
        let (with, steps) = optimized(&t, q, true);
        let (without, _) = optimized(&t, q, false);
        let notes: Vec<&String> = steps
            .iter()
            .filter(|s| s.rule == "PayloadAboveAggregate")
            .map(|s| &s.note)
            .collect();
        // The rule fired: it is the only difference between the two plans.
        assert!(
            notes.iter().any(|n| n.starts_with("moved")) && !with.structurally_eq(&without),
            "{q}\n{notes:?}\n{}",
            with.root()
        );
        let got = sorted(&run(&physical(&t, &with).unwrap()).unwrap());
        let want = sorted(&run(&physical(&t, &without).unwrap()).unwrap());
        assert!(!want.is_empty(), "{q}");
        assert_eq!(got, want, "{q}\n{}", with.root());
    }
}
