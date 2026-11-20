// SPDX-License-Identifier: Apache-2.0
//! `narrow_scan_instances` (study 38, lever A): each instance of a table read twice keeps only the
//! columns its own parent reads, also below a join, and the pass changes nothing else.

use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_optimizer::{narrow_scan_instances, optimize_with_udfs};
use krishiv_plan::Plan;
use krishiv_udf_sdk::UdfLibrary;

use crate::join_order::fixture;
use crate::support;

fn q11() -> Plan {
    let plan = support::resolve(&support::tpch_sql(11), &support::ctx());
    let mut config = Config::new();
    config
        .set("krishiv.optimizer.shareRepeatedSubtrees", "false")
        .unwrap();
    optimize_with_udfs(&plan, &fixture(), &UdfLibrary::new(), &config).unwrap()
}

/// The column lists of every `partsupp` scan in `text`.
fn partsupp_scans(text: &str) -> Vec<String> {
    text.lines()
        .filter(|l| l.contains("Scan spark_catalog.default.partsupp"))
        .map(|l| l.trim().to_owned())
        .collect()
}

/// q11 reads `partsupp ⋈ supplier ⋈ nation` in the main block and in its scalar subquery; the
/// subquery sums a product and never reads `ps_partkey`, so its instance drops that column.
#[test]
fn a_join_read_twice_narrows_each_instance_to_its_parent() {
    let plan = q11();
    let narrowed = narrow_scan_instances(plan.root(), None);
    let scans = partsupp_scans(&narrowed.to_string());
    assert_eq!(scans.len(), 2, "{narrowed}");
    assert_ne!(scans[0], scans[1], "the instances differ: {scans:?}");
    assert_eq!(
        scans.iter().filter(|s| s.contains("ps_partkey")).count(),
        1,
        "only the main block's instance reads ps_partkey: {scans:?}"
    );
    // The result is still a valid plan.
    Plan::try_new(Arc::clone(&narrowed)).unwrap();
}

/// The pass reads a graph with shared subtrees once per node, and a second run changes nothing.
#[test]
fn narrowing_is_a_fixed_point() {
    let plan = q11();
    let once = narrow_scan_instances(plan.root(), None);
    let twice = narrow_scan_instances(&once, None);
    assert_eq!(once.to_string(), twice.to_string());
}
