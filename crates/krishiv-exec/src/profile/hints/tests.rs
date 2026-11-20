// SPDX-License-Identifier: Apache-2.0
//! The hint rules at their thresholds, on synthetic metrics and plan trees.

use std::sync::Arc;
use std::time::Duration;

use super::*;
use crate::metrics::{OperatorMetrics, OperatorRole};
use crate::planner::{FilterGap, InputFacts};

fn node(
    id: u32,
    name: &str,
    detail: &str,
    facts: Option<NodeFacts>,
    children: Vec<ProfileNode>,
) -> ProfileNode {
    ProfileNode {
        id: Some(id),
        name: name.to_owned(),
        detail: detail.to_owned(),
        children,
        facts: facts.map(Arc::new),
    }
}

fn op(m: &mut QueryMetrics, id: u32, name: &'static str, node: u32, metrics: OperatorMetrics) {
    m.operators.insert(
        OperatorId(id),
        OperatorEntry {
            name: name.into(),
            node: Some(node),
            role: OperatorRole::Operator,
            metrics,
            ..OperatorEntry::default()
        },
    );
}

fn spill(bytes: u64) -> OperatorMetrics {
    OperatorMetrics {
        spill_write_bytes: bytes,
        spilled_bytes: bytes,
        drivers: 1,
        ..OperatorMetrics::default()
    }
}

fn agg_plan() -> ProfileNode {
    node(0, "HashAggregate", "[k#1]", None, vec![])
}

fn metrics_with_peak(peak: u64) -> QueryMetrics {
    let mut m = QueryMetrics::default();
    m.memory.peak_reserved = peak;
    m
}

#[test]
fn hint_spill_raised_at_the_boundary() {
    let s = HintSettings::default();
    let mb = 1 << 20;
    // Below the byte floor: nothing, whatever the share.
    let mut m = metrics_with_peak(100 * mb);
    op(&mut m, 1, "HashAggregate", 0, spill(63 * mb));
    assert!(hints(&agg_plan(), &m, &s).is_empty());
    // At the floor and above 10% of the peak.
    let mut m = metrics_with_peak(100 * mb);
    op(&mut m, 1, "HashAggregate", 0, spill(64 * mb));
    let h = hints(&agg_plan(), &m, &s);
    assert_eq!(h.len(), 1);
    assert_eq!(h[0].code, HintCode::Spill);
    assert_eq!(h[0].node, Some(0));
    assert_eq!(h[0].operator, Some(OperatorId(1)));
    assert_eq!(h[0].evidence[0], ("spill_write_bytes".to_owned(), 64 * mb));
    // A spill of exactly 10% of a large peak is quiet, one byte more is not.
    let mut m = metrics_with_peak(1000 * mb);
    op(&mut m, 1, "HashAggregate", 0, spill(100 * mb));
    assert!(hints(&agg_plan(), &m, &s).is_empty());
    let mut m = metrics_with_peak(1000 * mb);
    op(&mut m, 1, "HashAggregate", 0, spill(101 * mb));
    assert_eq!(hints(&agg_plan(), &m, &s).len(), 1);
}

#[test]
fn hint_memory_wait_raised_at_the_boundary() {
    let s = HintSettings::default();
    let wait = |blocked_ms: u64, compute_ms: u64| {
        let mut m = QueryMetrics::default();
        op(
            &mut m,
            1,
            "HashAggregate",
            0,
            OperatorMetrics {
                blocked_on_memory: Duration::from_millis(blocked_ms),
                compute_time: Duration::from_millis(compute_ms),
                drivers: 1,
                ..OperatorMetrics::default()
            },
        );
        hints(&agg_plan(), &m, &s)
    };
    assert!(wait(99, 1).is_empty(), "under the floor");
    assert!(wait(100, 100).is_empty(), "not more than compute");
    let h = wait(150, 100);
    assert_eq!(h.len(), 1);
    assert_eq!(h[0].code, HintCode::MemoryWait);
}

#[test]
fn hint_skew_raised_at_the_boundary() {
    let s = HintSettings::default();
    let skewed = |name: &'static str, max_ms: u64, total_ms: u64, drivers: u64| {
        let mut m = QueryMetrics::default();
        op(
            &mut m,
            1,
            name,
            0,
            OperatorMetrics {
                drivers,
                compute_time: Duration::from_millis(total_ms),
                max_driver_compute_time: Duration::from_millis(max_ms),
                ..OperatorMetrics::default()
            },
        );
        hints(&agg_plan(), &m, &s)
    };
    // 4 drivers, the slowest 400 of 1,000 ms: 1.6x the even share, under 4x the mean of the others.
    assert!(skewed("Window", 400, 1000, 4).is_empty());
    // At factor 4 the slowest driver must do 4/7 of the work at 4 drivers, 2/3 at 3 and 4/5 at 2.
    assert!(skewed("Window", 570, 1000, 4).is_empty());
    assert_eq!(skewed("Window", 580, 1000, 4).len(), 1);
    assert!(skewed("Window", 660, 1000, 3).is_empty());
    assert_eq!(skewed("Window", 670, 1000, 3).len(), 1);
    assert!(skewed("Window", 790, 1000, 2).is_empty());
    assert_eq!(skewed("Window", 800, 1000, 2).len(), 1);
    // The slowest 1,000 of 1,000 ms over 4 drivers.
    let h = skewed("Window", 1000, 1000, 4);
    assert_eq!(h.len(), 1);
    assert_eq!(h[0].code, HintCode::Skew);
    assert!(!h[0].message.contains("k#1"), "{}", h[0].message);
    // Under the time floor, or one driver: quiet.
    assert!(skewed("Window", 199, 199, 4).is_empty());
    assert!(skewed("Window", 1000, 1000, 1).is_empty());
    // Scans, hash joins and hash aggregates share morsels on a node: never skew (TPC-H q04, q05, q21).
    for name in [
        "FileScan",
        "HashJoinBuild",
        "HashJoinProbe",
        "HashAggregate",
    ] {
        assert!(skewed(name, 1000, 1000, 4).is_empty(), "{name}");
    }
    assert_eq!(skewed("ExchangeRead", 1000, 1000, 4).len(), 1);
}

#[test]
fn skew_by_rows_needs_the_row_floor() {
    let s = HintSettings::default();
    let by_rows = |rows: u64| {
        let mut m = QueryMetrics::default();
        op(
            &mut m,
            1,
            "Window",
            0,
            OperatorMetrics {
                drivers: 8,
                // Slow enough to count, but not the time test's skew: the rows decide.
                compute_time: Duration::from_secs(10),
                max_driver_compute_time: Duration::from_millis(300),
                rows_out: rows,
                max_driver_rows_out: rows * 9 / 10,
                ..OperatorMetrics::default()
            },
        );
        hints(&agg_plan(), &m, &s)
    };
    assert!(by_rows(99_000).is_empty());
    assert_eq!(by_rows(1_000_000).len(), 1);
}

fn join_plan(facts: NodeFacts) -> ProfileNode {
    node(
        0,
        "HashJoin",
        "Inner, [a#1] = [b#2]",
        Some(facts),
        vec![
            node(1, "Scan", "big", None, vec![]),
            node(2, "Scan", "small", None, vec![]),
        ],
    )
}

fn join_metrics(probe_in: u64, probe_out: u64, build_in: u64) -> QueryMetrics {
    let mut m = QueryMetrics::default();
    op(
        &mut m,
        1,
        "HashJoinProbe",
        0,
        OperatorMetrics {
            rows_in: probe_in,
            rows_out: probe_out,
            drivers: 1,
            ..OperatorMetrics::default()
        },
    );
    op(
        &mut m,
        2,
        "HashJoinBuild",
        0,
        OperatorMetrics {
            rows_in: build_in,
            drivers: 1,
            ..OperatorMetrics::default()
        },
    );
    m
}

fn input(rows: f64, stats: bool, tables: &[&str]) -> InputFacts {
    InputFacts {
        estimated_rows: rows,
        statistics: stats,
        tables_without_stats: tables.iter().map(|t| (*t).to_owned()).collect(),
    }
}

#[test]
fn hint_missing_stats_raised_at_the_boundary() {
    let s = HintSettings::default();
    let plan = |est: f64| {
        join_plan(NodeFacts {
            inputs: vec![input(1.0e6, true, &[]), input(est, false, &["sales_csv"])],
            ..NodeFacts::default()
        })
    };
    // Estimate 1,000,000 against 250,001 actual: q just under 4.
    assert!(
        hints(
            &plan(1.0e6),
            &join_metrics(5_000_000, 5_000_000, 250_001),
            &s
        )
        .is_empty()
    );
    // At q = 4.
    let h = hints(
        &plan(1.0e6),
        &join_metrics(5_000_000, 5_000_000, 250_000),
        &s,
    );
    assert_eq!(h.len(), 1);
    assert_eq!(h[0].code, HintCode::MissingStatistics);
    assert!(
        h[0].message
            .contains("ANALYZE TABLE sales_csv COMPUTE STATISTICS")
    );
    assert!(h[0].message.contains("1,000,000"));
    assert_eq!(h[0].operator, Some(OperatorId(2)));
    // Small tables stay quiet however wrong the estimate is.
    assert!(hints(&plan(100.0), &join_metrics(10, 10, 1), &s).is_empty());
    // A table with statistics never raises it.
    let with_stats = join_plan(NodeFacts {
        inputs: vec![input(1.0e6, true, &[]), input(1.0e6, true, &[])],
        ..NodeFacts::default()
    });
    assert!(hints(&with_stats, &join_metrics(5_000_000, 5_000_000, 10), &s).is_empty());
}

#[test]
fn missing_stats_keeps_the_worst_node_per_table() {
    let s = HintSettings::default();
    let facts = |est| NodeFacts {
        inputs: vec![input(est, false, &["t"]), input(1.0e6, true, &[])],
        ..NodeFacts::default()
    };
    let inner = node(1, "HashJoin", "", Some(facts(1.0e6)), vec![]);
    let plan = node(0, "HashJoin", "", Some(facts(1.0e6)), vec![inner]);
    let mut m = join_metrics(50_000_000, 1, 1);
    op(
        &mut m,
        3,
        "HashJoinProbe",
        1,
        OperatorMetrics {
            rows_in: 20_000_000,
            ..OperatorMetrics::default()
        },
    );
    let h = hints(&plan, &m, &s);
    assert_eq!(h.len(), 1);
    assert_eq!(h[0].node, Some(0));
}

fn gaps(reason: FilterGapReason) -> ProfileNode {
    join_plan(NodeFacts {
        filter_gaps: vec![FilterGap {
            key: 0,
            probe_key: "(l_partkey + 0)".to_owned(),
            reason,
        }],
        ..NodeFacts::default()
    })
}

#[test]
fn hint_runtime_filter_raised_at_the_boundary() {
    let s = HintSettings::default();
    let none = |probe_in, probe_out| {
        hints(
            &gaps(FilterGapReason::NoScanColumn),
            &join_metrics(probe_in, probe_out, 10),
            &s,
        )
    };
    // 999,999 probe rows: under the floor.
    assert!(none(999_999, 0).is_empty());
    // Dropping under half: under the fraction.
    assert!(none(1_000_000, 500_001).is_empty());
    let h = none(1_000_000, 500_000);
    assert_eq!(h.len(), 1);
    assert_eq!(h[0].code, HintCode::RuntimeFilterOpportunity);
    assert!(h[0].message.contains("join key #1"), "{}", h[0].message);
    assert!(!h[0].message.contains("l_partkey"), "{}", h[0].message);
    let h = hints(
        &gaps(FilterGapReason::Disabled),
        &join_metrics(1_000_000, 0, 10),
        &s,
    );
    assert!(
        h[0].message
            .contains("krishiv.runtimeFilter.enabled is false")
    );
    // No gap, no hint: the filter was published.
    let published = join_plan(NodeFacts::default());
    assert!(hints(&published, &join_metrics(1_000_000, 0, 10), &s).is_empty());
}

#[test]
fn no_hint_on_clean_metrics_and_when_disabled() {
    let s = HintSettings::default();
    let mut m = QueryMetrics::default();
    op(
        &mut m,
        1,
        "HashAggregate",
        0,
        OperatorMetrics {
            drivers: 4,
            rows_out: 10,
            ..OperatorMetrics::default()
        },
    );
    assert!(hints(&agg_plan(), &m, &s).is_empty());
    let mut m = metrics_with_peak(100 << 20);
    op(&mut m, 1, "HashAggregate", 0, spill(100 << 20));
    let off = HintSettings {
        enabled: false,
        ..s
    };
    assert!(hints(&agg_plan(), &m, &off).is_empty());
}

#[test]
fn hints_are_ordered_and_capped() {
    let s = HintSettings::default();
    let mb = 1 << 20;
    let mut m = metrics_with_peak(100 * mb);
    op(&mut m, 5, "HashAggregate", 0, spill(90 * mb));
    op(
        &mut m,
        2,
        "Window",
        0,
        OperatorMetrics {
            blocked_on_memory: Duration::from_secs(1),
            ..OperatorMetrics::default()
        },
    );
    op(&mut m, 1, "HashAggregate", 0, spill(80 * mb));
    let h = hints(&agg_plan(), &m, &s);
    let order: Vec<_> = h
        .iter()
        .map(|h| (h.code, h.operator.map(|o| o.0)))
        .collect();
    assert_eq!(
        order,
        [
            (HintCode::Spill, Some(1)),
            (HintCode::Spill, Some(5)),
            (HintCode::MemoryWait, Some(2))
        ]
    );
    let capped = HintSettings { max_hints: 2, ..s };
    assert_eq!(hints(&agg_plan(), &m, &capped).len(), 2);
}

#[test]
fn settings_read_the_config_keys() {
    let mut c = Config::default();
    c.set("krishiv.advisor.spill.minPercent", "50").unwrap();
    c.set("krishiv.advisor.spill.minBytes", "1k").unwrap();
    c.set("krishiv.advisor.skew.minTime", "5ms").unwrap();
    c.set("krishiv.advisor.enabled", "false").unwrap();
    let s = HintSettings::from_config(&c);
    assert_eq!(s.spill_min_percent, 50);
    assert_eq!(s.spill_min_bytes, 1024);
    assert_eq!(s.skew_min_time, Duration::from_millis(5));
    assert!(!s.enabled);
    assert_eq!(HintSettings::default().max_hints, 32);
}

#[test]
fn a_skew_hint_never_carries_text_from_the_node_detail() {
    // `SELECT 'acme_corp', x ... PARTITION BY` names the column after the literal: the detail holds it.
    let plan = node(0, "Window", "[acme_corp#7]", None, vec![]);
    let mut m = QueryMetrics::default();
    op(
        &mut m,
        1,
        "Window",
        0,
        OperatorMetrics {
            drivers: 4,
            compute_time: Duration::from_millis(1000),
            max_driver_compute_time: Duration::from_millis(1000),
            ..OperatorMetrics::default()
        },
    );
    let h = hints(&plan, &m, &HintSettings::default());
    assert_eq!(h.len(), 1);
    assert!(!h[0].message.contains("acme_corp"), "{}", h[0].message);
}

fn settings_with(pairs: &[(&str, &str)]) -> HintSettings {
    let mut c = Config::default();
    for (k, v) in pairs {
        assert!(c.set(k, v).is_ok(), "{k}");
    }
    HintSettings::from_config(&c)
}

#[test]
fn krishiv_advisor_missing_stats_min_rows_lowers_the_row_floor() {
    let plan = join_plan(NodeFacts {
        inputs: vec![input(1.0e6, true, &[]), input(100.0, false, &["small_t"])],
        ..NodeFacts::default()
    });
    let m = join_metrics(10, 10, 1);
    assert!(hints(&plan, &m, &HintSettings::default()).is_empty());
    let s = settings_with(&[("krishiv.advisor.missingStats.minRows", "5")]);
    assert_eq!(s.missing_stats_min_rows, 5);
    let h = hints(&plan, &m, &s);
    assert_eq!(h.len(), 1);
    assert_eq!(h[0].code, HintCode::MissingStatistics);
}

#[test]
fn krishiv_advisor_runtime_filter_min_probe_rows_lowers_the_probe_floor() {
    let plan = gaps(FilterGapReason::NoScanColumn);
    let m = join_metrics(1_000, 0, 10);
    assert!(hints(&plan, &m, &HintSettings::default()).is_empty());
    let s = settings_with(&[("krishiv.advisor.runtimeFilter.minProbeRows", "100")]);
    assert_eq!(s.runtime_filter_min_probe_rows, 100);
    let h = hints(&plan, &m, &s);
    assert_eq!(h.len(), 1);
    assert_eq!(h[0].code, HintCode::RuntimeFilterOpportunity);
}

#[test]
fn krishiv_advisor_runtime_filter_min_drop_percent_lowers_the_drop_floor() {
    let plan = gaps(FilterGapReason::NoScanColumn);
    // 30% of the probe rows dropped: under the default 50%.
    let m = join_metrics(1_000_000, 700_000, 10);
    assert!(hints(&plan, &m, &HintSettings::default()).is_empty());
    let s = settings_with(&[("krishiv.advisor.runtimeFilter.minDropPercent", "20")]);
    assert_eq!(s.runtime_filter_min_drop_percent, 20);
    assert_eq!(hints(&plan, &m, &s).len(), 1);
}

#[test]
fn krishiv_advisor_max_hints_cuts_the_list() {
    let mb = 1024 * 1024;
    let mut m = metrics_with_peak(400 * mb);
    op(&mut m, 1, "HashAggregate", 0, spill(80 * mb));
    op(&mut m, 5, "HashAggregate", 0, spill(90 * mb));
    assert_eq!(hints(&agg_plan(), &m, &HintSettings::default()).len(), 2);
    let s = settings_with(&[("krishiv.advisor.maxHints", "1")]);
    assert_eq!(s.max_hints, 1);
    assert_eq!(hints(&agg_plan(), &m, &s).len(), 1);
}

fn wait_metrics(blocked_ms: u64, compute_ms: u64) -> QueryMetrics {
    let mut m = QueryMetrics::default();
    op(
        &mut m,
        1,
        "HashAggregate",
        0,
        OperatorMetrics {
            blocked_on_memory: Duration::from_millis(blocked_ms),
            compute_time: Duration::from_millis(compute_ms),
            drivers: 1,
            ..OperatorMetrics::default()
        },
    );
    m
}

#[test]
fn krishiv_advisor_memory_wait_factor_scales_the_compute_comparison() {
    let m = wait_metrics(150, 100);
    assert_eq!(hints(&agg_plan(), &m, &HintSettings::default()).len(), 1);
    let s = settings_with(&[("krishiv.advisor.memoryWait.factor", "2")]);
    assert!(hints(&agg_plan(), &m, &s).is_empty());
}

#[test]
fn krishiv_advisor_memory_wait_min_time_sets_the_time_floor() {
    let m = wait_metrics(60, 10);
    assert!(hints(&agg_plan(), &m, &HintSettings::default()).is_empty());
    let s = settings_with(&[("krishiv.advisor.memoryWait.minTime", "50ms")]);
    assert_eq!(s.memory_wait_min_time, Duration::from_millis(50));
    assert_eq!(hints(&agg_plan(), &m, &s).len(), 1);
}

#[test]
fn krishiv_advisor_skew_factor_sets_how_lopsided_a_driver_must_be() {
    let mut m = QueryMetrics::default();
    op(
        &mut m,
        1,
        "Window",
        0,
        OperatorMetrics {
            drivers: 4,
            compute_time: Duration::from_millis(1000),
            max_driver_compute_time: Duration::from_millis(580),
            ..OperatorMetrics::default()
        },
    );
    assert_eq!(hints(&agg_plan(), &m, &HintSettings::default()).len(), 1);
    let s = settings_with(&[("krishiv.advisor.skew.factor", "8")]);
    assert!(hints(&agg_plan(), &m, &s).is_empty());
}

#[test]
fn krishiv_advisor_missing_stats_min_q_error_sets_how_wrong_the_estimate_must_be() {
    let plan = join_plan(NodeFacts {
        inputs: vec![input(1.0e6, true, &[]), input(1.0e6, false, &["sales_csv"])],
        ..NodeFacts::default()
    });
    // Estimate 1,000,000 against 250,000 actual: q = 4.
    let m = join_metrics(5_000_000, 5_000_000, 250_000);
    assert_eq!(hints(&plan, &m, &HintSettings::default()).len(), 1);
    let s = settings_with(&[("krishiv.advisor.missingStats.minQError", "8")]);
    assert!(hints(&plan, &m, &s).is_empty());
}
