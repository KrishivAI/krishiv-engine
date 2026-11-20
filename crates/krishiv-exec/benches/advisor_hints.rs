// SPDX-License-Identifier: Apache-2.0
//! The advisor (P3-T073) over a 500-operator profile: `advisor_hints_500_operators` times `hints()` on a plan
//! of 500 nodes whose operators spill, skew and wait on memory, so the SPILL, SKEW and MEMORY_WAIT rules run
//! their full path; the default `max_hints` caps that output, so the timed run emits capped, SPILL-only hints.
//! `advisor_hints_join_facts` covers the fact-dependent rules (MISSING_STATISTICS, RUNTIME_FILTER_OPPORTUNITY)
//! on a join plan, after an untimed check that both codes are raised with the cap lifted. The acceptance is
//! under 1 ms at p99 added to a finished query: criterion reports a mean, so `bench` also times each call of
//! the 500-operator run on its own and prints the p99.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_exec::planner::{FilterGap, FilterGapReason, InputFacts, NodeFacts};
use krishiv_exec::profile::{HintCode, HintSettings, ProfileNode, hints};
use krishiv_exec::{OperatorEntry, OperatorId, OperatorMetrics, OperatorRole, QueryMetrics};

const OPERATORS: u32 = 500;

fn fixture() -> (ProfileNode, QueryMetrics) {
    let mut m = QueryMetrics::default();
    m.memory.peak_reserved = 1 << 30;
    let mut plan = ProfileNode {
        id: Some(OPERATORS - 1),
        name: "Scan".into(),
        detail: String::new(),
        children: vec![],
        facts: None,
    };
    for id in 0..OPERATORS {
        let name = ["Window", "HashAggregate", "Sort", "HashJoin"][(id % 4) as usize];
        m.operators.insert(
            OperatorId(id),
            OperatorEntry {
                name: name.into(),
                node: Some(id),
                role: OperatorRole::Operator,
                metrics: OperatorMetrics {
                    spill_write_bytes: 200 << 20,
                    spilled_bytes: 200 << 20,
                    drivers: 4,
                    compute_time: Duration::from_millis(1000),
                    max_driver_compute_time: Duration::from_millis(900),
                    blocked_on_memory: Duration::from_millis(1500),
                    ..OperatorMetrics::default()
                },
                ..OperatorEntry::default()
            },
        );
        if id + 1 < OPERATORS {
            plan = ProfileNode {
                id: Some(OPERATORS - 2 - id),
                name: name.into(),
                detail: "[k#1]".into(),
                children: vec![plan],
                facts: None,
            };
        }
    }
    (plan, m)
}

/// A hash join (node 1) over a scan (node 0): the probe reads 1M rows and drops 90% of them with no runtime
/// filter, and the planner guessed 10 rows for an input whose table has no statistics.
fn join_fixture() -> (ProfileNode, QueryMetrics) {
    let facts = NodeFacts {
        estimated_rows: Some(10.0),
        inputs: vec![
            InputFacts {
                estimated_rows: 10.0,
                statistics: false,
                tables_without_stats: vec!["t".to_owned()],
            },
            InputFacts {
                estimated_rows: 10.0,
                statistics: true,
                tables_without_stats: Vec::new(),
            },
        ],
        filter_gaps: vec![FilterGap {
            key: 0,
            probe_key: "k#1".to_owned(),
            reason: FilterGapReason::Blocked,
        }],
        tables_without_stats: Vec::new(),
    };
    let scan = ProfileNode {
        id: Some(0),
        name: "Scan".into(),
        detail: String::new(),
        children: vec![],
        facts: None,
    };
    let plan = ProfileNode {
        id: Some(1),
        name: "HashJoin".into(),
        detail: "[k#1]".into(),
        children: vec![scan],
        facts: Some(Arc::new(facts)),
    };
    let mut m = QueryMetrics::default();
    for (id, name, rows_in, rows_out) in [
        (0, "HashJoinProbe", 1_000_000, 100_000),
        (1, "HashJoinBuild", 1_000_000, 1_000_000),
    ] {
        m.operators.insert(
            OperatorId(id),
            OperatorEntry {
                name: name.into(),
                node: Some(1),
                role: OperatorRole::Operator,
                metrics: OperatorMetrics {
                    rows_in,
                    rows_out,
                    ..OperatorMetrics::default()
                },
                ..OperatorEntry::default()
            },
        );
    }
    (plan, m)
}

fn bench(c: &mut Criterion) {
    let (plan, metrics) = fixture();
    let settings = HintSettings::default();
    c.bench_function("advisor_hints_500_operators", |b| {
        b.iter(|| black_box(hints(black_box(&plan), black_box(&metrics), &settings)));
    });
    // Criterion reports a mean; the acceptance is a p99, so time each call and report it.
    let mut times: Vec<Duration> = (0..2000)
        .map(|_| {
            let t = Instant::now();
            black_box(hints(black_box(&plan), black_box(&metrics), &settings));
            t.elapsed()
        })
        .collect();
    times.sort();
    #[expect(clippy::print_stdout, reason = "a bench reports its p99")]
    {
        println!(
            "advisor_hints_500_operators p99 = {:?} (target < 1ms)",
            times[times.len() * 99 / 100]
        );
    }

    let (join_plan, join_metrics) = join_fixture();
    let uncapped = HintSettings {
        max_hints: usize::MAX,
        ..HintSettings::default()
    };
    let found = hints(&join_plan, &join_metrics, &uncapped);
    for code in [
        HintCode::MissingStatistics,
        HintCode::RuntimeFilterOpportunity,
    ] {
        assert!(
            found.iter().any(|h| h.code == code),
            "the join fixture raised no {code:?}"
        );
    }
    c.bench_function("advisor_hints_join_facts", |b| {
        b.iter(|| {
            black_box(hints(
                black_box(&join_plan),
                black_box(&join_metrics),
                &uncapped,
            ))
        });
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
