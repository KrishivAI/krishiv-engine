// SPDX-License-Identifier: Apache-2.0
//! The advisor (P3-T073) over a 500-operator profile: `advisor_hints_500_operators` times `hints()` on a plan
//! of 500 nodes whose operators spill, skew and wait on memory, so every rule runs its full path. The
//! acceptance is under 1 ms at p99 added to a finished query.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_exec::profile::{HintSettings, ProfileNode, hints};
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

fn bench(c: &mut Criterion) {
    let (plan, metrics) = fixture();
    let settings = HintSettings::default();
    c.bench_function("advisor_hints_500_operators", |b| {
        b.iter(|| black_box(hints(black_box(&plan), black_box(&metrics), &settings)));
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
