// SPDX-License-Identifier: Apache-2.0
//! Structural-key cost for a deep `withColumn`-style chain: each operator
//! references the previous DataFrame's `plan_id`, the quadratic case of the
//! canonical encoding; run under CodSpeed in CI.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_plan::{ColumnRef, Expr, ExprNode, LogicalPlan, Origin, Plan, PlanId};

#[expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]
fn chain(depth: i64) -> Plan {
    let mut plan = LogicalPlan::table(["t"]).with_origin(Origin::plan(PlanId(0)));
    for i in 1..depth {
        let reference = Expr::new(ExprNode::Column(ColumnRef {
            name: vec!["a".to_owned()],
            plan_id: Some(PlanId(i - 1)),
        }));
        plan = plan.filter(reference).with_origin(Origin::plan(PlanId(i)));
    }
    Plan::try_new(plan).expect("chain within the depth cap")
}

fn plan_key(c: &mut Criterion) {
    let deep = chain(900);
    c.bench_function("structural_hash_chain_900", |b| {
        b.iter(|| black_box(&deep).structural_hash());
    });
}

criterion_group!(benches, plan_key);
criterion_main!(benches);
