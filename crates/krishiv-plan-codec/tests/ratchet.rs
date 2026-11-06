// SPDX-License-Identifier: Apache-2.0
//! The round-trip ratchet (P2-T173, design P2-T171 section 8.1): every
//! physical plan the planner produces for the TPC-H and TPC-DS queries
//! must encode to `krishiv.plan.v1` and decode to an equal plan, and the
//! decoded TPC-H plans must instantiate on a worker. There is no
//! allowlist of plans that may fail to round-trip: a node, expression or
//! option the codec cannot carry fails this test. CI runs it on every pull
//! request.
#![expect(clippy::unwrap_used, clippy::panic, reason = "test code")]

#[path = "common/suites.rs"]
mod suites;

use krishiv_exec::physical::{PhysNode, PhysicalPlan, WorkerContext, instantiate, plan_schemas};
use krishiv_plan_codec::{decode, encode};

use suites::{Suite, queries};

/// TPC-DS queries the engine cannot plan yet (not a codec limit: they fail
/// before a physical plan exists). The list may only shrink: the test
/// fails when a query on it plans, so its entry is removed in the change
/// that makes it plan.
const TPCDS_NOT_PLANNED: &[&str] = &[
    "q16", "q30", "q32", "q50", "q62", "q92", "q94", "q95", "q99",
];

fn assert_round_trips(name: &str, plan: &PhysicalPlan) -> Vec<u8> {
    let bytes = encode(plan).unwrap_or_else(|e| panic!("{name}: encode: {e}"));
    let back = decode(&bytes).unwrap_or_else(|e| panic!("{name}: decode: {e}"));
    assert!(back == *plan, "{name}: decode(encode(plan)) != plan");
    assert_eq!(encode(&back).unwrap(), bytes, "{name}: re-encoding differs");
    bytes
}

/// The planner's exchange schemas (plan data, T176 unit 1 review m4) are
/// the instantiated ones: a fragmenter uses them in place of instantiating
/// the plan.
fn assert_planned_schemas(name: &str, plan: &PhysicalPlan, worker: &WorkerContext) {
    let built = plan_schemas(plan, worker).unwrap_or_else(|e| panic!("{name}: schemas: {e}"));
    let mut stack: Vec<&PhysNode> = plan.roots().collect();
    let mut exchanges = 0;
    while let Some(n) = stack.pop() {
        if let PhysNode::Exchange(e) = n {
            let planned = e
                .schema
                .as_ref()
                .unwrap_or_else(|| panic!("{name}: exchange {} without a schema", e.id.0));
            assert_eq!(
                planned.fields(),
                built.exchanges[&e.id].fields(),
                "{name}: exchange {}",
                e.id.0
            );
            exchanges += 1;
        }
        stack.extend(n.children());
    }
    assert_eq!(exchanges, built.exchanges.len(), "{name}");
}

#[test]
fn tpch_plans_round_trip_and_instantiate() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let suite = Suite::tpch(&rt);
    let all = queries("tpch");
    assert_eq!(all.len(), 22);
    for (name, sql) in all {
        let plan = suite
            .plan(&sql)
            .unwrap_or_else(|e| panic!("tpch {name}: {e}"));
        let bytes = assert_round_trips(&format!("tpch {name}"), &plan);
        let back = decode(&bytes).unwrap();
        // The worker side: the decoded plan instantiates against the
        // worker's own tables.
        let worker = WorkerContext::new(suite.tables());
        instantiate(&back, &worker).unwrap_or_else(|e| panic!("tpch {name}: instantiate: {e}"));
    }
}

#[test]
fn tpcds_plans_round_trip() {
    let suite = Suite::tpcds();
    let all = queries("tpcds");
    assert_eq!(all.len(), 99);
    let worker = WorkerContext::new(suite.tables());
    let mut not_planned = Vec::new();
    for (name, sql) in &all {
        match suite.plan(sql) {
            Ok(plan) => {
                assert_round_trips(&format!("tpcds {name}"), &plan);
                assert_planned_schemas(&format!("tpcds {name}"), &plan, &worker);
            }
            Err(_) => not_planned.push(name.as_str()),
        }
    }
    assert_eq!(
        not_planned, TPCDS_NOT_PLANNED,
        "the TPC-DS queries that do not plan changed: update TPCDS_NOT_PLANNED (it may only shrink)"
    );
}

/// TPC-DS q10 and q35 (`EXISTS (store) AND (EXISTS (web) OR EXISTS
/// (catalog))`): a semi join and two existence joins, which encode and
/// decode (P2-T115 unit 2a).
#[test]
fn q10_and_q35_run_two_existence_joins() {
    let suite = Suite::tpcds();
    let all = queries("tpcds");
    for q in ["q10", "q35"] {
        let sql = &all.iter().find(|(n, _)| n == q).unwrap().1;
        let plan = suite.plan(sql).unwrap_or_else(|e| panic!("{q}: {e}"));
        let text = format!("{plan:?}");
        assert_eq!(text.matches("Existence").count(), 2, "{q}");
        assert_eq!(text.matches("LeftSemi").count(), 1, "{q}");
        assert_round_trips(q, &plan);
    }
}
