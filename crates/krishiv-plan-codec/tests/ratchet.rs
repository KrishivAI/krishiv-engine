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

use krishiv_exec::physical::{PhysicalPlan, WorkerContext, instantiate};
use krishiv_plan_codec::{CodecError, decode, encode};

use suites::{Suite, queries};

/// TPC-DS queries the engine cannot plan yet (not a codec limit: they fail
/// before a physical plan exists). The list may only shrink: the test
/// fails when a query on it plans, so its entry is removed in the change
/// that makes it plan.
const TPCDS_NOT_PLANNED: &[&str] = &[
    "q16", "q30", "q32", "q41", "q50", "q62", "q92", "q94", "q95", "q99",
];

/// TPC-DS queries that plan but use a moment aggregate (`stddev_samp`),
/// which `krishiv.plan.v1` has no code for yet: the encoder refuses them
/// with `CodecError::Unsupported`, so none reaches a worker. The list may
/// only shrink: the test fails when one of them encodes, so its entry is
/// removed in the change that adds the aggregate to the format.
const TPCDS_NOT_ENCODABLE: &[&str] = &["q17", "q39"];

fn assert_round_trips(name: &str, plan: &PhysicalPlan) -> Vec<u8> {
    let bytes = encode(plan).unwrap_or_else(|e| panic!("{name}: encode: {e}"));
    let back = decode(&bytes).unwrap_or_else(|e| panic!("{name}: decode: {e}"));
    assert!(back == *plan, "{name}: decode(encode(plan)) != plan");
    assert_eq!(encode(&back).unwrap(), bytes, "{name}: re-encoding differs");
    bytes
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
    let mut not_planned = Vec::new();
    let mut not_encodable = Vec::new();
    for (name, sql) in &all {
        match suite.plan(sql) {
            Ok(plan) if TPCDS_NOT_ENCODABLE.contains(&name.as_str()) => match encode(&plan) {
                Err(CodecError::Unsupported(what)) if what.contains("aggregate function") => {
                    not_encodable.push(name.as_str());
                }
                other => panic!("tpcds {name}: expected a refused aggregate: {other:?}"),
            },
            Ok(plan) => {
                assert_round_trips(&format!("tpcds {name}"), &plan);
            }
            Err(_) => not_planned.push(name.as_str()),
        }
    }
    assert_eq!(not_encodable, TPCDS_NOT_ENCODABLE);
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
