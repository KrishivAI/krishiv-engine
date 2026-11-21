// SPDX-License-Identifier: Apache-2.0
//! The T003 acceptance cases that run on the controller alone (design §9): selectors, caps,
//! cancel while queued, dispatch order, scan quota and the config refusals.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};

use futures::task::noop_waker_ref;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::WorkloadClass;
use krishiv_plan::analyzer::{Principal, PrincipalKind};
use krishiv_sql::StatementClass;

use super::{GroupSlot, ResourceGroups};
use crate::admission::{AdmissionController, Permit, QueryRequest};

type Pending<'a> = Pin<Box<dyn Future<Output = Result<Permit, KrishivError>> + Send + 'a>>;

const MEM: usize = 1 << 30;

fn request(user: &str) -> QueryRequest {
    let mut p = Principal::new(user, PrincipalKind::User);
    p.groups = vec!["analysts".to_owned()];
    QueryRequest {
        principal: Arc::new(p),
        tags: Vec::new(),
        class: WorkloadClass::Interactive,
        source: None,
        statement_class: None,
        estimate: None,
    }
}

fn poll(f: &mut Pending<'_>) -> Option<Result<Permit, KrishivError>> {
    match f.as_mut().poll(&mut Context::from_waker(noop_waker_ref())) {
        Poll::Ready(r) => Some(r),
        Poll::Pending => None,
    }
}

/// One root with a hard limit of 2, queue of 3, and one leaf.
const CAPS: &str = r#"{
  "rootGroups": [{"name": "root", "hardConcurrencyLimit": 2, "maxQueued": 3,
                  "subGroups": [{"name": "q", "hardConcurrencyLimit": 2, "maxQueued": 3}]}],
  "defaultGroup": "root.q"
}"#;

#[test]
fn concurrency_and_queue_caps() {
    let rg = ResourceGroups::from_json(CAPS, MEM).unwrap_or_else(|e| panic!("{e}"));
    let req = request("u");
    let mut futs: Vec<Pending<'_>> = (0..6).map(|_| rg.admit(&req)).collect();
    let mut permits = Vec::new();
    let mut results = Vec::new();
    for f in &mut futs {
        results.push(poll(f));
    }
    // two run, three queue, one is refused
    assert!(matches!(results[0], Some(Ok(_))) && matches!(results[1], Some(Ok(_))));
    assert!(results[2].is_none() && results[3].is_none() && results[4].is_none());
    let refused = results[5].take().unwrap_or_else(|| panic!("sixth pending"));
    let err = refused.unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivResourceGroupQueueFull);
    assert_eq!(rg.counts("root.q"), Some((2, 3)));
    for r in results.into_iter().flatten() {
        permits.push(r.unwrap_or_else(|e| panic!("{e}")));
    }
    // the oldest queued query starts when one ends
    drop(permits.remove(0));
    let third = poll(&mut futs[2]);
    assert!(third.as_ref().is_some_and(Result::is_ok));
    assert!(poll(&mut futs[3]).is_none());
    assert_eq!(rg.counts("root.q"), Some((2, 2)));
}

#[test]
fn a_dropped_waiter_frees_its_place() {
    let rg = ResourceGroups::from_json(CAPS, MEM).unwrap_or_else(|e| panic!("{e}"));
    let req = request("u");
    let mut a: Pending<'_> = rg.admit(&req);
    let mut b: Pending<'_> = rg.admit(&req);
    let held = [poll(&mut a), poll(&mut b)];
    let mut queued: Vec<Pending<'_>> = (0..3).map(|_| rg.admit(&req)).collect();
    for q in &mut queued {
        assert!(poll(q).is_none());
    }
    let mut fourth: Pending<'_> = rg.admit(&req);
    assert!(
        poll(&mut fourth).is_some_and(|r| r.is_err()),
        "full queue refuses"
    );
    drop(queued.pop());
    assert_eq!(rg.counts("root.q"), Some((2, 2)));
    let mut again: Pending<'_> = rg.admit(&req);
    assert!(poll(&mut again).is_none(), "the freed place is usable");
    drop(held);
}

const SELECTORS: &str = r#"{
  "rootGroups": [{"name": "root", "hardConcurrencyLimit": 10, "maxQueued": 10, "subGroups": [
    {"name": "admin", "hardConcurrencyLimit": 1, "maxQueued": 1},
    {"name": "etl", "hardConcurrencyLimit": 1, "maxQueued": 1},
    {"name": "bi", "hardConcurrencyLimit": 1, "maxQueued": 1},
    {"name": "tagged", "hardConcurrencyLimit": 1, "maxQueued": 1},
    {"name": "writes", "hardConcurrencyLimit": 1, "maxQueued": 1},
    {"name": "rest", "hardConcurrencyLimit": 1, "maxQueued": 1}]}],
  "selectors": [
    {"user": "admin|root", "group": "root.admin"},
    {"user": "etl_.*", "source": "airflow.*", "group": "root.etl"},
    {"userGroup": "bi", "group": "root.bi"},
    {"clientTags": ["priority"], "group": "root.tagged"},
    {"clientTags": ["team=x", "env=prod"], "group": "root.tagged"},
    {"queryType": "WRITE", "group": "root.writes"}
  ],
  "defaultGroup": "root.rest"
}"#;

fn routed(rg: &ResourceGroups, r: &QueryRequest) -> String {
    super::select::select(&rg.current().config, r).to_owned()
}

#[test]
fn selectors_route_queries() {
    let rg = ResourceGroups::from_json(SELECTORS, MEM).unwrap_or_else(|e| panic!("{e}"));
    let case = |name: &str, f: &dyn Fn(&mut QueryRequest), want: &str| {
        let mut r = request(name);
        f(&mut r);
        assert_eq!(routed(&rg, &r), want, "{name}");
    };
    case("admin", &|_| {}, "root.admin");
    case("root", &|_| {}, "root.admin");
    case("administrator", &|_| {}, "root.rest"); // anchored
    case(
        "etl_a",
        &|r| r.source = Some("airflow-2".into()),
        "root.etl",
    );
    case("etl_a", &|_| {}, "root.rest"); // no source
    case("etl_a", &|r| r.source = Some("cli".into()), "root.rest");
    case(
        "someone",
        &|r| r.principal = Arc::new(group_user("bi")),
        "root.bi",
    );
    case(
        "u",
        &|r| r.tags = vec![("priority".into(), String::new())],
        "root.tagged",
    );
    case(
        "u",
        &|r| r.tags = vec![("priority".into(), "high".into())],
        "root.tagged",
    );
    case(
        "u",
        &|r| r.tags = vec![("team".into(), "x".into()), ("env".into(), "prod".into())],
        "root.tagged",
    );
    case(
        "u",
        &|r| r.tags = vec![("team".into(), "x".into())],
        "root.rest",
    ); // needs both
    case(
        "u",
        &|r| r.tags = vec![("team".into(), "y".into()), ("env".into(), "prod".into())],
        "root.rest",
    );
    case(
        "u",
        &|r| r.statement_class = Some(StatementClass::Write),
        "root.writes",
    );
    case(
        "u",
        &|r| r.statement_class = Some(StatementClass::Read),
        "root.rest",
    );
    // first match wins: admin beats the WRITE selector
    case(
        "admin",
        &|r| r.statement_class = Some(StatementClass::Write),
        "root.admin",
    );
}

fn group_user(group: &str) -> Principal {
    let mut p = Principal::new("someone", PrincipalKind::User);
    p.groups = vec![group.to_owned()];
    p
}

/// Two saturated siblings, weights 3:1, four slots.
const FAIR: &str = r#"{
  "rootGroups": [{"name": "root", "hardConcurrencyLimit": 4, "maxQueued": 1000,
    "subGroups": [
      {"name": "a", "hardConcurrencyLimit": 4, "maxQueued": 1000, "schedulingWeight": 3},
      {"name": "b", "hardConcurrencyLimit": 4, "maxQueued": 1000, "schedulingWeight": 1}]}],
  "selectors": [{"user": "a", "group": "root.a"}],
  "defaultGroup": "root.b"
}"#;

#[test]
fn weighted_fair_dispatch_is_three_to_one() {
    let rg = ResourceGroups::from_json(FAIR, MEM).unwrap_or_else(|e| panic!("{e}"));
    let (ra, rb) = (request("a"), request("b"));
    let mut a: Vec<Pending<'_>> = (0..300).map(|_| rg.admit(&ra)).collect();
    let mut b: Vec<Pending<'_>> = (0..300).map(|_| rg.admit(&rb)).collect();
    let mut running = std::collections::VecDeque::new();
    let (mut ia, mut ib) = (0, 0);
    let grab = |a: &mut Vec<Pending<'_>>,
                ib_: &mut Vec<Pending<'_>>,
                ia: &mut usize,
                ib: &mut usize,
                running: &mut std::collections::VecDeque<Permit>|
     -> Option<char> {
        if let Some(p) = a.get_mut(*ia).and_then(poll) {
            running.push_back(p.unwrap_or_else(|e| panic!("{e}")));
            *ia += 1;
            return Some('a');
        }
        if let Some(p) = ib_.get_mut(*ib).and_then(poll) {
            running.push_back(p.unwrap_or_else(|e| panic!("{e}")));
            *ib += 1;
            return Some('b');
        }
        None
    };
    // Fill the four slots, then register the rest in the queues.
    for _ in 0..4 {
        grab(&mut a, &mut b, &mut ia, &mut ib, &mut running);
    }
    for f in a.iter_mut().skip(ia).chain(b.iter_mut().skip(ib)) {
        assert!(poll(f).is_none());
    }
    // Queries end oldest first; each freed slot goes to whichever group the dispatcher picks.
    let mut order = String::new();
    for _ in 0..200 {
        drop(running.pop_front());
        order.extend(grab(&mut a, &mut b, &mut ia, &mut ib, &mut running));
    }
    let tail = &order[order.len() - 160..];
    let na = tail.chars().filter(|c| *c == 'a').count();
    assert!(
        (na as f64 / 160.0 - 0.75).abs() < 0.10,
        "a took {na} of 160: {tail}"
    );
}

const PRIORITY: &str = r#"{
  "rootGroups": [{"name": "root", "hardConcurrencyLimit": 1, "maxQueued": 10,
    "schedulingPolicy": "query_priority",
    "subGroups": [
      {"name": "low", "hardConcurrencyLimit": 1, "maxQueued": 10, "priority": 10},
      {"name": "high", "hardConcurrencyLimit": 1, "maxQueued": 10, "priority": 90}]}],
  "selectors": [{"user": "h", "group": "root.high"}],
  "defaultGroup": "root.low"
}"#;

#[test]
fn query_priority_dispatches_the_higher_priority_first() {
    let rg = ResourceGroups::from_json(PRIORITY, MEM).unwrap_or_else(|e| panic!("{e}"));
    let (low, high) = (request("l"), request("h"));
    let mut first: Pending<'_> = rg.admit(&low);
    let running = poll(&mut first);
    let mut queued_low: Pending<'_> = rg.admit(&low); // older
    let mut queued_high: Pending<'_> = rg.admit(&high);
    assert!(poll(&mut queued_low).is_none() && poll(&mut queued_high).is_none());
    drop(running);
    let got = poll(&mut queued_high);
    assert!(got.as_ref().is_some_and(Result::is_ok));
    assert!(poll(&mut queued_low).is_none());
}

const QUOTA: &str = r#"{
  "rootGroups": [{"name": "root", "hardConcurrencyLimit": 4, "maxQueued": 4,
    "scanQuota": 1000, "scanQuotaPeriod": "1h"}],
  "defaultGroup": "root"
}"#;

#[test]
fn scan_quota_refuses_once_spent_and_resets_with_the_window() {
    let now = Arc::new(AtomicU64::new(1_700_000_000_000));
    let clock = Arc::clone(&now);
    let rg =
        ResourceGroups::with_clock(QUOTA, MEM, Arc::new(move || clock.load(Ordering::Relaxed)))
            .unwrap_or_else(|e| panic!("{e}"));
    let req = request("u");
    let mut f: Pending<'_> = rg.admit(&req);
    let permit = poll(&mut f)
        .and_then(Result::ok)
        .unwrap_or_else(|| panic!("admitted"));
    permit
        .slot::<GroupSlot>()
        .unwrap_or_else(|| panic!("a group slot"))
        .add_bytes_read(1000);
    drop(permit);
    let mut g: Pending<'_> = rg.admit(&req);
    let err = poll(&mut g).map(Result::unwrap_err);
    let err = err.unwrap_or_else(|| panic!("quota not enforced"));
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivResourceGroupQuotaExhausted
    );
    assert!(err.to_string().contains("2023-11-1"), "{err}");
    now.fetch_add(3_600_000, Ordering::Relaxed);
    let mut h: Pending<'_> = rg.admit(&req);
    assert!(poll(&mut h).is_some_and(|r| r.is_ok()), "the window reset");
}

#[test]
fn per_query_memory_is_the_permit_budget() {
    let json = r#"{"rootGroups": [{"name": "root", "hardConcurrencyLimit": 2, "maxQueued": 1,
        "perQueryMemoryLimit": 4096}], "defaultGroup": "root"}"#;
    let rg = ResourceGroups::from_json(json, MEM).unwrap_or_else(|e| panic!("{e}"));
    let req = request("u");
    let mut f: Pending<'_> = rg.admit(&req);
    let p = poll(&mut f).and_then(Result::ok);
    assert_eq!(p.and_then(|p| p.memory_budget()), Some(4096));
}

#[test]
fn soft_memory_limit_makes_new_queries_wait() {
    let json = r#"{"rootGroups": [{"name": "root", "hardConcurrencyLimit": 4, "maxQueued": 4,
        "softMemoryLimit": "50%"}], "defaultGroup": "root"}"#;
    let rg = ResourceGroups::from_json(json, 1000).unwrap_or_else(|e| panic!("{e}"));
    let req = request("u");
    let mut f: Pending<'_> = rg.admit(&req);
    let p = poll(&mut f)
        .and_then(Result::ok)
        .unwrap_or_else(|| panic!("admitted"));
    p.slot::<GroupSlot>()
        .unwrap_or_else(|| panic!("a group slot"))
        .set_reserved(500);
    let mut g: Pending<'_> = rg.admit(&req);
    assert!(
        poll(&mut g).is_none(),
        "at the soft limit a new query waits"
    );
}

#[test]
fn a_parents_scan_quota_counts_when_its_leaf_has_none() {
    let json = r#"{"rootGroups": [{"name": "root", "hardConcurrencyLimit": 4, "maxQueued": 4,
        "scanQuota": 1000, "scanQuotaPeriod": "1h",
        "subGroups": [{"name": "leaf", "hardConcurrencyLimit": 4, "maxQueued": 4}]}],
        "defaultGroup": "root.leaf"}"#;
    let rg = ResourceGroups::from_json(json, MEM).unwrap_or_else(|e| panic!("{e}"));
    let req = request("u");
    let mut f: Pending<'_> = rg.admit(&req);
    let permit = poll(&mut f)
        .and_then(Result::ok)
        .unwrap_or_else(|| panic!("admitted"));
    permit
        .slot::<GroupSlot>()
        .unwrap_or_else(|| panic!("a group slot"))
        .add_bytes_read(1000);
    drop(permit);
    let mut g: Pending<'_> = rg.admit(&req);
    let err = poll(&mut g).map(Result::unwrap_err);
    assert_eq!(
        err.map(|e| e.error_class()),
        Some(ErrorClass::KrishivResourceGroupQuotaExhausted)
    );
}

#[test]
fn max_queued_zero_still_runs_a_query_that_can_start() {
    let json = r#"{"rootGroups": [{"name": "root", "hardConcurrencyLimit": 1, "maxQueued": 0}],
        "defaultGroup": "root"}"#;
    let rg = ResourceGroups::from_json(json, MEM).unwrap_or_else(|e| panic!("{e}"));
    let req = request("u");
    let mut a: Pending<'_> = rg.admit(&req);
    let held = poll(&mut a);
    assert!(
        held.as_ref().is_some_and(Result::is_ok),
        "a free slot runs it"
    );
    let mut b: Pending<'_> = rg.admit(&req);
    let err = poll(&mut b).map(Result::unwrap_err);
    assert_eq!(
        err.map(|e| e.error_class()),
        Some(ErrorClass::KrishivResourceGroupQueueFull)
    );
}

#[test]
fn a_lower_reservation_wakes_a_query_waiting_on_the_soft_limit() {
    let json = r#"{"rootGroups": [{"name": "root", "hardConcurrencyLimit": 4, "maxQueued": 4,
        "softMemoryLimit": "50%"}], "defaultGroup": "root"}"#;
    let rg = ResourceGroups::from_json(json, 1000).unwrap_or_else(|e| panic!("{e}"));
    let req = request("u");
    let mut f: Pending<'_> = rg.admit(&req);
    let p = poll(&mut f)
        .and_then(Result::ok)
        .unwrap_or_else(|| panic!("admitted"));
    let slot = p
        .slot::<GroupSlot>()
        .unwrap_or_else(|| panic!("a group slot"));
    slot.set_reserved(500);
    let mut g: Pending<'_> = rg.admit(&req);
    assert!(poll(&mut g).is_none(), "waits at the soft limit");
    slot.set_reserved(100);
    assert!(
        poll(&mut g).is_some_and(|r| r.is_ok()),
        "starts once the reservation drops, with the first query still running"
    );
}

#[test]
fn bytes_scanned_across_a_window_boundary_are_charged() {
    let now = Arc::new(AtomicU64::new(1_700_000_000_000));
    let clock = Arc::clone(&now);
    let rg =
        ResourceGroups::with_clock(QUOTA, MEM, Arc::new(move || clock.load(Ordering::Relaxed)))
            .unwrap_or_else(|e| panic!("{e}"));
    let req = request("u");
    let mut f: Pending<'_> = rg.admit(&req);
    let permit = poll(&mut f)
        .and_then(Result::ok)
        .unwrap_or_else(|| panic!("admitted"));
    permit
        .slot::<GroupSlot>()
        .unwrap_or_else(|| panic!("a group slot"))
        .add_bytes_read(1000);
    // the window ends while the query runs; it finishes in the next one
    now.fetch_add(3_600_000, Ordering::Relaxed);
    drop(permit);
    let mut g: Pending<'_> = rg.admit(&req);
    let err = poll(&mut g).map(Result::unwrap_err);
    assert_eq!(
        err.map(|e| e.error_class()),
        Some(ErrorClass::KrishivResourceGroupQuotaExhausted)
    );
}

#[test]
fn a_grant_to_a_future_dropped_before_it_polls_frees_the_slot() {
    let rg = ResourceGroups::from_json(CAPS, MEM).unwrap_or_else(|e| panic!("{e}"));
    let req = request("u");
    let mut a: Pending<'_> = rg.admit(&req);
    let mut b: Pending<'_> = rg.admit(&req);
    let held = [poll(&mut a), poll(&mut b)];
    let mut waiting: Pending<'_> = rg.admit(&req);
    assert!(poll(&mut waiting).is_none());
    drop(held);
    // the freed slot went to `waiting`; dropping it unpolled must give the slot back
    drop(waiting);
    assert_eq!(rg.counts("root.q"), Some((0, 0)));
    let mut next: Pending<'_> = rg.admit(&req);
    assert!(poll(&mut next).is_some_and(|r| r.is_ok()), "no slot leaked");
}

#[test]
fn a_spent_quota_refuses_a_query_that_was_already_queued() {
    let json = r#"{"rootGroups": [{"name": "root", "hardConcurrencyLimit": 1, "maxQueued": 2,
        "scanQuota": 1000, "scanQuotaPeriod": "1h"}], "defaultGroup": "root"}"#;
    let rg = ResourceGroups::from_json(json, MEM).unwrap_or_else(|e| panic!("{e}"));
    let req = request("u");
    let mut a: Pending<'_> = rg.admit(&req);
    let first = poll(&mut a)
        .and_then(Result::ok)
        .unwrap_or_else(|| panic!("admitted"));
    let mut b: Pending<'_> = rg.admit(&req);
    assert!(poll(&mut b).is_none(), "queued behind the first");
    first
        .slot::<GroupSlot>()
        .unwrap_or_else(|| panic!("a group slot"))
        .add_bytes_read(1000);
    drop(first);
    let err = poll(&mut b).map(Result::unwrap_err);
    assert_eq!(
        err.map(|e| e.error_class()),
        Some(ErrorClass::KrishivResourceGroupQuotaExhausted),
        "checked again when dispatched"
    );
}

#[test]
fn an_ancestors_hard_limit_and_queue_cap_apply_to_its_leaves() {
    let json = r#"{"rootGroups": [{"name": "root", "hardConcurrencyLimit": 1, "maxQueued": 1,
        "subGroups": [{"name": "a", "hardConcurrencyLimit": 5, "maxQueued": 5},
                      {"name": "b", "hardConcurrencyLimit": 5, "maxQueued": 5}]}],
        "selectors": [{"user": "ua", "group": "root.a"}, {"user": "ub", "group": "root.b"}],
        "defaultGroup": "root.a"}"#;
    let rg = ResourceGroups::from_json(json, MEM).unwrap_or_else(|e| panic!("{e}"));
    let (ra, rb) = (request("ua"), request("ub"));
    let mut f1: Pending<'_> = rg.admit(&ra);
    let held = poll(&mut f1);
    assert!(held.as_ref().is_some_and(Result::is_ok));
    // root allows one running query, so a leaf with room still queues
    let mut f2: Pending<'_> = rg.admit(&rb);
    assert!(poll(&mut f2).is_none(), "root's hard limit holds b back");
    // root's queue holds one, so the next is refused although b has room
    let mut f3: Pending<'_> = rg.admit(&rb);
    let err = poll(&mut f3).map(Result::unwrap_err);
    assert_eq!(
        err.map(|e| e.error_class()),
        Some(ErrorClass::KrishivResourceGroupQueueFull)
    );
}

fn refused(json: &str) -> String {
    let e = ResourceGroups::from_json(json, MEM).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivResourceGroupConfigInvalid
    );
    e.to_string()
}

#[test]
fn the_file_is_checked() {
    let g = |extra: &str| {
        format!(
            r#"{{"rootGroups":[{{"name":"root","hardConcurrencyLimit":1,"maxQueued":1{extra}}}],"defaultGroup":"root"}}"#
        )
    };
    assert!(refused(&g(r#","softCpuLimit":"1m""#)).contains("softCpuLimit"));
    assert!(refused(&g(r#","bogus":1"#)).contains("bogus"));
    assert!(refused(r#"{"rootGroups":[{"name":"${USER}","hardConcurrencyLimit":1,"maxQueued":1}],"defaultGroup":"x"}"#).contains("templates"));
    assert!(
        refused(r#"{"rootGroups":[{"name":"root","hardConcurrencyLimit":1,"maxQueued":1}]}"#)
            .contains("defaultGroup")
    );
    assert!(
        refused(
            r#"{"rootGroups":[{"name":"root","hardConcurrencyLimit":1}],"defaultGroup":"root"}"#
        )
        .contains("maxQueued")
    );
    assert!(refused(r#"{"rootGroups":[{"name":"root","hardConcurrencyLimit":1,"maxQueued":1}],"selectors":[{"group":"nope"}],"defaultGroup":"root"}"#).contains("nope"));
    assert!(refused(&g(r#","scanQuota":0"#)).contains("scanQuota"));
    // Trino-only knobs are ignored
    assert!(
        ResourceGroups::from_json(&g(r#","jmxExport":true,"softConcurrencyLimit":1"#), MEM).is_ok()
    );
}

#[test]
fn a_selector_tag_matches_a_key_or_the_whole_pair() {
    use super::tag_matches;
    // A bare key matches any value; `k=v` the exact pair.
    assert!(tag_matches("team", "team", "x"));
    assert!(tag_matches("team=x", "team", "x"));
    assert!(!tag_matches("team=y", "team", "x"));
    // A Connect tag is opaque: `job=42` is a key, and the selector `job=42` matches it as written.
    assert!(tag_matches("job=42", "job=42", ""));
    // The over-match documented in the design: the selector `job=42` also matches the session
    // pair `("job", "42")`, and a key containing `=` matches whatever its value.
    assert!(tag_matches("job=42", "job", "42"));
    assert!(tag_matches("team=x", "team=x", "anything"));
    // Prefixes are not matches.
    assert!(!tag_matches("teams", "team", "x"));
    assert!(!tag_matches("team=", "team", "x"));
}

#[test]
fn rows_carry_the_subtree_counters_and_reserved_memory_to_every_ancestor() {
    let rg = ResourceGroups::from_json(CAPS, MEM).unwrap_or_else(|e| panic!("{e}"));
    let req = request("u");
    let mut f: Pending<'_> = rg.admit(&req);
    let permit = poll(&mut f)
        .and_then(Result::ok)
        .unwrap_or_else(|| panic!("admitted"));
    permit
        .slot::<GroupSlot>()
        .unwrap_or_else(|| panic!("a group slot"))
        .set_reserved(4096);
    let rows = rg.rows();
    let by_id = |id: &str| {
        rows.iter()
            .find(|r| r.group_id == id)
            .unwrap_or_else(|| panic!("row {id}"))
    };
    let (root, leaf) = (by_id("root"), by_id("root.q"));
    assert_eq!(root.parent_id, None);
    assert_eq!(leaf.parent_id.as_deref(), Some("root"));
    assert_eq!((root.running, leaf.running), (1, 1));
    assert_eq!(
        (root.memory_reserved_bytes, leaf.memory_reserved_bytes),
        (4096, 4096),
        "the parent row sums its subtree"
    );
    drop(permit);
    assert_eq!(rg.rows()[0].memory_reserved_bytes, 0);
}

// ---- reload (T066: the tree swap that keeps counters by group id) ----------------------------

fn two_leaves(a_hard: usize) -> String {
    format!(
        r#"{{"rootGroups": [{{"name": "root", "hardConcurrencyLimit": 10, "maxQueued": 10,
            "subGroups": [{{"name": "a", "hardConcurrencyLimit": {a_hard}, "maxQueued": 5}},
                          {{"name": "b", "hardConcurrencyLimit": 2, "maxQueued": 5}}]}}],
          "selectors": [{{"user": "bob", "group": "root.b"}}],
          "defaultGroup": "root.a"}}"#
    )
}

fn run(rg: &ResourceGroups, user: &str) -> Permit {
    let req = request(user);
    let mut f = rg.admit(&req);
    poll(&mut f)
        .unwrap_or_else(|| panic!("{user} queued"))
        .unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn a_reload_keeps_the_counters_of_the_groups_it_keeps_and_takes_their_new_limits() {
    let rg = ResourceGroups::from_json(&two_leaves(1), MEM).unwrap_or_else(|e| panic!("{e}"));
    let first = run(&rg, "u");
    let req = request("u");
    let mut waiting = rg.admit(&req);
    assert!(poll(&mut waiting).is_none());
    assert_eq!(rg.counts("root.a"), Some((1, 1)));
    // Raising the limit starts the waiting query: the running count carried over.
    rg.reload_json(&two_leaves(3))
        .unwrap_or_else(|e| panic!("{e}"));
    let second = poll(&mut waiting).unwrap_or_else(|| panic!("still queued"));
    assert!(second.is_ok());
    assert_eq!(rg.counts("root.a"), Some((2, 0)));
    drop((first, second));
    assert_eq!(rg.counts("root.a"), Some((0, 0)));
}

#[test]
fn a_removed_group_takes_nothing_new_and_goes_with_its_last_query() {
    let rg = ResourceGroups::from_json(&two_leaves(2), MEM).unwrap_or_else(|e| panic!("{e}"));
    let held = run(&rg, "bob");
    assert_eq!(rg.counts("root.b"), Some((1, 0)));
    let only_a = r#"{"rootGroups": [{"name": "root", "hardConcurrencyLimit": 10, "maxQueued": 10,
        "subGroups": [{"name": "a", "hardConcurrencyLimit": 2, "maxQueued": 5}]}],
        "defaultGroup": "root.a"}"#;
    rg.reload_json(only_a).unwrap_or_else(|e| panic!("{e}"));
    let states: Vec<_> = rg
        .rows()
        .iter()
        .map(|r| (r.group_id.clone(), r.state))
        .collect();
    assert_eq!(
        states,
        [
            ("root".to_owned(), "active"),
            ("root.a".to_owned(), "active"),
            ("root.b".to_owned(), "removed"),
        ]
    );
    let removed = rg.rows().into_iter().find(|r| r.state == "removed");
    assert!(
        removed.is_some_and(|r| r.config_etag.is_empty() && r.loaded_at_ms == 0),
        "a removed row carries no etag or load time"
    );
    // The selector for bob is gone: his next query goes to the default group.
    assert_eq!(routed(&rg, &request("bob")), "root.a");
    // The running query still counts in the parent, and ends cleanly.
    assert_eq!(rg.counts("root"), Some((1, 0)));
    drop(held);
    assert_eq!(rg.counts("root.b"), None, "freed with its last query");
    assert_eq!(rg.rows().len(), 2);
    // A new group reuses the freed place.
    rg.reload_json(&two_leaves(2))
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(rg.counts("root.b"), Some((0, 0)));
    assert_eq!(rg.rows().len(), 3);
}

#[test]
fn an_invalid_document_leaves_the_tree_and_its_etag_as_they_were() {
    let rg = ResourceGroups::from_json(&two_leaves(2), MEM).unwrap_or_else(|e| panic!("{e}"));
    let (etag, document) = (rg.etag(), rg.document());
    let held = run(&rg, "u");
    let err = rg.reload_json(r#"{"rootGroups": []}"#).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivResourceGroupConfigInvalid
    );
    assert_eq!((rg.etag(), rg.document()), (etag.clone(), document));
    assert_eq!(rg.counts("root.a"), Some((1, 0)));
    // A different document has another etag; the same document spelled differently has the same.
    let respelled = two_leaves(2).replace(" ", "  ");
    rg.reload_json(&respelled).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(rg.etag(), etag);
    rg.reload_json(&two_leaves(3))
        .unwrap_or_else(|e| panic!("{e}"));
    assert_ne!(rg.etag(), etag);
    drop(held);
}

#[test]
fn a_queued_query_in_a_removed_group_still_runs() {
    let rg = ResourceGroups::from_json(&two_leaves(2), MEM).unwrap_or_else(|e| panic!("{e}"));
    let (b1, b2) = (run(&rg, "bob"), run(&rg, "bob"));
    let req = request("bob");
    let mut waiting = rg.admit(&req);
    assert!(poll(&mut waiting).is_none());
    let only_a = r#"{"rootGroups": [{"name": "root", "hardConcurrencyLimit": 10, "maxQueued": 10,
        "subGroups": [{"name": "a", "hardConcurrencyLimit": 2, "maxQueued": 5}]}],
        "defaultGroup": "root.a"}"#;
    rg.reload_json(only_a).unwrap_or_else(|e| panic!("{e}"));
    drop(b1);
    let third = poll(&mut waiting).unwrap_or_else(|| panic!("the waiter never started"));
    assert!(third.is_ok());
    drop((b2, third));
    assert_eq!(rg.counts("root.b"), None);
}

// ---- replace (T066: a new document written over the group file) -------------------------------

fn group_file(text: &str) -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
    let path = dir.path().join("groups.json");
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("{e}"));
    let path = path.to_string_lossy().into_owned();
    (dir, path)
}

fn class_of(e: &KrishivError) -> ErrorClass {
    e.error_class()
}

#[test]
fn a_replace_writes_the_file_and_swaps_the_tree() {
    let (_dir, path) = group_file(&two_leaves(1));
    let rg = ResourceGroups::from_file(&path, MEM).unwrap_or_else(|e| panic!("{e}"));
    let old = rg.etag();
    assert_eq!(ResourceGroups::file_etag(&path), Some(old.clone()));
    rg.replace_file(&path, &two_leaves(4), &old)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_ne!(rg.etag(), old);
    assert_eq!(ResourceGroups::file_etag(&path), Some(rg.etag()));
    let on_disk = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(on_disk, two_leaves(4));
    let a = rg.rows().into_iter().find(|r| r.group_id == "root.a");
    assert_eq!(a.map(|r| r.hard_concurrency_limit), Some(4));
    // no temporary file is left beside it
    let left = std::fs::read_dir(
        std::path::Path::new(&path)
            .parent()
            .unwrap_or_else(|| panic!("dir")),
    )
    .unwrap_or_else(|e| panic!("{e}"))
    .count();
    assert_eq!(left, 1);
}

#[test]
fn a_replace_with_a_stale_etag_or_an_edited_file_changes_nothing() {
    let (_dir, path) = group_file(&two_leaves(1));
    let rg = ResourceGroups::from_file(&path, MEM).unwrap_or_else(|e| panic!("{e}"));
    let etag = rg.etag();
    let e = rg.replace_file(&path, &two_leaves(4), "0000").unwrap_err();
    assert_eq!(class_of(&e), ErrorClass::KrishivAdminPreconditionFailed);
    assert!(e.to_string().contains("stale_etag"), "{e}");
    // a hand edit of the file since it was loaded
    std::fs::write(&path, two_leaves(2)).unwrap_or_else(|e| panic!("{e}"));
    let e = rg.replace_file(&path, &two_leaves(4), &etag).unwrap_err();
    assert_eq!(class_of(&e), ErrorClass::KrishivAdminPreconditionFailed);
    assert!(e.to_string().contains("file_changed"), "{e}");
    assert_eq!(rg.etag(), etag);
    assert_eq!(
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}")),
        two_leaves(2)
    );
}

#[test]
fn a_replace_refuses_a_link_a_read_only_file_and_an_invalid_document() {
    let (dir, path) = group_file(&two_leaves(1));
    let rg = ResourceGroups::from_file(&path, MEM).unwrap_or_else(|e| panic!("{e}"));
    let etag = rg.etag();
    // invalid document: the file stays
    let e = rg.replace_file(&path, "{}", &etag).unwrap_err();
    assert_eq!(class_of(&e), ErrorClass::KrishivResourceGroupConfigInvalid);
    assert_eq!(rg.etag(), etag);
    // a symbolic link (a mounted ConfigMap)
    let link = dir.path().join("link.json");
    std::os::unix::fs::symlink(&path, &link).unwrap_or_else(|e| panic!("{e}"));
    let link = link.to_string_lossy().into_owned();
    let via_link = ResourceGroups::from_file(&link, MEM).unwrap_or_else(|e| panic!("{e}"));
    let e = via_link
        .replace_file(&link, &two_leaves(4), &via_link.etag())
        .unwrap_err();
    assert_eq!(class_of(&e), ErrorClass::KrishivResourceGroupConfigReadOnly);
    // a read-only file
    let mut perms = std::fs::metadata(&path)
        .unwrap_or_else(|e| panic!("{e}"))
        .permissions();
    perms.set_readonly(true);
    std::fs::set_permissions(&path, perms).unwrap_or_else(|e| panic!("{e}"));
    let e = rg.replace_file(&path, &two_leaves(4), &etag).unwrap_err();
    assert_eq!(class_of(&e), ErrorClass::KrishivResourceGroupConfigReadOnly);
    assert_eq!(rg.etag(), etag);
}

#[test]
fn queries_queued_on_a_group_that_gains_subgroups_still_start() {
    let rg = ResourceGroups::from_json(&two_leaves(1), MEM).unwrap_or_else(|e| panic!("{e}"));
    let first = run(&rg, "u");
    let req = request("u");
    let mut waiting = rg.admit(&req);
    assert!(poll(&mut waiting).is_none());
    // root.a becomes a parent; the query it queued as a leaf stays its own.
    let parent = r#"{"rootGroups": [{"name": "root", "hardConcurrencyLimit": 10, "maxQueued": 10,
        "subGroups": [{"name": "a", "hardConcurrencyLimit": 1, "maxQueued": 5,
                       "subGroups": [{"name": "c", "hardConcurrencyLimit": 1, "maxQueued": 5}]},
                      {"name": "b", "hardConcurrencyLimit": 2, "maxQueued": 5}]}],
        "selectors": [{"user": "bob", "group": "root.b"}],
        "defaultGroup": "root.a.c"}"#;
    rg.reload_json(parent).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(rg.counts("root.a"), Some((1, 1)));
    drop(first);
    let second = poll(&mut waiting).unwrap_or_else(|| panic!("the queued query never started"));
    assert!(second.is_ok());
    assert_eq!(rg.counts("root.a"), Some((1, 0)));
}

#[test]
fn a_replace_in_a_directory_that_cannot_be_written_is_read_only() {
    use std::os::unix::fs::PermissionsExt as _;
    let (dir, path) = group_file(&two_leaves(1));
    let rg = ResourceGroups::from_file(&path, MEM).unwrap_or_else(|e| panic!("{e}"));
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o555))
        .unwrap_or_else(|e| panic!("{e}"));
    let result = rg.replace_file(&path, &two_leaves(4), &rg.etag());
    // a superuser writes anyway; the check is for everyone else
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755))
        .unwrap_or_else(|e| panic!("{e}"));
    if let Err(e) = result {
        assert_eq!(class_of(&e), ErrorClass::KrishivResourceGroupConfigReadOnly);
    }
}

#[test]
fn a_waiting_query_reports_the_group_it_queued_in() {
    let rg = ResourceGroups::from_json(CAPS, MEM).unwrap_or_else(|e| panic!("{e}"));
    let holders: Vec<QueryRequest> = (0..2).map(|_| request("u")).collect();
    let _running: Vec<_> = holders
        .iter()
        .map(|r| poll(&mut rg.admit(r)).map(Result::unwrap))
        .collect();
    let waiting = request("w");
    let mut fut = rg.admit(&waiting);
    assert!(poll(&mut fut).is_none());
    assert_eq!(rg.queued_group(&waiting).as_deref(), Some("root.q"));
    drop(fut);
    // the entry is gone once the wait ends: the answer is a fresh selection again
    assert_eq!(rg.queued_group(&waiting).as_deref(), Some("root.q"));
    assert!(rg.waiting.lock().unwrap().is_empty());
}
