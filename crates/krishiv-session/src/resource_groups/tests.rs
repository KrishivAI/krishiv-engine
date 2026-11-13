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
    super::select::select(&rg.config, r).to_owned()
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
    // Trino-only knobs are ignored
    assert!(
        ResourceGroups::from_json(&g(r#","jmxExport":true,"softConcurrencyLimit":1"#), MEM).is_ok()
    );
}
