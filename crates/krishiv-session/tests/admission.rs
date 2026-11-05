// SPDX-License-Identifier: Apache-2.0
//! The admission seam (E5): the `queued` state, refusal, cancellation while queued, and the
//! permit's slot.
#![cfg(test)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use krishiv_common::error::{ErrorClass, KrishivError, KrishivUnsupportedFeature};
use krishiv_session::{AdmissionController, BoxFuture, Permit, QueryRequest, Session};
use tokio::sync::Semaphore;

/// Counts the slots in use: one is taken by a permit and given back when it drops.
#[derive(Debug, Default)]
struct Slots(Arc<AtomicUsize>);

struct Slot(Arc<AtomicUsize>);

impl Drop for Slot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Makes the first query wait until released; admits every later one at once.
#[derive(Debug)]
struct Gate {
    first: AtomicBool,
    release: Semaphore,
    slots: Slots,
    refuse: AtomicBool,
    requests: std::sync::Mutex<Vec<String>>,
}

impl Gate {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            first: AtomicBool::new(true),
            release: Semaphore::new(0),
            slots: Slots::default(),
            refuse: AtomicBool::new(false),
            requests: std::sync::Mutex::default(),
        })
    }
}

impl AdmissionController for Gate {
    fn admit<'a>(
        &'a self,
        request: &'a QueryRequest,
    ) -> BoxFuture<'a, Result<Permit, KrishivError>> {
        Box::pin(async move {
            self.requests
                .lock()
                .unwrap()
                .push(request.principal.name.clone());
            if self.refuse.load(Ordering::SeqCst) {
                return Err(KrishivUnsupportedFeature {
                    feature: "\"too many queries\"".into(),
                }
                .into());
            }
            if self.first.swap(false, Ordering::SeqCst) {
                self.release.acquire().await.unwrap().forget();
            }
            self.slots.0.fetch_add(1, Ordering::SeqCst);
            Ok(Permit::with_slot(Slot(Arc::clone(&self.slots.0))))
        })
    }
}

async fn states(s: &Session) -> Vec<(String, String)> {
    let df = s
        .sql("SELECT CAST(query_text AS STRING), CAST(state AS STRING) FROM system.queries")
        .await
        .unwrap();
    let mut out = Vec::new();
    for b in df.collect().await.unwrap() {
        let text = krishiv_arrow(b.column(0));
        let state = krishiv_arrow(b.column(1));
        out.extend(text.into_iter().zip(state));
    }
    out
}

fn krishiv_arrow(c: &arrow_array::ArrayRef) -> Vec<String> {
    if let Some(v) = c.as_string_view_opt() {
        (0..v.len()).map(|i| v.value(i).to_owned()).collect()
    } else {
        let v = c.as_string::<i32>();
        (0..v.len()).map(|i| v.value(i).to_owned()).collect()
    }
}

fn state_of(rows: &[(String, String)], text: &str) -> Option<String> {
    rows.iter().find(|(t, _)| t == text).map(|(_, s)| s.clone())
}

/// Waits until the gate has seen its first request, so the observer queries that follow can
/// never be the one it holds back.
async fn first_request_seen(gate: &Gate) {
    for _ in 0..2000 {
        if !gate.requests.lock().unwrap().is_empty() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("the gated query never asked for admission");
}

async fn wait_for(s: &Session, text: &str, state: &str) {
    for _ in 0..500 {
        if state_of(&states(s).await, text).as_deref() == Some(state) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("{text} never reached {state}: {:?}", states(s).await);
}

#[tokio::test]
async fn the_default_controller_admits_at_once() {
    let s = Session::new().unwrap();
    s.sql("SELECT 1 AS one")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        state_of(&states(&s).await, "SELECT 1 AS one").as_deref(),
        Some("finished")
    );
}

#[tokio::test]
async fn a_waiting_query_is_queued_then_runs_and_gives_its_slot_back() {
    let gate = Gate::new();
    let s = Session::builder()
        .admission_controller(Arc::<Gate>::clone(&gate))
        .build()
        .unwrap();
    let runner = s.clone();
    let query = tokio::spawn(async move {
        runner
            .sql("SELECT 7 AS seven")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
            .len()
    });
    first_request_seen(&gate).await;
    wait_for(&s, "SELECT 7 AS seven", "queued").await;
    assert_eq!(
        gate.slots.0.load(Ordering::SeqCst),
        0,
        "no slot before admission"
    );
    gate.release.add_permits(1);
    assert!(query.await.unwrap() > 0);
    assert_eq!(
        state_of(&states(&s).await, "SELECT 7 AS seven").as_deref(),
        Some("finished")
    );
    assert_eq!(
        gate.slots.0.load(Ordering::SeqCst),
        0,
        "the slot is back once the query ended"
    );
    assert!(gate.requests.lock().unwrap().iter().all(|p| !p.is_empty()));
}

#[tokio::test]
async fn a_refused_query_fails_with_the_controllers_error() {
    let gate = Gate::new();
    gate.refuse.store(true, Ordering::SeqCst);
    // Only the gate's first admitted query waits; the observer below must not be it.
    gate.first.store(false, Ordering::SeqCst);
    let s = Session::builder()
        .admission_controller(Arc::<Gate>::clone(&gate))
        .build()
        .unwrap();
    let e = s
        .sql("SELECT 2 AS two")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    gate.refuse.store(false, Ordering::SeqCst);
    let rows = states(&s).await;
    assert_eq!(
        state_of(&rows, "SELECT 2 AS two").as_deref(),
        Some("failed")
    );
}

#[tokio::test]
async fn dropping_a_queued_query_cancels_it() {
    let gate = Gate::new();
    let s = Session::builder()
        .admission_controller(Arc::<Gate>::clone(&gate))
        .build()
        .unwrap();
    let runner = s.clone();
    let query = tokio::spawn(async move {
        let _ = runner
            .sql("SELECT 3 AS three")
            .await
            .unwrap()
            .collect()
            .await;
    });
    first_request_seen(&gate).await;
    wait_for(&s, "SELECT 3 AS three", "queued").await;
    query.abort();
    let _ = query.await;
    wait_for(&s, "SELECT 3 AS three", "cancelled").await;
    assert_eq!(gate.slots.0.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_failed_query_gives_its_slot_back() {
    let gate = Gate::new();
    gate.first.store(false, Ordering::SeqCst);
    let s = Session::builder()
        .config("spark.sql.ansi.enabled", "true")
        .admission_controller(Arc::<Gate>::clone(&gate))
        .build()
        .unwrap();
    let e = s
        .sql("SELECT 1 / 0 AS boom")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DivideByZero);
    assert_eq!(
        gate.slots.0.load(Ordering::SeqCst),
        0,
        "the slot is back after the failure"
    );
    assert_eq!(
        state_of(&states(&s).await, "SELECT 1 / 0 AS boom").as_deref(),
        Some("failed")
    );
}
