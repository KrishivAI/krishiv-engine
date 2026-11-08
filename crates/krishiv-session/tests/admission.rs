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
    /// The estimate each request carried: `Some(scan_bytes)` or `None` when unknown.
    estimates: std::sync::Mutex<Vec<Option<Option<u64>>>>,
}

impl Gate {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            first: AtomicBool::new(true),
            release: Semaphore::new(0),
            slots: Slots::default(),
            refuse: AtomicBool::new(false),
            requests: std::sync::Mutex::default(),
            estimates: std::sync::Mutex::default(),
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
            self.estimates
                .lock()
                .unwrap()
                .push(request.estimate.as_ref().map(|e| e.scan_bytes));
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

#[tokio::test]
async fn a_dropped_stream_gives_its_slot_back_once_the_query_ended() {
    let gate = Gate::new();
    gate.first.store(false, Ordering::SeqCst);
    let s = Session::builder()
        .admission_controller(Arc::<Gate>::clone(&gate))
        .build()
        .unwrap();
    let df = s.sql("SELECT id FROM range(1000000)").await.unwrap();
    let mut stream = df.execute_stream().await.unwrap();
    // Take one batch, then drop the stream with the query still producing.
    let _ = futures::StreamExt::next(&mut stream).await;
    assert_eq!(
        gate.slots.0.load(Ordering::SeqCst),
        1,
        "the slot is held while it runs"
    );
    drop(stream);
    for _ in 0..2000 {
        if gate.slots.0.load(Ordering::SeqCst) == 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("the slot never came back after the stream was dropped");
}

#[tokio::test]
async fn a_refused_overwrite_leaves_the_existing_file_alone() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("out.txt");
    std::fs::write(&target, b"original").unwrap();
    let gate = Gate::new();
    gate.refuse.store(true, Ordering::SeqCst);
    gate.first.store(false, Ordering::SeqCst);
    let s = Session::builder()
        .admission_controller(Arc::<Gate>::clone(&gate))
        .build()
        .unwrap();
    let df = s.sql("SELECT 1 AS a").await;
    // The refusal may surface where the query is planned or where the write runs.
    if let Ok(df) = df {
        let r = df
            .write()
            .mode("overwrite")
            .parquet(target.to_str().unwrap())
            .await;
        assert!(r.is_err(), "the write must be refused");
    }
    assert_eq!(
        std::fs::read(&target).unwrap(),
        b"original",
        "the file survived the refusal"
    );
}

#[tokio::test]
async fn an_ignored_write_settles_its_row_as_finished() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("exists");
    std::fs::create_dir(&target).unwrap();
    let gate = Gate::new();
    gate.first.store(false, Ordering::SeqCst);
    let s = Session::builder()
        .admission_controller(Arc::<Gate>::clone(&gate))
        .build()
        .unwrap();
    let df = s.sql("SELECT 1 AS a").await.unwrap();
    df.write()
        .mode("ignore")
        .parquet(target.to_str().unwrap())
        .await
        .unwrap();
    assert_eq!(
        gate.slots.0.load(Ordering::SeqCst),
        0,
        "no slot is left held"
    );
    assert_eq!(
        state_of(&states(&s).await, "SELECT 1 AS a").as_deref(),
        Some("finished")
    );
}

#[tokio::test]
async fn a_refused_table_overwrite_keeps_the_old_table() {
    let gate = Gate::new();
    gate.first.store(false, Ordering::SeqCst);
    let dir = tempfile::tempdir().unwrap();
    let s = Session::builder()
        .config("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
        .admission_controller(Arc::<Gate>::clone(&gate))
        .build()
        .unwrap();
    s.sql("CREATE TABLE kept USING parquet AS SELECT 41 AS v")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    gate.requests.lock().unwrap().clear();
    // One admission for a successful overwrite: the write takes the one made before the drop.
    s.sql("SELECT 42 AS v")
        .await
        .unwrap()
        .write()
        .mode("overwrite")
        .save_as_table("kept")
        .await
        .unwrap();
    assert_eq!(gate.requests.lock().unwrap().len(), 1, "asked exactly once");
    assert_eq!(
        gate.slots.0.load(Ordering::SeqCst),
        0,
        "no slot is left held"
    );
    // A refused overwrite leaves the table as it was.
    gate.refuse.store(true, Ordering::SeqCst);
    let r = s
        .sql("SELECT 43 AS v")
        .await
        .unwrap()
        .write()
        .mode("overwrite")
        .save_as_table("kept")
        .await;
    assert!(r.is_err(), "the overwrite must be refused");
    gate.refuse.store(false, Ordering::SeqCst);
    let rows = s
        .sql("SELECT v FROM kept")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        rows.iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        1,
        "the table survived"
    );
}

/// A query cancelled while it waits for admission reports the time it waited as its wall time.
#[tokio::test]
async fn a_cancelled_queued_query_counts_its_queue_time_in_wall_ns() {
    let gate = Gate::new();
    let s = Session::builder()
        .admission_controller(Arc::<Gate>::clone(&gate))
        .build()
        .unwrap();
    let runner = s.clone();
    let started = std::time::Instant::now();
    let query = tokio::spawn(async move {
        let _ = runner
            .sql("SELECT 5 AS five")
            .await
            .unwrap()
            .collect()
            .await;
    });
    first_request_seen(&gate).await;
    wait_for(&s, "SELECT 5 AS five", "queued").await;
    tokio::time::sleep(Duration::from_millis(60)).await;
    query.abort();
    let _ = query.await;
    wait_for(&s, "SELECT 5 AS five", "cancelled").await;
    let df = s
        .sql("SELECT wall_ns FROM system.queries WHERE query_text = 'SELECT 5 AS five'")
        .await
        .unwrap();
    let wall: i64 = df
        .collect()
        .await
        .unwrap()
        .iter()
        .map(|b| {
            b.column(0)
                .as_primitive::<arrow_array::types::Int64Type>()
                .value(0)
        })
        .next()
        .expect("one row");
    // At least the 60 ms it slept while queued, at most the time since before it was submitted
    // (both on the monotonic clock).
    let most = i64::try_from(started.elapsed().as_nanos()).unwrap();
    assert!(
        (60_000_000..=most).contains(&wall),
        "the 60 ms in the queue count: {wall} ns, at most {most}"
    );
}

/// A query that reads only `system.*` skips the controller, even one that refuses everything;
/// a query that reads anything else, or nothing, does not.
#[tokio::test]
async fn a_query_reading_only_system_tables_skips_admission() {
    let gate = Gate::new();
    gate.first.store(false, Ordering::SeqCst);
    gate.refuse.store(true, Ordering::SeqCst);
    let s = Session::builder()
        .admission_controller(Arc::<Gate>::clone(&gate))
        .build()
        .unwrap();
    let rows = s
        .sql("SELECT query_id FROM system.queries")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert!(rows.iter().all(|b| b.num_columns() == 1));
    assert!(
        gate.requests.lock().unwrap().is_empty(),
        "the gate was not asked"
    );
    let refused = s.sql("SELECT 1 AS one").await.unwrap().collect().await;
    assert!(
        refused.is_err(),
        "a query reading nothing is admitted as usual"
    );
    assert_eq!(gate.requests.lock().unwrap().len(), 1);
    // A system table with rows of the query's own making, or with a user table, is admitted.
    s.sql("CREATE TEMPORARY VIEW t AS SELECT 1 AS id")
        .await
        .unwrap()
        .collect()
        .await
        .ok();
    let mixed = [
        "SELECT count(*) FROM system.queries CROSS JOIN range(10)",
        "SELECT q.query_id FROM system.queries q CROSS JOIN VALUES (1), (2) AS v(x)",
        "SELECT explode(array(1, 2)) FROM system.queries",
        "SELECT q.query_id FROM system.queries q JOIN t ON true",
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 3) \
         SELECT q.query_id FROM system.queries q CROSS JOIN r",
    ];
    for sql in mixed {
        let before = gate.requests.lock().unwrap().len();
        let refused = match s.sql(sql).await {
            Ok(df) => df.collect().await.is_err(),
            Err(e) => panic!("{sql}: {e}"),
        };
        assert!(
            refused,
            "{sql} is admitted, so the refusing gate refuses it"
        );
        assert_eq!(gate.requests.lock().unwrap().len(), before + 1, "{sql}");
    }
}

/// The request carries the estimate of the optimized plan for a controller that reads it, and
/// the default controller is never made to compute one.
#[tokio::test]
async fn the_request_carries_the_estimate_for_a_controller_that_wants_it() {
    let gate = Gate::new();
    gate.first.store(false, Ordering::SeqCst);
    let s = Session::builder()
        .admission_controller(Arc::<Gate>::clone(&gate))
        .build()
        .unwrap();
    s.sql("SELECT 1 AS one")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let seen = gate.estimates.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    // An estimate was computed: a query with no table scan reads no bytes.
    assert_eq!(seen, vec![Some(Some(0))]);
    assert!(!krishiv_session::Unlimited.uses_estimate());
    assert!(gate.uses_estimate());
}
