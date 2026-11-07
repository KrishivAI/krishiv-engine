// SPDX-License-Identifier: Apache-2.0
//! Memory and spill under one engine runtime (design P2-E1, unit R2): each
//! session's queries share a pool of its own under the process root
//! (`krishiv.session.memoryLimit`), and every session spills into one
//! directory under one `krishiv.spill.maxBytes`.
#![cfg(test)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::StreamExt;
use futures::executor::block_on;
use krishiv_session::{EngineRuntime, Session};

const FACT_ROWS: i64 = 300_000;
const KEYS: i64 = 60_000;
const MIB: usize = 1 << 20;

/// A sort of 300,000 joined rows that spills under a few MiB and streams its
/// result from its spill files.
const SORTED: &str = "SELECT f.k, f.v, d.k AS dk FROM f JOIN d ON f.k = d.k ORDER BY f.v, f.k";

/// A join, an aggregate and a sort that spill under a few MiB.
const QUERY: &str = "SELECT f.k, sum(f.v) AS s, count(*) AS n \
                     FROM f JOIN d ON f.k = d.k \
                     GROUP BY f.k ORDER BY s DESC, f.k";

fn runtime(spill: &std::path::Path, root: &str, spill_cap: Option<&str>) -> Arc<EngineRuntime> {
    let mut b = EngineRuntime::builder()
        .config("krishiv.spill.dir", spill.to_str().unwrap())
        .config("krishiv.memory.limit", root);
    if let Some(cap) = spill_cap {
        b = b.config("krishiv.spill.maxBytes", cap);
    }
    b.build().unwrap()
}

fn session(rt: &Arc<EngineRuntime>, limit: Option<&str>) -> Session {
    let mut b = Session::builder().runtime(Arc::clone(rt));
    if let Some(limit) = limit {
        b = b.config("krishiv.session.memoryLimit", limit);
    }
    let s = b.build().unwrap();
    let fact = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let rows: Vec<i64> = (0..FACT_ROWS).collect();
    let batches: Vec<RecordBatch> = rows
        .chunks(8192)
        .map(|c| {
            RecordBatch::try_new(
                Arc::clone(&fact),
                vec![
                    Arc::new(Int64Array::from_iter_values(c.iter().map(|i| i % KEYS))),
                    Arc::new(Int64Array::from_iter_values(c.iter().map(|i| i % 1000))),
                ],
            )
            .unwrap()
        })
        .collect();
    s.register_batches("f", fact, batches).unwrap();
    let dim = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
    let keys = RecordBatch::try_new(
        Arc::clone(&dim),
        vec![Arc::new(Int64Array::from_iter_values(0..KEYS))],
    )
    .unwrap();
    s.register_batches("d", dim, vec![keys]).unwrap();
    s
}

fn row_count(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

/// Spill bytes on disk now (the runtime's gauge).
fn spill_used(rt: &EngineRuntime) -> f64 {
    rt.metrics()
        .gauges
        .iter()
        .find(|g| g.name == "krishiv.spill.disk.used")
        .map_or(0.0, |g| g.value)
}

#[test]
fn a_session_pool_is_a_child_of_the_runtime_root() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(&dir.path().join("spill"), "100m", None);
    let limited = session(&rt, Some("48m"));
    let unlimited = session(&rt, None);
    // A pool of its own, bounded by its limit and the root.
    let rows = |s: &Session| {
        let df = block_on(s.sql(
            "SELECT path, parent, limit_bytes FROM system.memory \
             WHERE kind = 'pool' AND (parent IS NULL OR parent = 'process') ORDER BY path",
        ))
        .unwrap();
        block_on(df.collect()).unwrap()
    };
    let got = rows(&limited);
    assert_eq!(
        row_count(&got),
        1,
        "the session pool, the only child of the root it sees"
    );
    let paths = got[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .unwrap();
    assert!(
        paths.value(0).starts_with("process/session:"),
        "{}",
        paths.value(0)
    );
    let limits = got[0]
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(limits.value(0), 48 << 20);
    // Without a limit the session level is skipped: queries hang off the
    // root, as in a session built alone.
    let got = rows(&unlimited);
    let paths = got[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .unwrap();
    assert_eq!(paths.value(0), "process");
    // The limit is fixed when the session is built.
    let e = limited
        .conf()
        .set("krishiv.session.memoryLimit", "1g")
        .unwrap_err();
    assert_eq!(e.error_class().name(), "CANNOT_MODIFY_STATIC_CONFIG");
}

/// Two sessions with 6 MiB each under a 13 MiB root run spilling queries at
/// once: neither session ever holds more than its limit, the root never
/// more than its own, and both queries finish with the full result.
/// The limits are enforced by the pool tree (unit tests in `krishiv-memory`);
/// the peaks here are sampled, as evidence that the session level is wired.
#[test]
fn two_limited_sessions_spill_side_by_side_within_their_limits() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(&dir.path().join("spill"), "13m", None);
    let a = session(&rt, Some("6m"));
    let b = session(&rt, Some("6m"));
    let done = Arc::new(AtomicBool::new(false));
    let peaks: Vec<Arc<AtomicUsize>> = (0..3).map(|_| Arc::new(AtomicUsize::new(0))).collect();
    let sampler = {
        let (a, b, rt) = (a.clone(), b.clone(), Arc::clone(&rt));
        let (done, peaks) = (Arc::clone(&done), peaks.clone());
        std::thread::spawn(move || {
            while !done.load(Ordering::Acquire) {
                peaks[0].fetch_max(a.memory_reserved(), Ordering::Relaxed);
                peaks[1].fetch_max(b.memory_reserved(), Ordering::Relaxed);
                peaks[2].fetch_max(rt.memory().reserved(), Ordering::Relaxed);
                std::thread::yield_now();
            }
        })
    };
    let run = |s: Session| {
        std::thread::spawn(move || {
            let df = block_on(s.sql(QUERY)).unwrap();
            let (got, profile) = block_on(df.collect_with_profile()).unwrap();
            let spilled: u64 = profile
                .metrics
                .operators
                .values()
                .map(|e| e.metrics.spill_files)
                .sum();
            (row_count(&got), spilled)
        })
    };
    let (ta, tb) = (run(a), run(b));
    let (ra, rb) = (ta.join().unwrap(), tb.join().unwrap());
    done.store(true, Ordering::Release);
    sampler.join().unwrap();
    let keys = usize::try_from(KEYS).unwrap();
    assert_eq!(ra.0, keys);
    assert_eq!(rb.0, keys);
    assert!(ra.1 > 0 && rb.1 > 0, "both queries spilled: {ra:?} {rb:?}");
    assert!(peaks[0].load(Ordering::Relaxed) <= 6 * MIB);
    assert!(peaks[1].load(Ordering::Relaxed) <= 6 * MIB);
    assert!(peaks[2].load(Ordering::Relaxed) <= 13 * MIB);
}

/// One spill directory and one byte cap for the whole runtime: with one
/// session's spill files on disk (its sort is mid-stream), a cap sized for
/// a bit more than one session's spill fails the other session's query,
/// which alone fits under it.
#[test]
fn every_session_spills_under_one_cap() {
    // How much one query keeps on disk while its result streams.
    let probe = tempfile::tempdir().unwrap();
    let rt = runtime(&probe.path().join("spill"), "64m", None);
    let a = session(&rt, Some("16m"));
    let mut stream = block_on(block_on(a.sql(SORTED)).unwrap().execute_stream()).unwrap();
    if let Err(e) = block_on(stream.next()).unwrap() {
        panic!("{e}");
    }
    let one = spill_used(&rt);
    assert!(one > 0.0, "the query keeps spill files while it streams");
    drop(stream);

    let dir = tempfile::tempdir().unwrap();
    // Truncation is intended: a byte count well above one query's spill.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a positive byte count"
    )]
    let cap = (one * 1.5) as u64;
    let rt = runtime(&dir.path().join("spill"), "64m", Some(&cap.to_string()));
    let a = session(&rt, Some("16m"));
    let b = session(&rt, Some("16m"));
    // Alone, the second session's query fits.
    let alone = block_on(block_on(b.sql(SORTED)).unwrap().collect()).unwrap();
    assert_eq!(row_count(&alone), usize::try_from(FACT_ROWS).unwrap());
    // With the first one's files on disk, it does not.
    let mut held = block_on(block_on(a.sql(SORTED)).unwrap().execute_stream()).unwrap();
    assert!(block_on(held.next()).unwrap().is_ok());
    let e = block_on(block_on(b.sql(SORTED)).unwrap().collect()).unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_SPILL_CAP_EXCEEDED");
    drop(held);
}
