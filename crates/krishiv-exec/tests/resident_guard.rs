// SPDX-License-Identifier: Apache-2.0
//! An external sort under a permanent resident-set guard cut, with
//! another query holding all the ordinary memory the cut leaves: the sort
//! spills, reads its runs back and merges them in the memory above the
//! cut ceiling (spill I/O and merge keys are exempt from the cut, and its
//! arbitration rounds waive it once spilling is exhausted). The probe is
//! process-wide, so this binary holds one test.
#![cfg(not(loom))]
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::StreamExt;
use krishiv_exec::testing::{Break, Scan};
use krishiv_exec::{
    ExecConfig, ExecNode, ExecutorConfig, MemoryScan, QueryContext, QueryId, QueryRunner, Sort,
    SortKey, Source, TokioClock, TokioExecutor, WorkloadClass,
};
use krishiv_memory::{
    MemoryConfig, MemoryError, MemoryPool, SpillConfig, SpillManager, TreePool, set_resident_probe,
};
use krishiv_types::batch::Batch;

const LIMIT: usize = 16 << 20;

/// A process far over any allowance: every guarded root is cut by the
/// most the guard cuts, a quarter of its limit.
fn probe() -> Option<usize> {
    Some(usize::MAX / 4)
}

fn table(rows: u64) -> Vec<Batch> {
    let batch = |start: u64| {
        let ids = start..(start + 4096).min(rows);
        let k = Int64Array::from_iter_values(ids.clone().map(|i| ((i * 7919) % rows) as i64));
        let i = Int64Array::from_iter_values(ids.clone().map(|i| i as i64));
        let pad = StringArray::from_iter_values(ids.map(|i| format!("{i:0>64}")));
        let columns: [(_, ArrayRef); 3] = [
            ("k", Arc::new(k)),
            ("i", Arc::new(i)),
            ("pad", Arc::new(pad)),
        ];
        Batch::new(RecordBatch::try_from_iter(columns).unwrap())
    };
    (0..rows).step_by(4096).map(batch).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sort_spills_and_merges_under_the_cut() {
    assert!(set_resident_probe(probe));
    let root = TreePool::with_clock(
        &MemoryConfig {
            limit: Some(LIMIT),
            quantum: 4 << 10,
            spill_wait_timeout: Duration::from_secs(2),
            abort_release_timeout: Duration::from_secs(30),
            spill_scratch_reserve: krishiv_memory::DEFAULT_SPILL_SCRATCH_RESERVE,
            resident_guard: true,
        },
        Arc::new(TokioClock),
    );
    // Another query holds all the ordinary memory the cut leaves: what the
    // sort gets is above the cut ceiling.
    let other = root.child("other", None);
    let held = other.register("held", None);
    let mut filled = 0;
    while other.try_reserve(held, 64 << 10).is_ok() {
        filled += 64 << 10;
    }
    assert!(
        (LIMIT / 2..LIMIT - LIMIT / 4).contains(&filled),
        "the cut applies: {filled}"
    );
    assert!(matches!(
        other.try_reserve(held, 64 << 10),
        Err(MemoryError::Exhausted { .. })
    ));

    let runner = QueryRunner::new(
        TokioExecutor::new(&ExecutorConfig {
            compute_threads: 2,
            io_threads: 1,
            blocking_threads: 2,
            io_blocking_threads: 2,
        })
        .unwrap(),
    );
    let pool = root.child("q", None);
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Interactive,
        ExecConfig::default(),
        pool,
    );
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("resident-guard-sort");
    let manager = SpillManager::open(&SpillConfig {
        dir,
        ..SpillConfig::default()
    })
    .unwrap();
    let rows = 200_000;
    let sort = Sort::new(
        table(1)[0].schema(),
        &[SortKey::asc(0)],
        Some(manager.scope().unwrap()),
    )
    .unwrap();
    let sort = Arc::new(sort.with_max_fan_in(4));
    let schema = Source::schema(&*sort);
    let scan = Arc::new(Scan(Arc::new(
        MemoryScan::try_new(schema, table(rows)).unwrap(),
    )));
    let plan: Arc<dyn ExecNode> = Arc::new(Break(scan, Arc::clone(&sort), true));
    let mut stream = runner.execute(plan, Arc::clone(&q)).await.unwrap();
    let mut keys = Vec::new();
    while let Some(b) = stream.next().await {
        let b = b.unwrap();
        keys.extend(
            b.data()
                .column(0)
                .as_primitive::<Int64Type>()
                .values()
                .iter()
                .copied(),
        );
    }
    drop(stream);
    q.finished().await;
    assert_eq!(keys.len() as u64, rows);
    assert!(keys.is_sorted());
    let m = sort.metrics();
    assert!(m.runs_spilled >= 2, "{m:?}");
    assert!(q.metrics().memory.peak_reserved > 0);
    other.release(held, filled);
    assert_eq!(root.reserved(), 0);
}
