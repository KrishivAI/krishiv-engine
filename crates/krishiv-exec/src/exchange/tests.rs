// SPDX-License-Identifier: Apache-2.0
use std::sync::Arc;

use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};

use super::{reduce_partitions, route, split};
use crate::operators::SortKey;
use crate::physical::{HashScheme, Partitioning};

fn batch(n: i64) -> RecordBatch {
    let keys: Vec<i64> = (0..n).map(|i| i % 37).collect();
    let values: Vec<String> = (0..n).map(|i| format!("row {i}")).collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(Int64Array::from(keys)),
            Arc::new(StringArray::from(values)),
        ],
    )
    .unwrap()
}

fn hash(keys: Vec<usize>) -> Partitioning {
    Partitioning::Hash {
        keys,
        scheme: HashScheme::KeyGroupV1,
    }
}

#[test]
fn hash_routing_is_the_plans_routing_hash() {
    let b = batch(500);
    let mut got = Vec::new();
    route(&hash(vec![0]), &b, 8, &mut got).unwrap();
    let mut want = Vec::new();
    HashScheme::KeyGroupV1
        .partitions(&[b.column(0).as_ref()], 8, &mut want)
        .unwrap();
    assert_eq!(got, want);
    // Equal keys meet in one partition, and the rows spread.
    let keys = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    for (i, p) in got.iter().enumerate() {
        let first = keys
            .values()
            .iter()
            .position(|k| *k == keys.value(i))
            .unwrap();
        assert_eq!(*p, got[first]);
    }
    assert!(got.iter().any(|p| *p != got[0]));
    let unpinned = Partitioning::Hash {
        keys: vec![0],
        scheme: HashScheme::Unpinned,
    };
    assert!(route(&unpinned, &b, 8, &mut got).is_err());
}

#[test]
fn round_robin_routes_by_the_whole_row_whatever_the_batches() {
    // Q4: the same rows go to the same partitions however the input is cut
    // into batches (driver interleaving), so a retried task matches.
    let b = batch(300);
    let mut whole = Vec::new();
    route(&Partitioning::RoundRobin, &b, 4, &mut whole).unwrap();
    let mut pieces = Vec::new();
    for (start, len) in [(0, 7), (7, 200), (207, 93)] {
        let mut part = Vec::new();
        route(
            &Partitioning::RoundRobin,
            &b.slice(start, len),
            4,
            &mut part,
        )
        .unwrap();
        pieces.extend(part);
    }
    assert_eq!(whole, pieces);
    assert!(whole.iter().any(|p| *p != whole[0]));
}

#[test]
fn gathers_broadcasts_and_ranges_write_partition_zero() {
    let b = batch(50);
    let range = Partitioning::Range {
        keys: vec![SortKey::asc(0)],
    };
    for p in [Partitioning::Single, Partitioning::Broadcast, range] {
        let mut out = vec![9];
        route(&p, &b, 1, &mut out).unwrap();
        assert_eq!(out, vec![0; 50]);
        assert_eq!(reduce_partitions(&p, 8), 1);
    }
    // One reduce partition: everything to it, even for a hash exchange.
    let mut out = Vec::new();
    route(&hash(vec![0]), &b, 1, &mut out).unwrap();
    assert_eq!(out, vec![0; 50]);
    assert_eq!(reduce_partitions(&hash(vec![0]), 8), 8);
    assert_eq!(reduce_partitions(&Partitioning::RoundRobin, 0), 1);
}

#[test]
fn split_keeps_every_row_once_in_input_order() {
    let b = batch(100);
    let routes: Vec<u32> = (0..100).map(|i| [2, 0, 2, 3][i % 4]).collect();
    let parts = split(&b, &routes).unwrap();
    assert_eq!(
        parts.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
        vec![0, 2, 3]
    );
    let values = |rb: &RecordBatch| -> Vec<String> {
        let a = rb.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        (0..a.len()).map(|i| a.value(i).to_owned()).collect()
    };
    let all = values(&b);
    for (p, rows) in &parts {
        let want: Vec<String> = all
            .iter()
            .zip(&routes)
            .filter(|(_, r)| **r as usize == *p)
            .map(|(v, _)| v.clone())
            .collect();
        assert_eq!(values(rows), want);
    }
    assert_eq!(parts.iter().map(|(_, r)| r.num_rows()).sum::<usize>(), 100);
    // One partition: the batch as it is.
    let one = split(&b, &[1; 100]).unwrap();
    assert_eq!(one.len(), 1);
    assert_eq!(one[0], (1, b.clone()));
    assert!(split(&b.slice(0, 0), &[]).unwrap().is_empty());
}

/// An exchange recording what its one writer wrote and whether it
/// committed.
#[derive(Debug, Default)]
struct Recorded {
    writes: std::sync::Mutex<Vec<(u32, usize)>>,
    committed: std::sync::atomic::AtomicBool,
}

struct RecordedWriter(Arc<Recorded>);

impl super::ExchangeWriter for RecordedWriter {
    fn write(&mut self, reduce: u32, batch: &RecordBatch) -> crate::Result<()> {
        self.0
            .writes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((reduce, batch.num_rows()));
        Ok(())
    }

    fn commit(self: Box<Self>) -> crate::Result<()> {
        self.0
            .committed
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Debug, Default)]
struct RecordingExchange(Arc<Recorded>);

impl super::Exchange for RecordingExchange {
    fn writer(
        &self,
        _output: super::MapOutputId,
        _partitions: u32,
        _schema: arrow_schema::SchemaRef,
    ) -> crate::Result<Box<dyn super::ExchangeWriter>> {
        Ok(Box::new(RecordedWriter(Arc::clone(&self.0))))
    }

    fn reader(
        &self,
        _exchange: crate::physical::ExchangeId,
        _stage_attempt: u32,
        _map: u32,
        _reduce: u32,
    ) -> crate::Result<Box<dyn super::ExchangeReader>> {
        Err(crate::error::internal("write only".to_owned()))
    }
}

fn sink_ctx(limit: Option<usize>) -> (Arc<krishiv_memory::TreePool>, crate::TaskContext) {
    use krishiv_memory::{MemoryConfig, MemoryPool as _, SpillOwner, TreePool};
    let root = TreePool::new(&MemoryConfig {
        limit,
        ..MemoryConfig::default()
    });
    let q = crate::QueryContext::new(
        crate::QueryId(1),
        crate::WorkloadClass::Batch,
        crate::ExecConfig::default(),
        root.child("q", None),
    );
    let exec = crate::TokioExecutor::new(&crate::ExecutorConfig::default()).unwrap();
    (
        root,
        crate::TaskContext::new(q, exec, SpillOwner::new(), crate::OperatorId(1), "x"),
    )
}

fn sink_of(recorded: &Arc<Recorded>, b: &RecordBatch) -> super::ExchangeSink {
    super::ExchangeSink::new(
        Arc::new(RecordingExchange(Arc::clone(recorded))),
        super::MapOutputId {
            exchange: crate::physical::ExchangeId(1),
            stage_attempt: 0,
            map: 0,
            attempt: 0,
        },
        hash(vec![0]),
        4,
        b.schema(),
    )
    .unwrap()
}

#[test]
fn buffered_rows_are_charged_and_a_spill_writes_them_as_frames() {
    use crate::operator::{Sink as _, SinkStatus};
    use futures::FutureExt as _;
    use krishiv_memory::MemoryPool as _;
    let (root, ctx) = sink_ctx(None);
    let recorded = Arc::new(Recorded::default());
    let b = batch(1000);
    let sink = sink_of(&recorded, &b);
    let global = sink.global_state(&ctx).unwrap();
    let mut local = sink.local_state(&ctx, &global).unwrap();
    for _ in 0..3 {
        let status = sink
            .sink(
                krishiv_types::batch::Batch::new(b.clone()),
                &mut local,
                &global,
            )
            .unwrap();
        assert!(matches!(status, SinkStatus::NeedMoreInput));
    }
    // 3,000 rows over 4 partitions stay below a frame: buffered, charged.
    assert!(recorded.writes.lock().unwrap().is_empty());
    let held = root.reserved();
    assert!(held > 0);
    let freed = sink
        .spill(&mut local, usize::MAX, &ctx)
        .now_or_never()
        .unwrap()
        .unwrap();
    assert!(freed > 0);
    assert!(root.reserved() < held);
    let writes = recorded.writes.lock().unwrap().clone();
    assert_eq!(writes.iter().map(|(_, n)| n).sum::<usize>(), 3000);
    assert!(writes.iter().all(|(p, _)| *p < 4));
    // Nothing left to spill; the rest goes out on combine, and only
    // finalize commits.
    assert_eq!(
        sink.spill(&mut local, usize::MAX, &ctx)
            .now_or_never()
            .unwrap()
            .unwrap(),
        0
    );
    sink.sink(krishiv_types::batch::Batch::new(b), &mut local, &global)
        .unwrap();
    sink.combine(local, &global).unwrap();
    assert!(!recorded.committed.load(std::sync::atomic::Ordering::SeqCst));
    sink.finalize(&global, &ctx)
        .now_or_never()
        .unwrap()
        .unwrap();
    assert!(recorded.committed.load(std::sync::atomic::Ordering::SeqCst));
    let rows: usize = recorded.writes.lock().unwrap().iter().map(|(_, n)| n).sum();
    assert_eq!(rows, 4000);
    // Committing twice is an error, not a second commit.
    assert!(
        sink.finalize(&global, &ctx)
            .now_or_never()
            .unwrap()
            .is_err()
    );
    assert_eq!(root.reserved(), 0);
}

#[test]
fn a_full_pool_asks_for_memory_before_taking_a_batch() {
    use crate::operator::{Sink as _, SinkStatus};
    let (_root, ctx) = sink_ctx(Some(1024));
    let recorded = Arc::new(Recorded::default());
    let b = batch(5000);
    let sink = sink_of(&recorded, &b);
    let global = sink.global_state(&ctx).unwrap();
    let mut local = sink.local_state(&ctx, &global).unwrap();
    let status = sink
        .sink(krishiv_types::batch::Batch::new(b), &mut local, &global)
        .unwrap();
    assert!(
        matches!(status, SinkStatus::NeedMemory { .. }),
        "{status:?}"
    );
    assert!(recorded.writes.lock().unwrap().is_empty());
    // Several partitions over a gather are refused.
    let e = super::ExchangeSink::new(
        Arc::new(RecordingExchange::default()),
        super::MapOutputId {
            exchange: crate::physical::ExchangeId(1),
            stage_attempt: 0,
            map: 0,
            attempt: 0,
        },
        Partitioning::Single,
        2,
        batch(1).schema(),
    );
    assert!(e.is_err());
}
