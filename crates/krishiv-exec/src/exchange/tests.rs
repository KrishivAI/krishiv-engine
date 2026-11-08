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
fn rows_without_columns_are_routed_not_dropped() {
    let b = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(5)),
    )
    .unwrap();
    for p in [Partitioning::RoundRobin, hash(vec![])] {
        let mut out = Vec::new();
        route(&p, &b, 4, &mut out).unwrap();
        assert_eq!(out, vec![0; 5]);
    }
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
    // Each partition owns its rows' memory: together about the batch's.
    let big = batch(20_000);
    let routes: Vec<u32> = (0..20_000).map(|i| i % 4).collect();
    let owned: usize = split(&big, &routes)
        .unwrap()
        .iter()
        .map(|(_, r)| r.get_array_memory_size())
        .sum();
    assert!(
        owned <= big.get_array_memory_size() * 3 / 2,
        "{owned} for {}",
        big.get_array_memory_size()
    );
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
    sink_ctx_quantum(limit, krishiv_memory::MemoryConfig::default().quantum)
}

fn sink_ctx_quantum(
    limit: Option<usize>,
    quantum: usize,
) -> (Arc<krishiv_memory::TreePool>, crate::TaskContext) {
    use krishiv_memory::{MemoryConfig, MemoryPool as _, SpillOwner, TreePool};
    let root = TreePool::new(&MemoryConfig {
        limit,
        quantum,
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

#[test]
fn a_batch_spread_over_partitions_is_charged_once() {
    use crate::operator::Sink as _;
    use krishiv_memory::MemoryPool as _;
    let (root, ctx) = sink_ctx_quantum(None, 1);
    let recorded = Arc::new(Recorded::default());
    let b = batch(4000);
    let sink = sink_of(&recorded, &b);
    let global = sink.global_state(&ctx).unwrap();
    let mut local = sink.local_state(&ctx, &global).unwrap();
    sink.sink(
        krishiv_types::batch::Batch::new(b.clone()),
        &mut local,
        &global,
    )
    .unwrap();
    // Below a frame per partition, nothing went out: four partitions hold
    // the batch's rows once between them, not four times (a slice reports
    // its whole parent's buffers).
    assert!(recorded.writes.lock().unwrap().is_empty());
    let input = b.get_array_memory_size();
    assert!(
        root.reserved() <= 2 * input,
        "{} for {input}",
        root.reserved()
    );
    drop(local);
}

/// An exchange whose every reader yields one batch, then the end.
#[derive(Debug)]
struct OneBatch(RecordBatch);

struct OneBatchReader(Option<RecordBatch>);

impl super::ExchangeReader for OneBatchReader {
    fn next(&mut self) -> futures::future::BoxFuture<'_, crate::Result<super::ExchangeFrame>> {
        let frame = match self.0.take() {
            Some(b) => super::ExchangeFrame::Data(b),
            None => super::ExchangeFrame::End,
        };
        Box::pin(std::future::ready(Ok(frame)))
    }
}

impl super::Exchange for OneBatch {
    fn writer(
        &self,
        _output: super::MapOutputId,
        _partitions: u32,
        _schema: arrow_schema::SchemaRef,
    ) -> crate::Result<Box<dyn super::ExchangeWriter>> {
        Err(crate::error::internal("read only".to_owned()))
    }

    fn reader(
        &self,
        _exchange: crate::physical::ExchangeId,
        _stage_attempt: u32,
        _map: u32,
        _reduce: u32,
    ) -> crate::Result<Box<dyn super::ExchangeReader>> {
        Ok(Box::new(OneBatchReader(Some(self.0.clone()))))
    }
}

#[test]
fn a_reader_takes_every_pair_and_refuses_another_schema() {
    use crate::operator::Source as _;
    use futures::FutureExt as _;
    let (_root, ctx) = sink_ctx(None);
    let b = batch(10);
    let read = |exchange: OneBatch, ordered: bool| {
        super::ExchangeRead::new(
            Arc::new(exchange),
            crate::physical::ExchangeId(1),
            0,
            &[0, 2],
            3,
            b.schema(),
            ordered,
        )
    };
    // Reduce partitions 0 and 2 of maps 0..3: six pairs, one batch each.
    let source = read(OneBatch(b.clone()), false);
    let global = source.global_state(&ctx).unwrap();
    assert_eq!(source.max_parallelism(&global), 6);
    let mut local = source.local_state(&ctx, &global).unwrap();
    let mut rows = 0;
    while let Some(got) = source
        .get_data(&mut local, &global)
        .now_or_never()
        .unwrap()
        .unwrap()
    {
        rows += got.num_rows();
    }
    assert_eq!(rows, 60);
    assert_eq!(source.morsels_claimed(&local), 6);
    // A gather is read by one driver.
    let ordered = read(OneBatch(b.clone()), true);
    assert_eq!(ordered.max_parallelism(&global), 1);
    // A frame of other columns is an error, not rows.
    let other = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![1]))],
    )
    .unwrap();
    let wrong = read(OneBatch(other), false);
    let global = wrong.global_state(&ctx).unwrap();
    let mut local = wrong.local_state(&ctx, &global).unwrap();
    assert!(
        wrong
            .get_data(&mut local, &global)
            .now_or_never()
            .unwrap()
            .is_err()
    );
}

/// Runs a sink collecting a filter on `k` over one batch and finalizes it
/// under a pool of `finalize_limit` bytes; returns the published filter's form and what the
/// sink's state held of that pool until it was dropped.
fn published_filter_form(finalize_limit: Option<usize>) -> (krishiv_io::FilterForm, usize) {
    published_filter_form_with(finalize_limit, krishiv_io::KeyFilterOptions::default())
}

/// As [`published_filter_form`], collecting with `options`.
fn published_filter_form_with(
    finalize_limit: Option<usize>,
    options: krishiv_io::KeyFilterOptions,
) -> (krishiv_io::FilterForm, usize) {
    use crate::operator::Sink as _;
    use crate::values::{LocalValues, Producer, ValueId, ValueService as _};
    use futures::FutureExt as _;
    use krishiv_io::{ColumnPath, FilterId, FilterSnapshot};
    use krishiv_memory::MemoryPool as _;
    let (_root, ctx) = sink_ctx(None);
    let (finalize_root, finalize_ctx) = sink_ctx_quantum(finalize_limit, 1);
    let values = Arc::new(
        LocalValues::with_producers([(ValueId::Filter(FilterId(1)), 1)].into_iter().collect())
            .unwrap(),
    );
    values
        .declare_filter(FilterId(1), ColumnPath::new("k"), DataType::Int64)
        .unwrap();
    let recorded = Arc::new(Recorded::default());
    // Sparse keys: a dense integer range would already be exact as a range.
    let b = batch(100);
    let sparse: Vec<i64> = (0..100).map(|i| (i % 37) * 1000).collect();
    let b = RecordBatch::try_new(
        b.schema(),
        vec![Arc::new(Int64Array::from(sparse)), Arc::clone(b.column(1))],
    )
    .unwrap();
    let sink = sink_of(&recorded, &b).with_filters(
        vec![crate::physical::ExchangeFilterPlan {
            id: FilterId(1),
            column: 0,
            key_type: DataType::Int64,
            options,
            null_safe: false,
        }],
        Arc::clone(&values) as Arc<dyn crate::values::ValueService>,
        Producer {
            partition: 0,
            attempt: 0,
        },
    );
    let global = sink.global_state(&ctx).unwrap();
    let mut local = sink.local_state(&ctx, &global).unwrap();
    sink.sink(krishiv_types::batch::Batch::new(b), &mut local, &global)
        .unwrap();
    sink.combine(local, &global).unwrap();
    sink.finalize(&global, &finalize_ctx)
        .now_or_never()
        .unwrap()
        .unwrap();
    let held = finalize_root.reserved();
    let FilterSnapshot::Final(f) = values
        .filter(FilterId(1))
        .unwrap()
        .published()
        .snapshot
        .clone()
    else {
        panic!("filter not published")
    };
    drop(global);
    assert_eq!(finalize_root.reserved(), 0);
    (f.form(), held)
}

/// A map task is one producer of several: a large build's shared-array
/// size (design P2b-R1) is never built from its collected hashes, nor held
/// (M5, review blocker 1).
#[test]
fn an_exchange_filter_builds_no_shared_size_bloom() {
    let options = krishiv_io::KeyFilterOptions {
        max_values: 0,
        exact_max_values: 0,
        bloom_max_items: 1_000,
        bloom_bytes: Some(64 << 20),
        bloom_max_bytes: 512 << 20,
        ..krishiv_io::KeyFilterOptions::default()
    };
    assert!(options.shared_bloom_bytes().is_some());
    let (form, held) = published_filter_form_with(None, options);
    assert_eq!(form, krishiv_io::FilterForm::Range);
    assert!(held < 1 << 20, "{held}");
}

#[test]
fn an_exchange_filter_that_does_not_fit_is_published_as_its_range() {
    // Room for it: the 37 distinct sparse keys stay a set, reserved with the
    // sink's state until the state is dropped.
    let (form, held) = published_filter_form(None);
    assert!(
        matches!(
            form,
            krishiv_io::FilterForm::SmallSet(37) | krishiv_io::FilterForm::HashSet(37)
        ),
        "{form:?}"
    );
    assert!(held > 0);
    // No room at finalize: only the min/max range, which holds nothing.
    let (form, held) = published_filter_form(Some(1));
    assert_eq!(form, krishiv_io::FilterForm::Range);
    assert_eq!(held, 0);
}
