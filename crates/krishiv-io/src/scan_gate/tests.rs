// SPDX-License-Identifier: Apache-2.0
//! The scan gate with Parquet streams: shares, admission, over-commit and
//! revocation.

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use krishiv_memory::{MemoryConfig, MemoryPool, Reservation, TreePool};
use object_store::ObjectStore;
use object_store::memory::InMemory;
use object_store::path::Path;
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::ParquetMetaDataReader;
use parquet::file::properties::WriterProperties;

use super::ScanGate;
use crate::parquet::ParquetFormat;
use crate::{
    ByteRange, FileFormat, FileRef, FormatOptions, IoConfig, IoContext, IoScheduler, MetadataNeeds,
    ScanOptions, ScanRequest, ScanStream, StoreId,
};

const STORE: &str = "memory://gate";

struct Env {
    rt: tokio::runtime::Runtime,
    io: IoContext,
    store: Arc<InMemory>,
}

impl Env {
    fn new() -> Self {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()
            .unwrap();
        let store = Arc::new(InMemory::new());
        let sched = IoScheduler::new(IoConfig::default());
        sched.register_store(
            StoreId::new(STORE),
            Arc::clone(&store) as Arc<dyn ObjectStore>,
        );
        Self {
            rt,
            io: sched.context(),
            store,
        }
    }

    /// `k` = 0..rows in row groups of `rg_rows`, stored as `name`; with
    /// the file's row groups' expected shares (module docs of `scan_gate`).
    fn file(&self, name: &str, rows: i32, rg_rows: usize) -> (FileRef, Vec<usize>) {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from_iter_values(0..rows))],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(rg_rows))
            .build();
        let mut bytes = Vec::new();
        let mut w = ArrowWriter::try_new(&mut bytes, schema, Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
        let meta = ParquetMetaDataReader::new()
            .parse_and_finish(&bytes::Bytes::from(bytes.clone()))
            .unwrap();
        let shares = meta
            .row_groups()
            .iter()
            .map(|rg| {
                let c = rg.column(0);
                let n = |v: i64| usize::try_from(v).unwrap();
                let rows = n(rg.num_rows());
                let unc = n(c.uncompressed_size());
                n(c.compressed_size()) + unc.min(1 << 20) + unc / rows * rows.min(4096)
            })
            .collect();
        let file = self.rt.block_on(async {
            let path = Path::from(name);
            object_store::ObjectStoreExt::put(self.store.as_ref(), &path, bytes.into())
                .await
                .unwrap();
            let meta = object_store::ObjectStoreExt::head(self.store.as_ref(), &path)
                .await
                .unwrap();
            FileRef::from_meta(StoreId::new(STORE), &meta)
        });
        (file, shares)
    }

    fn request(&self, file: &FileRef) -> ScanRequest {
        let meta = self
            .rt
            .block_on(self.io.read_metadata(
                &ParquetFormat,
                file,
                MetadataNeeds::default(),
                &options(),
            ))
            .unwrap();
        ScanRequest::all(meta.schema())
    }

    /// A stream over the whole `file` without prefetch, sharing `gate`.
    async fn stream(
        &self,
        file: &FileRef,
        req: &ScanRequest,
        pool: &Arc<dyn MemoryPool>,
        gate: &Arc<ScanGate>,
    ) -> ScanStream {
        let id = pool.register("FileScan", None);
        ScanStream::open(
            &self.io,
            &ParquetFormat,
            file.clone(),
            ByteRange::whole(file.size),
            req,
            &options(),
            Reservation::new(Arc::clone(pool), id),
            ScanOptions {
                prefetch_bytes: 0,
                ..ScanOptions::default()
            },
        )
        .await
        .unwrap()
        .with_gate(Arc::clone(gate))
    }
}

fn options() -> FormatOptions {
    ParquetFormat
        .options()
        .parse(
            &std::collections::BTreeMap::new(),
            &krishiv_common::config::Config::new(),
        )
        .unwrap()
        .0
}

/// A query pool under a root with `limit`: exact grants, no scratch floor.
fn query_pool(limit: Option<usize>) -> (Arc<TreePool>, Arc<dyn MemoryPool>) {
    let root = TreePool::new(&MemoryConfig {
        limit,
        quantum: 1,
        spill_scratch_reserve: 0,
        resident_guard: true,
        ..MemoryConfig::default()
    });
    let query = root.child("query", None);
    (root, query)
}

/// Lets spawned tasks run until they block.
async fn settle() {
    for _ in 0..100 {
        tokio::task::yield_now().await;
    }
}

/// While a row group is decoded, the stream's share is its compressed
/// chunks, a page per column and one batch, reserved in the query's pool;
/// it is returned at the end. Row groups of different sizes check that each
/// is charged its own.
#[test]
fn a_units_working_set_is_the_streams_share_while_it_is_decoded() {
    let env = Env::new();
    let (file, shares) = env.file("ws.parquet", 10_000, 6_000);
    assert_ne!(shares[0], shares[1]);
    let req = env.request(&file);
    env.rt.block_on(async {
        let (_root, pool) = query_pool(None);
        let gate = Arc::new(ScanGate::new(&pool, usize::MAX));
        let mut s = env.stream(&file, &req, &pool, &gate).await;
        let mut seen = Vec::new();
        while let Some(b) = s.next_batch().await.unwrap() {
            seen.push((b.num_rows(), gate.bytes(), gate.reserved(), s.reserved()));
        }
        assert_eq!(
            seen,
            vec![
                (4_096, shares[0], shares[0], 0),
                (1_904, shares[0], shares[0], 0),
                (4_000, shares[1], shares[1], 0),
            ]
        );
        assert_eq!(pool.peak_reserved(), shares[0].max(shares[1]));
        assert_eq!((gate.bytes(), gate.active(), pool.reserved()), (0, 0, 0));
        assert_eq!((gate.overcommitted(), s.overcommitted()), (0, 0));
    });
}

/// With room in the pool, a second stream still waits while its unit would
/// take the shares past the budget, and starts once the first stream's
/// share is returned.
#[test]
fn the_budget_makes_a_stream_wait_for_another_streams_unit() {
    let env = Env::new();
    let (file, shares) = env.file("budget.parquet", 10_000, 5_000);
    let req = env.request(&file);
    env.rt.block_on(async {
        let (_root, pool) = query_pool(None);
        let gate = Arc::new(ScanGate::new(&pool, shares[0] * 3 / 2));
        let mut a = env.stream(&file, &req, &pool, &gate).await;
        a.next_batch().await.unwrap().unwrap();
        assert_eq!((gate.active(), gate.bytes()), (1, shares[0]));
        let mut b = env.stream(&file, &req, &pool, &gate).await;
        let waiting =
            tokio::spawn(async move { b.next_batch().await.unwrap().unwrap().num_rows() });
        settle().await;
        assert!(!waiting.is_finished(), "b waits for a's unit");
        drop(a);
        assert_eq!(waiting.await.unwrap(), 4_096);
        assert_eq!(gate.overcommitted(), 0);
    });
}

/// When the pool cannot fund a second unit, the stream waits for another
/// stream's share instead of failing, and nothing is over-committed.
#[test]
fn a_full_pool_makes_a_stream_wait_for_another_streams_unit() {
    let env = Env::new();
    let (file, shares) = env.file("full.parquet", 10_000, 5_000);
    let req = env.request(&file);
    env.rt.block_on(async {
        let limit = 16 << 20;
        let (_root, pool) = query_pool(Some(limit));
        let gate = Arc::new(ScanGate::new(&pool, usize::MAX));
        let mut a = env.stream(&file, &req, &pool, &gate).await;
        a.next_batch().await.unwrap().unwrap();
        // Leave less than one more unit free.
        let mut filler = Reservation::new(Arc::clone(&pool), pool.register("filler", None));
        filler
            .try_grow(limit - pool.reserved() - shares[0] / 2)
            .unwrap();
        let mut b = env.stream(&file, &req, &pool, &gate).await;
        let waiting =
            tokio::spawn(async move { b.next_batch().await.unwrap().unwrap().num_rows() });
        settle().await;
        assert!(!waiting.is_finished(), "b waits for a's unit");
        assert_eq!(gate.active(), 1);
        drop(a);
        assert_eq!(waiting.await.unwrap(), 4_096);
        assert_eq!(gate.overcommitted(), 0);
    });
}

/// A stream with no other to wait for decodes its unit even when nothing
/// can be spilled for it, counting the bytes as over-committed: the scan
/// slows down to one unit at a time instead of failing the query.
#[test]
fn a_lone_stream_over_commits_instead_of_failing() {
    let env = Env::new();
    let (file, shares) = env.file("alone.parquet", 10_000, 10_000);
    let req = env.request(&file);
    env.rt.block_on(async {
        let limit = 1 << 20;
        let (_root, pool) = query_pool(Some(limit));
        let gate = Arc::new(ScanGate::new(&pool, usize::MAX));
        let mut s = env.stream(&file, &req, &pool, &gate).await;
        let mut filler = Reservation::new(Arc::clone(&pool), pool.register("filler", None));
        filler.try_grow(limit - pool.reserved()).unwrap();
        let mut rows = 0;
        while let Some(b) = s.next_batch().await.unwrap() {
            rows += b.num_rows();
        }
        assert_eq!(rows, 10_000);
        assert_eq!(s.overcommitted(), gate.overcommitted());
        assert!(
            gate.overcommitted() >= shares[0] as u64,
            "{}",
            gate.overcommitted()
        );
        assert_eq!((gate.active(), gate.bytes()), (0, 0));
        assert_eq!(pool.reserved(), limit);
    });
}

/// An operator that needs memory the scans' shares hold, with nothing it
/// could spill, gets it by revoking the shares: no abort, the shares count
/// as over-committed until their units end, and the stream reads on.
#[test]
fn the_arbitrator_revokes_shares_instead_of_aborting_the_query() {
    let env = Env::new();
    let (file, shares) = env.file("revoke.parquet", 10_000, 10_000);
    let req = env.request(&file);
    env.rt.block_on(async {
        let limit = 4 << 20;
        let (root, pool) = query_pool(Some(limit));
        let gate = Arc::new(ScanGate::new(&pool, usize::MAX));
        let mut s = env.stream(&file, &req, &pool, &gate).await;
        s.next_batch().await.unwrap().unwrap();
        assert_eq!(gate.reserved(), shares[0]);
        let mut join = Reservation::new(Arc::clone(&pool), pool.register("join", None));
        join.grow(limit - shares[0] / 2).await.unwrap();
        assert_eq!(root.stats().aborts, 0);
        assert!(pool.check_aborted().is_ok());
        assert_eq!(gate.revoked(), shares[0] as u64);
        assert_eq!((gate.reserved(), gate.bytes()), (0, shares[0]));
        let mut rows = 4_096;
        while let Some(b) = s.next_batch().await.unwrap() {
            rows += b.num_rows();
        }
        assert_eq!(rows, 10_000);
        assert_eq!((gate.active(), gate.bytes()), (0, 0));
        assert_eq!(
            s.overcommitted(),
            shares[0] as u64,
            "the revoked share counts as over-committed once, when its unit ends"
        );
        drop(join);
        drop(s);
        assert_eq!(pool.reserved(), 0, "everything returned");
    });
}

/// Frees its reservation at once when asked to spill.
struct FreesAtOnce(std::sync::Mutex<Option<Reservation>>);

impl krishiv_memory::Spillable for FreesAtOnce {
    fn spillable_bytes(&self) -> usize {
        self.0.lock().unwrap().as_ref().map_or(0, Reservation::size)
    }

    fn request_spill(&self, target: usize) -> krishiv_memory::SpillTicket {
        let mut held = self.0.lock().unwrap();
        let r = held.as_mut().unwrap();
        let freed = target.min(r.size());
        r.shrink(freed);
        krishiv_memory::SpillTicket::ready(freed)
    }

    fn spill_requested(&self) -> Option<usize> {
        None
    }
}

/// A full pool, all of it held by a consumer that `spill` describes.
fn full_pool(limit: usize, spill: &Arc<FreesAtOnce>) -> (Arc<TreePool>, Arc<dyn MemoryPool>) {
    let (root, pool) = query_pool(Some(limit));
    let id = pool.register(
        "sort",
        Some(Arc::clone(spill) as Arc<dyn krishiv_memory::Spillable>),
    );
    let mut r = Reservation::new(Arc::clone(&pool), id);
    r.try_grow(limit).unwrap();
    *spill.0.lock().unwrap() = Some(r);
    (root, pool)
}

/// A lone stream whose share the pool cannot grant at once has operators
/// spill for it, and reserves the share instead of over-committing.
#[test]
fn a_lone_stream_has_operators_spill_for_its_share() {
    let env = Env::new();
    env.rt.block_on(async {
        let sort = Arc::new(FreesAtOnce(std::sync::Mutex::new(None)));
        let (root, pool) = full_pool(1 << 20, &sort);
        let gate = ScanGate::new(&pool, usize::MAX);
        let id = gate.stream_id();
        assert_eq!(gate.acquire(id, 256 << 10).await.unwrap(), 0);
        assert_eq!((gate.reserved(), gate.overcommitted()), (256 << 10, 0));
        assert_eq!(root.stats().aborts, 0);
        gate.shrink(id, 0);
        assert_eq!((gate.active(), gate.reserved()), (0, 0));
    });
}

/// A stream whose request for a share is dropped while the pool arbitrates
/// (its driver was cancelled) withdraws its claim: the gate does not count
/// it as active, so other streams do not wait for it.
#[test]
fn a_cancelled_request_withdraws_its_claim() {
    let env = Env::new();
    env.rt.block_on(async {
        // A victim that never begins its spill: the request waits.
        let (_root, pool) = query_pool(Some(1 << 20));
        let owner = krishiv_memory::SpillOwner::new();
        let signal = owner.signal();
        let id = pool.register(
            "sort",
            Some(Arc::clone(&signal) as Arc<dyn krishiv_memory::Spillable>),
        );
        let mut held = Reservation::new(Arc::clone(&pool), id);
        held.try_grow(1 << 20).unwrap();
        signal.set_spillable_bytes(1 << 20);
        let gate = ScanGate::new(&pool, usize::MAX);
        let stream = gate.stream_id();
        let waited = tokio::time::timeout(
            std::time::Duration::from_millis(10),
            gate.acquire(stream, 256 << 10),
        )
        .await;
        assert!(waited.is_err(), "the request was still waiting");
        assert_eq!((gate.active(), gate.bytes()), (0, 0));
    });
}
