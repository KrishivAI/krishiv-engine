// SPDX-License-Identifier: Apache-2.0
//! Where a cached relation lives (design P3-T083 §3.2): in memory, as a
//! [`HeldTable`] whose pool bytes are released when its last holder (the
//! entry, or a query reading it) drops, or on disk, as a [`SpilledTable`]
//! in a spill scope of its own ([`CacheDisk`]). [`CacheSpill`] is the
//! `relation_cache` consumer's [`Spillable`]: asked to free memory, it
//! spills `MEMORY_AND_DISK` entries and evicts `MEMORY_ONLY` ones, least
//! recently used first, on a task of its own.

use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use arrow_schema::SchemaRef;
use krishiv_common::error::KrishivError;
use krishiv_exec::planner::{MemoryTable, SpilledTable, TableProvider};
use krishiv_exec::{Executor, ScanPushdown, WorkloadClass};
use krishiv_memory::{
    ConsumerId, MemoryPool, Reservation, SpillConfig, SpillFile, SpillManager, SpillSignal,
    SpillTicket, Spillable,
};
use krishiv_optimizer::TableStats;
use krishiv_types::batch::Batch;

use super::RelationCache;

/// Pool bytes of one in-memory snapshot, released on drop.
#[derive(Debug)]
pub(crate) struct PoolHold {
    pub(super) pool: Arc<dyn MemoryPool>,
    pub(super) consumer: ConsumerId,
    pub(super) bytes: usize,
}

impl Drop for PoolHold {
    fn drop(&mut self) {
        if self.bytes > 0 {
            self.pool.release(self.consumer, self.bytes);
        }
    }
}

/// An in-memory snapshot: the table its scans read, its batches (what a
/// spill writes) and the pool bytes they hold. A query that scans it holds
/// it, so its bytes stay reserved until the last reader ends, whatever
/// happened to the entry meanwhile: the pool never counts as free memory
/// a running query still reads.
#[derive(Debug)]
pub(crate) struct HeldTable {
    pub(super) table: MemoryTable,
    pub(super) schema: SchemaRef,
    pub(super) batches: Vec<Batch>,
    pub(super) _hold: PoolHold,
}

impl TableProvider for HeldTable {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        self.table.scan()
    }

    fn statistics(&self) -> Option<TableStats> {
        self.table.statistics()
    }
}

/// Where an entry's snapshot is.
#[derive(Debug, Clone)]
pub(crate) enum Tier {
    /// In memory, under the `relation_cache` pool.
    Memory(Arc<HeldTable>),
    /// In a spill file of the entry's own scope.
    Disk(Arc<SpilledTable>),
}

impl Tier {
    /// The provider scans of the hidden name read.
    pub(crate) fn provider(&self) -> Arc<dyn TableProvider> {
        match self {
            Self::Memory(t) => Arc::clone(t) as Arc<dyn TableProvider>,
            Self::Disk(t) => Arc::clone(t) as Arc<dyn TableProvider>,
        }
    }

    /// Whether no query reads the in-memory snapshot (`false` on disk): its
    /// bytes are freed as soon as the entry lets go of it.
    pub(crate) fn unread_in_memory(&self) -> bool {
        matches!(self, Self::Memory(t) if Arc::strong_count(t) == 1)
    }
}

/// Where disk entries are written: the runtime's spill directory (opened by
/// the first spill, shared with queries, under `krishiv.spill.maxBytes`)
/// and the executor whose blocking pool runs the writes.
#[derive(Debug, Clone)]
pub(crate) struct CacheDisk {
    pub(crate) executor: Arc<dyn Executor>,
    pub(crate) spill: Arc<Mutex<Option<SpillManager>>>,
    pub(crate) config: SpillConfig,
}

impl CacheDisk {
    /// Writes `batches` (of `schema`) to a spill file in a new scope of its
    /// own, on the blocking pool; the scratch memory of the write is
    /// `scratch`. The scope's directory goes with the file.
    pub(crate) async fn write(
        &self,
        schema: SchemaRef,
        batches: Vec<Batch>,
        scratch: Reservation,
    ) -> Result<SpillFile, KrishivError> {
        let spill = Arc::clone(&self.spill);
        let config = self.config.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.executor
            .spawn_blocking(Box::new(move || {
                let written = (|| {
                    let scope = {
                        let mut slot = spill.lock().unwrap_or_else(PoisonError::into_inner);
                        if slot.is_none() {
                            *slot = Some(SpillManager::open(&config)?);
                        }
                        match slot.as_ref() {
                            Some(manager) => manager.scope()?,
                            None => return Err(KrishivError::from(internal_spill())),
                        }
                    };
                    let mut writer = scope.writer(schema, scratch.for_spill_scratch())?;
                    for batch in &batches {
                        writer.write(batch)?;
                    }
                    Ok(writer.finish()?)
                })();
                drop(tx.send(written));
            }))
            .await?;
        rx.await.map_err(|_| {
            KrishivError::from(krishiv_common::error::InternalError {
                message: "a cache spill write ended without a result".to_owned(),
            })
        })?
    }
}

fn internal_spill() -> krishiv_common::error::InternalError {
    krishiv_common::error::InternalError {
        message: "the spill directory is not open".to_owned(),
    }
}

/// The `relation_cache` consumer's spill hook (see the [module docs](self)).
/// The pool holds it weakly; the cache owns it.
#[derive(Debug)]
pub(crate) struct CacheSpill {
    pub(super) cache: OnceLock<Weak<RelationCache>>,
    pub(super) signal: SpillSignal,
    /// A spill task is running: requests join it (a lock, so the task's
    /// last look for a request and a request's look at this are ordered).
    running: Mutex<bool>,
    executor: Option<Arc<dyn Executor>>,
}

impl CacheSpill {
    pub(super) fn new(executor: Option<Arc<dyn Executor>>) -> Arc<Self> {
        Arc::new(Self {
            cache: OnceLock::new(),
            signal: SpillSignal::new(),
            running: Mutex::new(false),
            executor,
        })
    }

    /// Starts the spill task unless one runs. Without an executor (a cache
    /// built for a unit test), requests complete with nothing freed.
    fn start(self: &Arc<Self>) {
        let Some(executor) = &self.executor else {
            if let Some(service) = self.signal.begin() {
                service.complete(0);
            }
            return;
        };
        let mut running = self.running.lock().unwrap_or_else(PoisonError::into_inner);
        if *running {
            return;
        }
        *running = true;
        drop(running);
        let me = Arc::clone(self);
        // Detached: it ends when no request is pending.
        drop(executor.spawn(WorkloadClass::Background, Box::pin(me.serve())));
    }

    async fn serve(self: Arc<Self>) {
        loop {
            if let Some(service) = self.signal.begin() {
                let cache = self.cache.get().and_then(Weak::upgrade);
                let freed = match cache {
                    Some(cache) => cache.spill_for(service.target()).await,
                    None => 0,
                };
                service.complete(freed);
            }
            let mut running = self.running.lock().unwrap_or_else(PoisonError::into_inner);
            if self.signal.spill_requested().is_none() {
                *running = false;
                return;
            }
        }
    }
}

impl Spillable for CacheSpill {
    fn spillable_bytes(&self) -> usize {
        self.signal.spillable_bytes()
    }

    fn request_spill(&self, target: usize) -> SpillTicket {
        let ticket = self.signal.request_spill(target);
        if target > 0
            && let Some(me) = self
                .cache
                .get()
                .and_then(Weak::upgrade)
                .map(|c| Arc::clone(&c.spill))
        {
            me.start();
        }
        ticket
    }

    fn spill_requested(&self) -> Option<usize> {
        self.signal.spill_requested()
    }

    /// A cached snapshot is a finished output no running driver owns:
    /// spilling it costs one write and one read, so it goes before a
    /// running operator's state.
    fn idle(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use futures::executor::block_on;
    use krishiv_memory::Spillable;

    use crate::functions::{col, lit};
    use crate::{DataFrame, Session, StorageLevel};

    fn spill_dir(dir: &tempfile::TempDir) -> std::path::PathBuf {
        dir.path().join("spill")
    }

    fn session(dir: &tempfile::TempDir, pairs: &[(&str, &str)]) -> Session {
        let mut builder =
            Session::builder().config("krishiv.spill.dir", spill_dir(dir).to_str().unwrap());
        for (k, v) in pairs {
            builder = builder.config(*k, *v);
        }
        let s = builder.build().unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from((0..20_000).collect::<Vec<_>>()))],
        )
        .unwrap();
        s.register_batches("t", schema, vec![batch]).unwrap();
        s
    }

    fn ids(df: &DataFrame) -> Vec<i32> {
        let mut out: Vec<i32> = block_on(df.collect())
            .unwrap()
            .iter()
            .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
            .collect();
        out.sort_unstable();
        out
    }

    /// Asked to spill, the hook moves `MEMORY_AND_DISK` entries to disk and
    /// evicts `MEMORY_ONLY` ones; both read back the same rows (the evicted
    /// one by refilling).
    #[test]
    fn a_spill_request_moves_entries_to_disk_or_evicts_them() {
        let dir = tempfile::tempdir().unwrap();
        let s = session(&dir, &[]);
        let disk = s.table("t").cache();
        assert_eq!(block_on(disk.count()).unwrap(), 20_000);
        let memory = s
            .table("t")
            .filter(col("id").lt(lit(100)))
            .persist(StorageLevel::MEMORY_ONLY);
        assert_eq!(block_on(memory.count()).unwrap(), 100);
        let cache = &s.state.cache;
        let held = cache.stats().memory_bytes;
        assert!(held > 0);
        assert_eq!(cache.spill.spillable_bytes() as u64, held);
        let freed = block_on(cache.spill.request_spill(usize::MAX));
        assert_eq!(freed as u64, held);
        let stats = cache.stats();
        assert_eq!((stats.spills, stats.evictions), (1, 1));
        assert_eq!((stats.entries, stats.entries_pending), (1, 1));
        assert_eq!(stats.memory_bytes, 0);
        assert!(stats.disk_bytes > 0);
        assert_eq!(cache.reserved(), 0);
        assert_eq!(cache.spill.spillable_bytes(), 0);
        assert_eq!(ids(&disk), (0..20_000).collect::<Vec<_>>());
        assert_eq!(ids(&memory), (0..100).collect::<Vec<_>>());
        // The evicted one refilled on that use.
        assert_eq!(cache.stats().entries, 2);
    }

    /// A query reading an entry keeps its pool bytes: the entry is not
    /// spillable while it runs, and is once it ended.
    #[test]
    fn a_reader_in_flight_keeps_its_bytes_until_it_ends() {
        use futures::StreamExt as _;
        let dir = tempfile::tempdir().unwrap();
        let s = session(&dir, &[]);
        let df = s.table("t").cache();
        assert_eq!(block_on(df.count()).unwrap(), 20_000);
        let cache = &s.state.cache;
        assert!(cache.spill.spillable_bytes() > 0);
        let mut stream = block_on(df.execute_stream()).unwrap();
        assert!(block_on(stream.next()).unwrap().unwrap().num_rows() > 0);
        assert_eq!(cache.spill.spillable_bytes(), 0);
        assert_eq!(block_on(cache.spill.request_spill(usize::MAX)), 0);
        assert_eq!(cache.stats().spills, 0);
        while block_on(stream.next()).is_some() {}
        drop(stream);
        assert!(cache.spill.spillable_bytes() > 0);
        assert!(block_on(cache.spill.request_spill(usize::MAX)) > 0);
        assert_eq!(cache.stats().spills, 1);
        assert_eq!(cache.reserved(), 0);
    }

    /// A spill write the spill cap refuses evicts the entry; nothing fails.
    #[test]
    fn a_refused_spill_write_evicts() {
        let dir = tempfile::tempdir().unwrap();
        let s = session(&dir, &[("krishiv.spill.maxBytes", "1b")]);
        let df = s.table("t").cache();
        assert_eq!(block_on(df.count()).unwrap(), 20_000);
        let cache = &s.state.cache;
        assert!(block_on(cache.spill.request_spill(usize::MAX)) > 0);
        let stats = cache.stats();
        assert_eq!((stats.spills, stats.evictions), (0, 1));
        assert_eq!(cache.reserved(), 0);
        assert_eq!(ids(&df), (0..20_000).collect::<Vec<_>>());
    }
}
