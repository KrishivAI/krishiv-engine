// SPDX-License-Identifier: Apache-2.0
//! The session relation cache: `DataFrame.cache`, `persist` and `unpersist`.
//!
//! Registration is eager, materialization lazy (as Spark's `cacheQuery`):
//! `persist` queues the frame's plan, and every read of the registry
//! (`Session::register_pending_caches`) first analyzes the queued plans and
//! registers them by [`CacheKey`] with their storage level. `is_cached` and
//! `storage_level` answer from the registrations. Any action whose plan
//! contains a registered relation without data fills it: the whole plan as its
//! result streams, a proper subtree as a query of its own run first. One fill
//! per key runs at a time ([`FillClaim`]); a fill that is refused leaves the
//! registration, and the query runs uncached.
//!
//! A cached relation is one materialized snapshot: Arrow batches held under
//! the `relation_cache` memory pool (a pool consumer of that name), keyed by
//! the plan's [`CacheKey`] without table versions (normalized plan hash,
//! principal id, policy version, resolution configuration; [`key_of`]); the
//! versions of the tables it was filled from stay on the entry.
//! Materialization is lazy: the first action after `cache()` runs
//! the query and keeps its batches when the stream completes; later actions
//! substitute the cached snapshot for any matching (sub)plan (a subplan matches
//! when the versions its query pins of the tables the subplan reads equal those
//! it was filled from, else the next action refills it). Substitution
//! replaces the matching subtree by a scan of an in-memory table, the same
//! [`MemoryTable`] provider reads of created DataFrames use; no plan node is
//! added and nothing is shared between sessions (a recorded difference from
//! Spark's shared `CacheManager`: `docs/concepts/known-differences.md`).
//!
//! The cache is all-or-nothing. `krishiv.cache.maxBytes` (read from each
//! query's configuration) caps the bytes it holds in memory: a snapshot of a
//! memory-only level that would exceed it is not stored and the query runs
//! uncached, never failing; a level that uses disk is written to disk instead
//! (design P3-T083 §3.2). `DISK_ONLY` levels are written to disk when they
//! fill. A disk entry is one spill file in a spill scope of its own (the
//! runtime's spill directory, under `krishiv.spill.maxBytes`), read back by
//! [`SpilledTable`] a batch at a time under the reading query's memory; a
//! refused write stores nothing. Under memory pressure the arbitrator asks
//! the `relation_cache` consumer to spill ([`tier::CacheSpill`]): entries no
//! running query reads, least recently used first, move to disk
//! (`MEMORY_AND_DISK`) or are evicted (`MEMORY_ONLY`: the registration stays
//! and the next use refills it). A snapshot a query reads keeps its pool
//! bytes until that query ends, whatever happens to the entry.
//!
//! Entries are dropped by `unpersist`, `CLEAR CACHE`, `UNCACHE TABLE`, `REFRESH TABLE`
//! and `REFRESH path`, by the catalog calls behind them, and by the engine's
//! own writes: a successful file write drops the entries reading files. An
//! external change to a cached path is deliberately invisible: the key
//! carries no file version, as in Spark. Replacing or dropping a temporary
//! table or view drops the entries that read it, and an invalidation that
//! runs while a fill is in flight keeps that fill from being stored.
//! A data invalidation (a write, `REFRESH`) keeps the
//! registrations, which refill on use; `CLEAR CACHE`, `UNCACHE TABLE` and
//! replacing or dropping a temporary view drop them too. Dropping an entry
//! drops its spill scope (the directory and its file) once no query reads it.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use krishiv_common::config::{CACHE_MAX_BYTES, Config, ConfigValue};
use krishiv_exec::planner::{MemoryTable, SpilledTable, TableProvider};
use krishiv_memory::{ConsumerId, MemoryPool, Reservation, Spillable};
use krishiv_plan::analyzer::AnalyzedPlan;
use krishiv_plan::analyzer::TableIdent;
use krishiv_plan::cache::{CacheKey, unversioned_subtree_hash};
use krishiv_plan::{Expr, ExprNode, LogicalPlan, PlanNode, ReadSource, Scan, TableVersion};
use krishiv_types::batch::Batch;

use crate::StorageLevel;
use crate::dataframe::DataFrame;
use crate::dataframe::checkpoint::stands_on_materializing_checkpoint;
use crate::session::Session;

mod tier;

mod checkpoint_stats;
mod level;
pub(crate) use checkpoint_stats::CheckpointStats;
pub(crate) use tier::CacheDisk;
use tier::{CacheSpill, HeldTable, PoolHold, Tier};

/// What a registration keeps alive beside its plan (table providers and the like).
type KeepAlive = Vec<Arc<dyn std::any::Any + Send + Sync>>;

/// What a frame's `persist` queued: the plan to analyze and register.
struct PendingReg {
    plan: Arc<LogicalPlan>,
    keep: Vec<Arc<dyn std::any::Any + Send + Sync>>,
    level: StorageLevel,
}

impl std::fmt::Debug for PendingReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingReg")
            .field("level", &self.level)
            .finish_non_exhaustive()
    }
}

/// A queued `persist` one reader took off the queue. Dropped before it is
/// registered (the reader's future was dropped), it goes back to the front of
/// the queue; counted in flight, dropping it ends that and wakes the readers
/// waiting for it.
struct TakenReg<'a> {
    cache: &'a RelationCache,
    item: Option<PendingReg>,
    counted: bool,
}

/// Queued registrations a drain skipped; put back at the front, in order, when it ends.
struct DeferredRegs<'a> {
    cache: &'a RelationCache,
    items: Vec<PendingReg>,
}

impl Drop for DeferredRegs<'_> {
    fn drop(&mut self) {
        if self.items.is_empty() {
            return;
        }
        let mut state = self
            .cache
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        for item in self.items.drain(..).rev() {
            state.pending.push_front(item);
        }
    }
}

impl TakenReg<'_> {
    /// Counts the item in flight: readers that find the queue empty wait for it.
    fn count(&mut self) {
        self.cache
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .in_flight += 1;
        self.counted = true;
    }
}

impl Drop for TakenReg<'_> {
    fn drop(&mut self) {
        let mut state = self
            .cache
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let back = self.item.take();
        let wake = back.is_some() || self.counted;
        if let Some(item) = back {
            state.pending.push_front(item);
        }
        if self.counted {
            state.in_flight = state.in_flight.saturating_sub(1);
        }
        drop(state);
        if wake {
            self.cache.registered.notify_waiters();
        }
    }
}

/// A plan that was asked to be cached, with the level it was first asked at.
struct Registration {
    level: StorageLevel,
    /// The unanalyzed plan: a fill of it as a subtree runs this as a query.
    plan: Arc<LogicalPlan>,
    keep: Vec<Arc<dyn std::any::Any + Send + Sync>>,
    /// The table names (lower-case last parts) the plan reads.
    tables: Vec<String>,
    /// Whether a fill of this key is running (see [`FillClaim`]).
    filling: bool,
    /// Which registration of its key this is (set by `register`): a claim
    /// frees only the slot of the registration it was taken on.
    epoch: u64,
}

impl std::fmt::Debug for Registration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Registration")
            .field("level", &self.level)
            .field("tables", &self.tables)
            .field("filling", &self.filling)
            .finish_non_exhaustive()
    }
}

/// The right to fill one registered key: only one fill per key runs. Dropping
/// it (the fill ended, failed or was cancelled) frees the slot.
#[derive(Debug)]
pub(crate) struct FillClaim {
    cache: Arc<RelationCache>,
    key: CacheKey,
    /// The registration it was taken on ([`Registration::epoch`]).
    epoch: u64,
}

impl Drop for FillClaim {
    fn drop(&mut self) {
        let mut state = self
            .cache
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(reg) = state.registrations.get_mut(&self.key)
            && reg.epoch == self.epoch
        {
            reg.filling = false;
        }
    }
}

/// The prefix of the hidden table names cached snapshots are scanned as. No
/// user input spells them: reads of one behave as missing tables, and
/// creating a temporary table or view under one fails with
/// `KRISHIV_RESERVED_TEMP_VIEW_NAME`, as for snapshot pins.
pub(crate) const CACHE_PREFIX: &str = "__krishiv_cache_";

/// Whether `name` is a hidden cache snapshot name: it starts with
/// [`CACHE_PREFIX`], in any letter case.
pub(crate) fn is_cache_name(name: &str) -> bool {
    name.get(..CACHE_PREFIX.len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(CACHE_PREFIX))
}

/// A cached relation: one materialized snapshot.
#[derive(Debug)]
pub(crate) struct CacheEntry {
    /// The hidden table name its scans read.
    pub(crate) name: String,
    /// The snapshot, in memory or on disk.
    tier: Tier,
    /// The level it was registered at: whether pressure spills or evicts it.
    level: StorageLevel,
    /// Bytes of the snapshot in memory (0 on disk), counted against the cap.
    pub(crate) bytes: usize,
    /// Bytes of its spill file (0 in memory).
    disk_bytes: u64,
    /// The cache clock at its last use, for least-recently-used spills.
    last_used: AtomicU64,
    /// Whether the cached plan reads files by path.
    pub(crate) reads_files: bool,
    /// The table names (lower-case last parts) the cached plan reads. Pins
    /// only cover versioned tables; temporary tables leave no pins, so the
    /// entry carries what `REFRESH TABLE` / `UNCACHE TABLE` match on.
    pub(crate) tables: Vec<String>,
    /// The versions of the tables it was filled from: a query matches it only
    /// when its own pins of those tables are the same.
    pins: BTreeMap<TableIdent, TableVersion>,
}

impl CacheEntry {
    /// Whether the entry holds the data of `key`: filled from the versions it pins.
    fn holds(&self, key: &CacheKey) -> bool {
        self.pins == key.pins
    }
}

/// The registry's key of `key`: the same without the table versions.
/// Registrations, entries and hidden names are kept under it, so a write to a
/// cached table keeps the registration (Spark's `recacheByPlan`, design §3.1);
/// the versions an entry was filled from stay on the entry.
fn unpinned(key: &CacheKey) -> CacheKey {
    CacheKey {
        pins: BTreeMap::new(),
        ..key.clone()
    }
}

/// The relation-cache key of `node`, a (sub)plan of `analyzed`: its hash
/// without table versions, and the pins of the tables it reads (not those of
/// the whole query: a derived plan reading another versioned table still
/// matches a cached subtree).
pub(crate) fn key_of(node: &Arc<LogicalPlan>, analyzed: &AnalyzedPlan) -> CacheKey {
    let pins = if analyzed.pins.is_empty() {
        BTreeMap::new()
    } else {
        let tables = read_tables(node);
        analyzed
            .pins
            .iter()
            .filter(|(id, _)| tables.binary_search(&id.name.to_lowercase()).is_ok())
            .map(|(id, v)| (id.clone(), *v))
            .collect()
    };
    CacheKey {
        plan: unversioned_subtree_hash(node),
        pins,
        principal: analyzed.context.principal.clone(),
        policy: analyzed.context.policy,
        conf: analyzed.context.conf,
    }
}

/// The relations a session has cached, by key and by hidden scan name.
#[derive(Debug, Default)]
struct CacheState {
    entries: HashMap<CacheKey, Arc<CacheEntry>>,
    /// The plans asked to be cached, by key; an entry is the data of one.
    registrations: HashMap<CacheKey, Registration>,
    /// `persist` calls not yet analyzed and registered, in order.
    pending: VecDeque<PendingReg>,
    /// Queued `persist`s a reader took off `pending` and is analyzing now
    /// (their lazy checkpoints taken): another reader waits for them.
    in_flight: usize,
    names: HashMap<String, CacheKey>,
    /// Exact bytes held, for the cap: pool grants are quantised, so the
    /// pool's own counter would over-count small snapshots.
    used: usize,
}

/// What a fill was started under: [`RelationCache::insert`] refuses the
/// snapshot if an invalidation ran after `generation` or it exceeds `cap`,
/// and takes over the bytes the fill reserved from the pool.
#[derive(Debug)]
pub(crate) struct FillLimits {
    /// The cache generation before the fill's query was analyzed.
    pub(crate) generation: u64,
    /// The fill query's own `krishiv.cache.maxBytes`.
    pub(crate) cap: usize,
    /// The bytes held while the result streams.
    pub(crate) reservation: FillReservation,
}

/// A filled snapshot for [`RelationCache::store`].
#[derive(Debug)]
pub(crate) struct Snapshot {
    pub(crate) key: CacheKey,
    pub(crate) schema: SchemaRef,
    pub(crate) batches: Vec<RecordBatch>,
    /// Whether the cached plan reads files by path.
    pub(crate) reads_files: bool,
    /// The table names (lower-case last parts) the cached plan reads.
    pub(crate) tables: Vec<String>,
}

/// An entry about to be installed.
struct Placed {
    tier: Tier,
    level: StorageLevel,
    bytes: usize,
    disk_bytes: u64,
    reads_files: bool,
    tables: Vec<String>,
}

/// Bytes a running fill holds in the `relation_cache` pool consumer, so the
/// batches it buffers count against `krishiv.memory.limit` while the query
/// runs. Released on drop: an abandoned, failed or cancelled fill, or an
/// `insert` that refuses the snapshot, gives them back. A stored snapshot
/// keeps them ([`Self::adopt`]).
#[derive(Debug)]
pub(crate) struct FillReservation {
    pool: Arc<dyn MemoryPool>,
    consumer: ConsumerId,
    bytes: usize,
}

impl FillReservation {
    /// Reserves `bytes` more; `false` (nothing reserved) when the pool refuses.
    pub(crate) fn grow(&mut self, bytes: usize) -> bool {
        if self.pool.try_reserve(self.consumer, bytes).is_err() {
            return false;
        }
        self.bytes = self.bytes.saturating_add(bytes);
        true
    }

    /// The bytes held, handed over to the stored entry instead of released.
    fn adopt(mut self) -> usize {
        std::mem::take(&mut self.bytes)
    }
}

impl Drop for FillReservation {
    fn drop(&mut self) {
        if self.bytes > 0 {
            self.pool.release(self.consumer, self.bytes);
        }
    }
}

/// What `system.caches` shows of the relation cache.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RelationStats {
    /// Entries holding data, in memory or on disk.
    pub(crate) entries: u64,
    /// Registrations holding no data (not filled yet, being filled, or evicted).
    pub(crate) entries_pending: u64,
    pub(crate) memory_bytes: u64,
    pub(crate) disk_bytes: u64,
    /// Entries moved to disk, and entries dropped, under memory pressure.
    pub(crate) spills: u64,
    pub(crate) evictions: u64,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) invalidations: u64,
}

/// Where a table change reaches every cache of an engine runtime: its result
/// cache and each session's relation cache. The permanent catalog is the
/// runtime's (design P2-E1), so a write in one session must drop what any
/// session cached over the table, as Spark's shared `CacheManager` does.
/// Relation caches are held weakly: a session's cache ends with it.
#[derive(Debug)]
pub(crate) struct Invalidation {
    results: Arc<crate::result_cache::ResultCache>,
    relations: Mutex<Vec<std::sync::Weak<RelationCache>>>,
}

impl Invalidation {
    /// A hub over the runtime's `results`, with no relation cache yet.
    pub(crate) fn new(results: Arc<crate::result_cache::ResultCache>) -> Arc<Self> {
        Arc::new(Self {
            results,
            relations: Mutex::new(Vec::new()),
        })
    }

    /// The live relation caches (copied, so none is called under the lock).
    fn relations(&self) -> Vec<Arc<RelationCache>> {
        let mut all = self
            .relations
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        all.retain(|w| w.strong_count() > 0);
        all.iter().filter_map(std::sync::Weak::upgrade).collect()
    }

    /// Drops what every cache holds over the table `name`.
    pub(crate) fn invalidate_table(&self, name: &str) {
        self.results.invalidate_table(name);
        for cache in self.relations() {
            cache.invalidate_local(name);
        }
    }

    /// Drops the registrations and entries over the table `name` of every cache.
    pub(crate) fn forget_table(&self, name: &str) {
        self.results.invalidate_table(name);
        for cache in self.relations() {
            cache.forget_local(name);
        }
    }

    /// Drops every entry of every cache (`CLEAR CACHE`).
    pub(crate) fn clear(&self) {
        self.results.clear();
        for cache in self.relations() {
            cache.clear_local();
        }
    }
}

/// The session relation cache: snapshots under the `relation_cache` pool.
#[derive(Debug)]
pub(crate) struct RelationCache {
    /// The runtime's invalidation hub: a change seen here reaches every
    /// cache of the runtime.
    hub: Arc<Invalidation>,
    pool: Arc<dyn MemoryPool>,
    consumer: ConsumerId,
    hidden: AtomicU64,
    /// Bumped by every invalidation, so a fill that started before one is
    /// not stored after it (it may hold rows the invalidation meant to drop).
    generation: AtomicU64,
    /// Queries whose plan found a cached subplan, and queries that looked
    /// while the cache held entries but found none.
    hits: AtomicU64,
    misses: AtomicU64,
    /// Ticks at every store and hit: the entries' recency.
    clock: AtomicU64,
    spills: AtomicU64,
    evictions: AtomicU64,
    /// The next [`Registration::epoch`].
    epochs: AtomicU64,
    /// The session's checkpoints (reported in `system.caches`; not entries of this cache).
    pub(crate) checkpoints: Arc<CheckpointStats>,
    /// The consumer's spill hook (the pool holds it weakly).
    spill: Arc<CacheSpill>,
    /// Where disk entries go; `None` keeps every entry in memory (unit tests).
    disk: Option<CacheDisk>,
    state: Mutex<CacheState>,
    /// Notified when a queued `persist` taken off `pending` is registered
    /// (or dropped), so a reader that finds one in flight waits for it
    /// instead of missing it. No lock is held while a queued plan is
    /// analyzed: analysis may run queries (a lazy checkpoint takes its rows),
    /// and they read the registry too.
    registered: tokio::sync::Notify,
}

impl RelationCache {
    /// An empty cache under `memory` with a hub of its own (tests: sessions
    /// use [`Self::on`]).
    #[cfg(test)]
    pub(crate) fn new(memory: &Arc<dyn MemoryPool>) -> Arc<Self> {
        Self::on(
            memory,
            &Invalidation::new(Arc::clone(crate::result_cache::global())),
            None,
        )
    }

    /// An empty cache under `memory`, registered with the runtime's `hub`,
    /// writing disk entries through `disk`.
    pub(crate) fn on(
        memory: &Arc<dyn MemoryPool>,
        hub: &Arc<Invalidation>,
        disk: Option<CacheDisk>,
    ) -> Arc<Self> {
        let pool = memory.child("relation_cache", None);
        let spill = CacheSpill::new(disk.as_ref().map(|d| Arc::clone(&d.executor)));
        let consumer = pool.register(
            "relation_cache",
            Some(Arc::clone(&spill) as Arc<dyn Spillable>),
        );
        let cache = Arc::new(Self {
            hub: Arc::clone(hub),
            pool,
            consumer,
            hidden: AtomicU64::new(0),
            generation: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            clock: AtomicU64::new(0),
            spills: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            epochs: AtomicU64::new(0),
            checkpoints: Arc::default(),
            spill: Arc::clone(&spill),
            disk,
            state: Mutex::new(CacheState::default()),
            registered: tokio::sync::Notify::new(),
        });
        let _ = spill.cache.set(Arc::downgrade(&cache));
        hub.relations
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Arc::downgrade(&cache));
        cache
    }

    /// The `krishiv.cache.maxBytes` cap of a query's configuration snapshot.
    pub(crate) fn cap_of(config: &Config) -> usize {
        let cap = config
            .get(&CACHE_MAX_BYTES)
            .and_then(ConfigValue::as_size)
            .unwrap_or(u64::MAX);
        usize::try_from(cap).unwrap_or(usize::MAX)
    }

    /// An empty reservation in the cache's pool consumer, for one fill.
    pub(crate) fn reserve_fill(&self) -> FillReservation {
        FillReservation {
            pool: Arc::clone(&self.pool),
            consumer: self.consumer,
            bytes: 0,
        }
    }

    /// The invalidation generation: a fill captures it when it starts and
    /// [`Self::insert`] refuses the snapshot if an invalidation ran since.
    pub(crate) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn invalidated(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    /// The entry for `key`, if cached.
    pub(crate) fn lookup(&self, key: &CacheKey) -> Option<Arc<CacheEntry>> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entries
            .get(&unpinned(key))
            .filter(|e| e.holds(key))
            .cloned()
    }

    /// The snapshot the hidden table `name` reads, if cached.
    pub(crate) fn table(&self, name: &str) -> Option<Arc<dyn TableProvider>> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let key = state.names.get(name)?;
        state.entries.get(key).map(|e| e.tier.provider())
    }

    /// Whether the hidden table `name` is a snapshot in a spill file (a driver-local file).
    pub(crate) fn is_on_disk(&self, name: &str) -> bool {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state
            .names
            .get(name)
            .and_then(|key| state.entries.get(key))
            .is_some_and(|e| matches!(e.tier, Tier::Disk(_)))
    }

    fn tick(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed)
    }

    /// Publishes the bytes a spill could free now: in-memory entries no
    /// query reads. Called after every change of the entries and when a
    /// query lets go of the snapshots it read.
    fn publish_spillable(&self, state: &CacheState) {
        let bytes = state
            .entries
            .values()
            .filter(|e| e.tier.unread_in_memory())
            .map(|e| e.bytes)
            .fold(0_usize, usize::saturating_add);
        self.spill.signal.set_spillable_bytes(bytes);
    }

    /// A query let go of (or took) the snapshots it reads.
    pub(crate) fn readers_changed(&self) {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.publish_spillable(&state);
    }

    /// Queues `plan` to be registered at `level` by the next read of the
    /// registry. [`StorageLevel::NONE`] queues nothing, as in Spark.
    pub(crate) fn request(
        &self,
        plan: Arc<LogicalPlan>,
        keep: Vec<Arc<dyn std::any::Any + Send + Sync>>,
        level: StorageLevel,
    ) {
        if level == StorageLevel::NONE {
            return;
        }
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pending
            .push_back(PendingReg { plan, keep, level });
    }

    /// Registers `reg` under `key`; `false` (nothing changes) when the plan
    /// is registered already: the first level wins.
    fn register(&self, key: &CacheKey, mut reg: Registration) -> bool {
        let key = unpinned(key);
        reg.epoch = self.epochs.fetch_add(1, Ordering::Relaxed);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.registrations.contains_key(&key) {
            return false;
        }
        state.registrations.insert(key, reg);
        true
    }

    /// The level `key` is registered at, if it is.
    pub(crate) fn level_of(&self, key: &CacheKey) -> Option<StorageLevel> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registrations
            .get(&unpinned(key))
            .map(|r| r.level)
    }

    /// The fill slot of `key`: `Some` when it is registered, holds no data
    /// of `key`'s table versions and no other fill runs.
    pub(crate) fn claim_fill(self: &Arc<Self>, key: &CacheKey) -> Option<FillClaim> {
        let reg_key = unpinned(key);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.entries.get(&reg_key).is_some_and(|e| e.holds(key)) {
            return None;
        }
        let reg = state.registrations.get_mut(&reg_key)?;
        if reg.filling {
            return None;
        }
        reg.filling = true;
        let epoch = reg.epoch;
        Some(FillClaim {
            cache: Arc::clone(self),
            key: reg_key,
            epoch,
        })
    }

    /// The registered nodes of `analyzed`, top-down: each node's key, whether
    /// it is the root, and whether it still needs a fill (no data, none
    /// running). Empty when nothing is registered.
    fn registered_nodes(&self, analyzed: &AnalyzedPlan) -> Vec<(CacheKey, bool, bool)> {
        if self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registrations
            .is_empty()
        {
            return Vec::new();
        }
        let root = Arc::clone(analyzed.plan.root());
        let mut found = Vec::new();
        let walked: Result<Arc<LogicalPlan>, Infallible> =
            Arc::clone(&root).transform_down(&mut |node| {
                let key = key_of(&node, analyzed);
                let reg_key = unpinned(&key);
                let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                if let Some(reg) = state.registrations.get(&reg_key) {
                    let needs_fill =
                        !reg.filling && !state.entries.get(&reg_key).is_some_and(|e| e.holds(&key));
                    found.push((key, Arc::ptr_eq(&node, &root), needs_fill));
                }
                drop(state);
                Ok(node)
            });
        drop(walked);
        found
    }

    /// The registered proper subtrees of `analyzed` that need a fill.
    fn unfilled_subtrees(&self, analyzed: &AnalyzedPlan) -> Vec<CacheKey> {
        self.registered_nodes(analyzed)
            .into_iter()
            .filter(|(_, root, needs_fill)| !root && *needs_fill)
            .map(|(key, _, _)| key)
            .collect()
    }

    /// The plan registered under `key` if it still needs a fill.
    fn fill_source(&self, key: &CacheKey) -> Option<(Arc<LogicalPlan>, KeepAlive)> {
        let reg_key = unpinned(key);
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let reg = state.registrations.get(&reg_key)?;
        (!reg.filling && !state.entries.get(&reg_key).is_some_and(|e| e.holds(key)))
            .then(|| (Arc::clone(&reg.plan), reg.keep.clone()))
    }

    /// Stores the snapshot under its key and returns the hidden table name
    /// its scans read. A memory level is kept in memory when it fits the
    /// `cap` (the fill query's own `krishiv.cache.maxBytes`), else refused; a
    /// level that uses disk is written to a spill file when it does not fit,
    /// or always when it does not use memory (`DISK_ONLY`). Stores nothing
    /// and returns `None` (the query ran uncached, it never fails) when the
    /// cap is 0 (caching disabled), an invalidation ran since the fill
    /// started (`generation`), the key is no longer registered, a disk write
    /// fails (the spill cap, the file system), or two of its columns share a
    /// name (a scan finds its columns by name, so it could not tell them
    /// apart). Replaces any entry already stored under the key.
    pub(crate) async fn store(
        self: Arc<Self>,
        snapshot: Snapshot,
        limits: FillLimits,
    ) -> Option<String> {
        // Any early return drops the reservation, which releases the bytes.
        let FillLimits {
            generation,
            cap,
            reservation,
        } = limits;
        let Snapshot {
            key,
            schema,
            batches,
            reads_files,
            tables,
        } = snapshot;
        let mut names = std::collections::HashSet::new();
        if !schema.fields().iter().all(|f| names.insert(f.name())) {
            return None;
        }
        let bytes = reservation.bytes;
        let batches: Vec<Batch> = batches.into_iter().map(Batch::new).collect();
        let level = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            // A cap of 0 disables caching, the empty result included. The
            // generation is read under the state lock, which every
            // invalidation also holds while it bumps it.
            if cap == 0 || generation != self.generation() {
                return None;
            }
            let reg_key = unpinned(&key);
            let level = state.registrations.get(&reg_key)?.level;
            let old = state.entries.get(&reg_key).map(|e| e.bytes).unwrap_or(0);
            let fits = state.used.saturating_sub(old).saturating_add(bytes) <= cap;
            let to_disk = level.use_disk && (!level.use_memory || !fits) && self.disk.is_some();
            if !to_disk {
                if !fits {
                    return None;
                }
                let held = HeldTable {
                    table: MemoryTable::new(Arc::clone(&schema), batches.clone()),
                    schema,
                    batches,
                    _hold: PoolHold {
                        pool: Arc::clone(&self.pool),
                        consumer: self.consumer,
                        bytes: reservation.adopt(),
                    },
                };
                let entry = Placed {
                    tier: Tier::Memory(Arc::new(held)),
                    level,
                    bytes,
                    disk_bytes: 0,
                    reads_files,
                    tables,
                };
                return Some(self.install(&mut state, &key, entry));
            }
            level
        };
        let disk = self.disk.as_ref()?;
        let scratch = Reservation::new(Arc::clone(&self.pool), self.consumer);
        let file = match disk.write(Arc::clone(&schema), batches, scratch).await {
            Ok(file) => file,
            Err(error) => {
                tracing::warn!(target: "krishiv::session", %error, "a cached relation could not be written to disk; it stays uncached");
                return None;
            }
        };
        // The rows are on disk: the memory the fill buffered goes back.
        drop(reservation);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if generation != self.generation() || !state.registrations.contains_key(&unpinned(&key)) {
            return None;
        }
        let entry = Placed {
            disk_bytes: file.bytes(),
            tier: Tier::Disk(Arc::new(SpilledTable::new(file))),
            level,
            bytes: 0,
            reads_files,
            tables,
        };
        Some(self.install(&mut state, &key, entry))
    }

    /// Puts `entry` under `key`'s registry key with `key`'s pins (replacing
    /// any entry there, which keeps its hidden name: plans substituted
    /// earlier still read the same name, and each holds the table it
    /// resolved) and returns the hidden name.
    fn install(&self, state: &mut CacheState, key: &CacheKey, entry: Placed) -> String {
        let pins = key.pins.clone();
        let key = unpinned(key);
        if let Some(old) = state.entries.get(&key) {
            state.used = state.used.saturating_sub(old.bytes);
        }
        let name = match state
            .names
            .iter()
            .find(|(_, k)| *k == &key)
            .map(|(n, _)| n.clone())
        {
            Some(name) => name,
            None => format!(
                "{CACHE_PREFIX}{}",
                self.hidden.fetch_add(1, Ordering::AcqRel)
            ),
        };
        state.used = state.used.saturating_add(entry.bytes);
        state.names.insert(name.clone(), key.clone());
        state.entries.insert(
            key,
            Arc::new(CacheEntry {
                name: name.clone(),
                tier: entry.tier,
                level: entry.level,
                bytes: entry.bytes,
                disk_bytes: entry.disk_bytes,
                last_used: AtomicU64::new(self.tick()),
                reads_files: entry.reads_files,
                tables: entry.tables,
                pins,
            }),
        );
        self.publish_spillable(state);
        name
    }

    /// Frees at least `target` bytes of the cache's memory if it can: the
    /// in-memory entries no query reads, least recently used first, move to
    /// disk (levels that use disk) or are evicted (memory-only levels, or a
    /// refused write). Returns the bytes freed. The writes run on the
    /// blocking pool; an entry dropped or replaced while it is written keeps
    /// what happened to it (the file is discarded).
    pub(crate) async fn spill_for(self: &Arc<Self>, target: usize) -> usize {
        let mut freed = 0_usize;
        let mut tried: std::collections::HashSet<String> = std::collections::HashSet::new();
        while freed < target {
            let victim = {
                let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                state
                    .entries
                    .iter()
                    .filter(|(_, e)| e.tier.unread_in_memory() && !tried.contains(&e.name))
                    .min_by_key(|(_, e)| e.last_used.load(Ordering::Relaxed))
                    .map(|(k, e)| (k.clone(), Arc::clone(e)))
            };
            let Some((key, entry)) = victim else {
                break;
            };
            tried.insert(entry.name.clone());
            let Tier::Memory(held) = &entry.tier else {
                continue;
            };
            let file = match (&self.disk, entry.level.use_disk) {
                (Some(disk), true) => {
                    let scratch = Reservation::new(Arc::clone(&self.pool), self.consumer);
                    disk.write(Arc::clone(&held.schema), held.batches.clone(), scratch)
                        .await
                        .map_err(|error| {
                            tracing::warn!(target: "krishiv::session", %error, "a cached relation could not be spilled; it is evicted");
                        })
                        .ok()
                }
                _ => None,
            };
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            if !state
                .entries
                .get(&key)
                .is_some_and(|current| Arc::ptr_eq(current, &entry))
            {
                continue;
            }
            if entry.tier.unread_in_memory() {
                freed = freed.saturating_add(entry.bytes);
            }
            state.used = state.used.saturating_sub(entry.bytes);
            match file {
                Some(file) => {
                    self.spills.fetch_add(1, Ordering::Relaxed);
                    let disk = Arc::new(CacheEntry {
                        name: entry.name.clone(),
                        disk_bytes: file.bytes(),
                        tier: Tier::Disk(Arc::new(SpilledTable::new(file))),
                        level: entry.level,
                        bytes: 0,
                        last_used: AtomicU64::new(entry.last_used.load(Ordering::Relaxed)),
                        reads_files: entry.reads_files,
                        tables: entry.tables.clone(),
                        pins: entry.pins.clone(),
                    });
                    state.entries.insert(key, disk);
                }
                None => {
                    self.evictions.fetch_add(1, Ordering::Relaxed);
                    state.entries.remove(&key);
                    state.names.retain(|_, k| k != &key);
                }
            }
            self.publish_spillable(&state);
            drop(state);
            // The last holder of the snapshot: its pool bytes go back now.
            drop(entry);
        }
        freed
    }

    /// Drops the registration of `key` and its entry; whether one existed.
    pub(crate) fn remove(&self, key: &CacheKey) -> bool {
        let key = &unpinned(key);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.invalidated();
        let registration = state.registrations.remove(key);
        let Some(entry) = state.entries.remove(key) else {
            drop(state);
            return registration.is_some();
        };
        state.names.retain(|_, k| k != key);
        state.used = state.used.saturating_sub(entry.bytes);
        self.publish_spillable(&state);
        drop(state);
        // Dropped after the lock: its pool bytes (or its spill scope, whose
        // directory is deleted) go now, unless a running query still reads it.
        drop((entry, registration));
        true
    }

    /// Drops every entry of every cache of the runtime (`CLEAR CACHE`).
    pub(crate) fn clear(&self) {
        self.hub.clear();
    }

    /// Drops every registration and entry of this cache (its session closed).
    pub(crate) fn close(&self) {
        self.clear_local();
    }

    /// Drops every entry of this cache.
    fn clear_local(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.invalidated();
        let dropped = (
            std::mem::take(&mut state.entries),
            std::mem::take(&mut state.registrations),
            std::mem::take(&mut state.pending),
        );
        state.names.clear();
        state.used = 0;
        self.publish_spillable(&state);
        drop(state);
        // After the lock: dropping entries deletes their spill directories.
        drop(dropped);
    }

    /// Drops the entries reading the table `name` (any spelling): what they
    /// read may have changed. Matches the entry's tables and its pins, so
    /// versioned and temporary tables both clear.
    pub(crate) fn invalidate_table(&self, name: &str) {
        self.hub.invalidate_table(name);
    }

    /// [`Self::invalidate_table`] in this cache only.
    fn invalidate_local(&self, name: &str) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.invalidated();
        let lowered = name.to_lowercase();
        let short = lowered.rsplit('.').next().unwrap_or(&lowered);
        let matches = |table: &str| table == lowered || table == short;
        let stale: Vec<CacheKey> = state
            .entries
            .iter()
            .filter(|(_, entry)| {
                entry.tables.iter().any(|t| matches(t))
                    || entry.pins.keys().any(|id| matches(&id.name.to_lowercase()))
            })
            .map(|(key, _)| key.clone())
            .collect();
        let mut dropped = Vec::with_capacity(stale.len());
        for key in &stale {
            if let Some(entry) = state.entries.remove(key) {
                state.used = state.used.saturating_sub(entry.bytes);
                dropped.push(entry);
            }
        }
        state.names.retain(|_, k| !stale.contains(k));
        self.publish_spillable(&state);
        drop(state);
        // After the lock: dropping entries deletes their spill directories.
        drop(dropped);
    }

    /// Drops the registrations and entries reading the table `name` (any
    /// spelling) in every cache of the runtime, and the queued `persist`s that
    /// read it: `UNCACHE TABLE`, or the table or temporary view was replaced or
    /// dropped, so the plans no longer resolve to it.
    pub(crate) fn forget_table(&self, name: &str) {
        self.hub.forget_table(name);
    }

    /// [`Self::forget_table`] in this cache only.
    fn forget_local(&self, name: &str) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.invalidated();
        let lowered = name.to_lowercase();
        let short = lowered.rsplit('.').next().unwrap_or(&lowered);
        let matches = |table: &str| table == lowered || table == short;
        let stale: Vec<CacheKey> = state
            .registrations
            .iter()
            .filter(|(_, reg)| reg.tables.iter().any(|t| matches(t)))
            .map(|(key, _)| key.clone())
            .collect();
        let mut entries = Vec::with_capacity(stale.len());
        let mut registrations = Vec::with_capacity(stale.len());
        for key in &stale {
            registrations.extend(state.registrations.remove(key));
            if let Some(entry) = state.entries.remove(key) {
                state.used = state.used.saturating_sub(entry.bytes);
                entries.push(entry);
            }
        }
        state.names.retain(|_, k| !stale.contains(k));
        self.publish_spillable(&state);
        let (queued, kept): (VecDeque<PendingReg>, VecDeque<PendingReg>) =
            std::mem::take(&mut state.pending)
                .into_iter()
                .partition(|p| unresolved_tables(&p.plan).iter().any(|t| matches(t)));
        state.pending = kept;
        drop(state);
        // After the lock: dropping entries deletes their spill directories.
        drop((entries, registrations, queued));
    }

    /// Drops the entries reading files by path: the engine wrote files, so
    /// what they read may have changed. An external change drops nothing.
    pub(crate) fn invalidate_files(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.invalidated();
        let stale: Vec<CacheKey> = state
            .entries
            .iter()
            .filter(|(_, entry)| entry.reads_files)
            .map(|(key, _)| key.clone())
            .collect();
        let mut dropped = Vec::with_capacity(stale.len());
        for key in &stale {
            if let Some(entry) = state.entries.remove(key) {
                state.used = state.used.saturating_sub(entry.bytes);
                dropped.push(entry);
            }
        }
        state.names.retain(|_, k| !stale.contains(k));
        self.publish_spillable(&state);
        drop(state);
        // After the lock: dropping entries deletes their spill directories.
        drop(dropped);
    }

    /// The cache's size and counters now, for `system.caches`.
    pub(crate) fn stats(&self) -> RelationStats {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let pending = state
            .registrations
            .keys()
            .filter(|k| !state.entries.contains_key(*k))
            .count();
        RelationStats {
            entries: state.entries.len() as u64,
            entries_pending: pending as u64,
            memory_bytes: state.used as u64,
            disk_bytes: state.entries.values().map(|e| e.disk_bytes).sum(),
            spills: self.spills.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            invalidations: self.generation.load(Ordering::Acquire),
        }
    }

    /// Bytes reserved from the pool for the cache, rounded up to the pool's
    /// grant size (the cap is checked against exact bytes).
    pub(crate) fn reserved(&self) -> usize {
        self.pool.reserved()
    }

    /// The analyzed plan with every cached (sub)plan replaced by a scan of
    /// its snapshot: the same [`MemoryTable`] provider reads of created
    /// DataFrames use. Returns `None` when nothing matched; else also the
    /// snapshots the plan now scans, by hidden name, so the query keeps them
    /// even if the cache is cleared before it is planned.
    pub(crate) fn substitute(&self, analyzed: &AnalyzedPlan) -> Option<Substituted> {
        self.substitute_counting(analyzed, true)
    }

    /// Whether the plan contains a registered relation, with or without data (it reads, or
    /// fills, this session's cache): a probe (is this plan eligible for the shared result
    /// cache?), not a lookup of the execution.
    pub(crate) fn would_substitute(&self, analyzed: &AnalyzedPlan) -> bool {
        !self.registered_nodes(analyzed).is_empty()
    }

    fn substitute_counting(&self, analyzed: &AnalyzedPlan, count: bool) -> Option<Substituted> {
        // Nothing cached: skip hashing every node of the plan.
        if self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entries
            .is_empty()
        {
            return None;
        }
        let root = Arc::clone(analyzed.plan.root());
        let mut tables: Vec<(String, Arc<dyn TableProvider>)> = Vec::new();
        let out: Arc<LogicalPlan> = Arc::clone(&root)
            .transform_down(&mut |node| {
                let key = key_of(&node, analyzed);
                let found = {
                    let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                    let entry = state.entries.get(&unpinned(&key)).filter(|e| e.holds(&key));
                    entry.map(|e| {
                        e.last_used.store(self.tick(), Ordering::Relaxed);
                        (e.name.clone(), e.tier.provider())
                    })
                };
                let Some((name, table)) = found else {
                    return Ok::<_, Infallible>(node);
                };
                let Some(output) = node.resolved_output() else {
                    return Ok::<_, Infallible>(node);
                };
                tables.push((name.clone(), table));
                Ok(Arc::new(
                    LogicalPlan::new(PlanNode::Scan(Scan {
                        source: ReadSource::Table {
                            name: vec![name],
                            options: BTreeMap::new(),
                        },
                        output,
                        version: None,
                    }))
                    .with_origin(node.origin),
                ))
            })
            .unwrap_or(Arc::clone(&root));
        if Arc::ptr_eq(&out, &root) {
            if count {
                self.misses.fetch_add(1, Ordering::Relaxed);
            }
            None
        } else {
            if count {
                self.hits.fetch_add(1, Ordering::Relaxed);
            }
            Some(Substituted { plan: out, tables })
        }
    }
}

/// A plan with cached subplans replaced by scans, and the snapshots those
/// scans read.
pub(crate) struct Substituted {
    pub(crate) plan: Arc<LogicalPlan>,
    pub(crate) tables: Vec<(String, Arc<dyn TableProvider>)>,
}

/// The snapshots one action reads, held until it ends: its query's catalog
/// keeps them pinned by hidden name (planning and instantiation resolve the
/// names there), and the snapshots stay alive, so an entry spilled, evicted
/// or dropped meanwhile keeps its pool bytes (or its file) until the action
/// lets go. Dropping the lease unpins them and republishes what the cache
/// could spill.
#[derive(Default)]
pub(crate) struct SnapshotLease {
    catalog: Option<Arc<crate::catalog::QueryCatalog>>,
    names: Vec<String>,
    tables: Vec<Arc<dyn TableProvider>>,
    cache: std::sync::Weak<RelationCache>,
}

impl SnapshotLease {
    /// Pins `tables` (by hidden name) in `catalog` for one action.
    pub(crate) fn pin(
        catalog: &Arc<crate::catalog::QueryCatalog>,
        cache: &Arc<RelationCache>,
        tables: Vec<(String, Arc<dyn TableProvider>)>,
    ) -> Self {
        let mut lease = Self {
            catalog: Some(Arc::clone(catalog)),
            names: Vec::with_capacity(tables.len()),
            tables: Vec::with_capacity(tables.len()),
            cache: Arc::downgrade(cache),
        };
        for (name, table) in tables {
            catalog.pin_cached(name.clone(), Arc::clone(&table));
            lease.names.push(name);
            lease.tables.push(table);
        }
        cache.readers_changed();
        lease
    }
}

impl Drop for SnapshotLease {
    fn drop(&mut self) {
        if let Some(catalog) = &self.catalog {
            for name in &self.names {
                catalog.unpin_cached(name);
            }
        }
        self.tables.clear();
        if let Some(cache) = self.cache.upgrade() {
            cache.readers_changed();
        }
    }
}

/// The bytes of `batch` that are its own: slices of a shared buffer count
/// their slice, not the whole buffer, so a table scanned in many morsels is
/// not counted many times over.
pub(crate) fn batch_bytes(batch: &RecordBatch) -> usize {
    batch
        .columns()
        .iter()
        .map(|c| {
            c.to_data()
                .get_slice_memory_size()
                .unwrap_or_else(|_| c.get_array_memory_size())
        })
        .sum()
}

/// The table names (lower-case last parts) the resolved `plan` reads, for
/// [`RelationCache::invalidate_table`]: every scan's table, subquery plans
/// included.
pub(crate) fn read_tables(plan: &Arc<LogicalPlan>) -> Vec<String> {
    enum Step<'a> {
        Plan(&'a Arc<LogicalPlan>),
        Expr(&'a Arc<Expr>),
    }
    let mut out = Vec::new();
    let mut stack = vec![Step::Plan(plan)];
    // `plan` outlives the walk; every pushed reference borrows from it or
    // from a reference already on the stack.
    while let Some(step) = stack.pop() {
        match step {
            Step::Plan(p) => {
                if let PlanNode::Scan(s) = &p.node
                    && let ReadSource::Table { name, .. } = &s.source
                    && let Some(last) = name.last()
                {
                    out.push(last.to_lowercase());
                }
                p.for_each_child(&mut |c| stack.push(Step::Plan(c)));
                p.for_each_expr(&mut |e| stack.push(Step::Expr(e)));
            }
            Step::Expr(e) => {
                if let ExprNode::Subquery { plan, .. } = &e.node {
                    stack.push(Step::Plan(plan));
                }
                e.for_each_child(&mut |c| stack.push(Step::Expr(c)));
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// The table names (lower-case last parts) the unresolved `plan` reads:
/// what `REFRESH TABLE` / `UNCACHE TABLE` name. A view reads as its own
/// name here while resolving to its snapshot pin, so entries over views
/// clear by the view's name too.
pub(crate) fn unresolved_tables(plan: &Arc<LogicalPlan>) -> Vec<String> {
    enum Step<'a> {
        Plan(&'a Arc<LogicalPlan>),
        Expr(&'a Arc<Expr>),
    }
    let mut out = Vec::new();
    let mut stack = vec![Step::Plan(plan)];
    while let Some(step) = stack.pop() {
        match step {
            Step::Plan(p) => {
                if let PlanNode::Read(ReadSource::Table { name, .. }) = &p.node
                    && let Some(last) = name.last()
                {
                    out.push(last.to_lowercase());
                }
                p.for_each_child(&mut |c| stack.push(Step::Plan(c)));
                p.for_each_expr(&mut |e| stack.push(Step::Expr(e)));
            }
            Step::Expr(e) => {
                if let ExprNode::Subquery { plan, .. } = &e.node {
                    stack.push(Step::Plan(plan));
                }
                e.for_each_child(&mut |c| stack.push(Step::Expr(c)));
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

impl Session {
    /// Analyzes the `persist` calls queued since the last read of the registry and registers
    /// them, in order (Spark registers at the call; analysis is asynchronous here, so every
    /// reader of the registry catches up first). A plan that fails analysis is dropped with a
    /// warning: its frame raises the error at its own first action.
    ///
    /// Items are taken one at a time and no lock is held across an analysis: taking a lazy
    /// checkpoint's rows runs a query, which reads the registry again (it skips the item being
    /// registered, which is off the queue and not yet counted in flight). A reader that finds
    /// the queue empty while another analyzes an item waits for that item. Dropping the future
    /// puts the item it holds back at the front of the queue.
    pub(crate) async fn register_pending_caches(&self) {
        let cache = &self.state.cache;
        // Items standing on a lazy checkpoint that is taking its rows (this drain runs inside that
        // query): registering them would wait for the rows. They go back to the queue when the
        // drain ends; the action that takes the rows registers them afterwards.
        let mut deferred = DeferredRegs {
            cache,
            items: Vec::new(),
        };
        loop {
            // Created before the state is read: a registration that ends after the read wakes it.
            let notified = cache.registered.notified();
            let next = {
                let mut state = cache.state.lock().unwrap_or_else(PoisonError::into_inner);
                let position = state
                    .pending
                    .iter()
                    .position(|p| !stands_on_materializing_checkpoint(&p.keep));
                match position.and_then(|i| state.pending.remove(i)) {
                    Some(item) => Some(item),
                    None if state.in_flight == 0 => {
                        deferred.items = std::mem::take(&mut state.pending).into();
                        return;
                    }
                    None => None,
                }
            };
            match next {
                Some(item) => {
                    self.register_one(TakenReg {
                        cache,
                        item: Some(item),
                        counted: false,
                    })
                    .await;
                }
                None => notified.await,
            }
        }
    }

    /// Analyzes one queued `persist` and registers it.
    async fn register_one(&self, mut taken: TakenReg<'_>) {
        let Some(pending) = taken.item.as_ref() else {
            return;
        };
        let frame = DataFrame::over_plan(
            self.clone(),
            Arc::clone(&pending.plan),
            pending.keep.clone(),
        );
        // The lazy checkpoints first, not counted in flight: their queries read the registry
        // and must not wait for this item.
        if let Err(error) = frame.plan_with_checkpoints().await {
            taken.item = None;
            tracing::warn!(target: "krishiv::session", %error, "Asked to cache a plan that does not analyze");
            return;
        }
        taken.count();
        let analysis = match frame.analysis().await {
            Ok(a) => a,
            Err(error) => {
                taken.item = None;
                tracing::warn!(target: "krishiv::session", %error, "Asked to cache a plan that does not analyze");
                return;
            }
        };
        let key = key_of(analysis.analyzed.plan.root(), &analysis.analyzed);
        let mut tables = read_tables(analysis.analyzed.plan.root());
        tables.extend(unresolved_tables(&frame.plan));
        tables.sort();
        tables.dedup();
        let Some(pending) = taken.item.take() else {
            return;
        };
        let registration = Registration {
            level: pending.level,
            plan: Arc::clone(&frame.plan),
            keep: frame.keep.clone(),
            tables,
            filling: false,
            epoch: 0,
        };
        if !self.state.cache.register(&key, registration) {
            tracing::warn!(target: "krishiv::session", "Asked to cache already cached data.");
        }
    }

    /// Runs, as queries of their own, the registered proper subtrees of `analyzed` that hold no
    /// data, so the plan reads their snapshots (Spark's "first use materializes"). A subtree whose
    /// fill is refused stays registered and runs inside the plan; one that fails is left to fail
    /// there.
    pub(crate) fn fill_registered_subtrees<'a>(
        &'a self,
        analyzed: &'a AnalyzedPlan,
    ) -> futures::future::BoxFuture<'a, ()> {
        Box::pin(async move {
            for key in self.state.cache.unfilled_subtrees(analyzed) {
                // Re-checked: an earlier fill (a registered subtree inside this one) may have
                // filled it, or another action is filling it now.
                let Some((plan, keep)) = self.state.cache.fill_source(&key) else {
                    continue;
                };
                let frame = DataFrame::over_plan(self.clone(), plan, keep);
                let Ok(mut stream) = frame.execute_stream().await else {
                    continue;
                };
                while let Some(Ok(_)) = futures::StreamExt::next(&mut stream).await {}
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use futures::FutureExt as _;
    use futures::executor::block_on;
    use krishiv_plan::TableVersion;
    use krishiv_plan::analyzer::{PolicyVersion, ResolutionConf, TableIdent};

    use super::{CacheKey, FillLimits, Registration, RelationCache, Snapshot};
    use crate::StorageLevel;
    use crate::session::Session;

    fn session() -> Session {
        let s = Session::new().unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        s.register_batches("t", schema, vec![batch]).unwrap();
        s
    }

    /// A key of the plan 7 over `default.src` at `version`.
    fn pinned(version: i64) -> CacheKey {
        CacheKey {
            plan: 7,
            pins: std::collections::BTreeMap::from([(
                TableIdent {
                    catalog: "spark_catalog".to_owned(),
                    namespace: vec!["default".to_owned()],
                    name: "src".to_owned(),
                },
                TableVersion(version),
            )]),
            principal: "p".to_owned(),
            policy: PolicyVersion(0),
            conf: ResolutionConf::default(),
        }
    }

    /// Fills `key` with one row, as an action's fill does.
    fn fill(cache: &Arc<RelationCache>, key: &CacheKey) -> Option<String> {
        let claim = cache.claim_fill(key)?;
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();
        let stored = block_on(Arc::clone(cache).store(
            Snapshot {
                key: key.clone(),
                schema,
                batches: vec![batch],
                reads_files: false,
                tables: vec!["src".to_owned()],
            },
            FillLimits {
                generation: cache.generation(),
                cap: usize::MAX,
                reservation: cache.reserve_fill(),
            },
        ));
        drop(claim);
        stored
    }

    /// Spark's `recacheByPlan` (design §3.1): a registration is not tied to the table versions
    /// it was filled under. After a write to its table the level stays, the next action
    /// refills it under the new versions, and no registration is left pending (review B2).
    #[test]
    fn a_registration_outlives_the_table_versions_it_was_filled_under() {
        let s = session();
        let cache = &s.state.cache;
        let reg = Registration {
            level: StorageLevel::MEMORY_ONLY,
            plan: Arc::clone(&s.table("t").plan),
            keep: Vec::new(),
            tables: vec!["src".to_owned()],
            filling: false,
            epoch: 0,
        };
        assert!(cache.register(&pinned(1), reg));
        assert!(fill(cache, &pinned(1)).is_some());
        assert!(cache.lookup(&pinned(1)).is_some());
        // A write to `src`: its version moves to 2.
        cache.invalidate_table("src");
        assert_eq!(cache.level_of(&pinned(2)), Some(StorageLevel::MEMORY_ONLY));
        assert!(cache.lookup(&pinned(2)).is_none());
        let stats = cache.stats();
        assert_eq!((stats.entries, stats.entries_pending), (0, 1));
        assert!(
            fill(cache, &pinned(2)).is_some(),
            "the next action refills it"
        );
        assert!(cache.lookup(&pinned(2)).is_some());
        assert!(
            cache.lookup(&pinned(1)).is_none(),
            "an entry of other versions does not match"
        );
        let stats = cache.stats();
        assert_eq!((stats.entries, stats.entries_pending), (1, 0));
        // An entry of other versions (an external change) is no hit and is refilled.
        assert!(fill(cache, &pinned(3)).is_some());
        assert!(cache.lookup(&pinned(2)).is_none());
        assert!(cache.remove(&pinned(3)));
        let stats = cache.stats();
        assert_eq!((stats.entries, stats.entries_pending), (0, 0));
    }

    /// A fill claim of a registration that was dropped and made again does not free the slot of
    /// the new registration's fill (review L3): one fill per key.
    #[test]
    fn an_old_fill_claim_does_not_free_a_new_registrations_slot() {
        let s = session();
        let cache = &s.state.cache;
        let reg = || Registration {
            level: StorageLevel::MEMORY_ONLY,
            plan: Arc::clone(&s.table("t").plan),
            keep: Vec::new(),
            tables: vec!["src".to_owned()],
            filling: false,
            epoch: 0,
        };
        assert!(cache.register(&pinned(1), reg()));
        let old = cache.claim_fill(&pinned(1)).unwrap();
        assert!(cache.remove(&pinned(1)));
        assert!(cache.register(&pinned(1), reg()));
        let new = cache.claim_fill(&pinned(1)).unwrap();
        drop(old);
        assert!(
            cache.claim_fill(&pinned(1)).is_none(),
            "the new registration's fill still runs"
        );
        drop(new);
        assert!(cache.claim_fill(&pinned(1)).is_some());
    }

    /// A registry read dropped while it registers a queued `persist` (here: while the frame's
    /// lazy checkpoint takes its rows) leaves that `persist` queued for the next read (review B3).
    #[test]
    fn a_dropped_registry_read_keeps_the_queued_persist() {
        let s = session();
        let cp = block_on(s.table("t").local_checkpoint(false, None)).unwrap();
        let cached = cp.cache();
        let reader = s.register_pending_caches();
        let mut reader = Box::pin(reader);
        assert!(
            (&mut reader).now_or_never().is_none(),
            "the read stops in the checkpoint's query"
        );
        drop(reader);
        assert!(block_on(cached.is_cached()).unwrap());
    }
}
