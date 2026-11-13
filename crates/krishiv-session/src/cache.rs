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
//! [`CacheKey`] (normalized plan hash, pinned versions, principal id, policy
//! version). Materialization is lazy: the first action after `cache()` runs
//! the query and keeps its batches when the stream completes; later actions
//! substitute the cached snapshot for any matching (sub)plan (a subplan matches
//! when the pinned table versions of its query equal those it was cached under). Substitution
//! replaces the matching subtree by a scan of an in-memory table, the same
//! [`MemoryTable`] provider reads of created DataFrames use; no plan node is
//! added and nothing is shared between sessions.
//!
//! The cache is all-or-nothing and capped by `krishiv.cache.maxBytes` (read
//! from each query's configuration): when a snapshot would exceed the cap it
//! is not stored and the query runs uncached, never failing. Entries are
//! dropped by `unpersist`, `CLEAR CACHE`, `UNCACHE TABLE`, `REFRESH TABLE`
//! and `REFRESH path`, by the catalog calls behind them, and by the engine's
//! own writes: a successful file write drops the entries reading files. An
//! external change to a cached path is deliberately invisible: the key
//! carries no file version, as in Spark. Replacing or dropping a temporary
//! table or view drops the entries that read it, and an invalidation that
//! runs while a fill is in flight keeps that fill from being stored.
//! Entries are never evicted otherwise: they stay until dropped or the cap
//! refuses new ones. A data invalidation (a write, `REFRESH`) keeps the
//! registrations, which refill on use; `CLEAR CACHE`, `UNCACHE TABLE` and
//! replacing or dropping a temporary view drop them too.

use std::collections::{BTreeMap, HashMap};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use krishiv_common::config::{CACHE_MAX_BYTES, Config, ConfigValue};
use krishiv_exec::planner::MemoryTable;
use krishiv_memory::{ConsumerId, MemoryPool};
use krishiv_plan::analyzer::AnalyzedPlan;
use krishiv_plan::cache::{CacheKey, normalized_subtree_hash};
use krishiv_plan::{Expr, ExprNode, LogicalPlan, PlanNode, ReadSource, Scan};
use krishiv_types::batch::Batch;

use crate::StorageLevel;
use crate::dataframe::DataFrame;
use crate::session::Session;

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
}

impl Drop for FillClaim {
    fn drop(&mut self) {
        let mut state = self
            .cache
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(reg) = state.registrations.get_mut(&self.key) {
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
    /// The snapshot, as the in-memory table its scans read.
    pub(crate) table: Arc<MemoryTable>,
    /// Bytes reserved from the pool for the snapshot.
    pub(crate) bytes: usize,
    /// Whether the cached plan reads files by path.
    pub(crate) reads_files: bool,
    /// The table names (lower-case last parts) the cached plan reads. Pins
    /// only cover versioned tables; temporary tables leave no pins, so the
    /// entry carries what `REFRESH TABLE` / `UNCACHE TABLE` match on.
    pub(crate) tables: Vec<String>,
}

/// The relations a session has cached, by key and by hidden scan name.
#[derive(Debug, Default)]
struct CacheState {
    entries: HashMap<CacheKey, Arc<CacheEntry>>,
    /// The plans asked to be cached, by key; an entry is the data of one.
    registrations: HashMap<CacheKey, Registration>,
    /// `persist` calls not yet analyzed and registered, in order.
    pending: Vec<PendingReg>,
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
    pub(crate) entries: u64,
    pub(crate) bytes: u64,
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
    state: Mutex<CacheState>,
    /// Held while the pending registrations are analyzed, so a reader that
    /// finds the queue taken waits for the registrations instead of missing them.
    registering: futures::lock::Mutex<()>,
}

impl RelationCache {
    /// An empty cache under `memory` with a hub of its own (tests: sessions
    /// use [`Self::on`]).
    #[cfg(test)]
    pub(crate) fn new(memory: &Arc<dyn MemoryPool>) -> Arc<Self> {
        Self::on(
            memory,
            &Invalidation::new(Arc::clone(crate::result_cache::global())),
        )
    }

    /// An empty cache under `memory`, registered with the runtime's `hub`.
    pub(crate) fn on(memory: &Arc<dyn MemoryPool>, hub: &Arc<Invalidation>) -> Arc<Self> {
        let pool = memory.child("relation_cache", None);
        let consumer = pool.register("relation_cache", None);
        let cache = Arc::new(Self {
            hub: Arc::clone(hub),
            pool,
            consumer,
            hidden: AtomicU64::new(0),
            generation: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            state: Mutex::new(CacheState::default()),
            registering: futures::lock::Mutex::new(()),
        });
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
            .get(key)
            .cloned()
    }

    /// The snapshot the hidden table `name` reads, if cached.
    pub(crate) fn table(&self, name: &str) -> Option<Arc<MemoryTable>> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let key = state.names.get(name)?;
        state.entries.get(key).map(|e| Arc::clone(&e.table))
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
            .push(PendingReg { plan, keep, level });
    }

    /// The level `key` is registered at, if it is.
    pub(crate) fn level_of(&self, key: &CacheKey) -> Option<StorageLevel> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registrations
            .get(key)
            .map(|r| r.level)
    }

    /// The fill slot of `key`: `Some` when it is registered, holds no data
    /// and no other fill runs.
    pub(crate) fn claim_fill(self: &Arc<Self>, key: &CacheKey) -> Option<FillClaim> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.entries.contains_key(key) {
            return None;
        }
        let reg = state.registrations.get_mut(key)?;
        if reg.filling {
            return None;
        }
        reg.filling = true;
        Some(FillClaim {
            cache: Arc::clone(self),
            key: key.clone(),
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
                let key = CacheKey {
                    plan: normalized_subtree_hash(&node),
                    pins: analyzed.pins.clone(),
                    principal: analyzed.context.principal.clone(),
                    policy: analyzed.context.policy,
                    conf: analyzed.context.conf,
                };
                let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                if let Some(reg) = state.registrations.get(&key) {
                    let needs_fill = !reg.filling && !state.entries.contains_key(&key);
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
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let reg = state.registrations.get(key)?;
        (!reg.filling && !state.entries.contains_key(key))
            .then(|| (Arc::clone(&reg.plan), reg.keep.clone()))
    }

    /// Stores the snapshot (`schema`, `batches`) under `key` and returns the
    /// hidden table name its scans read. When the snapshot would exceed the
    /// `cap` (the fill query's own `krishiv.cache.maxBytes`; 0 disables
    /// caching), an invalidation ran since the fill
    /// started (`generation`), or two of its columns share a name (a scan finds its columns by
    /// name, so it could not tell them apart), stores nothing and returns
    /// `None`: the query runs uncached, it never fails. Replaces any entry
    /// already stored under `key`.
    pub(crate) fn insert(
        &self,
        key: CacheKey,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
        reads_files: bool,
        tables: Vec<String>,
        limits: FillLimits,
    ) -> Option<String> {
        // Any early return drops the reservation, which releases the bytes.
        let FillLimits {
            generation,
            cap,
            reservation,
        } = limits;
        let mut names = std::collections::HashSet::new();
        if !schema.fields().iter().all(|f| names.insert(f.name())) {
            return None;
        }
        let bytes = reservation.bytes;
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        // A cap of 0 disables caching, the empty result included. The
        // generation is read under the state lock, which every invalidation
        // also holds while it bumps it.
        if cap == 0 || generation != self.generation() || !state.registrations.contains_key(&key) {
            return None;
        }
        let old = state.entries.get(&key).map(|e| e.bytes).unwrap_or(0);
        if (state.used.saturating_sub(old)).saturating_add(bytes) > cap {
            return None;
        }
        reservation.adopt();
        if let Some(old) = state.entries.get(&key) {
            self.pool.release(self.consumer, old.bytes);
            state.used = state.used.saturating_sub(old.bytes);
        }
        // A replaced entry keeps its hidden name: plans substituted earlier
        // still read the same name, and the plan holds the table it resolved.
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
        let entry = Arc::new(CacheEntry {
            name: name.clone(),
            table: Arc::new(MemoryTable::new(
                schema,
                batches.into_iter().map(Batch::new).collect(),
            )),
            bytes,
            reads_files,
            tables,
        });
        state.used = state.used.saturating_add(bytes);
        state.names.insert(name.clone(), key.clone());
        state.entries.insert(key, entry);
        Some(name)
    }

    /// Drops the registration of `key` and its entry; whether one existed.
    pub(crate) fn remove(&self, key: &CacheKey) -> bool {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.invalidated();
        let registered = state.registrations.remove(key).is_some();
        let Some(entry) = state.entries.remove(key) else {
            return registered;
        };
        state.names.retain(|_, k| k != key);
        state.used = state.used.saturating_sub(entry.bytes);
        self.pool.release(self.consumer, entry.bytes);
        true
    }

    /// Drops every entry of every cache of the runtime (`CLEAR CACHE`).
    pub(crate) fn clear(&self) {
        self.hub.clear();
    }

    /// Drops every entry of this cache.
    fn clear_local(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.invalidated();
        for entry in state.entries.values() {
            self.pool.release(self.consumer, entry.bytes);
        }
        state.entries.clear();
        state.names.clear();
        state.registrations.clear();
        state.pending.clear();
        state.used = 0;
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
            .filter(|(key, entry)| {
                entry.tables.iter().any(|t| matches(t))
                    || key.pins.keys().any(|id| matches(&id.name.to_lowercase()))
            })
            .map(|(key, _)| key.clone())
            .collect();
        for key in &stale {
            if let Some(entry) = state.entries.remove(key) {
                state.used = state.used.saturating_sub(entry.bytes);
                self.pool.release(self.consumer, entry.bytes);
            }
        }
        state.names.retain(|_, k| !stale.contains(k));
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
            .filter(|(key, reg)| {
                reg.tables.iter().any(|t| matches(t))
                    || key.pins.keys().any(|id| matches(&id.name.to_lowercase()))
            })
            .map(|(key, _)| key.clone())
            .collect();
        for key in &stale {
            state.registrations.remove(key);
            if let Some(entry) = state.entries.remove(key) {
                state.used = state.used.saturating_sub(entry.bytes);
                self.pool.release(self.consumer, entry.bytes);
            }
        }
        state.names.retain(|_, k| !stale.contains(k));
        state
            .pending
            .retain(|p| !unresolved_tables(&p.plan).iter().any(|t| matches(t)));
    }

    /// Drops the entries reading files by path: the engine wrote files, so
    /// what they read may have changed. An external change drops nothing.
    pub(crate) fn invalidate_files(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.invalidated();
        let mut freed = 0_usize;
        state.entries.retain(|_, entry| {
            if entry.reads_files {
                freed = freed.saturating_add(entry.bytes);
                false
            } else {
                true
            }
        });
        state.used = state.used.saturating_sub(freed);
        self.pool.release(self.consumer, freed);
        let live: Vec<CacheKey> = state.entries.keys().cloned().collect();
        state.names.retain(|_, k| live.contains(k));
    }

    /// The cache's size and counters now, for `system.caches`.
    pub(crate) fn stats(&self) -> RelationStats {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        RelationStats {
            entries: state.entries.len() as u64,
            bytes: state.used as u64,
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
        let pins = analyzed.pins.clone();
        let principal = analyzed.context.principal.clone();
        let policy = analyzed.context.policy;
        let conf = analyzed.context.conf;
        let root = Arc::clone(analyzed.plan.root());
        let mut tables: Vec<(String, Arc<MemoryTable>)> = Vec::new();
        let out: Arc<LogicalPlan> = Arc::clone(&root)
            .transform_down(&mut |node| {
                let key = CacheKey {
                    plan: normalized_subtree_hash(&node),
                    pins: pins.clone(),
                    principal: principal.clone(),
                    policy,
                    conf,
                };
                let found = {
                    let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                    state
                        .entries
                        .get(&key)
                        .map(|e| (e.name.clone(), Arc::clone(&e.table)))
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
    pub(crate) tables: Vec<(String, Arc<MemoryTable>)>,
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
    pub(crate) async fn register_pending_caches(&self) {
        let cache = &self.state.cache;
        let _turn = cache.registering.lock().await;
        loop {
            let batch = std::mem::take(
                &mut cache
                    .state
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .pending,
            );
            if batch.is_empty() {
                return;
            }
            for pending in batch {
                let frame = DataFrame::over_plan(self.clone(), pending.plan, pending.keep);
                let analysis = match frame.analysis().await {
                    Ok(a) => a,
                    Err(error) => {
                        tracing::warn!(target: "krishiv::session", %error, "Asked to cache a plan that does not analyze");
                        continue;
                    }
                };
                let key = CacheKey::of(&analysis.analyzed);
                let mut tables = read_tables(analysis.analyzed.plan.root());
                tables.extend(unresolved_tables(&frame.plan));
                tables.sort();
                tables.dedup();
                let mut state = cache.state.lock().unwrap_or_else(PoisonError::into_inner);
                if state.registrations.contains_key(&key) {
                    tracing::warn!(target: "krishiv::session", "Asked to cache already cached data.");
                    continue;
                }
                state.registrations.insert(
                    key,
                    Registration {
                        level: pending.level,
                        plan: Arc::clone(&frame.plan),
                        keep: frame.keep.clone(),
                        tables,
                        filling: false,
                    },
                );
            }
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
