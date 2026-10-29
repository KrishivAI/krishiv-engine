// SPDX-License-Identifier: Apache-2.0
//! The session relation cache: `DataFrame.cache`, `persist` and `unpersist`.
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
//! refuses new ones.

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

/// The session relation cache: snapshots under the `relation_cache` pool.
#[derive(Debug)]
pub(crate) struct RelationCache {
    pool: Arc<dyn MemoryPool>,
    consumer: ConsumerId,
    hidden: AtomicU64,
    /// Bumped by every invalidation, so a fill that started before one is
    /// not stored after it (it may hold rows the invalidation meant to drop).
    generation: AtomicU64,
    state: Mutex<CacheState>,
}

impl RelationCache {
    /// An empty cache under `memory`.
    pub(crate) fn new(memory: &Arc<dyn MemoryPool>) -> Self {
        let pool = memory.child("relation_cache", None);
        let consumer = pool.register("relation_cache", None);
        Self {
            pool,
            consumer,
            hidden: AtomicU64::new(0),
            generation: AtomicU64::new(0),
            state: Mutex::new(CacheState::default()),
        }
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

    /// Whether `key` is cached.
    pub(crate) fn contains(&self, key: &CacheKey) -> bool {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entries
            .contains_key(key)
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
        if cap == 0 || generation != self.generation() {
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

    /// Drops the entry for `key`; whether one existed.
    pub(crate) fn remove(&self, key: &CacheKey) -> bool {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.invalidated();
        let Some(entry) = state.entries.remove(key) else {
            return false;
        };
        state.names.retain(|_, k| k != key);
        state.used = state.used.saturating_sub(entry.bytes);
        self.pool.release(self.consumer, entry.bytes);
        true
    }

    /// Drops every entry.
    pub(crate) fn clear(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.invalidated();
        for entry in state.entries.values() {
            self.pool.release(self.consumer, entry.bytes);
        }
        state.entries.clear();
        state.names.clear();
        state.used = 0;
    }

    /// Drops the entries reading the table `name` (any spelling): what they
    /// read may have changed. Matches the entry's tables and its pins, so
    /// versioned and temporary tables both clear.
    pub(crate) fn invalidate_table(&self, name: &str) {
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
                    state.names.iter().find_map(|(n, k)| {
                        (k == &key)
                            .then(|| {
                                state
                                    .entries
                                    .get(k)
                                    .map(|e| (n.clone(), Arc::clone(&e.table)))
                            })
                            .flatten()
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
            None
        } else {
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
