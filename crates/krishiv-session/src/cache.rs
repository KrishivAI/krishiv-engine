// SPDX-License-Identifier: Apache-2.0
//! The session relation cache: `DataFrame.cache`, `persist` and `unpersist`
//! (design P3-T083 3.1).
//!
//! Registration is eager, materialization lazy, as in Spark's
//! `CacheManager`. `persist(level)` puts the frame's plan on the cache's
//! pending list (the call stays synchronous); every entry point that reads
//! the registry (an action, `is_cached`, `storage_level`, `unpersist`, `CACHE
//! TABLE`) first analyzes and registers the pending plans in order, so no
//! read sees the state before the call. A registration is keyed by
//! [`RegKey`]: the normalized hash of the resolved plan with table versions
//! left out, the principal id and the policy version. Its state is
//! `Pending`, `Filling` (one action materializes it), `Memory` (a snapshot)
//! or `Evicted`.
//!
//! Any action whose plan contains a registered plan uses it: a `Memory`
//! snapshot taken under the same resolution configuration and compatible
//! table versions replaces the matching subtree by a scan of a hidden
//! in-memory table (the [`MemoryTable`] provider created DataFrames read). A
//! pending whole plan is filled while its result streams; a pending proper
//! subtree is run first as its own query and stored, then substituted. A
//! registration another action is filling is run uncached: one fill per key.
//! Nothing is shared between sessions.
//!
//! A snapshot is all-or-nothing and capped by `krishiv.cache.maxBytes` (read
//! from each query's configuration): a refused fill (the cap, the memory
//! pool, an invalidation during the fill, two columns of one name) leaves
//! the registration pending and the query runs uncached, never failing.
//! `unpersist` removes a registration and its data; `CLEAR CACHE`, `UNCACHE
//! TABLE` and a replaced or dropped temporary table or view remove the
//! registrations reading it. `REFRESH TABLE`, `REFRESH path` and the
//! engine's own writes drop only the data: the registration is pending again
//! and the next action refills it (Spark's recache). An external change to a
//! cached path is deliberately invisible: the key carries no file version,
//! as in Spark. Snapshots are never evicted otherwise.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use krishiv_common::config::{CACHE_MAX_BYTES, Config, ConfigValue};
use krishiv_exec::planner::MemoryTable;
use krishiv_memory::{ConsumerId, MemoryPool};
use krishiv_plan::analyzer::{AnalyzedPlan, PolicyVersion, ResolutionConf, TableIdent};
use krishiv_plan::cache::normalized_subtree_hash;
use krishiv_plan::{Expr, ExprNode, LogicalPlan, PlanNode, ReadSource, Scan, TableVersion};
use krishiv_types::batch::Batch;

use crate::StorageLevel;

mod pending;

pub(crate) use pending::PendingRegistration;

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

/// The registry key of a cached plan: what Spark's `sameResult` compares.
/// The table versions a plan reads are left out (a write to a cached table
/// keeps its registration, as Spark's recache does), and so is the
/// resolution configuration; both are kept on the snapshot instead
/// ([`CacheEntry::pins`], [`CacheEntry::conf`]), which is served only to a
/// query that matches them.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct RegKey {
    plan: u128,
    principal: String,
    policy: PolicyVersion,
}

impl RegKey {
    /// The key of the whole analyzed plan.
    pub(crate) fn of(analyzed: &AnalyzedPlan) -> Self {
        Self::of_node(analyzed.plan.root(), analyzed)
    }

    /// The key of `node`, a subtree of `analyzed`.
    fn of_node(node: &Arc<LogicalPlan>, analyzed: &AnalyzedPlan) -> Self {
        Self {
            plan: unversioned_hash(node, !analyzed.pins.is_empty()),
            principal: analyzed.context.principal.clone(),
            policy: analyzed.context.policy,
        }
    }
}

/// [`normalized_subtree_hash`] of `node` with the pinned versions of its
/// scans left out (only `versioned` plans carry any).
fn unversioned_hash(node: &Arc<LogicalPlan>, versioned: bool) -> u128 {
    if !versioned {
        return normalized_subtree_hash(node);
    }
    let stripped = Arc::clone(node)
        .transform_down(&mut |n| {
            if let PlanNode::Scan(scan) = &n.node
                && scan.version.is_some()
            {
                return Ok::<_, Infallible>(Arc::new(
                    LogicalPlan::new(PlanNode::Scan(Scan {
                        version: None,
                        ..scan.clone()
                    }))
                    .with_origin(n.origin),
                ));
            }
            Ok(n)
        })
        .unwrap_or_else(|never| match never {});
    normalized_subtree_hash(&stripped)
}

/// A cached relation: one materialized snapshot.
#[derive(Debug)]
pub(crate) struct CacheEntry {
    /// The snapshot, as the in-memory table its scans read.
    pub(crate) table: Arc<MemoryTable>,
    /// Bytes reserved from the pool for the snapshot.
    pub(crate) bytes: usize,
    /// The table versions the filling query pinned: a query pinning another
    /// version of one of them is not served this snapshot.
    pub(crate) pins: BTreeMap<TableIdent, TableVersion>,
    /// The resolution configuration it was computed under.
    pub(crate) conf: ResolutionConf,
}

impl CacheEntry {
    /// Whether a query pinning `pins` under `conf` may read this snapshot.
    fn serves(&self, pins: &BTreeMap<TableIdent, TableVersion>, conf: &ResolutionConf) -> bool {
        self.conf == *conf
            && self
                .pins
                .iter()
                .all(|(table, version)| pins.get(table).is_none_or(|v| v == version))
    }
}

/// Where a registration's data is.
#[derive(Debug)]
pub(crate) enum FillState {
    /// Registered, not materialized: the next action that uses it fills it.
    Pending,
    /// An action is filling it; the token is its [`FillClaim`]'s.
    Filling(u64),
    /// Materialized in memory.
    Memory(Arc<CacheEntry>),
    /// Materialized once, then evicted under memory pressure: the next use
    /// refills it (design P3-T083 3.2, the next step).
    #[expect(
        dead_code,
        reason = "eviction under memory pressure comes with the disk levels"
    )]
    Evicted,
}

/// A registered plan.
#[derive(Debug)]
struct Registration {
    level: StorageLevel,
    state: FillState,
    /// The hidden table name its snapshot is scanned as, fixed at
    /// registration so plans substituted earlier still read the same name.
    name: String,
    /// The table names (lower-case last parts) the plan reads. Pins only
    /// cover versioned tables; temporary tables leave no pins, so the
    /// registration carries what `REFRESH TABLE` / `UNCACHE TABLE` match on.
    tables: Vec<String>,
    /// Whether the plan reads files by path.
    reads_files: bool,
}

impl Registration {
    /// The pool bytes its data holds.
    fn bytes(&self) -> usize {
        match &self.state {
            FillState::Memory(entry) => entry.bytes,
            _ => 0,
        }
    }
}

/// The registrations of a session, by key and by hidden scan name.
#[derive(Debug, Default)]
struct CacheState {
    regs: HashMap<RegKey, Registration>,
    names: HashMap<String, RegKey>,
    /// Exact bytes held, for the cap: pool grants are quantised, so the
    /// pool's own counter would over-count small snapshots.
    used: usize,
}

/// What a fill was started under: [`RelationCache::insert`] refuses the
/// snapshot if an invalidation ran after `generation`, it exceeds `cap`, or
/// `claim` no longer holds the registration, and takes over the bytes the
/// fill reserved from the pool.
#[derive(Debug)]
pub(crate) struct FillLimits {
    /// The cache generation before the fill's query was analyzed.
    pub(crate) generation: u64,
    /// The fill query's own `krishiv.cache.maxBytes`.
    pub(crate) cap: usize,
    /// The bytes held while the result streams.
    pub(crate) reservation: FillReservation,
    /// The fill's hold on its registration.
    pub(crate) claim: FillClaim,
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

/// One action's hold on a registration it fills (state `Filling`): other
/// actions run the plan uncached meanwhile. Dropped without a stored
/// snapshot (a refused, failed or cancelled fill), it sets the registration
/// back to `Pending`.
#[derive(Debug)]
pub(crate) struct FillClaim {
    cache: Arc<RelationCache>,
    key: RegKey,
    /// 0 once the snapshot is stored.
    token: u64,
}

impl Drop for FillClaim {
    fn drop(&mut self) {
        if self.token == 0 {
            return;
        }
        let mut state = self
            .cache
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(reg) = state.regs.get_mut(&self.key)
            && matches!(reg.state, FillState::Filling(t) if t == self.token)
        {
            reg.state = FillState::Pending;
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

    /// Drops what every cache holds over the table `name`; registrations
    /// stay.
    pub(crate) fn invalidate_table(&self, name: &str) {
        self.results.invalidate_table(name);
        for cache in self.relations() {
            cache.invalidate_local(name);
        }
    }

    /// Drops every registration and entry of every cache (`CLEAR CACHE`).
    pub(crate) fn clear(&self) {
        self.results.clear();
        for cache in self.relations() {
            cache.clear_local();
        }
    }
}

/// The session relation cache: registrations and their snapshots under the
/// `relation_cache` pool.
#[derive(Debug)]
pub(crate) struct RelationCache {
    /// The runtime's invalidation hub: a change seen here reaches every
    /// cache of the runtime.
    hub: Arc<Invalidation>,
    pool: Arc<dyn MemoryPool>,
    consumer: ConsumerId,
    hidden: AtomicU64,
    /// The next [`FillClaim`] token (never 0).
    tokens: AtomicU64,
    /// Bumped by every invalidation, so a fill that started before one is
    /// not stored after it (it may hold rows the invalidation meant to drop).
    generation: AtomicU64,
    /// Queries whose plan found a cached subplan, and queries that looked
    /// while the cache held entries but found none.
    hits: AtomicU64,
    misses: AtomicU64,
    state: Mutex<CacheState>,
    /// `persist` calls not registered yet (see [`pending`]).
    pending: Mutex<Vec<PendingRegistration>>,
    /// Held while the pending list is registered, so a reader waits for a
    /// registration another task has taken off the list.
    registering: tokio::sync::Mutex<()>,
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
            tokens: AtomicU64::new(1),
            generation: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            state: Mutex::new(CacheState::default()),
            pending: Mutex::new(Vec::new()),
            registering: tokio::sync::Mutex::new(()),
        });
        hub.relations
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Arc::downgrade(&cache));
        cache
    }

    fn state(&self) -> std::sync::MutexGuard<'_, CacheState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
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

    /// Registers `key` at `level`; `false` (and the first level kept) when
    /// it is registered already.
    pub(crate) fn register(
        &self,
        key: RegKey,
        level: StorageLevel,
        tables: Vec<String>,
        reads_files: bool,
    ) -> bool {
        let mut state = self.state();
        if state.regs.contains_key(&key) {
            return false;
        }
        let name = format!(
            "{CACHE_PREFIX}{}",
            self.hidden.fetch_add(1, Ordering::AcqRel)
        );
        state.names.insert(name.clone(), key.clone());
        state.regs.insert(
            key,
            Registration {
                level,
                state: FillState::Pending,
                name,
                tables,
                reads_files,
            },
        );
        true
    }

    /// The registered level of `key`, if registered.
    pub(crate) fn level(&self, key: &RegKey) -> Option<StorageLevel> {
        self.state().regs.get(key).map(|r| r.level)
    }

    /// Removes `key`'s registration and its data; whether it was registered.
    pub(crate) fn unregister(&self, key: &RegKey) -> bool {
        let mut state = self.state();
        self.invalidated();
        let Some(reg) = state.regs.remove(key) else {
            return false;
        };
        self.forget(&mut state, &reg);
        true
    }

    /// Releases what the removed `reg` held.
    fn forget(&self, state: &mut CacheState, reg: &Registration) {
        state.names.remove(&reg.name);
        let bytes = reg.bytes();
        if bytes > 0 {
            state.used = state.used.saturating_sub(bytes);
            self.pool.release(self.consumer, bytes);
        }
    }

    /// Drops `reg`'s data, keeping it registered (pending again).
    fn drop_data(&self, used: &mut usize, reg: &mut Registration) {
        if let FillState::Memory(entry) = &reg.state {
            *used = used.saturating_sub(entry.bytes);
            self.pool.release(self.consumer, entry.bytes);
            reg.state = FillState::Pending;
        }
    }

    /// The snapshot the hidden table `name` reads, if filled.
    pub(crate) fn table(&self, name: &str) -> Option<Arc<MemoryTable>> {
        let state = self.state();
        let key = state.names.get(name)?;
        match &state.regs.get(key)?.state {
            FillState::Memory(entry) => Some(Arc::clone(&entry.table)),
            _ => None,
        }
    }

    /// Takes `key` for one fill when its data is missing (or was computed
    /// under another configuration or table versions than `pins`/`conf`);
    /// `None` when it is not registered, another action fills it, or its
    /// snapshot serves the query.
    pub(crate) fn claim(
        self: &Arc<Self>,
        key: &RegKey,
        pins: &BTreeMap<TableIdent, TableVersion>,
        conf: &ResolutionConf,
    ) -> Option<FillClaim> {
        let mut guard = self.state();
        let state = &mut *guard;
        let reg = state.regs.get_mut(key)?;
        match &reg.state {
            FillState::Filling(_) => return None,
            FillState::Memory(entry) if entry.serves(pins, conf) => return None,
            FillState::Memory(_) => self.drop_data(&mut state.used, reg),
            FillState::Pending | FillState::Evicted => {}
        }
        let token = self.tokens.fetch_add(1, Ordering::AcqRel);
        reg.state = FillState::Filling(token);
        Some(FillClaim {
            cache: Arc::clone(self),
            key: key.clone(),
            token,
        })
    }

    /// Stores the snapshot (`schema`, `batches`, computed under `pins` and
    /// `conf`) for the registration `limits.claim` fills and returns the
    /// hidden table name its scans read. When the snapshot would exceed the
    /// `cap` (the fill query's own `krishiv.cache.maxBytes`; 0 disables
    /// caching), an invalidation ran since the fill started (`generation`),
    /// the registration is gone or no longer this fill's, or two of its
    /// columns share a name (a scan finds its columns by name, so it could
    /// not tell them apart), stores nothing and returns `None`: the
    /// registration is pending again and the query runs uncached, it never
    /// fails.
    pub(crate) fn insert(
        &self,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
        pins: BTreeMap<TableIdent, TableVersion>,
        conf: ResolutionConf,
        limits: FillLimits,
    ) -> Option<String> {
        // Any early return drops the reservation, which releases the bytes,
        // and the claim, which sets the registration back to pending.
        let FillLimits {
            generation,
            cap,
            reservation,
            mut claim,
        } = limits;
        let mut names = HashSet::new();
        if !schema.fields().iter().all(|f| names.insert(f.name())) {
            return None;
        }
        let bytes = reservation.bytes;
        let mut guard = self.state();
        let state = &mut *guard;
        // A cap of 0 disables caching, the empty result included. The
        // generation is read under the state lock, which every invalidation
        // also holds while it bumps it.
        if cap == 0 || generation != self.generation() {
            return None;
        }
        if state.used.saturating_add(bytes) > cap {
            return None;
        }
        let reg = state.regs.get_mut(&claim.key)?;
        if !matches!(reg.state, FillState::Filling(t) if t == claim.token) {
            return None;
        }
        reservation.adopt();
        reg.state = FillState::Memory(Arc::new(CacheEntry {
            table: Arc::new(MemoryTable::new(
                schema,
                batches.into_iter().map(Batch::new).collect(),
            )),
            bytes,
            pins,
            conf,
        }));
        claim.token = 0;
        state.used = state.used.saturating_add(bytes);
        Some(reg.name.clone())
    }

    /// Drops every registration of every cache of the runtime (`CLEAR CACHE`).
    pub(crate) fn clear(&self) {
        self.hub.clear();
    }

    /// Drops every registration of this cache.
    fn clear_local(&self) {
        let mut state = self.state();
        self.invalidated();
        let regs = std::mem::take(&mut state.regs);
        for reg in regs.values() {
            self.forget(&mut state, reg);
        }
        state.names.clear();
        state.used = 0;
    }

    /// Drops the data of the registrations reading the table `name` (any
    /// spelling), in every cache of the runtime: what they read may have
    /// changed. The registrations stay and refill on their next use.
    pub(crate) fn invalidate_table(&self, name: &str) {
        self.hub.invalidate_table(name);
    }

    /// Removes the registrations of this cache reading the table `name`, and
    /// their data (`UNCACHE TABLE`, a temporary table or view replaced or
    /// dropped: the plans no longer resolve to what was registered); then
    /// drops what every cache of the runtime holds over it.
    pub(crate) fn uncache_table(&self, name: &str) {
        {
            let mut state = self.state();
            self.invalidated();
            let matches = table_matcher(name);
            let gone: Vec<RegKey> = state
                .regs
                .iter()
                .filter(|(_, reg)| reads_table(reg, &matches))
                .map(|(key, _)| key.clone())
                .collect();
            for key in gone {
                if let Some(reg) = state.regs.remove(&key) {
                    self.forget(&mut state, &reg);
                }
            }
        }
        self.hub.invalidate_table(name);
    }

    /// [`Self::invalidate_table`] in this cache only.
    fn invalidate_local(&self, name: &str) {
        let mut guard = self.state();
        let state = &mut *guard;
        self.invalidated();
        let matches = table_matcher(name);
        for reg in state.regs.values_mut() {
            if reads_table(reg, &matches) {
                self.drop_data(&mut state.used, reg);
            }
        }
    }

    /// Drops the data of the registrations reading files by path: the
    /// engine wrote files, so what they read may have changed. An external
    /// change drops nothing.
    pub(crate) fn invalidate_files(&self) {
        let mut guard = self.state();
        let state = &mut *guard;
        self.invalidated();
        for reg in state.regs.values_mut() {
            if reg.reads_files {
                self.drop_data(&mut state.used, reg);
            }
        }
    }

    /// The cache's size and counters now, for `system.caches`.
    pub(crate) fn stats(&self) -> RelationStats {
        let state = self.state();
        RelationStats {
            entries: state
                .regs
                .values()
                .filter(|r| matches!(r.state, FillState::Memory(_)))
                .count() as u64,
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

    fn is_empty(&self) -> bool {
        self.state().regs.is_empty()
    }

    /// The analyzed plan with every filled (sub)plan replaced by a scan of
    /// its snapshot: the same [`MemoryTable`] provider reads of created
    /// DataFrames use. Returns `None` when nothing matched; else also the
    /// snapshots the plan now scans, by hidden name, so the query keeps them
    /// even if the cache is cleared before it is planned.
    pub(crate) fn substitute(self: &Arc<Self>, analyzed: &AnalyzedPlan) -> Option<Substituted> {
        self.substitute_claiming(analyzed, true, false).0
    }

    /// Whether the plan uses a registered (sub)plan in any state, without
    /// counting a hit or a miss: a probe (is this plan eligible for the
    /// shared result cache?) is not a lookup of the execution.
    pub(crate) fn uses_registered(&self, analyzed: &AnalyzedPlan) -> bool {
        if self.is_empty() {
            return false;
        }
        let mut found = false;
        walk(analyzed.plan.root(), &mut |node| {
            let key = RegKey::of_node(node, analyzed);
            found = found || self.state().regs.contains_key(&key);
            if found { Walk::Stop } else { Walk::Children }
        });
        found
    }

    /// [`Self::substitute`] (counting a hit or a miss if `count`), and with
    /// `claim_root` a claim on the whole plan's registration when its data
    /// is missing: the caller fills it from the result stream.
    pub(crate) fn substitute_claiming(
        self: &Arc<Self>,
        analyzed: &AnalyzedPlan,
        count: bool,
        claim_root: bool,
    ) -> (Option<Substituted>, Option<FillClaim>) {
        // Nothing registered: skip hashing every node of the plan.
        if self.is_empty() {
            return (None, None);
        }
        let pins = &analyzed.pins;
        let conf = &analyzed.context.conf;
        let root = Arc::clone(analyzed.plan.root());
        let mut tables: Vec<(String, Arc<MemoryTable>)> = Vec::new();
        let mut root_claim = None;
        let mut first = true;
        let out: Arc<LogicalPlan> = Arc::clone(&root)
            .transform_down(&mut |node| {
                let at_root = std::mem::take(&mut first);
                let key = RegKey::of_node(&node, analyzed);
                let found = {
                    let state = self.state();
                    match state.regs.get(&key).map(|r| (&r.state, &r.name)) {
                        Some((FillState::Memory(entry), name)) if entry.serves(pins, conf) => {
                            Some((name.clone(), Arc::clone(&entry.table)))
                        }
                        _ => None,
                    }
                };
                let Some((name, table)) = found else {
                    if at_root && claim_root {
                        root_claim = self.claim(&key, pins, conf);
                    }
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
            (None, root_claim)
        } else {
            if count {
                self.hits.fetch_add(1, Ordering::Relaxed);
            }
            (Some(Substituted { plan: out, tables }), root_claim)
        }
    }

    /// Claims the registered proper subtrees of the plan whose data is
    /// missing, outermost first: each is run as its own query and stored
    /// before the plan is substituted. A subtree with a snapshot that serves
    /// the query, or that is claimed here, is not looked into; one another
    /// action fills is run uncached, so its own subtrees are looked into.
    pub(crate) fn claim_subtrees(
        self: &Arc<Self>,
        analyzed: &AnalyzedPlan,
    ) -> Vec<(Arc<LogicalPlan>, FillClaim)> {
        if self.is_empty() {
            return Vec::new();
        }
        let root = analyzed.plan.root();
        let mut claims = Vec::new();
        walk(root, &mut |node| {
            if Arc::ptr_eq(node, root) {
                return Walk::Children;
            }
            let key = RegKey::of_node(node, analyzed);
            let served = {
                let state = self.state();
                match state.regs.get(&key).map(|r| &r.state) {
                    None => return Walk::Children,
                    Some(FillState::Memory(entry)) => {
                        entry.serves(&analyzed.pins, &analyzed.context.conf)
                    }
                    Some(_) => false,
                }
            };
            if served {
                return Walk::Skip;
            }
            match self.claim(&key, &analyzed.pins, &analyzed.context.conf) {
                Some(claim) => {
                    claims.push((Arc::clone(node), claim));
                    Walk::Skip
                }
                None => Walk::Children,
            }
        });
        claims
    }
}

/// What [`walk`] does after visiting a node.
enum Walk {
    /// Visit its children.
    Children,
    /// Do not visit its children.
    Skip,
    /// End the walk.
    Stop,
}

/// Visits the relation nodes of `root` top-down (pre-order), each shared
/// node once: the nodes `transform_down` visits.
fn walk(root: &Arc<LogicalPlan>, visit: &mut impl FnMut(&Arc<LogicalPlan>) -> Walk) {
    let mut seen: HashSet<*const LogicalPlan> = HashSet::new();
    let mut stack = vec![Arc::clone(root)];
    while let Some(node) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&node)) {
            continue;
        }
        match visit(&node) {
            Walk::Stop => return,
            Walk::Skip => {}
            Walk::Children => {
                let mut children = Vec::new();
                node.for_each_child(&mut |c| children.push(Arc::clone(c)));
                // Reversed, so the first child is visited first.
                stack.extend(children.into_iter().rev());
            }
        }
    }
}

/// A matcher of the table `name` (any spelling) against lower-case last
/// parts and pinned identifiers.
fn table_matcher(name: &str) -> impl Fn(&str) -> bool {
    let lowered = name.to_lowercase();
    let short = lowered.rsplit('.').next().unwrap_or(&lowered).to_owned();
    move |table: &str| table == lowered || table == short
}

/// Whether `reg` reads the table `matches` matches: by its tables, or by
/// the pins of its snapshot, so versioned and temporary tables both match.
fn reads_table(reg: &Registration, matches: &impl Fn(&str) -> bool) -> bool {
    reg.tables.iter().any(|t| matches(t))
        || matches!(&reg.state, FillState::Memory(entry)
            if entry.pins.keys().any(|id| matches(&id.name.to_lowercase())))
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
