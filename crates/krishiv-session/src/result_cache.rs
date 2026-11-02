// SPDX-License-Identifier: Apache-2.0
//! The process-wide result cache (P2-T094, `design/P2-T094-result-cache.md`).
//!
//! With `krishiv.resultCache.enabled` a query's rows are kept for later runs of the same query by
//! any session of the process: the key is [`CacheKey`] (normalized plan hash, pinned versions,
//! principal, policy version, resolution configuration) plus the identity of every permanent
//! table the plan reads. Only a plan that reads permanent catalog tables alone and calls no
//! non-deterministic function is cached; a temporary view, an in-memory registration or a read by
//! path is session-local or unversioned and is never stored here.
//!
//! Entries expire after `krishiv.resultCache.ttl` and are dropped at once by the calls that drop
//! relation-cache entries (`REFRESH TABLE`, `CLEAR CACHE`, every engine write to a table they
//! read). The byte cap is `krishiv.resultCache.maxBytes`, held outside the query memory pool: the
//! store is shared by sessions and outlives their pools. A fill guarded by the store's generation
//! (taken before the query is analyzed) is dropped if an invalidation ran meanwhile.

use std::collections::HashMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use arrow_array::RecordBatch;
use futures::Stream;
use krishiv_common::config::{
    Config, ConfigValue, RESULT_CACHE_ENABLED, RESULT_CACHE_MAX_BYTES, RESULT_CACHE_TTL,
};
use krishiv_common::error::KrishivError;
use krishiv_plan::cache::CacheKey;
use krishiv_plan::{Expr, ExprNode, LogicalPlan, PlanNode, ReadSource, ResolvedExpr};

use crate::dataframe::Analysis;
use crate::session::{QueryEnv, RecordBatchStream, Session};
use crate::store::CatalogStore as _;
use crate::table_commands::TableName;

/// The identity of a permanent table's data: a drop and re-create changes `created_ms`, a moved
/// table changes `location`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct TableId {
    namespace: String,
    name: String,
    location: PathBuf,
    created_ms: i64,
}

/// A result-cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ResultKey {
    plan: CacheKey,
    tables: Vec<TableId>,
    /// The settings that change what a read or an execution returns and that the analysis
    /// configuration of `plan` does not carry.
    settings: Vec<(&'static str, String)>,
}

/// The settings of a query that change its rows without changing its plan.
const RESULT_SETTINGS: &[&str] = &[
    "krishiv.deterministicOrder",
    "spark.sql.files.ignoreCorruptFiles",
    "spark.sql.files.ignoreMissingFiles",
    "spark.sql.parquet.binaryAsString",
    "spark.sql.parquet.int96AsTimestamp",
    "spark.sql.parquet.datetimeRebaseModeInRead",
    "spark.sql.parquet.int96RebaseModeInRead",
    "spark.sql.parquet.inferTimestampNTZ.enabled",
    "spark.sql.legacy.timeParserPolicy",
    "spark.sql.mapKeyDedupPolicy",
    "spark.sql.storeAssignmentPolicy",
    "spark.sql.legacy.sizeOfNull",
];

/// The least a stored result is charged: the entry, its key and its table list.
const ENTRY_OVERHEAD: usize = 1024;

/// The settings of a query's configuration snapshot.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Settings {
    pub(crate) ttl: Duration,
    pub(crate) max_bytes: usize,
}

impl Settings {
    /// `Some` when `krishiv.resultCache.enabled` is set in `config`.
    pub(crate) fn of(config: &Config) -> Option<Self> {
        let enabled = config
            .get(&RESULT_CACHE_ENABLED)
            .and_then(ConfigValue::as_bool)
            .unwrap_or(false);
        enabled.then(|| Self {
            ttl: config
                .get(&RESULT_CACHE_TTL)
                .and_then(ConfigValue::as_duration)
                .unwrap_or(Duration::from_secs(24 * 60 * 60)),
            max_bytes: usize::try_from(
                config
                    .get(&RESULT_CACHE_MAX_BYTES)
                    .and_then(ConfigValue::as_size)
                    .unwrap_or(64 << 20),
            )
            .unwrap_or(usize::MAX),
        })
    }
}

#[derive(Debug)]
struct Entry {
    batches: Vec<RecordBatch>,
    stored: Instant,
    /// The `krishiv.resultCache.ttl` the entry was stored under (`0s`: never).
    ttl: Duration,
    bytes: usize,
    /// The lower-case names of the tables read (`namespace.name` and `name`).
    tables: Vec<String>,
}

#[derive(Debug, Default)]
struct State {
    entries: HashMap<ResultKey, Entry>,
    bytes: usize,
}

/// The store: one per process, see [`global`].
#[derive(Debug, Default)]
pub(crate) struct ResultCache {
    state: Mutex<State>,
    /// Served from the store, run because nothing usable was stored, stored, and dropped by an
    /// invalidation (counted per entry); for the process's whole life.
    hits: AtomicU64,
    misses: AtomicU64,
    stores: AtomicU64,
    invalidated: AtomicU64,
    /// Bumped by every invalidation, so a fill that started before one is not stored after it.
    generation: AtomicU64,
}

/// The process-wide store.
pub(crate) fn global() -> &'static ResultCache {
    static GLOBAL: OnceLock<ResultCache> = OnceLock::new();
    GLOBAL.get_or_init(ResultCache::default)
}

impl ResultCache {
    /// The generation to take before a query is analyzed.
    pub(crate) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// The batches stored under `key` that are younger than `ttl` (`0s`: never expire), and
    /// their age. An expired entry is dropped.
    pub(crate) fn lookup(
        &self,
        key: &ResultKey,
        ttl: Duration,
    ) -> Option<(Vec<RecordBatch>, Duration)> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(entry) = state.entries.get(key) else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let age = entry.stored.elapsed();
        if !ttl.is_zero() && age >= ttl {
            if let Some(gone) = state.entries.remove(key) {
                state.bytes = state.bytes.saturating_sub(gone.bytes);
            }
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        self.hits.fetch_add(1, Ordering::Relaxed);
        state.entries.get(key).map(|e| (e.batches.clone(), age))
    }

    /// `(entries, bytes, hits, misses, stores, invalidated)` now.
    pub(crate) fn stats(&self) -> (usize, usize, u64, u64, u64, u64) {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        (
            state.entries.len(),
            state.bytes,
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.stores.load(Ordering::Relaxed),
            self.invalidated.load(Ordering::Relaxed),
        )
    }

    /// The age of the entry for `key` if it would be served (no batches copied).
    pub(crate) fn age(&self, key: &ResultKey, ttl: Duration) -> Option<Duration> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let age = state.entries.get(key)?.stored.elapsed();
        (ttl.is_zero() || age < ttl).then_some(age)
    }

    /// Stores `batches` under `key` unless an invalidation ran after `generation`, they exceed
    /// `max_bytes`, or they would not fit once the oldest entries are dropped. Returns whether
    /// they were stored.
    pub(crate) fn insert(
        &self,
        key: ResultKey,
        batches: Vec<RecordBatch>,
        bytes: usize,
        generation: u64,
        max_bytes: usize,
        ttl: Duration,
    ) -> bool {
        // An empty result still costs its entry: bounded by the cap, the count stays bounded.
        let bytes = bytes.max(ENTRY_OVERHEAD);
        if bytes > max_bytes {
            return false;
        }
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if self.generation.load(Ordering::Acquire) != generation {
            return false;
        }
        // Expired entries go first: nothing else would free one nobody asks for again.
        let mut freed = 0;
        state.entries.retain(|_, e| {
            let expired = !e.ttl.is_zero() && e.stored.elapsed() >= e.ttl;
            if expired {
                freed += e.bytes;
            }
            !expired
        });
        state.bytes = state.bytes.saturating_sub(freed);
        if let Some(old) = state.entries.remove(&key) {
            state.bytes = state.bytes.saturating_sub(old.bytes);
        }
        while state.bytes.saturating_add(bytes) > max_bytes {
            let Some(oldest) = state
                .entries
                .iter()
                .min_by_key(|(_, e)| e.stored)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            if let Some(gone) = state.entries.remove(&oldest) {
                state.bytes = state.bytes.saturating_sub(gone.bytes);
            }
        }
        let tables = key
            .tables
            .iter()
            .flat_map(|t| {
                [
                    format!("{}.{}", t.namespace, t.name).to_lowercase(),
                    t.name.to_lowercase(),
                ]
            })
            .collect();
        state.bytes += bytes;
        self.stores.fetch_add(1, Ordering::Relaxed);
        state.entries.insert(
            key,
            Entry {
                batches,
                stored: Instant::now(),
                ttl,
                bytes,
                tables,
            },
        );
        true
    }

    /// Drops every entry that read the table `name` (`table` or `namespace.table`, any case).
    pub(crate) fn invalidate_table(&self, name: &str) {
        let lowered = name.to_lowercase();
        let short = lowered.rsplit('.').next().unwrap_or(&lowered).to_owned();
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.generation.fetch_add(1, Ordering::AcqRel);
        let mut freed = 0;
        let before = state.entries.len();
        state.entries.retain(|_, e| {
            let hit = e.tables.iter().any(|t| *t == lowered || *t == short);
            if hit {
                freed += e.bytes;
            }
            !hit
        });
        self.invalidated
            .fetch_add((before - state.entries.len()) as u64, Ordering::Relaxed);
        state.bytes = state.bytes.saturating_sub(freed);
    }

    /// Drops the entries that read a table whose directory contains `path` or lies inside it:
    /// a write by path (`df.write().parquet(dir)`) into a table's files changes what it reads.
    pub(crate) fn invalidate_path(&self, path: &std::path::Path) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.generation.fetch_add(1, Ordering::AcqRel);
        let mut freed = 0;
        state.entries.retain(|k, e| {
            let hit = k
                .tables
                .iter()
                .any(|t| t.location.starts_with(path) || path.starts_with(&t.location));
            if hit {
                freed += e.bytes;
            }
            !hit
        });
        state.bytes = state.bytes.saturating_sub(freed);
    }

    /// Drops every entry.
    pub(crate) fn clear(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.generation.fetch_add(1, Ordering::AcqRel);
        self.invalidated
            .fetch_add(state.entries.len() as u64, Ordering::Relaxed);
        state.entries.clear();
        state.bytes = 0;
    }

    /// Entries held, for tests.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entries
            .len()
    }
}

/// What a plan reads and calls, for eligibility.
#[derive(Default)]
struct Found {
    tables: Vec<Vec<String>>,
    files: bool,
    /// A call whose value is not a function of the data: a non-deterministic function, a clock or
    /// session function (the registry marks `current_timestamp` deterministic because Spark folds
    /// it per query), or a session-registered UDF (its identity is a name and a version the
    /// implementer chooses, not the code).
    nondeterministic: bool,
    session_udf: bool,
}

/// The functions that return a clock, the session or the environment: constant within a query,
/// different between runs, and marked deterministic by the registry.
const RUN_DEPENDENT: &[&str] = &[
    "current_timestamp",
    "now",
    "current_date",
    "curdate",
    "current_time",
    "localtimestamp",
    "localtime",
    "current_timezone",
    "current_user",
    "user",
    "session_user",
    "current_catalog",
    "current_database",
    "current_schema",
    "current_path",
    "version",
    "uuid",
    "reflect",
    "java_method",
    "try_reflect",
    "input_file_block_start",
    "input_file_block_length",
];

fn survey(plan: &Arc<LogicalPlan>) -> Found {
    enum Step<'a> {
        Plan(&'a Arc<LogicalPlan>),
        Expr(&'a Arc<Expr>),
    }
    let mut found = Found::default();
    let mut stack = vec![Step::Plan(plan)];
    while let Some(step) = stack.pop() {
        match step {
            Step::Plan(p) => {
                if let PlanNode::Scan(s) = &p.node {
                    match &s.source {
                        ReadSource::Table { name, .. } => found.tables.push(name.clone()),
                        _ => found.files = true,
                    }
                }
                p.for_each_child(&mut |c| stack.push(Step::Plan(c)));
                p.for_each_expr(&mut |e| stack.push(Step::Expr(e)));
            }
            Step::Expr(e) => {
                match &e.node {
                    ExprNode::Subquery { plan, .. } => stack.push(Step::Plan(plan)),
                    ExprNode::Resolved(ResolvedExpr::Function(f)) => {
                        if !f.deterministic || RUN_DEPENDENT.contains(&f.id.name()) {
                            found.nondeterministic = true;
                        }
                        if matches!(f.id, krishiv_plan::FunctionId::Udf { .. }) {
                            found.session_udf = true;
                        }
                    }
                    _ => {}
                }
                e.for_each_child(&mut |c| stack.push(Step::Expr(c)));
            }
        }
    }
    found
}

impl Session {
    /// The result-cache key of an analyzed query, or why it must not be cached: it reads
    /// anything but permanent catalog tables, or calls a non-deterministic function.
    pub(crate) fn result_verdict(&self, a: &Analysis) -> Result<ResultKey, &'static str> {
        let found = survey(a.analyzed.plan.root());
        if found.files {
            return Err("it reads files by path");
        }
        if found.nondeterministic {
            return Err("it calls a non-deterministic, clock or session function");
        }
        if found.session_udf {
            return Err("it calls a session-registered function");
        }
        let mut tables = Vec::new();
        for parts in &found.tables {
            let name = TableName::resolve(parts, &self.state.catalog.current_namespace(), &a.env)
                .map_err(|_| "it reads a table that does not resolve")?;
            if self.temp_entry_of(&name, &a.env) {
                return Err("it reads a temporary or in-memory table");
            }
            let table = self
                .state
                .catalog
                .store
                .table(&name.namespace, &name.name)
                .ok_or("it reads a table that is not in the catalog")?;
            if table.view.is_some() {
                return Err("it reads a view that is not inlined");
            }
            tables.push(TableId {
                namespace: table.namespace.clone(),
                name: table.name.clone(),
                location: table.location.clone(),
                created_ms: table.created_ms,
            });
        }
        tables.sort();
        tables.dedup();
        let settings = RESULT_SETTINGS
            .iter()
            .map(|k| (*k, crate::conf::get(&a.env.config, k).unwrap_or_default()))
            .collect();
        Ok(ResultKey {
            plan: CacheKey::of(&a.analyzed),
            tables,
            settings,
        })
    }

    /// [`Self::result_verdict`] as an option, for the action boundary.
    pub(crate) fn result_key(&self, a: &Analysis) -> Option<ResultKey> {
        self.result_verdict(a).ok()
    }

    /// The line `EXPLAIN` prints about the result cache, when it is enabled for the query.
    pub(crate) fn result_cache_note(&self, a: &Analysis) -> Option<String> {
        let settings = settings(&a.env)?;
        let verdict = match self.result_verdict(a) {
            Err(why) => format!("not cached: {why}"),
            Ok(key) => match global().age(&key, settings.ttl) {
                Some(age) => format!("hit (metadata_age_ms={})", age.as_millis()),
                None => "miss (the result is stored when the query has run)".to_owned(),
            },
        };
        let (entries, bytes, hits, misses, stores, invalidated) = global().stats();
        Some(format!(
            "{verdict}\nstore: {entries} entries, {bytes} bytes; since start: {hits} hits, {misses} misses, {stores} stored, {invalidated} invalidated"
        ))
    }
}

/// `stream`, storing its batches under `key` when it ends cleanly.
pub(crate) fn fill(
    stream: RecordBatchStream,
    key: ResultKey,
    generation: u64,
    settings: Settings,
) -> RecordBatchStream {
    Box::pin(FillStream {
        inner: stream,
        key: Some(key),
        batches: Vec::new(),
        bytes: 0,
        generation,
        max_bytes: settings.max_bytes,
        ttl: settings.ttl,
        abandoned: false,
    })
}

struct FillStream {
    inner: RecordBatchStream,
    key: Option<ResultKey>,
    batches: Vec<RecordBatch>,
    bytes: usize,
    generation: u64,
    max_bytes: usize,
    ttl: Duration,
    abandoned: bool,
}

impl Stream for FillStream {
    type Item = Result<RecordBatch, KrishivError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                if !self.abandoned {
                    // Whole buffers, not the slice a batch shows: a sliced batch keeps its parent alive.
                    self.bytes = self.bytes.saturating_add(batch.get_array_memory_size());
                    if self.bytes > self.max_bytes {
                        // Too big to keep: stop holding it.
                        self.abandoned = true;
                        self.batches = Vec::new();
                    } else {
                        self.batches.push(batch.clone());
                    }
                }
                Poll::Ready(Some(Ok(batch)))
            }
            Poll::Ready(Some(Err(e))) => {
                self.abandoned = true;
                self.batches = Vec::new();
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                if !self.abandoned
                    && let Some(key) = self.key.take()
                {
                    let batches = std::mem::take(&mut self.batches);
                    global().insert(
                        key,
                        batches,
                        self.bytes,
                        self.generation,
                        self.max_bytes,
                        self.ttl,
                    );
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// A stream over the stored `batches`.
pub(crate) fn replay(batches: Vec<RecordBatch>) -> RecordBatchStream {
    use futures::StreamExt as _;
    futures::stream::iter(batches.into_iter().map(Ok)).boxed()
}

/// Whether `env`'s configuration asks for the result cache.
pub(crate) fn settings(env: &QueryEnv) -> Option<Settings> {
    Settings::of(&env.config)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use arrow_array::cast::AsArray;
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use futures::executor::block_on;

    use super::*;
    use crate::{Principal, PrincipalKind, Session};
    use std::sync::OnceLock;

    fn session_as(user: &str) -> Session {
        let s = Session::builder()
            .principal(Principal::new(user, PrincipalKind::User))
            .build()
            .unwrap();
        s.conf().set("krishiv.resultCache.enabled", "true").unwrap();
        s
    }

    fn run(s: &Session, sql: &str) -> Vec<RecordBatch> {
        block_on(async { s.sql(sql).await.unwrap().collect().await.unwrap() })
    }

    fn rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(RecordBatch::num_rows).sum()
    }

    /// The key the session would cache `sql` under, or `None` when it must not.
    fn key_of(s: &Session, sql: &str) -> Option<ResultKey> {
        let (session, sql) = (s.clone(), sql.to_owned());
        block_on(s.on_executor(async move {
            let df = session.sql(&sql).await.unwrap();
            let a = df.analysis().await.unwrap();
            Ok::<_, KrishivError>(session.result_key(&a))
        }))
        .unwrap()
    }

    fn cached(key: &ResultKey) -> bool {
        global().lookup(key, Duration::ZERO).is_some()
    }

    fn table(s: &Session, name: &str, values: &str) {
        run(
            s,
            &format!("CREATE TABLE {name} USING parquet AS SELECT * FROM VALUES {values} AS v(a)"),
        );
    }

    #[test]
    fn it_is_off_by_default() {
        let s = Session::new().unwrap();
        table(&s, "rc_off", "(1), (2)");
        run(&s, "SELECT a FROM rc_off");
        let key = key_of(&s, "SELECT a FROM rc_off").unwrap();
        assert!(!cached(&key));
    }

    #[test]
    fn a_query_over_a_permanent_table_is_served_from_the_cache() {
        let s = session_as("ana");
        table(&s, "rc_hit", "(1), (2), (3)");
        let key = key_of(&s, "SELECT a FROM rc_hit WHERE a > 1").unwrap();
        assert!(!cached(&key));
        assert_eq!(rows(&run(&s, "SELECT a FROM rc_hit WHERE a > 1")), 2);
        assert!(cached(&key), "the rows were stored when the stream ended");
        // The second run is a hit: the stored batches are replayed.
        assert_eq!(rows(&run(&s, "SELECT a FROM rc_hit WHERE a > 1")), 2);
    }

    #[test]
    fn writes_refresh_and_clear_drop_the_entries_that_read_the_table() {
        let s = session_as("ana");
        table(&s, "rc_inv", "(1), (2)");
        let sql = "SELECT a FROM rc_inv";
        let key = key_of(&s, sql).unwrap();
        for step in [
            "INSERT INTO rc_inv VALUES (3)",
            "REFRESH TABLE rc_inv",
            "CLEAR CACHE",
        ] {
            run(&s, sql);
            assert!(cached(&key), "stored before `{step}`");
            run(&s, step);
            assert!(!cached(&key), "`{step}` dropped it");
        }
        // After the insert the next run sees the new row, not the old result.
        assert_eq!(rows(&run(&s, sql)), 3);
    }

    #[test]
    fn a_write_to_another_table_leaves_the_entry() {
        let s = session_as("ana");
        table(&s, "rc_keep", "(1)");
        table(&s, "rc_other", "(1)");
        let key = key_of(&s, "SELECT a FROM rc_keep").unwrap();
        run(&s, "SELECT a FROM rc_keep");
        run(&s, "INSERT INTO rc_other VALUES (2)");
        assert!(cached(&key));
    }

    #[test]
    fn what_is_not_deterministic_or_not_permanent_is_never_cached() {
        let s = session_as("ana");
        table(&s, "rc_nd", "(1)");
        assert!(key_of(&s, "SELECT a, rand() FROM rc_nd").is_none());
        assert!(key_of(&s, "SELECT uuid() AS u FROM rc_nd").is_none());
        // A temporary view over a permanent table is inlined: the plan scans the table, so its
        // key covers the view's query and it is eligible.
        run(&s, "CREATE TEMPORARY VIEW rc_tv AS SELECT a FROM rc_nd");
        assert!(key_of(&s, "SELECT a FROM rc_tv").is_some());
        // An in-memory registration, and a view over one, are session-local.
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();
        s.register_batches("rc_mem", schema, vec![batch]).unwrap();
        assert!(key_of(&s, "SELECT a FROM rc_mem").is_none());
        run(&s, "CREATE TEMPORARY VIEW rc_mv AS SELECT a FROM rc_mem");
        assert!(key_of(&s, "SELECT a FROM rc_mv").is_none());
        // A read by path has no version.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p").to_str().unwrap().to_owned();
        block_on(s.table("rc_nd").write().parquet(&path)).unwrap();
        assert!(key_of(&s, &format!("SELECT * FROM parquet.`{path}`")).is_none());
        // A plain deterministic one is eligible.
        assert!(key_of(&s, "SELECT a + 1 FROM rc_nd").is_some());
    }

    #[test]
    fn an_entry_expires_after_the_ttl() {
        let s = session_as("ana");
        table(&s, "rc_ttl", "(1)");
        let key = key_of(&s, "SELECT a FROM rc_ttl").unwrap();
        run(&s, "SELECT a FROM rc_ttl");
        assert!(global().lookup(&key, Duration::from_secs(3600)).is_some());
        std::thread::sleep(Duration::from_millis(20));
        assert!(
            global().lookup(&key, Duration::from_millis(5)).is_none(),
            "older than the ttl"
        );
        assert!(!cached(&key), "and dropped");
    }

    #[test]
    fn a_result_over_the_cap_is_not_stored() {
        let s = session_as("ana");
        s.conf().set("krishiv.resultCache.maxBytes", "1").unwrap();
        table(&s, "rc_cap", "(1), (2)");
        let key = key_of(&s, "SELECT a FROM rc_cap").unwrap();
        assert_eq!(rows(&run(&s, "SELECT a FROM rc_cap")), 2);
        assert!(!cached(&key));
    }

    #[test]
    fn a_fill_started_before_an_invalidation_is_dropped() {
        let cache = ResultCache::default();
        let s = session_as("ana");
        table(&s, "rc_race", "(1)");
        let key = key_of(&s, "SELECT a FROM rc_race").unwrap();
        let generation = cache.generation();
        cache.invalidate_table("other");
        assert!(
            !cache.insert(
                key.clone(),
                Vec::new(),
                10,
                generation,
                4096,
                Duration::ZERO
            ),
            "stale generation"
        );
        assert!(cache.insert(
            key,
            Vec::new(),
            10,
            cache.generation(),
            4096,
            Duration::ZERO
        ));
        assert_eq!(cache.len(), 1);
        // Oldest entries go first when the cap would be passed.
        let s2 = session_as("bob");
        table(&s2, "rc_race2", "(1)");
        let key2 = key_of(&s2, "SELECT a FROM rc_race2").unwrap();
        assert!(cache.insert(
            key2.clone(),
            Vec::new(),
            3500,
            cache.generation(),
            4096,
            Duration::ZERO
        ));
        assert_eq!(cache.len(), 1, "the first entry made room");
        assert!(cache.lookup(&key2, Duration::ZERO).is_some());
        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn principals_and_sessions_do_not_share_what_they_must_not() {
        let ana = session_as("ana");
        let bob = session_as("bob");
        table(&ana, "rc_who", "(1)");
        table(&bob, "rc_who", "(1), (2)");
        let sql = "SELECT a FROM rc_who";
        let (ka, kb) = (key_of(&ana, sql).unwrap(), key_of(&bob, sql).unwrap());
        assert_ne!(ka, kb, "another principal and other data: another key");
        // Each session still gets its own table's answer, cached or not, in either order.
        for _ in 0..2 {
            assert_eq!(rows(&run(&ana, sql)), 1);
            assert_eq!(rows(&run(&bob, sql)), 2);
        }
        // A relation cached in the session is substituted into the plans that match it: those
        // results depend on the session, so they stay out of the shared store.
        let q = "SELECT a FROM rc_who WHERE a < 5";
        let df = block_on(ana.sql(q)).unwrap().cache();
        block_on(df.collect()).unwrap(); // fills the session's relation (a fresh run, stored)
        let key = key_of(&ana, q).unwrap();
        global().clear();
        assert_eq!(rows(&run(&ana, q)), 1); // now served by substitution
        assert!(
            !cached(&key),
            "a plan that used the session's cached relation was not stored"
        );
    }

    fn explain(s: &Session, sql: &str) -> String {
        let batches = run(s, &format!("EXPLAIN {sql}"));
        let col = batches[0].column(0);
        match col.data_type() {
            DataType::Utf8View => col.as_string_view().value(0).to_owned(),
            DataType::LargeUtf8 => col.as_string::<i64>().value(0).to_owned(),
            _ => col.as_string::<i32>().value(0).to_owned(),
        }
    }

    #[test]
    fn explain_reports_the_cache_only_when_it_is_on() {
        let off = Session::new().unwrap();
        table(&off, "rc_ex_off", "(1)");
        assert!(!explain(&off, "SELECT a FROM rc_ex_off").contains("Result Cache"));

        let s = session_as("ana");
        table(&s, "rc_ex", "(1), (2)");
        let sql = "SELECT a FROM rc_ex";
        assert!(
            explain(&s, sql).contains("== Result Cache ==\nmiss"),
            "{}",
            explain(&s, sql)
        );
        run(&s, sql);
        let text = explain(&s, sql);
        assert!(
            text.contains("== Result Cache ==\nhit (metadata_age_ms="),
            "{text}"
        );
        let text = explain(&s, "SELECT a, rand() FROM rc_ex");
        assert!(
            text.contains("not cached: it calls a non-deterministic, clock or session function"),
            "{text}"
        );
        run(&s, "CREATE TEMPORARY VIEW rc_ex_v AS SELECT 1 AS a");
        let text = explain(&s, "SELECT a FROM rc_ex_v");
        assert!(!text.contains("hit (metadata_age_ms"), "{text}");
    }

    /// `add_one(x) = x + 1`, a session-registered function.
    #[derive(Debug)]
    struct AddOne;

    impl krishiv_udf_sdk::ScalarUdf for AddOne {
        fn name(&self) -> &str {
            "add_one"
        }

        fn version(&self) -> u32 {
            1
        }

        fn signature(&self) -> &krishiv_udf_sdk::Signature {
            static SIG: OnceLock<krishiv_udf_sdk::Signature> = OnceLock::new();
            SIG.get_or_init(|| {
                krishiv_udf_sdk::Signature::Exact(vec![krishiv_udf_sdk::UdfInputType::Exact(
                    krishiv_types::DataType::Integer,
                )])
            })
        }

        fn return_field(
            &self,
            _: &[krishiv_udf_sdk::ArgField],
        ) -> Result<krishiv_udf_sdk::ReturnField, krishiv_udf_sdk::UdfError> {
            Ok(krishiv_udf_sdk::ReturnField {
                data_type: krishiv_types::DataType::Integer,
                nullable: true,
            })
        }

        fn invoke(
            &self,
            args: &[krishiv_udf_sdk::UdfArg<'_>],
            _rows: usize,
            _: &krishiv_udf_sdk::InvokeContext,
        ) -> Result<arrow_array::ArrayRef, krishiv_udf_sdk::UdfError> {
            Ok(args[0].array())
        }
    }

    #[test]
    fn clock_session_and_udf_calls_are_never_cached() {
        let s = session_as("ana");
        table(&s, "rc_clock", "(1)");
        for call in [
            "current_timestamp()",
            "now()",
            "current_date()",
            "current_timezone()",
            "current_user()",
            "uuid()",
            "version()",
        ] {
            assert!(
                key_of(&s, &format!("SELECT a, {call} FROM rc_clock")).is_none(),
                "`{call}` made the query eligible"
            );
        }
        s.register_udf(AddOne).unwrap();
        assert!(key_of(&s, "SELECT add_one(a) FROM rc_clock").is_none());
        let why = block_on(s.on_executor({
            let s = s.clone();
            async move {
                let df = s.sql("SELECT add_one(a) FROM rc_clock").await.unwrap();
                let a = df.analysis().await.unwrap();
                Ok::<_, KrishivError>(s.result_verdict(&a).err())
            }
        }))
        .unwrap();
        assert_eq!(why, Some("it calls a session-registered function"));
    }

    #[test]
    fn settings_that_change_rows_are_part_of_the_key() {
        let s = session_as("ana");
        table(&s, "rc_set", "(1)");
        let a = key_of(&s, "SELECT a FROM rc_set").unwrap();
        s.conf().set("krishiv.deterministicOrder", "true").unwrap();
        let b = key_of(&s, "SELECT a FROM rc_set").unwrap();
        assert_ne!(a, b);
        s.conf()
            .set("spark.sql.files.ignoreCorruptFiles", "true")
            .unwrap();
        assert_ne!(b, key_of(&s, "SELECT a FROM rc_set").unwrap());
    }

    #[test]
    fn partition_and_path_changes_drop_the_entry() {
        let s = session_as("ana");
        run(
            &s,
            "CREATE TABLE rc_part (a INT, p INT) USING parquet PARTITIONED BY (p)",
        );
        run(&s, "INSERT INTO rc_part VALUES (1, 1)");
        let sql = "SELECT a FROM rc_part";
        let key = key_of(&s, sql).unwrap();
        // The partitioned read may be unsupported: the entry is only stored when the query ran.
        if block_on(async { s.sql(sql).await.unwrap().collect().await }).is_ok() && cached(&key) {
            run(&s, "ALTER TABLE rc_part ADD PARTITION (p = 2)");
            assert!(!cached(&key), "ADD PARTITION dropped it");
        }
        // A write by path into an unpartitioned table's directory.
        table(&s, "rc_dir", "(1), (2)");
        let key = key_of(&s, "SELECT a FROM rc_dir").unwrap();
        run(&s, "SELECT a FROM rc_dir");
        assert!(cached(&key));
        let location = key.tables[0].location.clone();
        block_on(
            s.table("rc_dir")
                .write()
                .mode("append")
                .parquet(location.to_str().unwrap()),
        )
        .unwrap();
        assert!(
            !cached(&key),
            "a write by path into the table's directory dropped it"
        );
    }

    #[test]
    fn empty_results_cost_an_entry_and_expired_entries_go_on_insert() {
        let cache = ResultCache::default();
        let s = session_as("ana");
        table(&s, "rc_mem1", "(1)");
        let k1 = key_of(&s, "SELECT a FROM rc_mem1").unwrap();
        let k2 = key_of(&s, "SELECT a + 1 FROM rc_mem1").unwrap();
        let gen_ = cache.generation();
        // A cap that holds two empty entries only if each is charged at least its overhead.
        assert!(cache.insert(
            k1.clone(),
            Vec::new(),
            0,
            gen_,
            2 * ENTRY_OVERHEAD,
            Duration::from_millis(5)
        ));
        std::thread::sleep(Duration::from_millis(15));
        // k1 expired: the next insert drops it without anyone asking for it.
        assert!(cache.insert(k2, Vec::new(), 0, gen_, 2 * ENTRY_OVERHEAD, Duration::ZERO));
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(&k1, Duration::ZERO).is_none());
    }

    #[test]
    fn the_store_counts_hits_misses_stores_and_invalidations() {
        let cache = ResultCache::default();
        let s = session_as("ana");
        table(&s, "rc_cnt", "(1)");
        let key = key_of(&s, "SELECT a FROM rc_cnt").unwrap();
        assert!(cache.lookup(&key, Duration::ZERO).is_none());
        assert!(cache.insert(
            key.clone(),
            Vec::new(),
            10,
            cache.generation(),
            1 << 20,
            Duration::ZERO
        ));
        assert!(cache.lookup(&key, Duration::ZERO).is_some());
        assert!(
            cache.lookup(&key, Duration::from_nanos(1)).is_none(),
            "expired counts as a miss"
        );
        assert!(cache.insert(
            key,
            Vec::new(),
            10,
            cache.generation(),
            1 << 20,
            Duration::ZERO
        ));
        cache.invalidate_table("rc_cnt");
        // entries, bytes, hits, misses, stores, invalidated
        assert_eq!(cache.stats(), (0, 0, 1, 2, 2, 1));
        cache.clear();
        assert_eq!(
            cache.stats().5,
            1,
            "clearing an empty store invalidates nothing"
        );
    }
}
