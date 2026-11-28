// SPDX-License-Identifier: Apache-2.0
//! The session's SQL-bodied functions (`CREATE TEMPORARY FUNCTION`): the
//! definitions a session holds, and the function registry a query resolves
//! against.
//!
//! Definitions live in a map of the session, next to its catalog of
//! temporary tables and views. The map is replaced, never edited: a query
//! takes the current map when it starts ([`SqlFunctions::registry`]) and
//! reads that one map throughout, whatever other statements run meanwhile.
//! Every change bumps the map's version, which the registry reports as its
//! [`FunctionRegistry::version`], so a cached resolved plan keyed by the
//! analysis context's fingerprint never outlives the function it inlined.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use krishiv_common::redact::RedactedSql;
use krishiv_plan::analyzer::{FunctionLookup, FunctionRegistry, SparkProfile, SqlFunction};
use krishiv_plan::function::Builtin;
use krishiv_plan::{Expr, ExprNode, LogicalPlan, PlanNode, SourceId};

use crate::catalog::{ReadCandidate, SessionCatalog, StoredViews, expand_views_in};
use crate::store::{BoundFunctions, CatalogStore, StoredFunctions};
use krishiv_common::error::KrishivError;

/// A defined function and the statement that defined it.
#[derive(Clone)]
pub(crate) struct Entry {
    pub(crate) def: Arc<SqlFunction>,
    /// The `CREATE` statement, redacted: the text the body's spans index.
    pub(crate) text: RedactedSql,
    /// What `DESCRIBE FUNCTION EXTENDED` shows beyond the definition.
    pub(crate) described: Arc<Described>,
}

/// The parts of a function's `DESCRIBE FUNCTION EXTENDED` rows (Spark's `SQLFunction` fields) that
/// the definition itself does not keep.
#[derive(Clone, Debug, Default)]
pub(crate) struct Described {
    /// The function's `COMMENT`.
    pub(crate) comment: Option<String>,
    /// Each parameter's `DEFAULT` as written, in parameter order.
    pub(crate) defaults: Vec<Option<String>>,
    /// Declared, or else the body's own (Spark: the analyzed body's `deterministic`).
    pub(crate) deterministic: bool,
    /// The SQL configuration captured at creation, as Spark's `CapturesConfig` takes it: every set
    /// `spark.sql.*` key it keeps, and always the session time zone and ANSI mode; sorted by key.
    pub(crate) configs: Vec<(String, String)>,
    /// When the function was created, in milliseconds since the epoch.
    pub(crate) created_ms: i64,
    /// The columns of a table function that declares none (`RETURNS TABLE` alone): its query's.
    pub(crate) columns: Vec<(String, krishiv_types::DataType)>,
}

/// `entry`'s description with the query columns a probe found, kept only for a table function
/// that declares no columns.
pub(crate) fn with_columns(
    entry: &Entry,
    columns: Vec<(String, krishiv_types::DataType)>,
) -> Arc<Described> {
    if entry.def.returns_table.as_ref().is_some_and(Vec::is_empty) {
        Arc::new(Described {
            columns,
            ..(*entry.described).clone()
        })
    } else {
        Arc::clone(&entry.described)
    }
}

impl Described {
    /// What `CREATE FUNCTION` `f` declares, at creation: its comment, its defaults as written,
    /// `deterministic` (settled later from the body when `f` declares nothing), and the
    /// configuration it captures now.
    pub(crate) fn declared(
        f: &krishiv_sql::ast::CreateFunction,
        deterministic: bool,
        redacted: &RedactedSql,
        conf: &crate::conf::ConfStore,
    ) -> Self {
        Self {
            comment: f.characteristics.comment.clone(),
            defaults: f
                .params
                .iter()
                .map(|p| {
                    p.default
                        .as_ref()
                        .map(|d| redacted.fragment(d.span.range()).trim().to_owned())
                })
                .collect(),
            deterministic,
            configs: captured_configs(conf),
            created_ms: crate::table_commands::now_ms(),
            columns: Vec::new(),
        }
    }
}

/// Whether the body of `def` calls no nondeterministic function (Spark: the analyzed body's
/// `deterministic`), for `DESCRIBE FUNCTION` of a function that declares neither. A call of a SQL
/// function is nondeterministic when that function is declared `NOT DETERMINISTIC` or its own body is
/// (Spark's `SQLScalarFunction.deterministic` is the declaration and its children, the inlined body
/// among them), so the walk goes into the bodies of the SQL functions the body calls, scalar and table.
pub(crate) fn body_deterministic(
    registry: &dyn FunctionRegistry,
    profile: SparkProfile,
    def: &SqlFunction,
) -> bool {
    let mut seen = BTreeSet::new();
    seen.insert(def.name.clone());
    !body_calls_nondeterministic(registry, profile, def, &mut seen)
}

/// [`body_deterministic`]'s walk. `seen` holds the SQL functions already on the walk, so a cycle
/// (Spark refuses one at creation; `CREATE OR REPLACE` could still close one here) ends.
fn body_calls_nondeterministic(
    registry: &dyn FunctionRegistry,
    profile: SparkProfile,
    def: &SqlFunction,
    seen: &mut BTreeSet<String>,
) -> bool {
    any_call(vec![&*def.body], |name: &[String]| {
        match registry.lookup(name, profile) {
            FunctionLookup::Sql(d) => {
                !d.deterministic
                    || (seen.insert(d.name.clone())
                        && body_calls_nondeterministic(registry, profile, &d, seen))
            }
            _ => name
                .last()
                .is_some_and(|n| registry.spark_deterministic(n) == Some(false)),
        }
    })
}

/// Keys Spark's `CapturesConfig` never captures (`configPrefixDenyList`, 4.3.0).
const UNCAPTURED: [&str; 12] = [
    "spark.sql.view.maxNestedViewDepth",
    "spark.sql.optimizer.",
    "spark.sql.codegen.",
    "spark.sql.execution.",
    "spark.sql.shuffle.",
    "spark.sql.adaptive.",
    "spark.sql.hive.convertMetastoreParquet",
    "spark.sql.hive.convertMetastoreOrc",
    "spark.sql.hive.convertInsertingPartitionedTable",
    "spark.sql.hive.convertInsertingUnpartitionedTable",
    "spark.sql.hive.convertMetastoreCtas",
    "spark.sql.additionalRemoteRepositories",
];

/// Keys Spark's `CapturesConfig` never captures by exact name (`singlePassResolverDenyList`, 4.3.0); the
/// other `spark.sql.analyzer.singlePassResolver.*` keys are captured.
const UNCAPTURED_EXACT: [&str; 2] = [
    "spark.sql.analyzer.singlePassResolver.enabledTentatively",
    "spark.sql.analyzer.singlePassResolver.dualRunWithLegacy",
];

/// The configuration a function created now captures (Spark's `sqlConfigsToProps`): the set
/// `spark.sql.*` keys that are not static outside
/// [`UNCAPTURED`] and [`UNCAPTURED_EXACT`] (`spark.sql.optimizer.disableHints` is kept), then the time
/// zone and ANSI mode when not set; sorted by key. Values show as `SET` shows them, redacted where it
/// redacts: Spark prints the raw captured value, and Krishiv hides a secret-like one on purpose.
///
/// Spark captures only keys `SQLConf.isModifiable` accepts, so not a key no SQL entry registers
/// (`spark.sql.my.flag`, a catalog declaration). Krishiv honours Spark keys it does not register
/// (`spark.sql.caseSensitive`), so its registry cannot tell them apart.
// TODO(R1-D-9 follow-up, front-1 queue): use the generated Spark entry table of #1080 (`spark_default`) here.
pub(crate) fn captured_configs(conf: &crate::conf::ConfStore) -> Vec<(String, String)> {
    const ALWAYS: [&str; 2] = ["spark.sql.session.timeZone", "spark.sql.ansi.enabled"];
    let mut out: BTreeMap<String, String> = conf
        .explicit_entries()
        .into_iter()
        .filter(|(k, _)| {
            k.starts_with("spark.sql.")
                && crate::conf::is_modifiable(k)
                && !UNCAPTURED_EXACT.contains(&k.as_str())
                && (k == "spark.sql.optimizer.disableHints"
                    || !UNCAPTURED.iter().any(|d| k.starts_with(d)))
        })
        .map(|(k, v)| {
            let shown = crate::conf::shown(&k, v);
            (k, shown)
        })
        .collect();
    for key in ALWAYS {
        if !out.contains_key(key)
            && let Some(v) = conf.value(key)
        {
            out.insert(key.to_owned(), v);
        }
    }
    out.into_iter().collect()
}

/// The functions of a session at one version.
#[derive(Clone, Default)]
pub(crate) struct FunctionMap {
    version: u64,
    /// By lower-case name: Spark's function names are case-insensitive
    /// whatever `spark.sql.caseSensitive` says.
    entries: BTreeMap<String, Entry>,
}

impl FunctionMap {
    /// The function `name`.
    pub(crate) fn get(&self, name: &str) -> Option<&Entry> {
        self.entries.get(&name.to_lowercase())
    }

    /// The defined functions' names, sorted (lower case).
    pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// This map with `entry` defined (replacing any of its name), at the
    /// next version.
    pub(crate) fn with(&self, entry: Entry) -> Self {
        let mut next = self.clone();
        next.entries.insert(entry.def.name.clone(), entry);
        next.version += 1;
        next
    }

    /// This map without `name`, at the next version.
    fn without(&self, name: &str) -> Self {
        let mut next = self.clone();
        next.entries.remove(&name.to_lowercase());
        next.version += 1;
        next
    }

    /// The version: how many changes the session has made.
    pub(crate) fn version(&self) -> u64 {
        self.version
    }
}

/// The session's functions: the current [`FunctionMap`].
#[derive(Default)]
pub(crate) struct SqlFunctions {
    map: RwLock<Arc<FunctionMap>>,
}

impl SqlFunctions {
    /// Functions starting from this session's current map; later changes
    /// to either do not reach the other (the map is replaced, never edited).
    pub(crate) fn copy(&self) -> Self {
        Self {
            map: RwLock::new(self.snapshot()),
        }
    }

    /// The current map; what a query reads throughout.
    pub(crate) fn snapshot(&self) -> Arc<FunctionMap> {
        Arc::clone(&self.map.read().unwrap_or_else(PoisonError::into_inner))
    }

    /// Defines `entry`, replacing a function of its name.
    ///
    /// `expected` is the version of the map the caller checked the
    /// definition against: when another statement changed the functions
    /// meanwhile, nothing is defined and `false` is returned, so the caller
    /// checks again against the new map (a definition is never validated
    /// against a map that is no longer current).
    pub(crate) fn define(&self, entry: Entry, expected: u64) -> bool {
        let mut map = self.map.write().unwrap_or_else(PoisonError::into_inner);
        if map.version() != expected {
            return false;
        }
        *map = Arc::new(map.with(entry));
        true
    }

    /// Drops `name`; whether it was defined.
    pub(crate) fn drop_function(&self, name: &str) -> bool {
        let mut map = self.map.write().unwrap_or_else(PoisonError::into_inner);
        if map.get(name).is_none() {
            return false;
        }
        *map = Arc::new(map.without(name));
        true
    }
}

/// The registry of one query: the built-in functions, then the session's
/// SQL-bodied functions as of the map taken when the query started.
/// Built-ins rank first, as in Spark's default resolution order.
///
/// A function's body inlines the temporary views it reads, as the session
/// does for the query itself (the analyzer sees no view); the result is
/// kept per function for the query, so the analyzer meets one body however
/// often it asks.
pub(crate) struct QueryFunctions {
    base: Arc<dyn FunctionRegistry>,
    map: Arc<FunctionMap>,
    catalog: Arc<SessionCatalog>,
    case_sensitive: bool,
    /// The current catalog is another than `spark_catalog` (`system`, which has no namespaces, or a
    /// declared one): a two-part call is no stored function, and a stored body's calls are
    /// `spark_catalog`'s.
    outside_session: bool,
    /// The functions stored in the catalog's namespaces, as of the start of
    /// the query, and the namespace a bare name resolves in.
    stored: Arc<StoredFunctions>,
    /// The SQL path (`SET PATH`) a bare name resolves through.
    path: Vec<Vec<String>>,
    /// What reading a permanent view in a body needs; `None` for a registry
    /// that inlines temporary views only.
    views: Option<ViewSource>,
    /// The bodies expanded so far, what their expansion looked up and the views it inlined: shared
    /// with [`Self::recording_fork`]s, so a definer view's analysis and the query's meet one body
    /// (and one record of its lookups) however often it is called.
    bodies: Arc<Bodies>,
    /// The stored functions a lookup resolved so far (namespace, name as stored), each once: a
    /// call needs `Select` on the function (P3-T062 §2.4).
    called: Mutex<Vec<(String, String)>>,
    /// The stored function a `CREATE FUNCTION` is checking (namespace, name): its probe calls
    /// it, which is no call by the creator, so it is not recorded in `called`.
    defining: Option<(String, String)>,
    /// The stored functions a definer view's text calls (asked of its definer, review r5 M7), and
    /// of those the ones only such texts call: the query's principal is not asked those.
    definer_calls: Mutex<DefinerCalls>,
}

/// Stored functions by (namespace, name as stored), each once.
pub(crate) type Calls = Vec<(String, String)>;

/// [`QueryFunctions::definer_calls`].
#[derive(Default)]
struct DefinerCalls {
    /// The stored functions definer views' texts call, asked of their definers.
    asked: Calls,
    /// Of [`Self::asked`], those only such texts call.
    skipped: Calls,
    /// The bodies (by `cache_key`) the query calls outside definer views' texts, directly or
    /// through bodies; `None` until settled, or when unknown (every body is the principal's).
    own_bodies: Option<Vec<String>>,
}

/// What a definer view's text reaches through stored functions ([`QueryFunctions::definer_scope`]),
/// all asked of its definer (review r5 M7, r6 M8).
pub(crate) struct DefinerScope {
    /// The stored functions it calls, directly or through bodies.
    pub(crate) calls: Calls,
    /// What the bodies it reaches look up, merged.
    pub(crate) body: BodyReads,
    /// Each stored function body it reaches, also asked of that function's owner (review r7 M9).
    pub(crate) owned: Vec<OwnedBody>,
}

/// One stored function body a definer view's text reaches: what it looks up and calls itself,
/// asked of the function's owner too (review r7 M9).
pub(crate) struct OwnedBody {
    /// The function (namespace, name).
    pub(crate) function: (String, String),
    /// The body's own lookups.
    pub(crate) reads: BodyReads,
    /// The stored functions the body calls itself.
    pub(crate) calls: Calls,
}

/// [`QueryFunctions::bodies`].
#[derive(Default)]
struct Bodies {
    /// Each body as expanded, by `cache_key`.
    expanded: Mutex<HashMap<String, Arc<SqlFunction>>>,
    /// What each body's expansion looked up, by `cache_key`, in order.
    reads: Mutex<Vec<BodyReads>>,
    /// The views the bodies inlined, for the query's view side table (`AnalyzedPlan::views`).
    views_read: Mutex<Vec<krishiv_plan::analyzer::views::ViewExpansion>>,
    /// What the temporary views the bodies expanded so far kept
    /// ([`crate::catalog::TempTable::View`]'s keeps: an observed frame's observations).
    keeps_read: Mutex<Vec<crate::catalog::Keep>>,
    /// The stored views the query's own expansions inlined, with their versions
    /// ([`QueryFunctions::note_views`], review r10 M10).
    views_noted: Mutex<BTreeMap<(String, String), u64>>,
}

/// The catalog-store lookups of one body's expansion (see [`crate::catalog::Expansion`]).
#[derive(Clone, Default)]
pub(crate) struct BodyReads {
    /// The body's `cache_key`.
    key: String,
    /// The stored function whose body it is (namespace, name), `None` for a temporary one.
    pub(crate) function: Option<(String, String)>,
    /// Its own lookups.
    pub(crate) reads: Vec<ReadCandidate>,
    /// The lookups of the texts of the definer views it reads (read as invoker ones, fails closed):
    /// a refusal of one names what the reader did not write, so it is redacted.
    pub(crate) definer_reads: Vec<ReadCandidate>,
    /// The first failure inside a stored view's text, and whether it is inside a definer's.
    pub(crate) deferred: Option<(KrishivError, bool)>,
    /// The stored views it inlined, at any depth (review r10 M10).
    pub(crate) views: Vec<crate::catalog::ReachedView>,
}

impl BodyReads {
    /// `self` with `other`'s lookups and failure added, each once.
    fn merge(mut self, other: &Self) -> Self {
        for read in &other.reads {
            if !self.reads.contains(read) {
                self.reads.push(read.clone());
            }
        }
        for read in &other.definer_reads {
            if !self.definer_reads.contains(read) {
                self.definer_reads.push(read.clone());
            }
        }
        if self.deferred.is_none() {
            self.deferred.clone_from(&other.deferred);
        }
        self.views.extend(other.views.iter().cloned());
        self
    }
}

/// How a body's permanent views are read: the parser options of the query
/// and the source ids of the views' lowered text.
#[derive(Clone)]
pub(crate) struct ViewSource {
    pub(crate) parser: krishiv_sql::ParserOptions,
    pub(crate) next_source: Arc<dyn Fn() -> SourceId + Send + Sync>,
}

impl QueryFunctions {
    pub(crate) fn new(
        base: Arc<dyn FunctionRegistry>,
        map: Arc<FunctionMap>,
        catalog: Arc<SessionCatalog>,
        case_sensitive: bool,
        views: Option<ViewSource>,
    ) -> Self {
        let stored = catalog.store.functions();
        let path = catalog.sql_path("");
        let outside_session = !catalog
            .current_catalog()
            .eq_ignore_ascii_case(crate::catalog::CATALOG);
        Self {
            outside_session,
            path,
            base,
            map,
            catalog,
            case_sensitive,
            stored,
            views,
            bodies: Arc::default(),
            called: Mutex::new(Vec::new()),
            defining: None,
            definer_calls: Mutex::default(),
        }
    }

    /// This registry with nothing looked up yet: what an analysis through it calls is recorded
    /// apart from the query's (a definer view's query, review r5 M7). The bodies it expands are the
    /// query's ([`Self::bodies`]).
    pub(crate) fn recording_fork(&self) -> Self {
        Self {
            base: Arc::clone(&self.base),
            map: Arc::clone(&self.map),
            catalog: Arc::clone(&self.catalog),
            case_sensitive: self.case_sensitive,
            outside_session: self.outside_session,
            stored: Arc::clone(&self.stored),
            path: self.path.clone(),
            views: self.views.clone(),
            bodies: Arc::clone(&self.bodies),
            called: Mutex::new(Vec::new()),
            defining: self.defining.clone(),
            definer_calls: Mutex::default(),
        }
    }

    /// Records `calls`, the stored functions a definer view's text called (its definer was asked).
    pub(crate) fn note_definer_calls(&self, calls: Vec<(String, String)>) {
        let mut noted = self
            .definer_calls
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        for call in calls {
            if !noted.asked.contains(&call) {
                noted.asked.push(call);
            }
        }
    }

    /// Of the stored functions definer views' texts called, those `plan` (the query, its definer
    /// views' texts `texts` aside) does not call itself, directly or through a body: the query's
    /// principal is not asked about them (review r5 M7), nor about what the bodies only those texts
    /// reach look up (review r6 M8: their definers were). A plan holding SQL text (not walked here)
    /// keeps every call and body the principal's.
    pub(crate) fn settle_definer_calls(
        &self,
        plan: &Arc<LogicalPlan>,
        texts: &krishiv_plan::analyzer::views::DefinerTexts,
        profile: SparkProfile,
    ) {
        let mut noted = self
            .definer_calls
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if noted.asked.is_empty() {
            return;
        }
        match self.calls_outside(plan, texts, profile) {
            Some((own, bodies)) => {
                noted.skipped = noted
                    .asked
                    .iter()
                    .filter(|c| !own.contains(c))
                    .cloned()
                    .collect();
                noted.own_bodies = Some(bodies);
            }
            None => {
                noted.skipped = Vec::new();
                noted.own_bodies = None;
            }
        }
    }

    /// What a definer view's query `query` calls and looks up through bodies, the texts `nested` of
    /// the definer views inside it aside (each is its own definer's, review r6 m1), as analyzed
    /// through this registry (a [`Self::recording_fork`]): the stored functions, then the bodies'
    /// lookups, all asked of the view's definer (review r5 M7, r6 M8). A query holding SQL text
    /// gives everything this registry called and every body's lookups (asked of the definer too,
    /// fails closed).
    pub(crate) fn definer_scope(
        &self,
        query: &Arc<LogicalPlan>,
        nested: &krishiv_plan::analyzer::views::DefinerTexts,
        profile: SparkProfile,
    ) -> DefinerScope {
        let reads = self
            .bodies
            .reads
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let (calls, reached) = match self.calls_outside(query, nested, profile) {
            Some((calls, bodies)) => (
                calls,
                reads
                    .into_iter()
                    .filter(|r| bodies.contains(&r.key))
                    .collect::<Vec<_>>(),
            ),
            None => (self.stored_calls(), reads),
        };
        let owned = reached
            .iter()
            .filter_map(|body| {
                let function = body.function.clone()?;
                Some(OwnedBody {
                    calls: self.direct_calls(&body.key, profile),
                    function,
                    reads: body.clone(),
                })
            })
            .collect();
        DefinerScope {
            body: reached.iter().fold(BodyReads::default(), BodyReads::merge),
            calls,
            owned,
        }
    }

    /// The stored functions the body expanded under `key` calls itself (not through the bodies of
    /// the functions it calls), as this registry resolves them.
    fn direct_calls(&self, key: &str, profile: SparkProfile) -> Calls {
        let Some(def) = self
            .bodies
            .expanded
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(key)
            .cloned()
        else {
            return Calls::new();
        };
        let fork = self.recording_fork();
        let mut plans: Vec<Arc<LogicalPlan>> = Vec::new();
        let mut exprs: Vec<Arc<Expr>> = std::iter::once(Arc::clone(&def.body))
            .chain(def.params.iter().filter_map(|p| p.default.clone()))
            .collect();
        loop {
            if let Some(p) = plans.pop() {
                if let PlanNode::TableFunction { name, .. } = &p.node {
                    let parts: Vec<String> = name
                        .split(krishiv_plan::TABLE_FUNCTION_NAME_SEP)
                        .map(str::to_owned)
                        .collect();
                    fork.lookup(&parts, profile);
                }
                p.for_each_expr(&mut |e| exprs.push(Arc::clone(e)));
                p.for_each_child(&mut |c| plans.push(Arc::clone(c)));
            } else if let Some(e) = exprs.pop() {
                match &e.node {
                    ExprNode::Function(call) => {
                        let mut parts = call.qualifier.clone();
                        parts.push(call.name.clone());
                        fork.lookup(&parts, profile);
                    }
                    ExprNode::Subquery { plan, .. } => plans.push(Arc::clone(plan)),
                    _ => {}
                }
                e.for_each_child(&mut |c| exprs.push(Arc::clone(c)));
            } else {
                break;
            }
        }
        fork.stored_calls()
    }

    /// The stored functions `plan` calls outside the subtrees `texts`, directly or through the
    /// bodies of the functions it calls, as this registry resolves them, and the bodies it reaches
    /// so (by `cache_key`); `None` for a plan holding SQL text.
    fn calls_outside(
        &self,
        plan: &Arc<LogicalPlan>,
        texts: &krishiv_plan::analyzer::views::DefinerTexts,
        profile: SparkProfile,
    ) -> Option<(Calls, Vec<String>)> {
        let fork = self.recording_fork();
        let mut bodies: Vec<String> = Vec::new();
        let mut plans: Vec<Arc<LogicalPlan>> = vec![Arc::clone(plan)];
        let mut exprs: Vec<Arc<Expr>> = Vec::new();
        let mut body = |found: FunctionLookup, exprs: &mut Vec<Arc<Expr>>| {
            if let FunctionLookup::Sql(def) = found
                && !bodies.contains(&cache_key(&def))
            {
                bodies.push(cache_key(&def));
                exprs.push(Arc::clone(&def.body));
                exprs.extend(def.params.iter().filter_map(|p| p.default.clone()));
            }
        };
        loop {
            if let Some(p) = plans.pop() {
                if texts.contains(&p) {
                    continue;
                }
                match &p.node {
                    PlanNode::Sql { .. } => return None,
                    PlanNode::TableFunction { name, .. } => {
                        let parts: Vec<String> = name
                            .split(krishiv_plan::TABLE_FUNCTION_NAME_SEP)
                            .map(str::to_owned)
                            .collect();
                        body(fork.lookup(&parts, profile), &mut exprs);
                    }
                    _ => {}
                }
                p.for_each_expr(&mut |e| exprs.push(Arc::clone(e)));
                p.for_each_child(&mut |c| plans.push(Arc::clone(c)));
            } else if let Some(e) = exprs.pop() {
                match &e.node {
                    ExprNode::Function(call) => {
                        let mut parts = call.qualifier.clone();
                        parts.push(call.name.clone());
                        body(fork.lookup(&parts, profile), &mut exprs);
                    }
                    ExprNode::Subquery { plan, .. } => plans.push(Arc::clone(plan)),
                    _ => {}
                }
                e.for_each_child(&mut |c| exprs.push(Arc::clone(c)));
            } else {
                break;
            }
        }
        Some((fork.stored_calls(), bodies))
    }

    /// The stored functions a lookup resolved so far that the query's principal is asked about:
    /// all but those only a definer view's text calls (review r5 M7).
    pub(crate) fn principal_calls(&self) -> Vec<(String, String)> {
        let skipped = self
            .definer_calls
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .skipped
            .clone();
        self.stored_calls()
            .into_iter()
            .filter(|c| !skipped.contains(c))
            .collect()
    }

    /// The stored functions a lookup resolved so far, in order, each once (see `called`).
    pub(crate) fn stored_calls(&self) -> Vec<(String, String)> {
        self.called
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// This registry as the query snapshot saw the current catalog (`spark_catalog` or not).
    pub(crate) fn outside_session(mut self, outside_session: bool) -> Self {
        self.outside_session = outside_session;
        self
    }

    /// This registry resolving bare names through `path`.
    pub(crate) fn with_path(mut self, path: Vec<Vec<String>>) -> Self {
        self.path = path;
        self
    }

    /// This registry checking the definition of the stored function `namespace.name`: its own
    /// call is not one [`Self::stored_calls`] reports.
    pub(crate) fn defining(mut self, namespace: &str, name: &str) -> Self {
        self.defining = Some((namespace.to_owned(), name.to_owned()));
        self
    }

    /// This registry over `stored` instead (the candidate functions a new
    /// stored definition is checked in).
    pub(crate) fn over_stored(&self, stored: Arc<StoredFunctions>) -> Self {
        Self {
            stored,
            ..self.over(Arc::clone(&self.map))
        }
    }

    /// This registry over `map` instead (the candidate map a new
    /// definition is checked in).
    pub(crate) fn over(&self, map: Arc<FunctionMap>) -> Self {
        Self::new(
            Arc::clone(&self.base),
            map,
            Arc::clone(&self.catalog),
            self.case_sensitive,
            self.views.clone(),
        )
        .with_path(self.path.clone())
    }

    /// The catalog-store lookups the bodies expanded so far made, in order, and the first failure
    /// inside a stored view's text, of the bodies the query's principal reaches: not those only a
    /// definer view's text reaches (review r6 M8; [`Self::definer_scope`] gives those to its
    /// definer). A query asks the policy about the reads before it raises the failure.
    pub(crate) fn body_reads(&self) -> BodyReads {
        let own = self
            .definer_calls
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .own_bodies
            .clone();
        self.bodies
            .reads
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|r| own.as_ref().is_none_or(|own| own.contains(&r.key)))
            .fold(BodyReads::default(), BodyReads::merge)
    }

    /// The views the bodies expanded so far inlined, one entry per expansion (a body is expanded once per
    /// query, however often it is called).
    pub(crate) fn body_views(&self) -> Vec<krishiv_plan::analyzer::views::ViewExpansion> {
        self.bodies
            .views_read
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// What the temporary views the bodies expanded so far kept, so a query whose only read of
    /// an observed view is inside a function body completes its observations.
    pub(crate) fn body_keeps(&self) -> Vec<crate::catalog::Keep> {
        self.bodies
            .keeps_read
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The text a function's body spans index, for the source `id`.
    pub(crate) fn source_text(&self, id: SourceId) -> Option<RedactedSql> {
        self.map
            .entries
            .values()
            .chain(self.stored.iter().map(|f| &f.entry))
            .find(|e| e.def.source == id)
            .map(|e| e.text.clone())
    }

    /// `def` with the views its body reads inlined: the temporary ones for
    /// a temporary function, and the permanent ones of its namespace for a
    /// stored one (which sees no temporary view: it could not have been
    /// created reading one).
    ///
    /// A stored function's bare function names resolve in its own
    /// namespace, not the caller's, so no temporary function or session
    /// UDF of the same name shadows them.
    fn expanded(&self, def: &Arc<SqlFunction>, profile: SparkProfile) -> Arc<SqlFunction> {
        let mut cache = self
            .bodies
            .expanded
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let key = cache_key(def);
        if let Some(hit) = cache.get(&key) {
            return Arc::clone(hit);
        }
        let stored_in = def.qualifier.get(1).cloned();
        if stored_in.is_some() && self.views.is_none() {
            let out = self.qualified(def, profile);
            cache.insert(key, Arc::clone(&out));
            return out;
        }
        let wrapper = one_row_project(vec![Arc::clone(&def.body)]);
        let next_source = |views: &ViewSource| (views.next_source)();
        // The calls of a view a body reads resolve as in any stored view's text (review r5 Z1), and a
        // stored body's own calls, at every level, through its frozen path or namespace (r7 R7-1).
        let qualify = |e: &Arc<Expr>, namespace: &str, path: &[Vec<String>], rule: CallRule<'_>| {
            self.qualify_calls(e, namespace, path, profile, rule)
        };
        let qualify_table =
            |name: &str, namespace: &str, path: &[Vec<String>], rule: CallRule<'_>| {
                self.qualify_table_call(name, namespace, path, profile, rule)
            };
        let expansion = match &self.views {
            Some(views) => {
                let stored = StoredViews {
                    parser: views.parser,
                    // A body reads a view as written; schema binding modes
                    // apply to the views a query reads itself.
                    binding_enabled: false,
                    outside_session: self.outside_session,
                    // A body's definer views are not trusted by identity (no `DefinerTexts`).
                    cte_precedence: krishiv_plan::analyzer::CtePrecedencePolicy::Corrected,
                    path: if def.path.is_empty() {
                        self.path.clone()
                    } else {
                        def.path.clone()
                    },
                    next_source: &|| next_source(views),
                    qualify: Some(&qualify),
                    qualify_table: Some(&qualify_table),
                    stored_body: stored_in.is_some(),
                    namespace: stored_in
                        .clone()
                        .unwrap_or_else(|| self.catalog.current_namespace()),
                };
                expand_views_in(
                    &self.catalog,
                    &wrapper,
                    self.case_sensitive,
                    Some(&stored),
                    stored_in.clone(),
                    (!def.path.is_empty()).then(|| def.path.clone()),
                )
            }
            None => expand_views_in(
                &self.catalog,
                &wrapper,
                self.case_sensitive,
                None,
                None,
                None,
            ),
        };
        // A body no view expansion can read (a snapshot name, a cycle)
        // stays as written: the analyzer then reports the table missing.
        let body = match expansion {
            Ok(expansion) => {
                // A definer view read by a body is read as an invoker one (fails closed): the
                // lookups of its text are the caller's too (P3-T062 §2.4).
                let mine = BodyReads {
                    key: key.clone(),
                    function: stored_in.as_ref().map(|ns| (ns.clone(), def.name.clone())),
                    reads: expansion.reads,
                    definer_reads: expansion
                        .definers
                        .into_iter()
                        .flat_map(|d| d.reads)
                        .collect(),
                    deferred: expansion
                        .deferred
                        .map(|e| (e, expansion.deferred_in_definer)),
                    views: expansion.reached,
                };
                self.bodies
                    .reads
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(mine);
                self.bodies
                    .views_read
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .extend(expansion.views);
                self.bodies
                    .keeps_read
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .extend(expansion.keeps);
                match &expansion.plan.node {
                    PlanNode::Project { exprs, .. } => exprs.first().cloned(),
                    _ => None,
                }
            }
            Err(_) => None,
        };
        let out = match body {
            Some(body) if !Arc::ptr_eq(&body, &def.body) => Arc::new(SqlFunction {
                body,
                ..(**def).clone()
            }),
            _ => Arc::clone(def),
        };
        let out = self.qualified(&out, profile);
        cache.insert(key, Arc::clone(&out));
        out
    }

    /// `def` with, for a stored function, the bare calls of its body and of its parameters'
    /// defaults qualified (review r7 R7-1: nothing of the caller's session answers in them).
    fn qualified(&self, def: &Arc<SqlFunction>, profile: SparkProfile) -> Arc<SqlFunction> {
        let Some(namespace) = def.qualifier.last() else {
            return Arc::clone(def);
        };
        let qualify =
            |e: &Arc<Expr>| self.qualify_calls(e, namespace, &def.path, profile, CallRule::Body);
        let body = qualify(&def.body);
        let mut params = def.params.clone();
        let mut defaults = false;
        for p in &mut params {
            if let Some(d) = p.default.as_ref().and_then(qualify) {
                p.default = Some(d);
                defaults = true;
            }
        }
        if body.is_none() && !defaults {
            return Arc::clone(def);
        }
        Arc::new(SqlFunction {
            body: body.unwrap_or_else(|| Arc::clone(&def.body)),
            params,
            ..(**def).clone()
        })
    }
}

impl QueryFunctions {
    /// `body` with each bare call that is not a built-in qualified: by the
    /// first schema of the function's frozen `path` holding a stored function
    /// of that name, else by `namespace`; `None` when nothing changed.
    /// In a permanent view's text ([`CallRule::View`], [`CallRule::Bound`]) no temporary function
    /// or session UDF answers a call, whatever the path or the reader's session holds (review r5
    /// Z1; Spark v4.2.0 `SessionCatalog.scala` lines 2427-2441). A bare call goes to the first
    /// catalog schema of the path holding a stored function of its name, else to the path's first
    /// catalog schema (where it is unresolved), else, for a view stored with no path, to its own
    /// namespace; with a path of no catalog schema it stays bare and the analyzer refuses it
    /// (`AnalysisContext::stored_view_texts`). A definer view's ([`CallRule::Bound`]) goes where it
    /// was bound when the view was created, and stays bare (refused) when it was not bound.
    /// Calls inside subquery plans are not entered.
    pub(crate) fn qualify_calls(
        &self,
        body: &Arc<Expr>,
        namespace: &str,
        path: &[Vec<String>],
        profile: SparkProfile,
        rule: CallRule<'_>,
    ) -> Option<Arc<Expr>> {
        let out = Arc::clone(body)
            .transform_up(
                &mut |e: Arc<Expr>| -> Result<Arc<Expr>, std::convert::Infallible> {
                    let ExprNode::Function(call) = &e.node else {
                        return Ok(e);
                    };
                    // A call written `ns.f()` inside a stored body is the defining catalog's, whatever
                    // catalog is current when the body is expanded.
                    if self.outside_session
                        && let [q] = call.qualifier.as_slice()
                        && !q.eq_ignore_ascii_case("session")
                        && !q.eq_ignore_ascii_case("system")
                        && !q.eq_ignore_ascii_case("builtin")
                    {
                        let mut call = call.clone();
                        call.qualifier = vec![crate::catalog::CATALOG.to_owned(), q.clone()];
                        let mut next = (*e).clone();
                        next.node = ExprNode::Function(call);
                        return Ok(Arc::new(next));
                    }
                    // Canonical operators and built-ins stay bare; anything
                    // else, session UDFs included, is a user function.
                    let builtin = ["system".to_owned(), "builtin".to_owned(), call.name.clone()];
                    // `qualify` is lowering's QUALIFY marker, which the analyzer strips.
                    if !call.qualifier.is_empty()
                        || call.name == "qualify"
                        || Builtin::from_name(&call.name).is_some()
                        || !matches!(
                            self.base.lookup(&builtin, profile),
                            FunctionLookup::NotFound
                        )
                    {
                        return Ok(e);
                    }
                    let mut call = call.clone();
                    call.qualifier = match self.stored_qualifier(&call.name, namespace, path, rule)
                    {
                        Some(q) => q,
                        None => return Ok(e),
                    };
                    let mut next = (*e).clone();
                    next.node = ExprNode::Function(call);
                    Ok(Arc::new(next))
                },
            )
            .unwrap_or_else(|never| match never {});
        (!Arc::ptr_eq(&out, body)).then_some(out)
    }
}

impl QueryFunctions {
    /// The qualifier [`Self::qualify_calls`] gives a bare call of `name` that no built-in answers:
    /// the first catalog schema of `path` holding a stored function of that name; for a view, else
    /// the path's first catalog schema, else (a view stored with no path) its own `namespace`; for
    /// a body, else `namespace`; for a definer view, where it was bound ([`CallRule::Bound`]).
    /// `None`: a view whose path holds no catalog schema, or a definer view that bound no function
    /// of the name, keeps the call bare (the analyzer refuses it in a stored view's text, never the
    /// reader's).
    fn stored_qualifier(
        &self,
        name: &str,
        namespace: &str,
        path: &[Vec<String>],
        rule: CallRule<'_>,
    ) -> Option<Vec<String>> {
        let in_view = match rule {
            CallRule::Bound(bound) => return bound.names.get(name).cloned(),
            CallRule::View => true,
            CallRule::Body => false,
        };
        let name = fold(name, self.case_sensitive);
        let from_path = path.iter().find_map(|entry| match entry.as_slice() {
            [c, n] if c.eq_ignore_ascii_case(crate::catalog::CATALOG) => {
                let n = fold(n, self.case_sensitive);
                self.stored.get(&n, &name).map(|_| vec![c.clone(), n])
            }
            _ => None,
        });
        // A routine gone from the frozen path stays unresolved: it is looked up in the path's first
        // schema, never the namespace outside the path.
        let first = path.iter().find_map(|entry| match entry.as_slice() {
            [c, n] if c.eq_ignore_ascii_case(crate::catalog::CATALOG) => {
                Some(vec![c.clone(), fold(n, self.case_sensitive)])
            }
            _ => None,
        });
        // Only a view is held to the path's first schema; a stored function's body falls back to
        // its own namespace.
        match from_path.or(first.filter(|_| in_view)) {
            Some(q) => Some(q),
            // A view stored without a path reads its own namespace.
            None if in_view && path.is_empty() => Some(vec![
                crate::catalog::CATALOG.to_owned(),
                fold(namespace, self.case_sensitive),
            ]),
            None if in_view => None,
            None if self.outside_session => Some(vec![
                crate::catalog::CATALOG.to_owned(),
                namespace.to_owned(),
            ]),
            None => Some(vec![namespace.to_owned()]),
        }
    }

    /// The name of a table function a stored view's text or a stored function's body
    /// calls (`name` as written, its parts joined by [`krishiv_plan::TABLE_FUNCTION_NAME_SEP`])
    /// qualified as [`Self::qualify_calls`] qualifies a scalar call there (review r6 R6-1, r7
    /// R7-1): a bare name no built-in table function answers goes to the stored function its frozen
    /// `path` (or its `namespace`) holds, never to a temporary one or the reader's namespace. `None`
    /// when it stays as written.
    pub(crate) fn qualify_table_call(
        &self,
        name: &str,
        namespace: &str,
        path: &[Vec<String>],
        profile: SparkProfile,
        rule: CallRule<'_>,
    ) -> Option<String> {
        if name.contains(krishiv_plan::TABLE_FUNCTION_NAME_SEP) || builtin_table(name, profile) {
            return None;
        }
        let mut parts = self.stored_qualifier(name, namespace, path, rule)?;
        parts.push(name.to_owned());
        Some(parts.join(&krishiv_plan::TABLE_FUNCTION_NAME_SEP.to_string()))
    }
}

/// Where the bare calls of an expanded text resolve when not through the reader's session.
#[derive(Debug, Clone, Copy)]
pub(crate) enum CallRule<'a> {
    /// A stored function's body: through its frozen path, else its own namespace (review r7 R7-1).
    Body,
    /// An invoker view's text: through its frozen path at each read, as Spark resolves it.
    View,
    /// A definer view's text: where its functions were bound when it was created
    /// ([`crate::store::ViewDef::functions`], review r8 item 1).
    Bound(&'a BoundFunctions),
}

impl QueryFunctions {
    /// The functions of a definer view's text `plan`, bound now (review r8 item 1): each name it
    /// calls bare (scalar or table, at every level) that no built-in answers, with the catalog and
    /// schema [`CallRule::View`] resolves it to through the view's frozen `path` and `namespace`.
    /// A name that resolves nowhere is left out, so a read refuses its call. With them, the
    /// revision of every stored function they name and of every one this registry's lookups
    /// resolved so far (the view's analysis: its qualified calls, its invoker views' and the
    /// bodies' calls), which a read may call only at that revision (review r9 item 1).
    pub(crate) fn bound_calls(
        &self,
        plan: &Arc<LogicalPlan>,
        namespace: &str,
        path: &[Vec<String>],
        profile: SparkProfile,
    ) -> BoundFunctions {
        let mut bound = BoundFunctions::default();
        let mut bind = |name: &str| {
            if !bound.names.contains_key(name)
                && let Some(q) = self.stored_qualifier(name, namespace, path, CallRule::View)
            {
                if let [_, schema] = q.as_slice()
                    && let Some(f) = self.stored.get(schema, name)
                {
                    bound
                        .versions
                        .insert((f.namespace.clone(), f.entry.def.name.clone()), f.revision);
                }
                bound.names.insert(name.to_owned(), q);
            }
        };
        let mut plans = vec![plan];
        let mut exprs: Vec<&Arc<Expr>> = Vec::new();
        while let Some(p) = plans.pop() {
            if let PlanNode::TableFunction { name, .. } = &p.node
                && !name.contains(krishiv_plan::TABLE_FUNCTION_NAME_SEP)
                && !builtin_table(name, profile)
            {
                bind(name);
            }
            p.for_each_expr(&mut |e| exprs.push(e));
            p.for_each_child(&mut |c| plans.push(c));
            while let Some(e) = exprs.pop() {
                if let ExprNode::Function(call) = &e.node
                    && call.qualifier.is_empty()
                    && call.name != "qualify"
                    && Builtin::from_name(&call.name).is_none()
                    && matches!(
                        self.base.lookup(
                            &["system".to_owned(), "builtin".to_owned(), call.name.clone()],
                            profile
                        ),
                        FunctionLookup::NotFound
                    )
                {
                    bind(&call.name);
                }
                if let ExprNode::Subquery { plan, .. } = &e.node {
                    plans.push(plan);
                }
                e.for_each_child(&mut |c| exprs.push(c));
            }
        }
        // Every stored function the view's analysis reached (its invoker views' and bodies'
        // calls too), pinned at its revision (review r9 item 1).
        for (function, revision) in self.called_versions() {
            bound.versions.entry(function).or_insert(revision);
        }
        bound.views = self.viewed_versions();
        bound
    }
}

impl QueryFunctions {
    /// Each stored function a lookup resolved so far (see [`Self::stored_calls`]), with the
    /// revision this registry's snapshot holds: what a definer view being created reached, pinned
    /// with it (review r9 item 1).
    pub(crate) fn called_versions(&self) -> BTreeMap<(String, String), u64> {
        self.stored_calls()
            .into_iter()
            .filter_map(|(namespace, name)| {
                let revision = self.stored.get(&namespace, &name)?.revision;
                Some(((namespace, name), revision))
            })
            .collect()
    }

    /// Records `views`, stored views an expansion of the query inlined, with their versions: what a
    /// definer view being created reaches, pinned with it (review r10 M10).
    pub(crate) fn note_views(&self, views: &[crate::catalog::ReachedView]) {
        let mut noted = self
            .bodies
            .views_noted
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        for v in views {
            noted.entry(v.view.clone()).or_insert(v.revision);
        }
    }

    /// Each stored view the query's expansions and the bodies expanded so far inlined, with the
    /// version inlined (the first, if one was inlined twice).
    fn viewed_versions(&self) -> BTreeMap<(String, String), u64> {
        let mut all = self
            .bodies
            .views_noted
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let bodies = self
            .bodies
            .reads
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        for v in bodies.iter().flat_map(|b| &b.views) {
            all.entry(v.view.clone()).or_insert(v.revision);
        }
        all
    }

    /// The revision and author of the stored function `(namespace, name)` in this registry's
    /// snapshot; `None` when it holds none.
    pub(crate) fn stored_identity(
        &self,
        (namespace, name): &(String, String),
    ) -> Option<(u64, &str)> {
        self.stored
            .get(namespace, name)
            .map(|f| (f.revision, f.author.as_str()))
    }
}

/// Whether a built-in table function (or `json_tuple`) answers a bare table function call of `name`.
fn builtin_table(name: &str, profile: SparkProfile) -> bool {
    name.eq_ignore_ascii_case("json_tuple")
        || matches!(
            krishiv_plan::analyzer::TableFunctionRegistry::lookup(
                &krishiv_plan::analyzer::BuiltinTableFunctions::new(),
                name,
                profile,
            ),
            krishiv_plan::analyzer::TableFunctionLookup::Found(_)
        )
}

impl QueryFunctions {
    /// The first temporary SQL function or session UDF (Rust, Python, Connect) `plan` calls, as
    /// this registry resolves a call: a bare name a built-in does not answer and a session UDF
    /// or, earlier on the path than any stored function of that name, a temporary function does,
    /// or a `session.` / `system.session.` name. A permanent view may call none (review r5 Z2;
    /// Spark v4.2.0 `views.scala` lines 628-650). Lower case, as Spark names a function.
    pub(crate) fn temporary_call(
        &self,
        plan: &Arc<LogicalPlan>,
        profile: SparkProfile,
    ) -> Option<String> {
        let mut plans = vec![plan];
        let mut exprs: Vec<&Arc<Expr>> = Vec::new();
        while let Some(p) = plans.pop() {
            if let PlanNode::TableFunction { name, .. } = &p.node {
                let mut parts: Vec<String> = name
                    .split(krishiv_plan::TABLE_FUNCTION_NAME_SEP)
                    .map(str::to_owned)
                    .collect();
                let f = parts.pop().unwrap_or_default();
                if self.temporary_answers(&parts, &f, true, profile) {
                    return Some(f.to_lowercase());
                }
            }
            p.for_each_expr(&mut |e| exprs.push(e));
            p.for_each_child(&mut |c| plans.push(c));
            while let Some(e) = exprs.pop() {
                if let ExprNode::Function(call) = &e.node
                    && self.temporary_answers(&call.qualifier, &call.name, false, profile)
                {
                    return Some(call.name.to_lowercase());
                }
                if let ExprNode::Subquery { plan, .. } = &e.node {
                    plans.push(plan);
                }
                e.for_each_child(&mut |c| exprs.push(c));
            }
        }
        None
    }

    /// Whether a temporary function or session UDF answers a call of `name` written with
    /// `qualifier` (a table function's when `table`).
    fn temporary_answers(
        &self,
        qualifier: &[String],
        name: &str,
        table: bool,
        profile: SparkProfile,
    ) -> bool {
        let session = matches!(qualifier, [q] if q.eq_ignore_ascii_case("session"))
            || matches!(qualifier, [s, q]
                if s.eq_ignore_ascii_case("system") && q.eq_ignore_ascii_case("session"));
        if !qualifier.is_empty() && !session {
            return false;
        }
        let temporary = self
            .map
            .get(name)
            .is_some_and(|e| !table || e.def.returns_table.is_some());
        let udf = !table
            && !matches!(
                self.base
                    .lookup(&["session".to_owned(), name.to_owned()], profile),
                FunctionLookup::NotFound
            );
        if session {
            return temporary || udf;
        }
        let builtin = if table {
            builtin_table(name, profile)
        } else {
            name == "qualify"
                || Builtin::from_name(name).is_some()
                || !matches!(
                    self.base.lookup(
                        &["system".to_owned(), "builtin".to_owned(), name.to_owned()],
                        profile
                    ),
                    FunctionLookup::NotFound
                )
        };
        if builtin {
            return false;
        }
        // A session UDF answers a bare name right after the built-ins.
        if udf {
            return true;
        }
        if !temporary {
            return false;
        }
        let name = fold(name, self.case_sensitive);
        for entry in &self.path {
            match entry.as_slice() {
                [s, q] if s.eq_ignore_ascii_case("system") && q.eq_ignore_ascii_case("session") => {
                    return true;
                }
                [c, n]
                    if c.eq_ignore_ascii_case(crate::catalog::CATALOG)
                        && self
                            .stored
                            .get(&fold(n, self.case_sensitive), &name)
                            .is_some() =>
                {
                    return false;
                }
                _ => {}
            }
        }
        false
    }
}

impl FunctionRegistry for QueryFunctions {
    fn lookup(&self, name: &[String], profile: SparkProfile) -> FunctionLookup {
        match self.base.lookup(name, profile) {
            FunctionLookup::NotFound => {}
            found => return found,
        }
        // Temporary functions also answer under `session.f` and
        // `system.session.f`, as session UDFs do; stored ones under their
        // namespace (`default.f`, `spark_catalog.default.f`), and a bare
        // name after the temporary ones, in the current namespace.
        let lower: Vec<String> = name.iter().map(|p| p.to_lowercase()).collect();
        let ns = |i: usize| fold(&name[i], self.case_sensitive);
        let temporary = |f: &String| {
            self.map
                .get(f)
                .map(|entry| FunctionLookup::Sql(self.expanded(&entry.def, profile)))
        };
        let stored = |namespace: &str, f: &String| {
            self.stored.get(namespace, f).map(|stored| {
                let key = (stored.namespace.clone(), f.clone());
                let same = |a: &str, b: &str| {
                    if self.case_sensitive {
                        a == b
                    } else {
                        a.eq_ignore_ascii_case(b)
                    }
                };
                let own = self
                    .defining
                    .as_ref()
                    .is_some_and(|(n, d)| same(n, &key.0) && same(d, &key.1));
                let mut called = self.called.lock().unwrap_or_else(PoisonError::into_inner);
                if !own && !called.contains(&key) {
                    called.push(key);
                }
                drop(called);
                FunctionLookup::Sql(self.expanded(&stored.entry.def, profile))
            })
        };
        let found = match lower.as_slice() {
            [f] => self.path.iter().find_map(|entry| match entry.as_slice() {
                [s, q] if s.eq_ignore_ascii_case("system") && q.eq_ignore_ascii_case("session") => {
                    temporary(f)
                }
                [c, n] if c.eq_ignore_ascii_case(crate::catalog::CATALOG) => {
                    stored(&fold(n, self.case_sensitive), f)
                }
                _ => None,
            }),
            [q, f] if q == "session" => temporary(f),
            [s, q, f] if s == "system" && q == "session" => temporary(f),
            [_, f] if !self.outside_session => stored(&ns(0), f),
            [c, _, f] if c == crate::catalog::CATALOG => stored(&ns(1), f),
            _ => None,
        };
        found.unwrap_or(FunctionLookup::NotFound)
    }

    fn spark_deterministic(&self, name: &str) -> Option<bool> {
        self.base.spark_deterministic(name)
    }

    fn resolve_udf(&self, name: &str, version: u32) -> Option<Arc<dyn krishiv_udf_sdk::ScalarUdf>> {
        self.base.resolve_udf(name, version)
    }

    fn rejects_distinct(&self, name: &str) -> bool {
        self.base.rejects_distinct(name)
    }

    /// The built-ins' version and the session map's, mixed: a change to
    /// either is a different number.
    fn version(&self) -> u64 {
        self.base
            .version()
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(self.map.version())
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(self.stored.version())
    }
}

/// The temporary view a function body reads, if any: a stored function
/// may not depend on an object of the session.
pub(crate) fn reads_temporary_view(
    catalog: &SessionCatalog,
    body: &Arc<Expr>,
    case_sensitive: bool,
) -> Option<String> {
    crate::catalog::reads_temporary(
        catalog,
        &one_row_project(vec![Arc::clone(body)]),
        case_sensitive,
    )
}

/// Whether a function body reads a relation: a table, a view or a file
/// (not `range`, `VALUES` or a `WITH` name in scope where it is read).
pub(crate) fn reads_relation(body: &Arc<Expr>, case_sensitive: bool) -> bool {
    use krishiv_plan::{ExprNode, ReadSource};
    fn plan(p: &Arc<LogicalPlan>, cs: bool, scope: &mut Vec<String>) -> bool {
        if let PlanNode::Read(source) = &p.node {
            let shadowed = matches!(source, ReadSource::Table { name, .. }
            if matches!(name.as_slice(), [n] if scope.iter().any(|c| {
                c == n || !cs && c.eq_ignore_ascii_case(n)
            })));
            if !shadowed {
                return true;
            }
        }
        if let PlanNode::WithCte { input, ctes } = &p.node {
            let outer = scope.len();
            for cte in ctes.iter() {
                if plan(&cte.plan, cs, scope) {
                    return true;
                }
                scope.push(cte.name.clone());
            }
            let found = plan(input, cs, scope);
            scope.truncate(outer);
            return found;
        }
        let mut children = Vec::new();
        p.for_each_child(&mut |c| children.push(Arc::clone(c)));
        if children.iter().any(|c| plan(c, cs, scope)) {
            return true;
        }
        let mut exprs = Vec::new();
        p.for_each_expr(&mut |e| exprs.push(Arc::clone(e)));
        exprs.iter().any(|e| expr(e, cs, scope))
    }
    fn expr(e: &Arc<Expr>, cs: bool, scope: &mut Vec<String>) -> bool {
        if let ExprNode::Subquery { plan: p, .. } = &e.node
            && plan(p, cs, scope)
        {
            return true;
        }
        let mut children = Vec::new();
        e.for_each_child(&mut |c| children.push(Arc::clone(c)));
        children.iter().any(|c| expr(c, cs, scope))
    }
    expr(body, case_sensitive, &mut Vec::new())
}

/// A namespace name as the session compares them.
fn fold(namespace: &str, case_sensitive: bool) -> String {
    if case_sensitive {
        namespace.to_owned()
    } else {
        namespace.to_lowercase()
    }
}

/// What a function is cached under: its qualifier and name.
fn cache_key(def: &SqlFunction) -> String {
    format!("{}.{}", def.qualifier.join("."), def.name)
}

/// A `Project` of `exprs` over one row.
pub(crate) fn one_row_project(exprs: Vec<Arc<Expr>>) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(PlanNode::Project {
        input: Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
        exprs,
    }))
}

/// A call of `def` named `qualifier`.`name` with a typed `NULL` for each parameter it needs
/// (defaults fill the rest): `SELECT name(NULL, ...)` for a scalar function, the table function's
/// rows for a table one. What `CREATE FUNCTION` resolves to check a body, and what a definer
/// view's read analyzes to ask a body's writers the columns it reads (review r11 B1).
pub(crate) fn null_call(def: &SqlFunction, qualifier: &[String]) -> Arc<LogicalPlan> {
    let args: Vec<Expr> = def.params[..def.required()]
        .iter()
        .map(|p| Expr::literal(krishiv_plan::Literal::Null(Some(p.data_type.clone()))))
        .collect();
    if def.returns_table.is_some() {
        let name: Vec<&str> = qualifier
            .iter()
            .map(String::as_str)
            .chain(std::iter::once(def.name.as_str()))
            .collect();
        return Arc::new(LogicalPlan::new(PlanNode::TableFunction {
            name: name.join(&krishiv_plan::TABLE_FUNCTION_NAME_SEP.to_string()),
            args: args.into_iter().map(Arc::new).collect(),
        }));
    }
    let mut call = Expr::function(def.name.clone(), args);
    if let ExprNode::Function(function) = &mut call.node {
        function.qualifier = qualifier.to_vec();
    }
    one_row_project(vec![Arc::new(call)])
}

/// Whether `hit` holds for the name of a function the expressions `roots` call, in their
/// subqueries too (a table function of a subquery's plan included).
pub(crate) fn any_call(roots: Vec<&Expr>, mut hit: impl FnMut(&[String]) -> bool) -> bool {
    let mut stack = roots;
    let mut plans = Vec::new();
    while let Some(e) = stack.pop() {
        if let ExprNode::Function(call) = &e.node {
            let mut name = call.qualifier.clone();
            name.push(call.name.clone());
            if hit(&name) {
                return true;
            }
        }
        if let ExprNode::Subquery { plan, .. } = &e.node {
            plans.push(plan);
            while let Some(p) = plans.pop() {
                if let PlanNode::TableFunction { name, .. } = &p.node {
                    let parts: Vec<String> = name
                        .split(krishiv_plan::TABLE_FUNCTION_NAME_SEP)
                        .map(str::to_owned)
                        .collect();
                    if hit(&parts) {
                        return true;
                    }
                }
                p.for_each_expr(&mut |x| stack.push(x));
                p.for_each_child(&mut |c| plans.push(c));
            }
        }
        e.for_each_child(&mut |c| stack.push(c));
    }
    false
}

#[cfg(test)]
mod tests {
    use krishiv_plan::analyzer::BuiltinFunctions;
    use krishiv_plan::{Expr, Literal};

    use super::*;

    fn entry(name: &str, value: i32) -> Entry {
        Entry {
            def: Arc::new(SqlFunction {
                name: name.to_owned(),
                qualifier: Vec::new(),
                path: Vec::new(),
                returns_table: None,
                params: Vec::new(),
                returns: None,
                body: Arc::new(Expr::literal(Literal::Integer(value))),
                body_text: value.to_string(),
                source: SourceId(7),
                deterministic: true,
                reads_data: false,
                ansi: None,
            }),
            text: krishiv_sql::redact_sql("CREATE FUNCTION f() RETURN 1"),
            described: Arc::default(),
        }
    }

    fn registry(functions: &SqlFunctions) -> QueryFunctions {
        QueryFunctions::new(
            Arc::new(BuiltinFunctions::new()),
            functions.snapshot(),
            Arc::new(SessionCatalog::default()),
            false,
            None,
        )
    }

    fn body(r: &QueryFunctions, name: &str) -> Option<Arc<Expr>> {
        match r.lookup(&[name.to_owned()], SparkProfile::default()) {
            FunctionLookup::Sql(def) => Some(Arc::clone(&def.body)),
            _ => None,
        }
    }

    #[test]
    fn a_query_reads_one_snapshot_whatever_changes_after() {
        let functions = SqlFunctions::default();
        assert!(functions.define(entry("f", 1), 0));
        let before = registry(&functions);
        assert!(functions.define(entry("f", 2), 1));
        assert!(functions.drop_function("f"));
        // The registry made before the changes still answers with the
        // definition it started with; a new one sees the new map.
        let old = body(&before, "f").unwrap();
        assert!(matches!(
            old.node,
            krishiv_plan::ExprNode::Literal(Literal::Integer(1))
        ));
        assert!(body(&registry(&functions), "f").is_none());
        assert_ne!(before.version(), registry(&functions).version());
    }

    #[test]
    fn every_change_changes_the_version() {
        let functions = SqlFunctions::default();
        let mut seen = vec![registry(&functions).version()];
        assert!(functions.define(entry("f", 1), 0));
        seen.push(registry(&functions).version());
        assert!(functions.define(entry("f", 1), 1));
        seen.push(registry(&functions).version());
        assert!(functions.drop_function("f"));
        seen.push(registry(&functions).version());
        let mut distinct = seen.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), seen.len(), "{seen:?}");
        // Dropping what is not defined changes nothing.
        assert!(!functions.drop_function("f"));
        assert_eq!(registry(&functions).version(), *seen.last().unwrap());
    }

    #[test]
    fn a_definition_checked_against_an_old_map_is_not_installed() {
        let functions = SqlFunctions::default();
        let checked = functions.snapshot().version();
        assert!(functions.define(entry("a", 1), checked));
        assert!(!functions.define(entry("b", 2), checked));
        assert!(body(&registry(&functions), "b").is_none());
        assert!(functions.define(entry("b", 2), functions.snapshot().version()));
    }

    #[test]
    fn names_fold_case_and_built_ins_win() {
        let functions = SqlFunctions::default();
        assert!(functions.define(entry("f", 1), 0));
        assert!(functions.define(entry("abs", 2), 1));
        let r = registry(&functions);
        assert!(body(&r, "F").is_some());
        assert!(body(&r, "abs").is_none());
        // Only one-part names are functions of the session.
        let two = ["db".to_owned(), "f".to_owned()];
        assert!(matches!(
            r.lookup(&two, SparkProfile::default()),
            FunctionLookup::NotFound
        ));
        assert!(r.source_text(SourceId(7)).is_some());
        assert!(r.source_text(SourceId(8)).is_none());
    }
}

/// Spark's `SQL_TABLE_UDF_MISSING_COLUMN_NAMES` (`CreateSQLFunctionCommand`, checked after the
/// body analyzed): a table function without a column list (`RETURNS TABLE`) takes its columns'
/// names from the body. Spark checks the parsed body only: a bare `SELECT` (a `Project`) may not
/// hold an item it would name itself (`x + 1`, `IDENTIFIER('x')`, `m['k']`); any other shape
/// (`ORDER BY`, `LIMIT`, `GROUP BY`, `DISTINCT`, `WITH`, `WINDOW`, `QUALIFY`, a hint, a set
/// operation) is not checked and names its columns as a query does. A pipe is checked at its last
/// operator: `|> SELECT` and `|> EXTEND` are `Project`s too. `name` is the function's name as
/// written (Spark keeps its case there). `ids` are the spans of the names the
/// session substituted for `IDENTIFIER(..)` clauses, which Spark reads as unnamed expressions.
pub(crate) fn check_return_names(
    f: &krishiv_sql::ast::CreateFunction,
    name: &str,
    ids: &[krishiv_sql::ast::Span],
) -> Result<(), KrishivError> {
    use krishiv_sql::ast::{FunctionBodyKind, FunctionReturnsKind};
    let without_columns = matches!(
        f.returns.as_ref().map(|r| &r.kind),
        Some(FunctionReturnsKind::Table(None))
    );
    let query = match &f.body.kind {
        FunctionBodyKind::Query(q) => Some(&**q),
        other => bracketed_query(other),
    };
    match query {
        Some(q) if without_columns && lacks_column_names(q, ids) => Err(
            krishiv_common::error::UserDefinedFunctionsSqlTableUdfMissingColumnNames {
                function_name: format!("`{}`", name.replace('`', "``")),
            }
            .into(),
        ),
        _ => Ok(()),
    }
}

/// Whether `q` parses to Spark's `Project` with an `UnresolvedAlias` item (see
/// [`check_return_names`]).
fn lacks_column_names(q: &krishiv_sql::ast::Query, ids: &[krishiv_sql::ast::Span]) -> bool {
    use krishiv_sql::ast::{PipeOpKind, SetExprKind};
    // Each of these wraps the `Project` in another node (`Sort`, `Limit`, `WithCTE`, ...).
    let organized = q.with.is_some()
        || !q.order_by.is_empty()
        || !q.cluster_by.is_empty()
        || !q.distribute_by.is_empty()
        || !q.sort_by.is_empty()
        || !q.windows.is_empty()
        || q.limit.is_some()
        || q.offset.is_some();
    if organized {
        return false;
    }
    match &q.body.kind {
        SetExprKind::Select(s) => {
            // A hint is an `UnresolvedHint` over the `Project`.
            let projected = s.hints.is_empty()
                && s.group_by.is_none()
                && s.having.is_none()
                && s.windows.is_empty()
                && s.qualify.is_none()
                && !matches!(
                    s.quantifier,
                    Some(krishiv_sql::ast::SetQuantifier::Distinct)
                );
            projected && unnamed_item(&s.projection, ids)
        }
        // `(query)` is the query; `FROM t SELECT ...` with one body is that body over `t`.
        SetExprKind::Query(inner) => lacks_column_names(inner, ids),
        SetExprKind::FromStatement(from) => match &from.bodies[..] {
            [body] => lacks_column_names(body, ids),
            _ => false,
        },
        SetExprKind::Pipe(pipe) => match pipe.ops.last().map(|op| &op.kind) {
            Some(PipeOpKind::Select(s)) => {
                s.hints.is_empty()
                    && s.group_by.is_none()
                    && s.windows.is_empty()
                    && !matches!(
                        s.quantifier,
                        Some(krishiv_sql::ast::SetQuantifier::Distinct)
                    )
                    && unnamed_item(&s.projection, ids)
            }
            Some(PipeOpKind::Extend(items)) => unnamed_item(items, ids),
            _ => false,
        },
        _ => false,
    }
}

/// Whether an item of `items` has no alias and is not a named expression.
fn unnamed_item(items: &[krishiv_sql::ast::SelectItem], ids: &[krishiv_sql::ast::Span]) -> bool {
    items
        .iter()
        .any(|item| item.alias.is_none() && !named(&item.expr, ids))
}

/// Whether Spark's parser makes `e` a named expression (an attribute or a star), which a
/// `Project` keeps as it is, rather than wrapping it in an `UnresolvedAlias`.
fn named(e: &krishiv_sql::ast::Expr, ids: &[krishiv_sql::ast::Span]) -> bool {
    use krishiv_sql::ast::ExprKind;
    match &e.kind {
        ExprKind::Wildcard { .. } => true,
        _ => attribute(e, ids),
    }
}

/// Whether `e` is an `UnresolvedAttribute`: a column, in brackets or with fields taken
/// (`(c1).c2`), but not a name substituted for an `IDENTIFIER(..)` clause, which Spark parses as
/// an expression (`IDENTIFIER('c1').c2` is a field of one).
fn attribute(e: &krishiv_sql::ast::Expr, ids: &[krishiv_sql::ast::Span]) -> bool {
    use krishiv_sql::ast::ExprKind;
    match &e.kind {
        // `Span`'s `==` ignores positions: compare the offsets (as `lower_identified` does).
        ExprKind::Column(_) => !ids
            .iter()
            .any(|s| s.start == e.span.start && s.end <= e.span.end),
        ExprKind::Nested(inner) | ExprKind::Field(inner, _) => attribute(inner, ids),
        _ => false,
    }
}

/// The query of a function body written `RETURN (query)`, which the parser reads as a scalar
/// subquery expression; Spark's grammar reads it as the query (in brackets, as many as written).
pub(crate) fn bracketed_query(
    body: &krishiv_sql::ast::FunctionBodyKind,
) -> Option<&krishiv_sql::ast::Query> {
    use krishiv_sql::ast::{ExprKind, FunctionBodyKind};
    let FunctionBodyKind::Expr(e) = body else {
        return None;
    };
    let mut e: &krishiv_sql::ast::Expr = e;
    loop {
        match &e.kind {
            ExprKind::Nested(inner) => e = inner,
            ExprKind::Subquery(q) => return Some(q),
            _ => return None,
        }
    }
}

/// What `CREATE FUNCTION`'s `declare` lowers a function's defaults and body with.
#[derive(Clone, Copy)]
pub(crate) struct Declared<'a> {
    pub(crate) source: krishiv_plan::SourceId,
    pub(crate) case_sensitive: bool,
    pub(crate) ansi: bool,
    /// The spans of the names substituted for `IDENTIFIER(..)` clauses in the statement.
    pub(crate) ids: &'a [krishiv_sql::ast::Span],
}
