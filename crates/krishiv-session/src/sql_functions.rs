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

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use krishiv_common::redact::RedactedSql;
use krishiv_plan::analyzer::{FunctionLookup, FunctionRegistry, SparkProfile, SqlFunction};
use krishiv_plan::function::Builtin;
use krishiv_plan::{Expr, ExprNode, LogicalPlan, PlanNode, SourceId};

use crate::catalog::{ReadCandidate, SessionCatalog, StoredViews, expand_views_in};
use crate::store::{CatalogStore, StoredFunctions};
use krishiv_common::error::KrishivError;

/// A defined function and the statement that defined it.
#[derive(Clone)]
pub(crate) struct Entry {
    pub(crate) def: Arc<SqlFunction>,
    /// The `CREATE` statement, redacted: the text the body's spans index.
    pub(crate) text: RedactedSql,
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
    /// The functions stored in the catalog's namespaces, as of the start of
    /// the query, and the namespace a bare name resolves in.
    stored: Arc<StoredFunctions>,
    namespace: String,
    /// What reading a permanent view in a body needs; `None` for a registry
    /// that inlines temporary views only.
    views: Option<ViewSource>,
    expanded: Mutex<HashMap<String, Arc<SqlFunction>>>,
    /// The catalog-store lookups the bodies expanded so far made, and the
    /// first failure inside a stored view's text (see
    /// [`crate::catalog::Expansion`]).
    reads: Mutex<(Vec<ReadCandidate>, Option<KrishivError>)>,
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
        let namespace = fold(&catalog.current_namespace(), case_sensitive);
        Self {
            base,
            map,
            catalog,
            case_sensitive,
            stored,
            namespace,
            views,
            expanded: Mutex::new(HashMap::new()),
            reads: Mutex::new((Vec::new(), None)),
        }
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
    }

    /// The catalog-store lookups the bodies expanded so far made, in order,
    /// and the first failure inside a stored view's text. A query asks the
    /// policy about the reads before it raises the failure.
    pub(crate) fn body_reads(&self) -> (Vec<ReadCandidate>, Option<KrishivError>) {
        let reads = self.reads.lock().unwrap_or_else(PoisonError::into_inner);
        (reads.0.clone(), reads.1.clone())
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
        let mut cache = self.expanded.lock().unwrap_or_else(PoisonError::into_inner);
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
        let expansion = match &self.views {
            Some(views) => {
                let stored = StoredViews {
                    parser: views.parser,
                    // A body reads a view as written; schema binding modes
                    // apply to the views a query reads itself.
                    binding_enabled: false,
                    next_source: &|| next_source(views),
                    namespace: stored_in
                        .clone()
                        .unwrap_or_else(|| self.catalog.current_namespace()),
                };
                expand_views_in(
                    &self.catalog,
                    &wrapper,
                    self.case_sensitive,
                    Some(&stored),
                    stored_in,
                )
            }
            None => expand_views_in(&self.catalog, &wrapper, self.case_sensitive, None, None),
        };
        // A body no view expansion can read (a snapshot name, a cycle)
        // stays as written: the analyzer then reports the table missing.
        let body = match expansion {
            Ok(expansion) => {
                let mut seen = self.reads.lock().unwrap_or_else(PoisonError::into_inner);
                for read in expansion.reads {
                    if !seen.0.contains(&read) {
                        seen.0.push(read);
                    }
                }
                if let Some(e) = expansion.deferred {
                    seen.1.get_or_insert(e);
                }
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

    /// `def` with, for a stored function, its bare calls qualified.
    fn qualified(&self, def: &Arc<SqlFunction>, profile: SparkProfile) -> Arc<SqlFunction> {
        let Some(namespace) = def.qualifier.last() else {
            return Arc::clone(def);
        };
        match self.qualify_calls(&def.body, namespace, profile) {
            Some(body) => Arc::new(SqlFunction {
                body,
                ..(**def).clone()
            }),
            None => Arc::clone(def),
        }
    }
}

impl QueryFunctions {
    /// `body` with each bare call that is not a built-in qualified by
    /// `namespace`; `None` when nothing changed. Calls inside subquery
    /// plans are not entered.
    fn qualify_calls(
        &self,
        body: &Arc<Expr>,
        namespace: &str,
        profile: SparkProfile,
    ) -> Option<Arc<Expr>> {
        let out = Arc::clone(body)
            .transform_up(
                &mut |e: Arc<Expr>| -> Result<Arc<Expr>, std::convert::Infallible> {
                    let ExprNode::Function(call) = &e.node else {
                        return Ok(e);
                    };
                    // Canonical operators and built-ins stay bare; anything
                    // else, session UDFs included, is a user function.
                    let builtin = ["system".to_owned(), "builtin".to_owned(), call.name.clone()];
                    if !call.qualifier.is_empty()
                        || Builtin::from_name(&call.name).is_some()
                        || !matches!(
                            self.base.lookup(&builtin, profile),
                            FunctionLookup::NotFound
                        )
                    {
                        return Ok(e);
                    }
                    let mut call = call.clone();
                    call.qualifier = vec![namespace.to_owned()];
                    let mut next = (*e).clone();
                    next.node = ExprNode::Function(call);
                    Ok(Arc::new(next))
                },
            )
            .unwrap_or_else(|never| match never {});
        (!Arc::ptr_eq(&out, body)).then_some(out)
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
            self.stored
                .get(namespace, f)
                .map(|stored| FunctionLookup::Sql(self.expanded(&stored.entry.def, profile)))
        };
        let found = match lower.as_slice() {
            [f] => temporary(f).or_else(|| stored(&self.namespace, f)),
            [q, f] if q == "session" => temporary(f),
            [s, q, f] if s == "system" && q == "session" => temporary(f),
            [_, f] => stored(&ns(0), f),
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
                returns_table: None,
                params: Vec::new(),
                returns: None,
                body: Arc::new(Expr::literal(Literal::Integer(value))),
                body_text: value.to_string(),
                source: SourceId(7),
                deterministic: true,
                reads_data: false,
            }),
            text: krishiv_sql::redact_sql("CREATE FUNCTION f() RETURN 1"),
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
