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
use krishiv_plan::{Expr, LogicalPlan, PlanNode, SourceId};

use crate::catalog::{SessionCatalog, expand_temp_views};

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
    expanded: Mutex<HashMap<String, Arc<SqlFunction>>>,
}

impl QueryFunctions {
    pub(crate) fn new(
        base: Arc<dyn FunctionRegistry>,
        map: Arc<FunctionMap>,
        catalog: Arc<SessionCatalog>,
        case_sensitive: bool,
    ) -> Self {
        Self {
            base,
            map,
            catalog,
            case_sensitive,
            expanded: Mutex::new(HashMap::new()),
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
        )
    }

    /// The text a function's body spans index, for the source `id`.
    pub(crate) fn source_text(&self, id: SourceId) -> Option<RedactedSql> {
        self.map
            .entries
            .values()
            .find(|e| e.def.source == id)
            .map(|e| e.text.clone())
    }

    /// `def` with the temporary views its body reads inlined.
    fn expanded(&self, def: &Arc<SqlFunction>) -> Arc<SqlFunction> {
        let mut cache = self.expanded.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(hit) = cache.get(&def.name) {
            return Arc::clone(hit);
        }
        let wrapper = one_row_project(vec![Arc::clone(&def.body)]);
        // A body no view expansion can read (a snapshot name, a cycle)
        // stays as written: the analyzer then reports the table missing.
        let body = match expand_temp_views(&self.catalog, &wrapper, self.case_sensitive) {
            Ok(plan) => match &plan.node {
                PlanNode::Project { exprs, .. } => exprs.first().cloned(),
                _ => None,
            },
            Err(_) => None,
        };
        let out = match body {
            Some(body) if !Arc::ptr_eq(&body, &def.body) => Arc::new(SqlFunction {
                body,
                ..(**def).clone()
            }),
            _ => Arc::clone(def),
        };
        cache.insert(def.name.clone(), Arc::clone(&out));
        out
    }
}

impl FunctionRegistry for QueryFunctions {
    fn lookup(&self, name: &[String], profile: SparkProfile) -> FunctionLookup {
        match self.base.lookup(name, profile) {
            FunctionLookup::NotFound => {}
            found => return found,
        }
        let [name] = name else {
            return FunctionLookup::NotFound;
        };
        match self.map.get(name) {
            Some(entry) => FunctionLookup::Sql(self.expanded(&entry.def)),
            None => FunctionLookup::NotFound,
        }
    }

    /// The built-ins' version and the session map's, mixed: a change to
    /// either is a different number.
    fn version(&self) -> u64 {
        self.base
            .version()
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(self.map.version())
    }
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
                params: Vec::new(),
                returns: None,
                body: Arc::new(Expr::literal(Literal::Integer(value))),
                body_text: value.to_string(),
                source: SourceId(7),
                deterministic: true,
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
