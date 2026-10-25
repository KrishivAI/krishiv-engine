// SPDX-License-Identifier: Apache-2.0
//! Session variables: one named, typed value per [`Session`](crate::Session).
//!
//! A variable is declared with a type (or infers one from its default),
//! holds one value of that type, and resolves in expressions after columns
//! and outer references. The state lives in the session: one map per
//! session, no process-global or thread-local state. Every query reads one
//! snapshot, taken with [`SessionVariables::snapshot`] like the
//! configuration snapshot.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, PoisonError, RwLock};

use krishiv_plan::Literal;
use krishiv_plan::analyzer::{SessionVariable, VariableSnapshot, variable_name};
use krishiv_sql::ast::ObjectName;
use krishiv_types::DataType;

/// A declared session variable: its type, its current value (always of
/// that type) and the query re-evaluating its default.
#[derive(Debug, Clone)]
pub(crate) struct StoredVariable {
    /// The declared or inferred type.
    pub(crate) data_type: DataType,
    /// The current value, of [`Self::data_type`].
    pub(crate) value: Literal,
    /// The padded `SELECT` re-evaluating the default (`None` for a
    /// declaration without a default, whose default is NULL).
    pub(crate) default_query: Option<String>,
}

/// The search path a declaration resolves a name on: only the session
/// scope, as in Spark.
pub(crate) const DECLARE_SEARCH_PATH: &str = "[`system`.`session`]";

/// The search path an assignment or drop resolves a name on, as in Spark.
pub(crate) const RESOLVE_SEARCH_PATH: &str =
    "[`system`.`builtin`, `system`.`session`, `spark_catalog`.`default`]";

/// The session's variables, by lower-case unqualified name, with a version
/// bumped under the write lock after every change.
///
/// Each `put` and `remove` is atomic, but a command that evaluates and then
/// assigns (one `SET VARIABLE` with several targets) spans no lock: writers
/// must be externally synchronized, as a concurrent command can interleave
/// its snapshot between the evaluation and the assignment.
#[derive(Debug, Default)]
pub(crate) struct SessionVariables {
    map: RwLock<BTreeMap<String, StoredVariable>>,
    version: AtomicU64,
}

impl SessionVariables {
    /// The snapshot one query resolves, with the version it was taken at.
    /// The bump happens under the write lock, so a snapshot never pairs
    /// newer contents with an older version (only the reverse, which
    /// re-resolves). The snapshot clones every value: variables hold small
    /// scalars, and the clone is not memory-accounted.
    pub(crate) fn snapshot(&self) -> Arc<VariableSnapshot> {
        let map = self.map.read().unwrap_or_else(PoisonError::into_inner);
        let variables = map
            .iter()
            .map(|(name, var)| {
                (
                    name.clone(),
                    SessionVariable {
                        data_type: var.data_type.clone(),
                        value: var.value.clone(),
                    },
                )
            })
            .collect();
        Arc::new(VariableSnapshot {
            variables,
            version: self.version.load(Ordering::Acquire),
        })
    }

    /// The stored variable `key`, if declared.
    pub(crate) fn get(&self, key: &str) -> Option<StoredVariable> {
        self.map
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(key)
            .cloned()
    }

    /// Declares or replaces `key`, bumping the version.
    pub(crate) fn put(&self, key: String, var: StoredVariable) {
        let mut map = self.map.write().unwrap_or_else(PoisonError::into_inner);
        map.insert(key, var);
        self.version.fetch_add(1, Ordering::AcqRel);
    }

    /// Drops `key`, bumping the version when it existed.
    pub(crate) fn remove(&self, key: &str) -> Option<StoredVariable> {
        let mut map = self.map.write().unwrap_or_else(PoisonError::into_inner);
        let out = map.remove(key);
        if out.is_some() {
            self.version.fetch_add(1, Ordering::AcqRel);
        }
        out
    }
}

/// The storage key of `name` (`v`, `session.v` or `system.session.v`, in
/// any letter case): the lower-case unqualified name. `None` when the
/// parts are not a variable's shape, by [`variable_name`].
pub(crate) fn variable_key(name: &ObjectName) -> Option<String> {
    let parts: Vec<&str> = name.parts.iter().map(|p| p.value.as_str()).collect();
    variable_name(&parts).map(str::to_lowercase)
}

/// `name` as Spark quotes it in errors: `` `v` ``, `` `session`.`v` ``.
pub(crate) fn quote_name(name: &ObjectName) -> String {
    name.parts
        .iter()
        .map(|p| format!("`{}`", p.value.replace('`', "``")))
        .collect::<Vec<_>>()
        .join(".")
}

/// The normalized `` `system`.`session`.`v` `` display of storage `key`.
pub(crate) fn quote_normalized(key: &str) -> String {
    format!("`system`.`session`.`{}`", key.replace('`', "``"))
}
