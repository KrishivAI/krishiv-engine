// SPDX-License-Identifier: Apache-2.0
//! `SqliteCatalog`: the embedded catalog (ADR 0014 item 6; design P3-lakehouse §2.3), a `Catalog` and a `CatalogWrite` over
//! one SQLite file (the DuckLake pattern: metadata in a SQL database, data in files elsewhere). It keeps names, columns,
//! comments, defaults, properties, locations and partition columns; it never touches a table's files, so dropping a
//! table removes its row and nothing else (`purge` has nothing to purge here).
//!
//! * One transaction per change (`BEGIN IMMEDIATE`, WAL, `synchronous = FULL`, a busy timeout), so a process killed in
//!   the middle of a DDL leaves the catalog as it was before it, and two processes writing the same file take turns.
//! * The epoch lives in the file and is bumped inside the change's transaction: every change made through any process
//!   moves it, a refused or no-op call does not, and a second process sees the first one's changes at the next call
//!   (the `per_query` consistency of ADR 0037).
//! * A table's version is the epoch of the change that made it what it is: `alter_table` against a version that is no
//!   longer current fails, and a table dropped and created again is never at the version a reader saw. There is no
//!   history, so a load at any other version finds nothing (`time_travel` is off).
//! * Namespaces have one part. `default` exists in a new file and cannot be dropped, as in Spark's session catalog.
//!   Names are `[A-Za-z0-9_]+` and are stored folded (lower case unless the catalog is case sensitive). That rule is
//!   stored in the file and the one rule for every call: a file opened with the other value is refused, and the per-query
//!   `NameForm::case_sensitive` does not change how this catalog matches names.
//! * Column types are stored as Spark catalog strings and read back by the DDL parser; a type that does not round-trip
//!   is refused at create. NOT NULL is kept as declared (the writer enforces it, `krishiv-session`'s write check).
//! * A failure of SQLite (busy past the timeout, a full disk, a damaged value) is `KRISHIV_CATALOG_FAILURE` naming the
//!   catalog and the table; the detail is logged, cut short. A file written by a newer format is refused.
//!
//! The calls block for as long as SQLite takes (a local file; the timeout bounds contention), inside the returned
//! future, as the file catalog is "local and cheap" (ADR 0037).

mod functions;
mod store;
mod write;

pub use functions::{FunctionChange, StoredPythonFunction};

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use krishiv_common::error::{
    KrishivCatalogFailure, KrishivError, UnsupportedFeatureCatalogOperation,
};
use krishiv_plan::TableVersion;
use krishiv_plan::analyzer::{
    BoxFuture, Catalog, CatalogCapabilities, CatalogWrite, FileMetadata, NameForm, TableIdent,
    TableKind, TableMeta,
};
use rusqlite::Connection;

use crate::location::{LocationPolicy, check_location};
use store::{Failure, TableRow};

/// How long a call waits for another writer before it fails as a catalog failure.
const BUSY_TIMEOUT: Duration = Duration::from_secs(10);
/// The longest SQLite detail the log carries.
const LOGGED_DETAIL_CHARS: usize = 300;

/// What a table keeps besides the columns [`TableMeta`] carries.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TableDetails {
    /// The table's comment.
    pub comment: Option<String>,
    /// Where its data lives, as given at create (this catalog never touches it; [`SqliteCatalog::table_details`] checks it).
    pub location: Option<String>,
    /// Its properties.
    pub properties: BTreeMap<String, String>,
    /// Its partition columns, in order.
    pub partition_columns: Vec<String>,
}

/// The embedded SQLite catalog; see the module documentation.
pub struct SqliteCatalog {
    name: String,
    case_sensitive: bool,
    conn: Mutex<Connection>,
    /// What `epoch()` has answered, and the shift a read fault added to it.
    epoch_state: Mutex<EpochState>,
}

/// See [`SqliteCatalog::epoch`](Catalog::epoch).
#[derive(Debug, Default)]
struct EpochState {
    /// Added to the file's epoch.
    offset: u64,
    /// The highest answer so far.
    handed: u64,
}

impl std::fmt::Debug for SqliteCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteCatalog")
            .field("name", &self.name)
            .field("case_sensitive", &self.case_sensitive)
            .finish_non_exhaustive()
    }
}

fn quoted(part: &str) -> String {
    format!("`{}`", part.replace('`', "``"))
}

impl SqliteCatalog {
    /// The catalog `name` over the file at `path` (made when it is not there), names folded as `case_sensitive` says.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE` when the file cannot be opened or is not a catalog of a format this build reads.
    pub fn open(
        name: impl Into<String>,
        path: &Path,
        case_sensitive: bool,
    ) -> Result<Self, KrishivError> {
        let name = name.into();
        let conn = Connection::open(path).map_err(|e| fault(&name, "", &e.to_string()))?;
        Self::with_connection(name, conn, case_sensitive)
    }

    /// A catalog in memory, gone when it is dropped (tests and scratch sessions).
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE` when SQLite cannot make the database.
    pub fn open_in_memory(
        name: impl Into<String>,
        case_sensitive: bool,
    ) -> Result<Self, KrishivError> {
        let name = name.into();
        let conn = Connection::open_in_memory().map_err(|e| fault(&name, "", &e.to_string()))?;
        Self::with_connection(name, conn, case_sensitive)
    }

    fn with_connection(
        name: String,
        mut conn: Connection,
        case_sensitive: bool,
    ) -> Result<Self, KrishivError> {
        let setup = |conn: &mut Connection, timeout: Duration| -> Result<(), Failure> {
            conn.busy_timeout(timeout)?;
            conn.pragma_update(None, "foreign_keys", true)?;
            // WAL for a file (readers never block the writer); a memory database answers "memory", which is fine.
            conn.pragma_update_and_check(None, "journal_mode", "WAL", |_| Ok(()))?;
            conn.pragma_update(None, "synchronous", "FULL")?;
            store::migrate(conn, case_sensitive)
        };
        // SQLite does not run the busy handler for every lock conflict while a database is being set up: switching a
        // new file to WAL, and the first schema transaction, answer "database is locked" at once when another
        // connection is doing the same. Setup is idempotent (the migration is one transaction), so try again until
        // the busy timeout is spent.
        let started = std::time::Instant::now();
        loop {
            // Each attempt may wait only for what is left of the one budget, so the whole setup stays within it.
            let left = BUSY_TIMEOUT.saturating_sub(started.elapsed());
            match setup(&mut conn, left) {
                Ok(()) => break,
                Err(Failure::Sql(detail))
                    if detail.contains("locked")
                        && !left.is_zero()
                        && started.elapsed() < BUSY_TIMEOUT =>
                {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(f) => return Err(self_fault(&name, "", f)),
            }
        }
        Ok(Self {
            name,
            case_sensitive,
            conn: Mutex::new(conn),
            epoch_state: Mutex::new(EpochState::default()),
        })
    }

    fn fold(&self, text: &str) -> String {
        if self.case_sensitive {
            text.to_owned()
        } else {
            text.to_lowercase()
        }
    }

    fn lock(&self) -> MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn fault(&self, relation: &str, f: Failure) -> KrishivError {
        self_fault(&self.name, relation, f)
    }

    fn unsupported(&self, operation: &str) -> KrishivError {
        UnsupportedFeatureCatalogOperation {
            catalog_name: quoted(&self.name),
            operation: operation.to_owned(),
        }
        .into()
    }

    /// The one part of a namespace.
    fn namespace(&self, namespace: &[String]) -> Result<String, KrishivError> {
        match namespace {
            [one] => Ok(self.fold(one)),
            _ => Err(self.unsupported("a namespace of more than one part")),
        }
    }

    /// A read of the database.
    fn read<T>(
        &self,
        relation: &str,
        f: impl FnOnce(&Connection) -> Result<T, Failure>,
    ) -> Result<T, KrishivError> {
        let conn = self.lock();
        f(&conn).map_err(|failure| self.fault(relation, failure))
    }

    /// One change in one immediate transaction. `f` bumps the epoch (with `store::bump_epoch`) when it is about to
    /// change something and returns whether it did: only then is the transaction committed (and the epoch moves); a
    /// failure, a refusal or a no-op rolls everything back.
    fn change<T>(
        &self,
        relation: &str,
        f: impl FnOnce(&Connection) -> Result<(T, bool), Failure>,
    ) -> Result<T, KrishivError> {
        let mut conn = self.lock();
        let run = |conn: &mut Connection| -> Result<T, Failure> {
            let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            let (value, changed) = f(&tx)?;
            if changed {
                tx.commit()?;
            }
            Ok(value)
        };
        run(&mut conn).map_err(|failure| self.fault(relation, failure))
    }

    fn not_found(&self, namespace: &str, name: &str) -> KrishivError {
        krishiv_common::error::TableOrViewNotFound {
            relation_name: format!(
                "{}.{}.{}",
                quoted(&self.name),
                quoted(namespace),
                quoted(name)
            ),
            search_path: format!("[{}.`default`]", quoted(&self.name)),
        }
        .into()
    }

    fn ident(&self, namespace: &str, name: &str) -> TableIdent {
        TableIdent {
            catalog: self.name.clone(),
            namespace: vec![namespace.to_owned()],
            name: name.to_owned(),
        }
    }

    fn meta(&self, namespace: &str, name: &str, row: &TableRow) -> TableMeta {
        TableMeta {
            ident: self.ident(namespace, name),
            columns: row.columns.iter().map(|c| c.column.clone()).collect(),
            version: Some(TableVersion(row.version)),
            kind: TableKind::Table,
            file_metadata: FileMetadata::None,
            column_comments: row
                .columns
                .iter()
                .filter_map(|c| Some((c.column.name.clone(), c.comment.clone()?)))
                .collect(),
        }
    }

    /// What the table keeps besides its columns; `None` when it does not exist. The file may be written by another
    /// process, so a stored location goes through `policy` before it is returned (design 5.6).
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE`; `KRISHIV_CATALOG_LOCATION_REFUSED` when `policy` refuses the location.
    pub fn table_details(
        &self,
        id: &TableIdent,
        policy: &dyn LocationPolicy,
    ) -> Result<Option<TableDetails>, KrishivError> {
        let Ok(namespace) = self.namespace(&id.namespace) else {
            return Ok(None);
        };
        let name = self.fold(&id.name);
        let row = self.read(&id.name, |c| store::read_table(c, &namespace, &name))?;
        let Some(r) = row else {
            return Ok(None);
        };
        if let Some(location) = &r.location {
            check_location(&self.name, &quoted(&id.name), location, policy)?;
        }
        Ok(Some(TableDetails {
            comment: r.comment,
            location: r.location,
            properties: r.properties,
            partition_columns: r.partition_columns,
        }))
    }

    /// The names of the namespaces, sorted.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE`.
    pub fn list_namespaces(&self) -> Result<Vec<String>, KrishivError> {
        self.read("", store::namespace_names)
    }

    /// The names of the tables of `namespace`, sorted.
    ///
    /// # Errors
    /// `SCHEMA_NOT_FOUND`; `KRISHIV_CATALOG_FAILURE`.
    pub fn list_tables(&self, namespace: &[String]) -> Result<Vec<String>, KrishivError> {
        let namespace = self.namespace(namespace)?;
        self.read("", |c| {
            if store::read_namespace(c, &namespace)?.is_none() {
                return Err(Failure::Krishiv(
                    krishiv_common::error::SchemaNotFound {
                        schema_name: quoted(&namespace),
                    }
                    .into(),
                ));
            }
            store::table_names(c, &namespace)
        })
    }
}

/// A catalog fault with the detail in the log only; a rule's own error is returned as it is.
fn self_fault(catalog: &str, relation: &str, failure: Failure) -> KrishivError {
    match failure {
        Failure::Krishiv(e) => e,
        Failure::Sql(detail) => fault(catalog, relation, &detail),
    }
}

fn fault(catalog: &str, relation: &str, detail: &str) -> KrishivError {
    let mut cut: String = detail.chars().take(LOGGED_DETAIL_CHARS).collect();
    if detail.chars().count() > LOGGED_DETAIL_CHARS {
        cut.push('…');
    }
    tracing::warn!(
        target: "krishiv::catalog",
        catalog,
        table = relation,
        detail = cut.as_str(),
        "the SQLite catalog failed"
    );
    KrishivCatalogFailure {
        catalog: quoted(catalog),
        relation_name: relation.to_owned(),
        reason: "the call failed; the server log has the detail (target `krishiv::catalog`)"
            .to_owned(),
    }
    .into()
}

impl Catalog for SqliteCatalog {
    fn name(&self) -> &str {
        &self.name
    }

    /// The epoch in the file, which another process's change moves too. It never goes backwards and never repeats a
    /// value it handed out earlier: when the file cannot be read it answers one more than any earlier answer, and
    /// every later answer is shifted above that (by two, so not even an unchanged file lands on it). Nothing keyed
    /// by an earlier epoch is served as current after a fault, and a fault value never equals a later file epoch.
    fn epoch(&self) -> u64 {
        let read = store::read_epoch(&self.lock());
        let mut state = self
            .epoch_state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let answer = match read {
            Ok(file) => u64::try_from(file)
                .unwrap_or(0)
                .saturating_add(state.offset),
            Err(_) => {
                state.offset = state.offset.saturating_add(2);
                state.handed.saturating_add(1)
            }
        };
        state.handed = state.handed.max(answer);
        state.handed
    }

    fn single_part_namespace(&self) -> bool {
        true
    }

    fn capabilities(&self) -> CatalogCapabilities {
        CatalogCapabilities::read_write()
    }

    fn as_write(&self) -> Option<&dyn CatalogWrite> {
        Some(self)
    }

    fn load_table<'a>(
        &'a self,
        id: &'a TableIdent,
        version: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        Box::pin(async move {
            // A search path that names no schema of this catalog resolves only temporary relations, and it has none.
            if form.temporary_only {
                return Ok(None);
            }
            let [namespace] = id.namespace.as_slice() else {
                return Ok(None);
            };
            // One rule, the file's, whatever the query's `caseSensitive` says (see the module documentation).
            let (namespace, name) = (self.fold(namespace), self.fold(&id.name));
            let row = self.read(&id.name, |c| store::read_table(c, &namespace, &name))?;
            Ok(row
                .filter(|r| version.is_none_or(|v| v == TableVersion(r.version)))
                .map(|r| self.meta(&namespace, &name, &r)))
        })
    }
}

#[cfg(test)]
mod tests;
