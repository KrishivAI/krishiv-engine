// SPDX-License-Identifier: Apache-2.0
//! `spark.catalog` (Spark's `org.apache.spark.sql.catalog.Catalog`, plan row PR-46, T082): the embedded
//! catalog API every surface shares. Listings and lookups read the session catalog and the permanent store
//! directly and return Spark's structs (`Database`, `Table`, `Column`, `Function`); the mutating calls run
//! the SQL statement Spark's `CatalogImpl` runs, so the access checks, errors and cache invalidation are
//! those of the statement. Spark Connect lowers its `Catalog` relation arms to these functions.

use krishiv_common::error::KrishivError;

mod ops;

use crate::catalog::{GLOBAL_NAMESPACE, global_key};
use crate::session::Session;
use crate::store::{CatalogStore, CatalogTable};
use krishiv_plan::analyzer::Privilege;

/// Spark's `Database`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Database {
    /// The namespace's name.
    pub name: String,
    /// The catalog it belongs to.
    pub catalog: String,
    /// Its comment.
    pub description: Option<String>,
    /// Its location URI.
    pub location_uri: String,
}

/// Spark's `Table`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Table {
    /// The table's name.
    pub name: String,
    /// The catalog; `None` for a temporary view.
    pub catalog: Option<String>,
    /// The namespace parts: empty for a local temporary view, `[global_temp]` for a global one.
    pub namespace: Option<Vec<String>>,
    /// Its comment.
    pub description: Option<String>,
    /// `MANAGED`, `EXTERNAL`, `VIEW` or `TEMPORARY`.
    pub table_type: String,
    /// Whether it is a temporary view (local or global).
    pub is_temporary: bool,
}

/// Spark's `Column`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Column {
    /// The column's name.
    pub name: String,
    /// Its comment.
    pub description: Option<String>,
    /// Its type, as `DESCRIBE` prints it.
    pub data_type: String,
    /// Whether it accepts null.
    pub nullable: bool,
    /// Whether it is a partition column.
    pub is_partition: bool,
    /// Whether it is a bucket column.
    pub is_bucket: bool,
    /// Whether it is a clustering column.
    pub is_cluster: bool,
}

/// Spark's `Function`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Function {
    /// The function's name.
    pub name: String,
    /// The catalog; `None` for a built-in or temporary function.
    pub catalog: Option<String>,
    /// The namespace parts; `None` for a built-in or temporary function.
    pub namespace: Option<Vec<String>>,
    /// Its description.
    pub description: Option<String>,
    /// The implementing class, where Spark reports one.
    pub class_name: String,
    /// Whether it is built in or temporary.
    pub is_temporary: bool,
}

/// A name as `spark.catalog` takes it: `table`, `db.table` or `catalog.db.table`.
/// The parts of a multipart name as Spark parses it: dots separate parts outside backquotes, and a doubled
/// backquote inside backquotes is one.
fn parts(name: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    let mut chars = name.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '`' if quoted && chars.peek() == Some(&'`') => {
                cur.push('`');
                chars.next();
            }
            '`' => quoted = !quoted,
            '.' if !quoted => out.push(std::mem::take(&mut cur)),
            c => cur.push(c),
        }
    }
    out.push(cur);
    out
}

/// The partition columns of the files under the local directory `location`: the `key` of each level of
/// `key=value` directories, following the first one at each level. Empty for a path that is not a local
/// directory.
fn partition_keys(location: &str) -> Vec<String> {
    let mut dir = std::path::PathBuf::from(location.strip_prefix("file:").unwrap_or(location));
    let mut keys: Vec<String> = Vec::new();
    loop {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            break;
        };
        let mut next: Option<(String, std::path::PathBuf)> = None;
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if e.path().is_dir()
                && let Some((k, _)) = name.split_once('=')
                && !k.is_empty()
                && next.as_ref().is_none_or(|(_, p)| e.path() < *p)
            {
                next = Some((k.to_owned(), e.path()));
            }
        }
        let Some((key, path)) = next else { break };
        if keys.iter().any(|k| k.eq_ignore_ascii_case(&key)) {
            break;
        }
        keys.push(key);
        dir = path;
    }
    keys
}

fn unsupported_schema() -> KrishivError {
    krishiv_common::error::KrishivUnsupportedFeature {
        feature: "anything but a column list in the schema of createTable".to_owned(),
    }
    .into()
}

/// A SQL string literal of `v`: backslashes and single quotes escaped. Under
/// `spark.sql.parser.escapedStringLiterals` the parser keeps backslashes as written, so a value holding a
/// backslash or a quote cannot be written back exactly: that is refused rather than silently changed.
fn literal(v: &str, escaped_strings: bool) -> Result<String, KrishivError> {
    if escaped_strings && v.contains(['\\', '\'']) {
        return Err(krishiv_common::error::KrishivUnsupportedFeature {
            feature:
                "a value holding a quote or backslash under spark.sql.parser.escapedStringLiterals"
                    .to_owned(),
        }
        .into());
    }
    Ok(format!(
        "'{}'",
        v.replace('\\', "\\\\").replace('\'', "\\'")
    ))
}

fn quote(part: &str) -> String {
    format!("`{}`", part.replace('`', "``"))
}

fn quoted(name: &str) -> String {
    parts(name)
        .iter()
        .map(|p| quote(p))
        .collect::<Vec<_>>()
        .join(".")
}

/// The catalog API of one session ([`Session::catalog`]).
#[derive(Debug, Clone, Copy)]
pub struct Catalog<'a> {
    session: &'a Session,
}

impl Session {
    /// Spark's `spark.catalog`.
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// assert_eq!(session.catalog().current_database(), "default");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub const fn catalog(&self) -> Catalog<'_> {
        Catalog { session: self }
    }
}

impl Catalog<'_> {
    async fn run(&self, sql: &str) -> Result<(), KrishivError> {
        self.session.sql(sql).await?.collect().await?;
        Ok(())
    }

    /// Checks what the SQL statement that stands for this call needs (`Browse` on what it names), as
    /// `Session::sql` does before it runs one: the same policy, denial and audit record, so the catalog API
    /// shows a principal no more than `SHOW TABLES` or `DESCRIBE` would.
    async fn authorize_as(&self, sql: &str) -> Result<(), KrishivError> {
        let env = self.session.query_env();
        let statement = krishiv_sql::parse_statement(sql, &env.parser).map_err(|e| e.error)?;
        let class = krishiv_sql::classify(&statement);
        self.session
            .authorize_statement(&statement, class, &env)
            .await
    }

    fn case_sensitive(&self) -> bool {
        self.session.query_env().analysis.conf.case_sensitive
    }

    fn fold(&self, s: &str) -> String {
        if self.case_sensitive() {
            s.to_owned()
        } else {
            s.to_lowercase()
        }
    }

    /// The current catalog's name.
    #[must_use]
    pub fn current_catalog(&self) -> String {
        self.session.state.catalog.current_catalog()
    }

    /// Makes `name` the current catalog: `spark_catalog` or a declared catalog this principal sees.
    ///
    /// A catalog's name matches whatever its case, in either `spark.sql.caseSensitive` mode, as Spark's
    /// `CatalogManager` and the SQL `USE` do. (Divergence: Spark 4.2 resets the namespace to `default` when the name
    /// differs from the current catalog's only in case; here the current namespace is kept.) The `system` catalog is
    /// Krishiv's own and is reached with `USE system`, not here: Spark has no such catalog, so it is
    /// `CATALOG_NOT_FOUND` like any other name.
    ///
    /// # Errors
    /// `CATALOG_NOT_FOUND` for any other name (a namespace's name is not a catalog).
    pub async fn set_current_catalog(&self, name: &str) -> Result<(), KrishivError> {
        // `USE <name>` would switch the namespace for a namespace's name: the catalog is checked first.
        let Some(known) = self
            .list_catalogs()
            .into_iter()
            .find(|c| c.eq_ignore_ascii_case(name))
        else {
            return Err(krishiv_common::error::CatalogNotFound {
                catalog_name: krishiv_common::names::to_sql_id(name)?,
                config: format!("\"spark.sql.catalog.{name}\""),
            }
            .into());
        };
        self.run(&format!("USE {}", quoted(&known))).await
    }

    /// The catalogs, sorted: `spark_catalog` and the declared catalogs this principal sees.
    ///
    /// Spark returns the loaded catalogs and `spark_catalog`. The `system` catalog is not listed, see
    /// [`Catalog::set_current_catalog`].
    #[must_use]
    pub fn list_catalogs(&self) -> Vec<String> {
        let mut names = self.session.catalog_names();
        names.retain(|c| c != crate::system::SYSTEM);
        names
    }

    /// The current namespace.
    #[must_use]
    pub fn current_database(&self) -> String {
        self.session.state.catalog.current_namespace()
    }

    /// Makes `name` the current namespace.
    ///
    /// # Errors
    /// `SCHEMA_NOT_FOUND` for an unknown namespace.
    pub async fn set_current_database(&self, name: &str) -> Result<(), KrishivError> {
        self.run(&format!("USE NAMESPACE {}", quoted(name))).await
    }

    /// Whether `catalog` names a declared catalog this principal sees: not `spark_catalog`, not
    /// `system`.
    fn is_declared(&self, catalog: &str) -> bool {
        !catalog.eq_ignore_ascii_case(crate::catalog::CATALOG)
            && !crate::system::is_system(catalog)
            && self
                .list_catalogs()
                .iter()
                .any(|c| c.eq_ignore_ascii_case(catalog))
    }

    /// Refuses a call about the namespaces or tables of the declared catalog `catalog`: a declared
    /// catalog resolves names in queries only for now (design P3-T067 PR-58a), and the embedded
    /// catalog's namespaces are never answered in its name.
    fn refuse_declared(catalog: &str) -> KrishivError {
        krishiv_common::error::KrishivUnsupportedFeature {
            feature: format!(
                "the namespaces and tables of the declared catalog {} through the Catalog API",
                quote(catalog)
            ),
        }
        .into()
    }

    /// [`Self::refuse_declared`] when the current catalog is a declared one.
    fn check_current(&self) -> Result<(), KrishivError> {
        let current = self.current_catalog();
        if self.is_declared(&current) {
            return Err(Self::refuse_declared(&current));
        }
        Ok(())
    }

    /// The namespaces are always the embedded catalog's, so `spark_catalog` names them.
    fn database_of(&self, name: &str, meta: &crate::store::NamespaceMeta) -> Database {
        Database {
            name: name.to_owned(),
            catalog: crate::catalog::CATALOG.to_owned(),
            description: meta.comment.clone(),
            location_uri: meta.location.clone().unwrap_or_else(|| {
                let wh = self.session.warehouse_dir().unwrap_or_default();
                let dir = if name == crate::store::DEFAULT_NAMESPACE {
                    wh
                } else {
                    wh.join(format!("{name}.db"))
                };
                format!("file:{}", dir.display())
            }),
        }
    }

    /// The namespaces of the current catalog, sorted by name.
    ///
    /// The namespaces of the current catalog, sorted by name, optionally only those matching the
    /// `*`/`|` pattern.
    ///
    /// # Errors
    /// The access error `SHOW NAMESPACES` would give.
    pub async fn list_databases(
        &self,
        pattern: Option<&str>,
    ) -> Result<Vec<Database>, KrishivError> {
        // A declared current catalog: its own namespaces (design P3-T067 PR-58b).
        let current = self.current_catalog();
        if self.is_declared(&current) {
            self.authorize_as(&format!("SHOW NAMESPACES IN {}", quoted(&current)))
                .await?;
            let found = match self.session.declared_namespaces(&current, &[]).await {
                Some(found) => found?,
                None => return Err(Self::refuse_declared(&current)),
            };
            let mut out: Vec<Database> = found
                .iter()
                .map(|n| n.join("."))
                .filter(|n| pattern.is_none_or(|p| crate::introspection::matches(p, n)))
                .map(|name| Database {
                    name,
                    catalog: current.clone(),
                    description: None,
                    location_uri: String::new(),
                })
                .collect();
            out.sort_by(|a, b| a.name.cmp(&b.name));
            return Ok(out);
        }
        self.authorize_as("SHOW NAMESPACES").await?;
        self.check_current()?;
        // Only what the principal may see, as `SHOW NAMESPACES` lists (P3-T062 §2.4).
        let mut out = Vec::new();
        for d in self.databases(pattern) {
            if self.session.may_list_namespace(&d.name).await {
                out.push(d);
            }
        }
        Ok(out)
    }

    /// The namespaces of the embedded catalog `spark_catalog` whatever the current catalog is,
    /// sorted by name: the Arrow Flight SQL `GetDbSchemas` listing (design P3-T101 §5), which
    /// names its catalog rather than following `USE`.
    ///
    /// # Errors
    /// The access error `SHOW NAMESPACES` gives with `spark_catalog` current: `Browse` on the
    /// catalog, audited as a read.
    pub async fn list_embedded_databases(&self) -> Result<Vec<Database>, KrishivError> {
        self.session
            .authorize_privilege(
                &krishiv_plan::analyzer::Securable::Namespace(vec![
                    crate::catalog::CATALOG.to_owned(),
                ]),
                Privilege::Browse,
                krishiv_sql::StatementClass::Read,
            )
            .await?;
        Ok(self.databases(None))
    }

    fn databases(&self, pattern: Option<&str>) -> Vec<Database> {
        let mut out: Vec<Database> = self
            .session
            .state
            .catalog
            .store
            .namespaces()
            .iter()
            .filter(|(n, _)| pattern.is_none_or(|p| crate::introspection::matches(p, n)))
            .map(|(n, m)| self.database_of(n, m))
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    /// The namespace `name`.
    ///
    /// # Errors
    /// `SCHEMA_NOT_FOUND`, or the access error `DESCRIBE NAMESPACE` would give.
    pub async fn get_database(&self, name: &str) -> Result<Database, KrishivError> {
        self.authorize_as(&format!("DESCRIBE NAMESPACE {}", quoted(name)))
            .await?;
        self.visible_database(name).await
    }

    /// [`Self::lookup_database`], with a namespace the principal may not list reading as missing
    /// (review B3; `Session::namespace_hidden`, the predicate of the SQL namespace lookups): every
    /// by-namespace entry point of this API resolves its namespace here.
    async fn visible_database(&self, name: &str) -> Result<Database, KrishivError> {
        let found = self.lookup_database(name)?;
        if self.session.namespace_hidden(&found.name).await {
            return Err(Self::database_not_found(name));
        }
        Ok(found)
    }

    /// `db` or `spark_catalog.db`, as Spark's `resolveNamespace` accepts it: the namespace's name.
    fn bare_database(name: &str) -> String {
        let p = parts(name);
        match p.as_slice() {
            [db] => db.clone(),
            [cat, db] if cat.eq_ignore_ascii_case(crate::catalog::CATALOG) => db.clone(),
            _ => name.to_owned(),
        }
    }

    /// `SCHEMA_NOT_FOUND` for the namespace `name`.
    fn database_not_found(name: &str) -> KrishivError {
        krishiv_common::error::SchemaNotFound {
            schema_name: quote(&Self::bare_database(name)),
        }
        .into()
    }

    fn lookup_database(&self, name: &str) -> Result<Database, KrishivError> {
        // A bare name is the current catalog's; a declared catalog's namespaces are not answered here.
        match parts(name).as_slice() {
            [_] => self.check_current()?,
            [cat, _] if self.is_declared(cat) => return Err(Self::refuse_declared(cat)),
            _ => {}
        }
        let bare = Self::bare_database(name);
        let name = bare.as_str();
        let want = self.fold(name);
        self.session
            .state
            .catalog
            .store
            .namespaces()
            .iter()
            .find(|(n, _)| self.fold(n) == want)
            .map(|(n, m)| self.database_of(n, m))
            .ok_or_else(|| Self::database_not_found(name))
    }

    /// Whether the namespace exists.
    ///
    /// # Errors
    /// The access error `DESCRIBE NAMESPACE` would give.
    pub async fn database_exists(&self, name: &str) -> Result<bool, KrishivError> {
        match self.get_database(name).await {
            Ok(_) => Ok(true),
            Err(e) if e.error_class().name() == "SCHEMA_NOT_FOUND" => Ok(false),
            Err(e) => Err(e),
        }
    }

    fn table_of(t: &CatalogTable) -> Table {
        // `CatalogImpl.makeTable` gives `tableType.name`.
        let table_type = if t
            .view
            .as_ref()
            .is_some_and(crate::store::ViewDef::is_metric)
        {
            "METRIC_VIEW"
        } else if t.view.is_some() {
            "VIEW"
        } else if t.managed {
            "MANAGED"
        } else {
            "EXTERNAL"
        };
        Table {
            name: t.name.clone(),
            catalog: Some(crate::catalog::CATALOG.to_owned()),
            namespace: Some(vec![t.namespace.clone()]),
            description: t.comment.clone(),
            table_type: table_type.to_owned(),
            is_temporary: false,
        }
    }

    fn temp_table(name: &str, global: bool) -> Table {
        Table {
            name: name.to_owned(),
            catalog: None,
            namespace: Some(if global {
                vec![GLOBAL_NAMESPACE.to_owned()]
            } else {
                Vec::new()
            }),
            description: None,
            table_type: "TEMPORARY".to_owned(),
            is_temporary: true,
        }
    }

    /// The tables, views and temporary views of `database` (default: the current one), sorted by name.
    /// Local temporary views are listed for every database, global ones for `global_temp`.
    ///
    /// Permanent tables and views come first, then the temporary views, each group sorted by name.
    ///
    /// # Errors
    /// `SCHEMA_NOT_FOUND` for an unknown namespace, or the access error `SHOW TABLES` would give.
    pub async fn list_tables(
        &self,
        database: Option<&str>,
        pattern: Option<&str>,
    ) -> Result<Vec<Table>, KrishivError> {
        // `cat.ns` of a declared catalog: the catalog's own listing (design P3-T067 PR-58b).
        if let Some(d) = database
            && let [catalog, ns @ ..] = parts(d).as_slice()
            && !ns.is_empty()
            && self.is_declared(catalog)
        {
            self.authorize_as(&format!("SHOW TABLES IN {}", quoted(d)))
                .await?;
            let found = match self.session.declared_tables(catalog, ns).await {
                Some(found) => found?,
                None => return Err(Self::refuse_declared(catalog)),
            };
            let mut out: Vec<Table> = found
                .into_iter()
                .filter(|n| pattern.is_none_or(|p| crate::introspection::matches(p, n)))
                .map(|name| Table {
                    name,
                    catalog: Some(catalog.clone()),
                    namespace: Some(ns.to_vec()),
                    description: None,
                    table_type: "EXTERNAL".to_owned(),
                    is_temporary: false,
                })
                .collect();
            out.sort_by(|a, b| a.name.cmp(&b.name));
            return Ok(out);
        }
        let statement = match database {
            Some(d) => format!("SHOW TABLES IN {}", quoted(d)),
            None => "SHOW TABLES".to_owned(),
        };
        self.authorize_as(&statement).await?;
        let keep = |n: &str| pattern.is_none_or(|p| crate::introspection::matches(p, n));
        let catalog = &self.session.state.catalog;
        let mut permanent = Vec::new();
        let mut temporary = Vec::new();
        // `system` is a read-only catalog of tables with no namespaces: after `USE system`, or for `IN system`,
        // its tables are listed (as `SHOW TABLES` does), not the session catalog's.
        let in_system = match database {
            Some(d) => crate::system::is_system(d),
            None => crate::system::is_system(&catalog.current_catalog()),
        };
        if in_system {
            for t in crate::system::table_names() {
                if keep(t) {
                    permanent.push(Table {
                        name: (*t).to_owned(),
                        catalog: Some(crate::system::SYSTEM.to_owned()),
                        namespace: Some(Vec::new()),
                        description: None,
                        table_type: "MANAGED".to_owned(),
                        is_temporary: false,
                    });
                }
            }
        } else if database.is_some_and(|d| d.eq_ignore_ascii_case(GLOBAL_NAMESPACE)) {
            // Spark lists the global views and the session's local ones for `global_temp`, as
            // `SHOW TABLES IN global_temp` does.
            for n in catalog.global_names() {
                if keep(&n) {
                    temporary.push(Self::temp_table(&n, true));
                }
            }
        } else {
            let db = match database {
                // As `SHOW TABLES IN`: one the principal may not list reads as missing (review r3 X1).
                Some(d) => self.visible_database(d).await?.name,
                None => {
                    self.check_current()?;
                    self.current_database()
                }
            };
            // Only what the principal may see, as `SHOW TABLES` lists (P3-T062 §2.4).
            let listed = self
                .session
                .listed_relations(catalog.store.tables_in(&db))
                .await;
            permanent.extend(
                listed
                    .iter()
                    .filter(|t| keep(&t.name))
                    .map(|t| Self::table_of(t)),
            );
        }
        for n in catalog.names() {
            if keep(&n) {
                temporary.push(Self::temp_table(&n, false));
            }
        }
        permanent.sort_by(|a, b| a.name.cmp(&b.name));
        temporary.sort_by(|a, b| a.name.cmp(&b.name));
        permanent.extend(temporary);
        Ok(permanent)
    }

    /// The table, view or temporary view `name` (`t`, `db.t` or `global_temp.v`).
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_NOT_FOUND`, or the access error `DESCRIBE TABLE` would give.
    pub async fn get_table(&self, name: &str) -> Result<Table, KrishivError> {
        self.authorize_as(&format!("DESCRIBE TABLE {}", quoted(name)))
            .await?;
        self.visible_table(name).await
    }

    /// [`Self::lookup_table`], with a stored table or view the principal may not list reading as
    /// missing (review B3: the rule of the listings, so no existence oracle).
    async fn visible_table(&self, name: &str) -> Result<Table, KrishivError> {
        let table = self.lookup_table(name)?;
        if !table.is_temporary
            && let Some(ns) = table.namespace.as_ref().and_then(|n| n.first())
            && let Some(stored) = self.session.state.catalog.store.table(ns, &table.name)
            && self.session.hidden(&Session::relation_securable(&stored))
        {
            return Err(self.table_not_found(name));
        }
        Ok(table)
    }

    /// `TABLE_OR_VIEW_NOT_FOUND` for `name`.
    fn table_not_found(&self, name: &str) -> KrishivError {
        krishiv_common::error::TableOrViewNotFound {
            relation_name: quoted(name),
            search_path: self.session.search_path(),
        }
        .into()
    }

    fn lookup_table(&self, name: &str) -> Result<Table, KrishivError> {
        let not_found = || self.table_not_found(name);
        let catalog = &self.session.state.catalog;
        let cs = self.case_sensitive();
        let p = parts(name);
        match p.as_slice() {
            [only] => {
                if let Some(n) = catalog
                    .names()
                    .into_iter()
                    .find(|n| self.fold(n) == self.fold(only))
                {
                    return Ok(Self::temp_table(&n, false));
                }
                self.check_current()?;
                catalog
                    .store
                    .table(&self.current_database(), &self.fold(only))
                    .map(|t| Self::table_of(&t))
                    .ok_or_else(not_found)
            }
            [db, t]
                if db.eq_ignore_ascii_case(GLOBAL_NAMESPACE)
                    && catalog.exists(&global_key(t), cs) =>
            {
                Ok(Self::temp_table(t, true))
            }
            [db, t] => {
                if !db.eq_ignore_ascii_case(GLOBAL_NAMESPACE) {
                    self.check_current()?;
                }
                catalog
                    .store
                    .table(&self.fold(db), &self.fold(t))
                    .map(|t| Self::table_of(&t))
                    .ok_or_else(not_found)
            }
            [cat, _, _] if self.is_declared(cat) => Err(Self::refuse_declared(cat)),
            [cat, db, t] if cat.eq_ignore_ascii_case(crate::catalog::CATALOG) => catalog
                .store
                .table(&self.fold(db), &self.fold(t))
                .map(|t| Self::table_of(&t))
                .ok_or_else(not_found),
            _ => Err(not_found()),
        }
    }

    /// Whether the table or view exists.
    ///
    /// # Errors
    /// The access error `DESCRIBE TABLE` would give: a principal that may not browse a table learns
    /// nothing about whether it exists.
    pub async fn table_exists(&self, name: &str) -> Result<bool, KrishivError> {
        match self.get_table(name).await {
            Ok(_) => Ok(true),
            Err(e) if e.error_class().name() == "TABLE_OR_VIEW_NOT_FOUND" => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// The columns of the table or view `name`, in table order.
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_NOT_FOUND`, or the access error `DESCRIBE TABLE` would give.
    pub async fn list_columns(&self, name: &str) -> Result<Vec<Column>, KrishivError> {
        self.authorize_as(&format!("DESCRIBE TABLE {}", quoted(name)))
            .await?;
        let table = self.visible_table(name).await?;
        if table.is_temporary {
            let df = self
                .session
                .sql(&format!("SELECT * FROM {}", quoted(name)))
                .await?;
            let schema = df.schema().await?;
            return schema
                .fields()
                .iter()
                .map(|f| {
                    Ok(Column {
                        name: f.name().clone(),
                        description: None,
                        // Spark's `dataType.catalogString`: `int`, `string`, `array<int>`.
                        data_type: krishiv_types::DataType::from_arrow_field(f)?.catalog_string(),
                        nullable: f.is_nullable(),
                        is_partition: false,
                        is_bucket: false,
                        is_cluster: false,
                    })
                })
                .collect();
        }
        let ns = table
            .namespace
            .and_then(|n| n.into_iter().next())
            .unwrap_or_default();
        let t = self
            .session
            .state
            .catalog
            .store
            .table(&ns, &table.name)
            .ok_or_else(|| -> KrishivError {
                krishiv_common::error::TableOrViewNotFound {
                    relation_name: quoted(name),
                    search_path: self.session.search_path(),
                }
                .into()
            })?;
        if t.view.is_some() {
            // As `DESCRIBE TABLE`: a permanent view's columns need `Browse` on the view itself.
            self.session
                .authorize_view(
                    &t.namespace,
                    &t.name,
                    Privilege::Browse,
                    &self.session.query_env(),
                )
                .await?;
        }
        let in_list = |list: &[String], c: &str| list.iter().any(|x| x.eq_ignore_ascii_case(c));
        Ok(t.columns
            .iter()
            .map(|c| Column {
                name: c.name.clone(),
                description: c.comment.clone(),
                data_type: type_string(c),
                nullable: c.nullable,
                is_partition: in_list(&t.layout.partition_columns, &c.name),
                is_bucket: t
                    .layout
                    .bucket
                    .as_ref()
                    .is_some_and(|b| in_list(&b.columns, &c.name)),
                is_cluster: false,
            })
            .collect())
    }

    /// The functions visible to the session, sorted by name.
    ///
    /// The functions: the built-ins, the session's temporary functions and the current namespace's
    /// stored ones, sorted by name.
    ///
    /// # Errors
    /// The access error `SHOW FUNCTIONS` would give.
    pub async fn list_functions(
        &self,
        pattern: Option<&str>,
    ) -> Result<Vec<Function>, KrishivError> {
        self.list_functions_in(None, pattern).await
    }

    /// `spark.catalog.listFunctions(db, pattern)`: the built-in and session functions, and the stored
    /// functions of `database` (the current one for `None`). The session's current database is not changed.
    ///
    /// # Errors
    /// `SCHEMA_NOT_FOUND` for an unknown `database`, or the access error `SHOW FUNCTIONS` would give.
    pub async fn list_functions_in(
        &self,
        database: Option<&str>,
        pattern: Option<&str>,
    ) -> Result<Vec<Function>, KrishivError> {
        Ok(self
            .list_functions_tagged(database, pattern)
            .await?
            .into_iter()
            .map(|(f, _)| f)
            .collect())
    }

    /// `list_functions_in`, each function with whether it is a built-in (its origin, not its name: a session
    /// function may share a built-in's name).
    pub(crate) async fn list_functions_tagged(
        &self,
        database: Option<&str>,
        pattern: Option<&str>,
    ) -> Result<Vec<(Function, bool)>, KrishivError> {
        self.authorize_as("SHOW FUNCTIONS").await?;
        let db = match database {
            Some(d) => self.get_database(d).await?.name,
            None => self.current_database(),
        };
        // A stored function is listed only to a principal that may see it (P3-T062 §2.4).
        let mut out = Vec::new();
        for (f, builtin) in self.functions_tagged(&db, pattern) {
            if self.function_listed(&f).await {
                out.push((f, builtin));
            }
        }
        Ok(out)
    }

    /// Whether the principal may see the function `f` (`Session::may_list`; built-in and
    /// temporary functions are not securables): what `list_functions` lists, and what
    /// `get_function` / `function_exists` find (review B3).
    async fn function_listed(&self, f: &Function) -> bool {
        match (&f.catalog, &f.namespace) {
            (Some(catalog), Some(namespace)) => {
                let mut parts = vec![catalog.clone()];
                parts.extend(namespace.iter().cloned());
                parts.push(f.name.clone());
                self.session
                    .may_list(&krishiv_plan::analyzer::Securable::Function(parts))
                    .await
            }
            _ => true,
        }
    }

    fn functions(&self, db: &str, pattern: Option<&str>) -> Vec<Function> {
        self.functions_tagged(db, pattern)
            .into_iter()
            .map(|(f, _)| f)
            .collect()
    }

    fn functions_tagged(&self, db: &str, pattern: Option<&str>) -> Vec<(Function, bool)> {
        let profile = self.session.state.profile;
        let keep = |n: &str| pattern.is_none_or(|p| crate::introspection::matches(p, n));
        let mut out: Vec<Function> = krishiv_functions::registry::builtins()
            .functions()
            .into_iter()
            // Only the names Spark itself lists: the engine's internal helpers (`__subscript`, ...) and the
            // grammar-owned `timestampadd` are no catalog functions.
            .filter(|d| {
                keep(d.name)
                    && profile >= d.since
                    && krishiv_functions::registry::builtins()
                        .spark_function(d.name)
                        .is_some()
            })
            .map(|d| Function {
                name: d.name.to_owned(),
                catalog: None,
                namespace: None,
                description: None,
                class_name: String::new(),
                is_temporary: true,
            })
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        // Spark lists the built-in functions before the session's and the stored ones.
        let builtin = out.len();
        for n in self.session.state.sql_functions.snapshot().names() {
            if keep(n) {
                out.push(Function {
                    name: n.to_owned(),
                    catalog: None,
                    namespace: None,
                    description: None,
                    class_name: String::new(),
                    is_temporary: true,
                });
            }
        }
        // Native UDFs and UDAFs registered on the session resolve in SQL, so they are listed as well.
        for n in self.session.state.udfs.names() {
            if keep(&n)
                && !out[builtin..]
                    .iter()
                    .any(|f| f.name.eq_ignore_ascii_case(&n))
            {
                out.push(Function {
                    name: n,
                    catalog: None,
                    namespace: None,
                    description: None,
                    class_name: String::new(),
                    is_temporary: true,
                });
            }
        }
        let stored = self.session.state.catalog.store.functions();
        for f in stored.in_namespace(db) {
            if keep(&f) {
                out.push(Function {
                    name: f,
                    catalog: Some(crate::catalog::CATALOG.to_owned()),
                    namespace: Some(vec![db.to_owned()]),
                    description: None,
                    class_name: String::new(),
                    is_temporary: false,
                });
            }
        }
        out[builtin..].sort_by(|a, b| a.name.cmp(&b.name));
        out.into_iter()
            .enumerate()
            .map(|(i, f)| (f, i < builtin))
            .collect()
    }

    /// The function `name`.
    ///
    /// # Errors
    /// `UNRESOLVED_ROUTINE`.
    pub async fn get_function(&self, name: &str) -> Result<Function, KrishivError> {
        self.authorize_as("SHOW FUNCTIONS").await?;
        let not_found = || -> KrishivError {
            krishiv_common::error::UnresolvedRoutine {
                routine_name: quoted(name),
                search_path: self.session.search_path(),
            }
            .into()
        };
        let p = parts(name);
        // `f` is a built-in, a temporary or a stored function of the current namespace; `db.f` and
        // `spark_catalog.db.f` name a stored function of that namespace only.
        let (database, last) = match p.as_slice() {
            [f] => (None, f.to_lowercase()),
            [db, f] => (Some(db.as_str()), f.to_lowercase()),
            [cat, db, f] if cat.eq_ignore_ascii_case(crate::catalog::CATALOG) => {
                (Some(db.as_str()), f.to_lowercase())
            }
            _ => return Err(not_found()),
        };
        let found = match database {
            None => self
                .functions(&self.current_database(), None)
                .into_iter()
                .find(|f| f.name == last),
            Some(db) => {
                let db = self.lookup_database(db).map_err(|_| not_found())?.name;
                self.stored_function(&db, &last)
            }
        };
        // One the principal may not list reads as missing (review B3).
        match found {
            Some(f) if self.function_listed(&f).await => Ok(f),
            _ => Err(not_found()),
        }
    }

    /// The stored function `last` (lower case) of the namespace `db`.
    fn stored_function(&self, db: &str, last: &str) -> Option<Function> {
        self.session
            .state
            .catalog
            .store
            .functions()
            .in_namespace(db)
            .into_iter()
            .find(|f| f.to_lowercase() == last)
            .map(|f| Function {
                name: f,
                catalog: Some(crate::catalog::CATALOG.to_owned()),
                namespace: Some(vec![db.to_owned()]),
                description: None,
                class_name: String::new(),
                is_temporary: false,
            })
    }

    /// Whether the function exists.
    ///
    /// # Errors
    /// The access error `SHOW FUNCTIONS` would give.
    pub async fn function_exists(&self, name: &str) -> Result<bool, KrishivError> {
        match self.get_function(name).await {
            Ok(_) => Ok(true),
            Err(e) if e.error_class().name() == "UNRESOLVED_ROUTINE" => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Drops the local temporary view; whether it existed.
    #[must_use = "whether the view existed"]
    pub fn drop_temp_view(&self, name: &str) -> bool {
        self.session.drop_temp_view(name)
    }

    /// Drops the global temporary view; whether it existed. Makes the access check of
    /// `DROP VIEW global_temp.name` (`Drop` on the view) first, so the API removes no more than the SQL does.
    ///
    /// # Errors
    /// The access error `DROP VIEW` would give.
    pub async fn drop_global_temp_view(&self, name: &str) -> Result<bool, KrishivError> {
        let catalog = &self.session.state.catalog;
        let key = global_key(name);
        if !catalog.exists(&key, self.case_sensitive()) {
            return Ok(false);
        }
        self.authorize_as(&format!(
            "DROP VIEW IF EXISTS {GLOBAL_NAMESPACE}.{}",
            quote(name)
        ))
        .await?;
        let dropped = catalog.drop_table(&key);
        self.session.state.cache.invalidate_table(name);
        Ok(dropped)
    }

    /// Checks that `name` can be cached (`CACHE TABLE name`): the access check of the statement, then the
    /// name is analysed, so a missing table is `TABLE_OR_VIEW_NOT_FOUND` here and not a failure that only a
    /// later query would meet. For a cache entry registered another way, such as at a storage level.
    ///
    /// # Errors
    /// The access error `CACHE TABLE` would give, `TABLE_OR_VIEW_NOT_FOUND`.
    pub async fn check_cache_target(&self, name: &str) -> Result<(), KrishivError> {
        self.authorize_as(&format!("CACHE TABLE {}", quoted(name)))
            .await?;
        self.session.table(name).schema_for_metadata().await?;
        Ok(())
    }

    /// `CACHE TABLE name`.
    ///
    /// # Errors
    /// The statement's.
    pub async fn cache_table(&self, name: &str) -> Result<(), KrishivError> {
        self.run(&format!("CACHE TABLE {}", quoted(name))).await
    }

    /// `spark.catalog.createTable(name, source, schema, description, options)`: creates a table of the data
    /// source `source` (`spark.sql.sources.default` when `None`). `schema` is a DDL column list
    /// (`"a INT, b STRING"`); without one a data source table takes its schema from its files. A `path`
    /// option makes the table external, at that location, as Spark's `CatalogImpl` does; the other options
    /// become the table's `OPTIONS`.
    ///
    /// # Errors
    /// The statement's: `TABLE_OR_VIEW_ALREADY_EXISTS`, `DATA_SOURCE_NOT_FOUND`, a parse error in `schema`.
    pub async fn create_table(
        &self,
        name: &str,
        source: Option<&str>,
        schema: Option<&str>,
        description: Option<&str>,
        options: &[(&str, &str)],
    ) -> Result<(), KrishivError> {
        let escaped = self.session.query_env().parser.escaped_string_literals;
        let mut sql = format!("CREATE TABLE {}", quoted(name));
        let given = schema.filter(|s| !s.trim().is_empty());
        if let Some(schema) = given {
            self.check_column_list(schema)?;
        }
        let location = options
            .iter()
            .rev()
            .find(|(k, _)| k.eq_ignore_ascii_case("path"))
            .map(|(_, v)| *v);
        // A data source table without a schema takes it from its files (Spark infers it when it creates the
        // table); `CREATE TABLE` here wants columns, so they are read from the files first.
        let default_source = crate::conf::get(
            &self.session.state.conf.snapshot(),
            "spark.sql.sources.default",
        );
        let inferred = match (given, source.or(default_source.as_deref()), location) {
            (None, Some(source), Some(location)) => {
                Some(self.infer_columns(source, location, options).await?)
            }
            _ => None,
        };
        if let Some(schema) = given.map(str::to_owned).or(inferred) {
            sql.push_str(&format!(" ({schema})"));
        }
        if let Some(source) = source {
            sql.push_str(&format!(" USING {}", quote(source)));
        }
        let (path, rest): (Vec<_>, Vec<_>) = options
            .iter()
            .partition(|(k, _)| k.eq_ignore_ascii_case("path"));
        if !rest.is_empty() {
            let list = rest
                .iter()
                .map(|(k, v)| Ok(format!("{} {}", quote(k), literal(v, escaped)?)))
                .collect::<Result<Vec<String>, KrishivError>>()?;
            sql.push_str(&format!(" OPTIONS ({})", list.join(", ")));
        }
        if let (Some((_, location)), true) = (path.last(), given.is_none()) {
            // Spark takes the partition columns of a table over files from their `key=value` directories.
            let keys = partition_keys(location);
            if !keys.is_empty() {
                let list: Vec<String> = keys.iter().map(|k| quote(k)).collect();
                sql.push_str(&format!(" PARTITIONED BY ({})", list.join(", ")));
            }
        }
        if let Some((_, location)) = path.last() {
            sql.push_str(&format!(" LOCATION {}", literal(location, escaped)?));
        }
        // Spark maps an empty description to no comment.
        if let Some(description) = description.filter(|d| !d.is_empty()) {
            sql.push_str(&format!(" COMMENT {}", literal(description, escaped)?));
        }
        self.run(&sql).await
    }

    /// Refuses `schema` unless it is only a column list: it is written into the `CREATE TABLE`, so text like
    /// `a INT) USING x LOCATION '/y' (b INT` must not become clauses of the statement.
    fn check_column_list(&self, schema: &str) -> Result<(), KrishivError> {
        let env = self.session.query_env();
        let statement =
            krishiv_sql::parse_statement(&format!("CREATE TABLE t ({schema})"), &env.parser)
                .map_err(|e| e.error)?;
        let krishiv_sql::ast::Statement::CreateTable(t) = statement else {
            return Err(unsupported_schema());
        };
        let only_columns = t.using.is_none()
            && t.options.is_empty()
            && t.partitioned_by.is_empty()
            && t.cluster_by.is_empty()
            && t.bucket.is_none()
            && t.skew.is_none()
            && t.row_format.is_none()
            && t.file_format.is_none()
            && t.location.is_none()
            && t.comment.is_none()
            && t.collation.is_none()
            && t.tblproperties.is_empty()
            && t.as_query.is_none();
        if only_columns {
            Ok(())
        } else {
            Err(unsupported_schema())
        }
    }

    /// The column list (`` `a` INT, `b` STRING ``) of the files at `location`, read as `source` with the table's
    /// `options` (a CSV `header` or `delimiter` changes the columns), as the table will read them.
    async fn infer_columns(
        &self,
        source: &str,
        location: &str,
        options: &[(&str, &str)],
    ) -> Result<String, KrishivError> {
        let mut reader = self.session.read().format(source);
        for (k, v) in options {
            if !k.eq_ignore_ascii_case("path") {
                reader = reader.option(k, v);
            }
        }
        let schema = reader
            .load([location.strip_prefix("file:").unwrap_or(location)])
            .schema()
            .await?;
        let mut columns = Vec::with_capacity(schema.fields().len());
        for field in schema.fields() {
            let ty = krishiv_types::DataType::from_arrow_field(field)?;
            // `sql()` quotes nested field names (`struct<`a b`:int>`); `catalog_string()` does not.
            columns.push(format!("{} {}", quote(field.name()), ty.sql()));
        }
        Ok(columns.join(", "))
    }

    /// Whether the table or view `name` is cached (`spark.catalog.isCached`).
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_NOT_FOUND` when `name` resolves to nothing.
    pub async fn is_cached(&self, name: &str) -> Result<bool, KrishivError> {
        self.session.table(name).is_cached().await
    }

    /// `UNCACHE TABLE name`.
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_NOT_FOUND`.
    pub async fn uncache_table(&self, name: &str) -> Result<(), KrishivError> {
        self.run(&format!("UNCACHE TABLE {}", quoted(name))).await
    }

    /// `CLEAR CACHE`.
    ///
    /// # Errors
    /// The access error of `CLEAR CACHE`, which drops the cached data of every session and so needs
    /// `Manage` on the cluster.
    pub async fn clear_cache(&self) -> Result<(), KrishivError> {
        self.authorize_as("CLEAR CACHE").await?;
        self.session.clear_cache();
        Ok(())
    }

    /// `REFRESH TABLE name`.
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_NOT_FOUND`.
    pub async fn refresh_table(&self, name: &str) -> Result<(), KrishivError> {
        self.run(&format!("REFRESH TABLE {}", quoted(name))).await
    }

    /// `REFRESH 'path'`: drops the cached relations reading files by that path.
    ///
    /// # Errors
    /// The statement's.
    pub async fn refresh_by_path(&self, path: &str) -> Result<(), KrishivError> {
        let escaped = self.session.query_env().parser.escaped_string_literals;
        self.run(&format!("REFRESH {}", literal(path, escaped)?))
            .await
    }

    /// `ALTER TABLE name RECOVER PARTITIONS` (`MSCK REPAIR TABLE`).
    ///
    /// # Errors
    /// The statement's.
    pub async fn recover_partitions(&self, name: &str) -> Result<(), KrishivError> {
        self.run(&format!("ALTER TABLE {} RECOVER PARTITIONS", quoted(name)))
            .await
    }
}

/// The type of a stored column as `DESCRIBE` prints it (`char(5)` for a `CHAR` column).
fn type_string(c: &crate::store::StoreColumn) -> String {
    c.char_varchar
        .clone()
        .unwrap_or_else(|| c.data_type.catalog_string())
}
