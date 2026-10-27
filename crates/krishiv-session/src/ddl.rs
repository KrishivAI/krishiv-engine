// SPDX-License-Identifier: Apache-2.0
//! `CREATE TABLE` (column list, CTAS, `LIKE`, `OR REPLACE`,
//! `IF NOT EXISTS`, external `LOCATION`), `DROP TABLE`,
//! `CREATE DATABASE|SCHEMA|NAMESPACE`, `DROP DATABASE|SCHEMA|NAMESPACE
//! [CASCADE]`, `USE` and `SHOW CURRENT NAMESPACE` as [`Session::sql`](crate::Session::sql)
//! runs them.
//!
//! Only the `parquet`, `csv`, `json` and `text` providers are supported;
//! with no `USING`, the provider is `spark.sql.sources.default` (`parquet`
//! unless configured). Anything else fails with `DATA_SOURCE_NOT_FOUND`.
//! `STORED AS` and `ROW FORMAT` fail with the error Spark raises without
//! Hive support, and `PARTITIONED BY`, `CLUSTER BY` and bucketing fail with
//! `KRISHIV_UNSUPPORTED_FEATURE`, which names partitioned tables. Table and
//! column constraints are accepted and ignored: they change nothing the
//! engine reads back.
//!
//! A table's rows always live in files: a read of one plans as an ordinary
//! read by path with the stored columns as its schema (see
//! [`CatalogTable`](crate::catalog::CatalogTable)), so analysis reads no
//! files. Managed locations come from the warehouse directory
//! (`spark.sql.warehouse.dir`, else the session's temporary directory);
//! dropping a managed table deletes its directory, while an external
//! table's directory is kept.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use krishiv_common::error::{
    ColumnAlreadyExists, DataSourceNotFound, InternalError, InvalidEmptyLocation, KrishivError,
    KrishivUnsupportedFeature, LocationAlreadyExists, NotSupportedCommandWithoutHiveSupport,
    RequiresSinglePartNamespace, SchemaAlreadyExists, SchemaNotFound, TableOrViewAlreadyExists,
    TableOrViewNotFound, UnsupportedFeatureSchemaLevelCollations,
};
use krishiv_common::redact::RedactedSql;
use krishiv_common::secret::OptionValue;
use krishiv_plan::analyzer::{TableColumn, TableIdent};
use krishiv_plan::{LogicalPlan, PlanNode, SourceId};
use krishiv_sql::ast::{
    ColumnName, CreateNamespace, CreateTable, CreateTableLike, DataTypeKind, DdlName, Describe,
    DescribeKind, DropBehavior, DropNamespace, DropTable, Expr, ExprKind, Ident, Literal,
    ObjectName, OptionKey, Show, ShowKind, Span, TableElement, TableOption, Use,
};
use krishiv_types::DataType;

use crate::catalog::{
    CATALOG, CatalogTable, CatalogTableKind, CreateMode, NAMESPACE, NamespaceMeta,
};
use crate::dataframe::DataFrame;
use crate::functions::lit;
use crate::session::{Session, WriteTarget, sql_context};

/// `KRISHIV_UNSUPPORTED_FEATURE` for a DDL shape this engine does not run.
fn unsupported(feature: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.to_owned(),
    }
    .into()
}

/// A name as Spark quotes it in errors: `` `name` ``.
fn quote(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// Name parts as Spark quotes a dotted name: `` `a`.`b` ``.
fn quoted(parts: &[&str]) -> String {
    parts.iter().map(|p| quote(p)).collect::<Vec<_>>().join(".")
}

/// A [`DdlName`] as written, quoted and dotted: what `TABLE_OR_VIEW_*`
/// errors report.
fn written_name(name: &DdlName, original: &str) -> String {
    if let DdlName::Parts(n) = name {
        quoted(&n.parts.iter().map(|p| p.value.as_str()).collect::<Vec<_>>())
    } else {
        original.to_owned()
    }
}

/// Spark's search path, as analysis reports it for a missing table in
/// `current`.
fn search_path(current: &str) -> String {
    format!("[`system`.`builtin`, `system`.`session`, `{CATALOG}`.`{current}`]")
}

/// `REQUIRES_SINGLE_PART_NAMESPACE` for `rest` (already stripped of the
/// catalog, when one was written).
fn requires_single_part(rest: &[&str]) -> KrishivError {
    RequiresSinglePartNamespace {
        session_catalog: CATALOG.to_owned(),
        identifier: quoted(rest),
    }
    .into()
}

/// The `(namespace, table)` a table [`DdlName`] names: one part reads the
/// current namespace, two parts name `ns.t`, three parts name
/// `spark_catalog.ns.t` (like the analyzer, any other shape fails with
/// `REQUIRES_SINGLE_PART_NAMESPACE`).
fn resolve_table_name(name: &DdlName, current: &str) -> Result<(String, String), KrishivError> {
    let DdlName::Parts(n) = name else {
        return Err(unsupported("dynamic names in DDL statements"));
    };
    let parts: Vec<&str> = n.parts.iter().map(|p| p.value.as_str()).collect();
    let cataloged = parts.len() > 1 && parts[0].eq_ignore_ascii_case(CATALOG);
    let rest = if cataloged { &parts[1..] } else { &parts[..] };
    match rest {
        [t] if !cataloged && !t.eq_ignore_ascii_case(CATALOG) => {
            Ok((current.to_owned(), (*t).to_owned()))
        }
        [ns, t] => Ok(((*ns).to_owned(), (*t).to_owned())),
        _ => Err(requires_single_part(if cataloged { rest } else { &parts })),
    }
}

/// The namespace a namespace [`DdlName`] names: one part, or two with the
/// session catalog first.
fn resolve_namespace_name(name: &DdlName) -> Result<String, KrishivError> {
    let DdlName::Parts(n) = name else {
        return Err(unsupported("dynamic names in DDL statements"));
    };
    let parts: Vec<&str> = n.parts.iter().map(|p| p.value.as_str()).collect();
    let cataloged = parts.len() > 1 && parts[0].eq_ignore_ascii_case(CATALOG);
    let rest = if cataloged { &parts[1..] } else { &parts[..] };
    match rest {
        [ns] => Ok((*ns).to_owned()),
        _ => Err(requires_single_part(if cataloged { rest } else { &parts })),
    }
}

/// The provider of a `CREATE TABLE`: `USING` as written, else
/// `spark.sql.sources.default` (`parquet` unless configured), lower-case.
/// Anything else fails with `DATA_SOURCE_NOT_FOUND`, as Spark does without
/// the connector (`delta`, `iceberg`) or for an unknown provider.
fn provider_name(
    using: Option<&krishiv_sql::ast::ObjectName>,
    default: Option<String>,
) -> Result<String, KrishivError> {
    let written = match using {
        Some(u) => u
            .parts
            .iter()
            .map(|p| p.value.clone())
            .collect::<Vec<_>>()
            .join("."),
        None => default.unwrap_or_else(|| "parquet".to_owned()),
    };
    if ["parquet", "csv", "json", "text"]
        .iter()
        .any(|f| f.eq_ignore_ascii_case(&written))
    {
        Ok(written.to_lowercase())
    } else {
        Err(DataSourceNotFound { provider: written }.into())
    }
}

/// The type of a `CREATE TABLE` column: `char`/`varchar` read and write as
/// strings (the engine has no padded character types yet), anything else
/// as [`DataTypeSpec`](krishiv_sql::ast::DataTypeSpec) denotes.
fn column_data_type(spec: &krishiv_sql::ast::DataTypeSpec) -> Result<DataType, KrishivError> {
    if let DataTypeKind::Primitive { name, .. } = &spec.kind
        && (name.eq_ignore_ascii_case("char")
            || name.eq_ignore_ascii_case("character")
            || name.eq_ignore_ascii_case("varchar"))
    {
        return Ok(DataType::String);
    }
    spec.to_data_type()
}

/// A declared column: its [`TableColumn`], its comment and its default as
/// SQL text.
struct DeclaredColumn {
    /// The column.
    column: TableColumn,
    /// Its comment, if written.
    comment: Option<String>,
    /// Its default, as SQL text, if written.
    default_sql: Option<String>,
}

/// The column definitions of a `CREATE TABLE`: constraints and generated
/// columns are accepted and ignored. A repeated name fails with
/// `COLUMN_ALREADY_EXISTS` (in lower case unless `case_sensitive`).
fn declared_columns(
    elements: &[TableElement],
    original: &str,
    case_sensitive: bool,
) -> Result<Vec<DeclaredColumn>, KrishivError> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for element in elements {
        let TableElement::Column(c) = element else {
            continue;
        };
        let ColumnName::Ident(i) = &c.name else {
            return Err(unsupported("dynamic column names in CREATE TABLE"));
        };
        let name = i.value.clone();
        let key = if case_sensitive {
            name.clone()
        } else {
            name.to_lowercase()
        };
        if !seen.insert(key.clone()) {
            return Err(ColumnAlreadyExists {
                column_name: format!("`{}`", key.replace('`', "``")),
            }
            .into());
        }
        out.push(DeclaredColumn {
            column: TableColumn {
                name,
                data_type: column_data_type(&c.data_type)?,
                nullable: !c.not_null,
            },
            comment: c.comment.clone(),
            default_sql: c
                .default
                .as_ref()
                .map(|e| original.get(e.span.range()).unwrap_or_default().to_owned()),
        });
    }
    Ok(out)
}

/// An option key: the dotted parts, or the string for a string key.
fn option_key(key: &OptionKey) -> String {
    if key.is_string {
        key.parts.first().map_or(String::new(), |p| p.value.clone())
    } else {
        key.parts
            .iter()
            .map(|p| p.value.clone())
            .collect::<Vec<_>>()
            .join(".")
    }
}

/// An option value as text: literals render as written; anything else keeps
/// its source text.
fn option_text(value: Option<&Expr>, original: &str) -> String {
    let Some(e) = value else {
        return String::new();
    };
    match &e.kind {
        ExprKind::Literal(Literal::String(s)) => s.clone(),
        ExprKind::Literal(Literal::Boolean(b)) => b.to_string(),
        ExprKind::Literal(Literal::Number { text, .. }) => text.clone(),
        ExprKind::Literal(Literal::Typed { value, .. }) => value.clone(),
        _ => original.get(e.span.range()).unwrap_or_default().to_owned(),
    }
}

/// The `OPTIONS` of a `CREATE TABLE`, typed as secrets where Spark redacts
/// them.
fn option_map(options: &[TableOption], original: &str) -> BTreeMap<String, OptionValue> {
    options
        .iter()
        .map(|o| {
            let key = option_key(&o.key);
            (
                key.clone(),
                OptionValue::from_user(&key, option_text(o.value.as_ref(), original)),
            )
        })
        .collect()
}

/// The `TBLPROPERTIES` (or `DBPROPERTIES`) of a statement.
fn properties_map(options: &[TableOption], original: &str) -> BTreeMap<String, String> {
    options
        .iter()
        .map(|o| (option_key(&o.key), option_text(o.value.as_ref(), original)))
        .collect()
}

/// `path` absolute: as is when absolute, else under `cwd`.
fn absolutize(path: &str, cwd: &str) -> String {
    let p = std::path::PathBuf::from(path);
    if p.is_absolute() {
        path.to_owned()
    } else {
        std::path::PathBuf::from(cwd)
            .join(p)
            .to_string_lossy()
            .into_owned()
    }
}

/// The managed location of `name` in `ns`: directly under the warehouse in
/// `default`, else under `<ns>.db`.
fn managed_location(warehouse: &std::path::Path, ns: &str, name: &str) -> String {
    let path = if ns.eq_ignore_ascii_case(NAMESPACE) {
        warehouse.join(name)
    } else {
        warehouse.join(format!("{ns}.db")).join(name)
    };
    path.to_string_lossy().into_owned()
}

/// Creates the directory `location` (and its parents).
fn ensure_dir(location: &str) -> Result<(), KrishivError> {
    std::fs::create_dir_all(location).map_err(|e| {
        KrishivError::from(InternalError {
            message: format!("cannot create the table directory {location}: {e}"),
        })
    })
}

/// Deletes the directory `location` when it exists.
fn remove_dir(location: &str) -> Result<(), KrishivError> {
    if std::fs::symlink_metadata(location).is_ok() {
        std::fs::remove_dir_all(location).map_err(|e| {
            KrishivError::from(InternalError {
                message: format!("cannot delete the table directory {location}: {e}"),
            })
        })?;
    }
    Ok(())
}

/// The current time, as seconds and nanos since the epoch.
fn now_string() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{}.{:09}", d.as_secs(), d.subsec_nanos()))
        .unwrap_or_else(|_| "0".to_owned())
}

impl Session {
    /// Runs `CREATE TABLE` (a column list or a CTAS): an existing table or
    /// temporary view fails with `TABLE_OR_VIEW_ALREADY_EXISTS`, unless
    /// `IF NOT EXISTS` (then nothing happens) or `OR REPLACE` (then it is
    /// replaced). A CTAS plans its query and writes it through the same
    /// path as [`DataFrameWriter::save`](crate::DataFrameWriter::save),
    /// inserting the metadata only after the write commits, so a failed
    /// write leaves no metadata and no files.
    pub(crate) async fn exec_create_table(
        &self,
        t: &CreateTable,
        original: &str,
        text: &RedactedSql,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let case_sensitive = env.analysis.conf.case_sensitive;
        if t.temporary {
            return Err(unsupported("temporary tables"));
        }
        let current = self.state.catalog.current_namespace();
        let (ns, name) = resolve_table_name(&t.name, &current)?;
        if !self.state.catalog.namespace_exists(&ns, case_sensitive) {
            return Err(SchemaNotFound {
                schema_name: quote(&ns),
            }
            .into());
        }
        if !t.partitioned_by.is_empty() || !t.cluster_by.is_empty() || t.bucket.is_some() {
            return Err(unsupported("partitioned tables"));
        }
        if t.file_format.is_some() || t.row_format.is_some() {
            return Err(NotSupportedCommandWithoutHiveSupport {
                cmd: if t.file_format.is_some() {
                    "STORED AS".to_owned()
                } else {
                    "ROW FORMAT".to_owned()
                },
            }
            .into());
        }
        if t.collation.is_some() {
            return Err(UnsupportedFeatureSchemaLevelCollations {}.into());
        }
        let default = crate::conf::get(&env.config, "spark.sql.sources.default");
        let provider = provider_name(t.using.as_ref(), default)?;
        // The parser rejects `OR REPLACE` together with `IF NOT EXISTS`.
        let mode = if t.or_replace {
            CreateMode::Replace
        } else if t.if_not_exists {
            CreateMode::IfNotExists
        } else {
            CreateMode::ErrorIfExists
        };
        let single = matches!(&t.name, DdlName::Parts(n) if n.parts.len() == 1);
        if single && self.state.catalog.exists(&name, case_sensitive) {
            match mode {
                CreateMode::Replace => {
                    self.state.catalog.drop_table(&name);
                }
                CreateMode::IfNotExists => return self.empty_frame().await,
                CreateMode::ErrorIfExists => {
                    return Err(TableOrViewAlreadyExists {
                        relation_name: written_name(&t.name, original),
                    }
                    .into());
                }
            }
        }
        let store_exists = self
            .state
            .catalog
            .get_table(&ns, &name, case_sensitive)
            .is_some();
        if store_exists && mode != CreateMode::Replace {
            if mode == CreateMode::IfNotExists {
                return self.empty_frame().await;
            }
            return Err(TableOrViewAlreadyExists {
                relation_name: written_name(&t.name, original),
            }
            .into());
        }
        let options = option_map(&t.options, original);
        let properties = properties_map(&t.tblproperties, original);
        let warehouse = self.state.catalog.warehouse_root(&env.config, self.cwd());
        let location = match &t.location {
            Some(l) if l.is_empty() => {
                return Err(InvalidEmptyLocation {
                    location: l.clone(),
                }
                .into());
            }
            Some(l) => absolutize(l, self.cwd()),
            None => managed_location(&warehouse, &ns, &name),
        };
        if let Some(q) = &t.as_query {
            return self
                .exec_ctas(
                    q,
                    text,
                    env,
                    CreateTableSpec {
                        ns,
                        name,
                        kind: if t.external {
                            CatalogTableKind::External
                        } else {
                            CatalogTableKind::Managed
                        },
                        provider,
                        location,
                        options,
                        properties,
                        comment: t.comment.clone(),
                        computed_location: t.location.is_none() && !t.external,
                        replace: t.or_replace,
                    },
                )
                .await;
        }
        let declared = declared_columns(&t.elements, original, case_sensitive)?;
        // The directory goes first, then the metadata: a failed create
        // leaves neither. Replacing reuses the directory.
        if !store_exists {
            if t.location.is_none() && !t.external && std::fs::symlink_metadata(&location).is_ok() {
                return Err(LocationAlreadyExists {
                    identifier: TableIdent {
                        catalog: CATALOG.to_owned(),
                        namespace: vec![ns.clone()],
                        name: name.clone(),
                    }
                    .to_string(),
                    location: location.clone(),
                }
                .into());
            }
            ensure_dir(&location)?;
        } else if t.location.is_some() || t.external {
            // `OR REPLACE` with an explicit `LOCATION`: the new directory
            // is created; the old one is left behind.
            ensure_dir(&location)?;
        }
        let table = CatalogTable {
            ident: TableIdent {
                catalog: CATALOG.to_owned(),
                namespace: vec![ns],
                name,
            },
            kind: if t.external {
                CatalogTableKind::External
            } else {
                CatalogTableKind::Managed
            },
            provider,
            location,
            columns: declared.iter().map(|d| d.column.clone()).collect(),
            col_comments: declared.iter().map(|d| d.comment.clone()).collect(),
            col_defaults: declared.iter().map(|d| d.default_sql.clone()).collect(),
            options,
            properties,
            comment: t.comment.clone(),
            created_at: now_string(),
            owner: self.owner_name().to_owned(),
        };
        // The existence checks above ordered the directory work; the store
        // re-checks under its lock, so a concurrent DDL still fails with
        // the right class. (`IF NOT EXISTS` lost a race reports nothing,
        // like the check above.)
        self.state
            .catalog
            .create_table(table, mode, case_sensitive)?;
        self.empty_frame().await
    }

    /// The `AS query` of a `CREATE TABLE`: plans `query`, writes it to
    /// `spec` (overwriting under `OR REPLACE`), and inserts the metadata
    /// only after the write commits.
    async fn exec_ctas(
        &self,
        query: &krishiv_sql::ast::Query,
        text: &RedactedSql,
        env: crate::session::QueryEnv,
        spec: CreateTableSpec,
    ) -> Result<DataFrame, KrishivError> {
        if !spec.replace
            && spec.computed_location
            && std::fs::symlink_metadata(&spec.location).is_ok()
        {
            return Err(LocationAlreadyExists {
                identifier: TableIdent {
                    catalog: CATALOG.to_owned(),
                    namespace: vec![spec.ns.clone()],
                    name: spec.name.clone(),
                }
                .to_string(),
                location: spec.location.clone(),
            }
            .into());
        }
        let plan = krishiv_sql::lower(query, SourceId(0))
            .map_err(|e| sql_context(e.error, text, e.span.start, e.span.end))?;
        let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
        let analysis = df.analyzed_in_with_text(env, Some(text)).await?;
        let (_, physical) = self.physical(&analysis.analyzed, &analysis.env).await?;
        self.write_files(
            physical,
            &analysis.env,
            WriteTarget {
                format: spec.provider.clone(),
                mode: if spec.replace {
                    "overwrite".to_owned()
                } else {
                    "error".to_owned()
                },
                path: OptionValue::from_user("path", &spec.location),
                options: spec.options.clone(),
                partition_by: Vec::new(),
            },
        )
        .await?;
        let columns: Vec<TableColumn> = analysis
            .analyzed
            .output
            .iter()
            .map(|c| TableColumn {
                name: c.name.clone(),
                data_type: c.data_type.clone(),
                nullable: c.nullable,
            })
            .collect();
        // One comment and default slot per column, as for declared
        // columns; a CTAS has neither.
        let width = columns.len();
        let table = CatalogTable {
            ident: TableIdent {
                catalog: CATALOG.to_owned(),
                namespace: vec![spec.ns],
                name: spec.name,
            },
            kind: spec.kind,
            provider: spec.provider,
            location: spec.location,
            columns,
            col_comments: vec![None; width],
            col_defaults: vec![None; width],
            options: spec.options,
            properties: spec.properties,
            comment: spec.comment,
            created_at: now_string(),
            owner: self.owner_name().to_owned(),
        };
        let case_sensitive = analysis.env.analysis.conf.case_sensitive;
        self.state
            .catalog
            .create_table(table, CreateMode::Replace, case_sensitive)?;
        self.empty_frame().await
    }

    /// Runs `CREATE TABLE ... LIKE`: an empty table copying the source's
    /// columns, provider and options (`USING` overrides the provider).
    pub(crate) async fn exec_create_table_like(
        &self,
        t: &CreateTableLike,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let case_sensitive = env.analysis.conf.case_sensitive;
        let current = self.state.catalog.current_namespace();
        let (ns, name) = resolve_table_name(&t.name, &current)?;
        if !self.state.catalog.namespace_exists(&ns, case_sensitive) {
            return Err(SchemaNotFound {
                schema_name: quote(&ns),
            }
            .into());
        }
        if t.file_format.is_some() || t.row_format.is_some() {
            return Err(NotSupportedCommandWithoutHiveSupport {
                cmd: if t.file_format.is_some() {
                    "STORED AS".to_owned()
                } else {
                    "ROW FORMAT".to_owned()
                },
            }
            .into());
        }
        let single = matches!(&t.name, DdlName::Parts(n) if n.parts.len() == 1);
        let mode = if t.if_not_exists {
            CreateMode::IfNotExists
        } else {
            CreateMode::ErrorIfExists
        };
        if single && self.state.catalog.exists(&name, case_sensitive) {
            if mode == CreateMode::IfNotExists {
                return self.empty_frame().await;
            }
            return Err(TableOrViewAlreadyExists {
                relation_name: written_name(&t.name, original),
            }
            .into());
        }
        let store_exists = self
            .state
            .catalog
            .get_table(&ns, &name, case_sensitive)
            .is_some();
        if store_exists {
            if mode == CreateMode::IfNotExists {
                return self.empty_frame().await;
            }
            return Err(TableOrViewAlreadyExists {
                relation_name: written_name(&t.name, original),
            }
            .into());
        }
        let (sns, sname) = resolve_table_name(&t.source, &current)?;
        let Some(source) = self.state.catalog.get_table(&sns, &sname, case_sensitive) else {
            return Err(TableOrViewNotFound {
                relation_name: written_name(&t.source, original),
                search_path: search_path(&current),
            }
            .into());
        };
        let provider = match &t.using {
            Some(u) => provider_name(Some(u), None)?,
            None => source.provider.clone(),
        };
        let warehouse = self.state.catalog.warehouse_root(&env.config, self.cwd());
        let location = match &t.location {
            Some(l) if l.is_empty() => {
                return Err(InvalidEmptyLocation {
                    location: l.clone(),
                }
                .into());
            }
            Some(l) => absolutize(l, self.cwd()),
            None => managed_location(&warehouse, &ns, &name),
        };
        if t.location.is_none() && std::fs::symlink_metadata(&location).is_ok() {
            return Err(LocationAlreadyExists {
                identifier: TableIdent {
                    catalog: CATALOG.to_owned(),
                    namespace: vec![ns.clone()],
                    name: name.clone(),
                }
                .to_string(),
                location: location.clone(),
            }
            .into());
        }
        ensure_dir(&location)?;
        let mut properties = source.properties.clone();
        properties.extend(properties_map(&t.tblproperties, original));
        let table = CatalogTable {
            ident: TableIdent {
                catalog: CATALOG.to_owned(),
                namespace: vec![ns],
                name,
            },
            kind: CatalogTableKind::Managed,
            provider,
            location,
            columns: source.columns.clone(),
            col_comments: source.col_comments.clone(),
            col_defaults: source.col_defaults.clone(),
            options: source.options.clone(),
            properties,
            comment: None,
            created_at: now_string(),
            owner: self.owner_name().to_owned(),
        };
        self.state
            .catalog
            .create_table(table, mode, case_sensitive)?;
        self.empty_frame().await
    }

    /// Runs `DROP TABLE [IF EXISTS] name [PURGE]` (`PURGE` is accepted: the
    /// engine keeps no trash). The metadata goes first, then a managed
    /// table's directory; an external table's directory is kept. A failed
    /// directory delete is reported, with the metadata already gone.
    pub(crate) async fn exec_drop_table(&self, t: &DropTable) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let case_sensitive = env.analysis.conf.case_sensitive;
        let current = self.state.catalog.current_namespace();
        let (ns, name) = resolve_table_name(&t.name, &current)?;
        let original = t.name.span().text("");
        let Some(table) = self
            .state
            .catalog
            .drop_store_table(&ns, &name, case_sensitive)
        else {
            if t.if_exists {
                return self.empty_frame().await;
            }
            return Err(TableOrViewNotFound {
                relation_name: written_name(&t.name, original),
                search_path: search_path(&current),
            }
            .into());
        };
        if table.kind == CatalogTableKind::Managed {
            remove_dir(&table.location)?;
        }
        self.empty_frame().await
    }

    /// Runs `CREATE DATABASE|SCHEMA|NAMESPACE [IF NOT EXISTS]`: the
    /// namespace and its directory. The `default` namespace always exists,
    /// so creating it fails with `SCHEMA_ALREADY_EXISTS`.
    pub(crate) async fn exec_create_namespace(
        &self,
        n: &CreateNamespace,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let case_sensitive = env.analysis.conf.case_sensitive;
        let name = resolve_namespace_name(&n.name)?;
        if n.collation.is_some() {
            return Err(UnsupportedFeatureSchemaLevelCollations {}.into());
        }
        // The `default` namespace always exists: creating it fails here,
        // before any directory is made.
        if self.state.catalog.namespace_exists(&name, case_sensitive) {
            if n.if_not_exists {
                return self.empty_frame().await;
            }
            return Err(SchemaAlreadyExists {
                schema_name: quote(&name),
            }
            .into());
        }
        let warehouse = self.state.catalog.warehouse_root(&env.config, self.cwd());
        let location = match &n.location {
            Some(l) if l.is_empty() => {
                return Err(InvalidEmptyLocation {
                    location: l.clone(),
                }
                .into());
            }
            Some(l) => absolutize(l, self.cwd()),
            None => warehouse
                .join(format!("{name}.db"))
                .to_string_lossy()
                .into_owned(),
        };
        ensure_dir(&location)?;
        self.state.catalog.create_namespace(
            NamespaceMeta {
                name,
                comment: n.comment.clone(),
                location,
                properties: properties_map(&n.properties, ""),
            },
            n.if_not_exists,
            case_sensitive,
        )?;
        self.empty_frame().await
    }

    /// Runs `DROP DATABASE|SCHEMA|NAMESPACE [IF EXISTS] [RESTRICT|CASCADE]`:
    /// without `CASCADE`, a non-empty namespace fails with
    /// `SCHEMA_NOT_EMPTY`; with it, the namespace and its tables' metadata
    /// go, then the managed tables' data. The `default` namespace cannot
    /// be dropped.
    pub(crate) async fn exec_drop_namespace(
        &self,
        n: &DropNamespace,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let case_sensitive = env.analysis.conf.case_sensitive;
        let name = resolve_namespace_name(&n.name)?;
        let cascade = matches!(n.behavior, Some(DropBehavior::Cascade));
        let ns_location = self.state.catalog.namespace_location(&name, case_sensitive);
        let dropped =
            self.state
                .catalog
                .drop_namespace(&name, n.if_exists, cascade, case_sensitive)?;
        let mut first_error = None;
        for table in &dropped {
            if table.kind == CatalogTableKind::Managed
                && let Err(e) = remove_dir(&table.location)
            {
                first_error.get_or_insert(e);
            }
        }
        // The namespace directory is bookkeeping: removing it is
        // best-effort, after the tables' data is gone.
        if let Some(location) = ns_location {
            drop(std::fs::remove_dir_all(location));
        }
        if let Some(e) = first_error {
            return Err(e);
        }
        self.empty_frame().await
    }

    /// Runs `USE name`: the current namespace of the session. A missing
    /// namespace fails with `SCHEMA_NOT_FOUND`.
    pub(crate) async fn exec_use(&self, u: &Use) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let case_sensitive = env.analysis.conf.case_sensitive;
        let name = resolve_namespace_name(&u.name)?;
        if !self.state.catalog.namespace_exists(&name, case_sensitive) {
            return Err(SchemaNotFound {
                schema_name: quote(&name),
            }
            .into());
        }
        self.state.catalog.set_current_namespace(&name);
        self.empty_frame().await
    }

    /// Runs `SHOW CURRENT NAMESPACE`: the one row of `catalog` and
    /// `namespace`, as in Spark.
    pub(crate) async fn exec_show(&self, s: &Show) -> Result<DataFrame, KrishivError> {
        let ShowKind::CurrentNamespace { .. } = &s.kind else {
            return Err(KrishivUnsupportedFeature {
                feature: "statements other than queries in sessions".to_owned(),
            }
            .into());
        };
        let env = self.query_env();
        let ns = self.state.catalog.current_namespace();
        let df = DataFrame::new(
            self.clone(),
            Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
        )
        .select([
            lit("spark_catalog").alias("catalog"),
            lit(ns.as_str()).alias("namespace"),
        ]);
        df.analyzed_in(env).await?;
        Ok(df)
    }

    /// Runs `DESCRIBE TABLE name`: the stored columns (`col_name`,
    /// `data_type`, `comment`), in order, of a temporary entry or a
    /// file-backed table. Types render as [`DataType::simple_string`], as
    /// Spark's `DESCRIBE` shows them; a column without a comment shows an
    /// empty string. Anything beyond the plain table form (`EXTENDED`,
    /// `FORMATTED`, `PARTITION`, a column, `AS JSON`, functions,
    /// namespaces and queries) fails with `KRISHIV_UNSUPPORTED_FEATURE`.
    /// (No golden covers the output shape or nullability; both follow the
    /// `SET` command's string frames.)
    pub(crate) async fn exec_describe_table(
        &self,
        d: &Describe,
    ) -> Result<DataFrame, KrishivError> {
        let DescribeKind::Table {
            mode,
            table,
            partition,
            column,
            as_json,
            ..
        } = &d.kind
        else {
            return Err(unsupported(
                "DESCRIBE FUNCTION, DESCRIBE NAMESPACE and DESCRIBE QUERY",
            ));
        };
        if mode.is_some() {
            return Err(unsupported(
                "DESCRIBE TABLE EXTENDED and DESCRIBE TABLE FORMATTED",
            ));
        }
        if partition.is_some() {
            return Err(unsupported("DESCRIBE TABLE PARTITION"));
        }
        if !column.is_empty() {
            return Err(unsupported("DESCRIBE TABLE column"));
        }
        if *as_json {
            return Err(unsupported("DESCRIBE TABLE AS JSON"));
        }
        let env = self.query_env();
        let case_sensitive = env.analysis.conf.case_sensitive;
        let current = self.state.catalog.current_namespace();
        let (ns, name) = resolve_table_name(table, &current)?;
        let original = table.span().text("");
        // Temporary entries resolve for unqualified names only, as reads do.
        let single = matches!(table, DdlName::Parts(n) if n.parts.len() == 1);
        if single && let Some(entry) = self.state.catalog.get(&name, case_sensitive) {
            let (columns, comments) = match &*entry {
                crate::catalog::TempTable::Memory { columns, .. }
                | crate::catalog::TempTable::Files { columns, .. }
                | crate::catalog::TempTable::View { columns, .. } => {
                    (columns.clone(), vec![None; columns.len()])
                }
            };
            return self.describe_frame(&columns, &comments).await;
        }
        if !self.state.catalog.namespace_exists(&ns, case_sensitive) {
            return Err(SchemaNotFound {
                schema_name: quote(&ns),
            }
            .into());
        }
        let Some(found) = self.state.catalog.get_table(&ns, &name, case_sensitive) else {
            return Err(TableOrViewNotFound {
                relation_name: written_name(table, original),
                search_path: search_path(&current),
            }
            .into());
        };
        self.describe_frame(&found.columns, &found.col_comments)
            .await
    }

    /// A frame of (`col_name`, `data_type`, `comment`) string rows for
    /// [`Self::exec_describe_table`], in column order.
    async fn describe_frame(
        &self,
        columns: &[TableColumn],
        comments: &[Option<String>],
    ) -> Result<DataFrame, KrishivError> {
        let one = || {
            DataFrame::new(
                self.clone(),
                Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
            )
        };
        let row = |name: &str, data_type: &str, comment: &str| {
            one().select([
                lit(name).alias("col_name"),
                lit(data_type).alias("data_type"),
                lit(comment).alias("comment"),
            ])
        };
        let df = if columns.is_empty() {
            row("", "", "").limit(0)
        } else if columns.len() == 1 {
            let data_type = columns[0].data_type.simple_string();
            let comment = comments.first().and_then(Option::as_deref).unwrap_or("");
            row(&columns[0].name, &data_type, comment)
        } else {
            let inputs: Vec<Arc<LogicalPlan>> = columns
                .iter()
                .zip(comments.iter().chain(std::iter::repeat(&None)))
                .map(|(c, comment)| {
                    Arc::clone(
                        &row(
                            &c.name,
                            &c.data_type.simple_string(),
                            comment.as_deref().unwrap_or(""),
                        )
                        .plan,
                    )
                })
                .collect();
            DataFrame::new(
                self.clone(),
                Arc::new(LogicalPlan::new(PlanNode::Union {
                    inputs,
                    by_name: false,
                    allow_missing_columns: false,
                })),
            )
        };
        let env = self.query_env();
        df.analyzed_in(env).await?;
        Ok(df)
    }

    /// Writes `df` as the table `spec.table`
    /// ([`DataFrameWriterV2`](crate::DataFrameWriterV2)): the same write
    /// and metadata path as `CREATE TABLE ... AS SELECT` ([`Self::exec_ctas`]),
    /// so `writeTo("t").create()` and `CREATE TABLE t AS SELECT ...` run one
    /// code path. `Create` fails on an existing table with
    /// `TABLE_OR_VIEW_ALREADY_EXISTS`, `Replace` fails on a missing one with
    /// `TABLE_OR_VIEW_NOT_FOUND`, and `CreateOrReplace` does either. The
    /// metadata is inserted only after the write commits, so a failed write
    /// leaves no table and no files. Partitioning and clustering fail with
    /// `KRISHIV_UNSUPPORTED_FEATURE`, as in [`Self::exec_create_table`];
    /// `INSERT` (append and overwrites) never reaches here: those methods
    /// fail before any command runs.
    pub(crate) async fn exec_write_v2(
        &self,
        df: &DataFrame,
        spec: V2WriteSpec,
        mode: V2Mode,
    ) -> Result<(), KrishivError> {
        let env = self.query_env();
        let case_sensitive = env.analysis.conf.case_sensitive;
        let parsed = krishiv_sql::parse_multipart_identifier(&spec.table, &self.parser_options())
            .map_err(|e| e.error)?;
        let written: Vec<String> = parsed.parts.iter().map(|p| p.value.clone()).collect();
        let ddl_name = DdlName::Parts(ObjectName {
            parts: parsed.parts,
            span: Span::default(),
        });
        let current = self.state.catalog.current_namespace();
        let (ns, name) = resolve_table_name(&ddl_name, &current)?;
        if !self.state.catalog.namespace_exists(&ns, case_sensitive) {
            return Err(SchemaNotFound {
                schema_name: quote(&ns),
            }
            .into());
        }
        if !spec.partitioned_by.is_empty() || !spec.cluster_by.is_empty() {
            return Err(unsupported("partitioned tables"));
        }
        let default = crate::conf::get(&env.config, "spark.sql.sources.default");
        let provider = match spec.provider {
            Some(p) => {
                let using = ObjectName {
                    parts: vec![Ident {
                        value: p,
                        quoted: false,
                        span: Span::default(),
                    }],
                    span: Span::default(),
                };
                provider_name(Some(&using), None)?
            }
            None => provider_name(None, default)?,
        };
        let refs: Vec<&str> = written.iter().map(String::as_str).collect();
        let single = matches!(&ddl_name, DdlName::Parts(n) if n.parts.len() == 1);
        if single && self.state.catalog.exists(&name, case_sensitive) {
            match mode {
                V2Mode::Create => {
                    return Err(TableOrViewAlreadyExists {
                        relation_name: quoted(&refs),
                    }
                    .into());
                }
                V2Mode::Replace | V2Mode::CreateOrReplace => {
                    self.state.catalog.drop_table(&name);
                }
            }
        }
        let store_exists = self
            .state
            .catalog
            .get_table(&ns, &name, case_sensitive)
            .is_some();
        match mode {
            V2Mode::Create if store_exists => {
                return Err(TableOrViewAlreadyExists {
                    relation_name: quoted(&refs),
                }
                .into());
            }
            V2Mode::Replace if !store_exists => {
                return Err(TableOrViewNotFound {
                    relation_name: quoted(&refs),
                    search_path: search_path(&current),
                }
                .into());
            }
            V2Mode::Create | V2Mode::Replace | V2Mode::CreateOrReplace => {}
        }
        let warehouse = self.state.catalog.warehouse_root(&env.config, self.cwd());
        let location = managed_location(&warehouse, &ns, &name);
        if matches!(mode, V2Mode::Create) && std::fs::symlink_metadata(&location).is_ok() {
            return Err(LocationAlreadyExists {
                identifier: TableIdent {
                    catalog: CATALOG.to_owned(),
                    namespace: vec![ns.clone()],
                    name: name.clone(),
                }
                .to_string(),
                location: location.clone(),
            }
            .into());
        }
        let analysis = df.analyzed_in(env).await?;
        let (_, physical) = self.physical(&analysis.analyzed, &analysis.env).await?;
        self.write_files(
            physical,
            &analysis.env,
            WriteTarget {
                format: provider.clone(),
                mode: if matches!(mode, V2Mode::Create) {
                    "error".to_owned()
                } else {
                    "overwrite".to_owned()
                },
                path: OptionValue::from_user("path", &location),
                options: spec.options.clone(),
                partition_by: Vec::new(),
            },
        )
        .await?;
        let columns: Vec<TableColumn> = analysis
            .analyzed
            .output
            .iter()
            .map(|c| TableColumn {
                name: c.name.clone(),
                data_type: c.data_type.clone(),
                nullable: c.nullable,
            })
            .collect();
        let width = columns.len();
        let table = CatalogTable {
            ident: TableIdent {
                catalog: CATALOG.to_owned(),
                namespace: vec![ns],
                name,
            },
            kind: CatalogTableKind::Managed,
            provider,
            location,
            columns,
            col_comments: vec![None; width],
            col_defaults: vec![None; width],
            options: spec.options,
            properties: spec.properties,
            comment: None,
            created_at: now_string(),
            owner: self.owner_name().to_owned(),
        };
        let case_sensitive = analysis.env.analysis.conf.case_sensitive;
        self.state
            .catalog
            .create_table(table, CreateMode::Replace, case_sensitive)?;
        Ok(())
    }
}

/// The metadata of a CTAS target, before its columns are known.
struct CreateTableSpec {
    /// The namespace, as written.
    ns: String,
    /// The name, as written.
    name: String,
    /// Managed or external.
    kind: CatalogTableKind,
    /// The file format, lower-case.
    provider: String,
    /// The table directory, absolute.
    location: String,
    /// The `OPTIONS`.
    options: BTreeMap<String, OptionValue>,
    /// The `TBLPROPERTIES`.
    properties: BTreeMap<String, String>,
    /// The `COMMENT`, if written.
    comment: Option<String>,
    /// Whether `location` was computed from the warehouse (and a leftover
    /// there fails with `LOCATION_ALREADY_EXISTS`).
    computed_location: bool,
    /// Whether `OR REPLACE` was written (the write overwrites).
    replace: bool,
}

/// What [`Session::exec_write_v2`] writes: the table name as written and
/// the builder state of [`DataFrameWriterV2`](crate::DataFrameWriterV2).
/// Tables are always managed (the writer has no `LOCATION`); options are
/// typed as secrets where Spark redacts them, as in `CREATE TABLE`.
pub(crate) struct V2WriteSpec {
    /// The table name as written (`t`, `db.t` or `spark_catalog.db.t`).
    pub table: String,
    /// `using`: `None` for `spark.sql.sources.default`.
    pub provider: Option<String>,
    /// `option`/`options`: the table's options and its read options.
    pub options: BTreeMap<String, OptionValue>,
    /// `tableProperty`: the `TBLPROPERTIES`.
    pub properties: BTreeMap<String, String>,
    /// `partitionedBy` columns.
    pub partitioned_by: Vec<String>,
    /// `clusterBy` columns.
    pub cluster_by: Vec<String>,
}

/// How [`Session::exec_write_v2`] treats an existing table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum V2Mode {
    /// Fail with `TABLE_OR_VIEW_ALREADY_EXISTS`.
    Create,
    /// Fail with `TABLE_OR_VIEW_NOT_FOUND` when there is no such table.
    Replace,
    /// Either.
    CreateOrReplace,
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use krishiv_common::error::ErrorClass;

    use super::*;
    use crate::session::Session;

    /// A session.
    fn session() -> Session {
        Session::new().unwrap()
    }

    /// Runs `sql`, unwrapping success.
    fn run(s: &Session, sql: &str) {
        block_on(s.sql(sql)).unwrap();
    }

    /// The error class of `sql`, which must fail.
    fn class(s: &Session, sql: &str) -> ErrorClass {
        block_on(s.sql(sql)).unwrap_err().error_class()
    }

    /// The schema of `sql`: names and nullability.
    fn schema_of(s: &Session, sql: &str) -> Vec<(String, bool)> {
        let df = block_on(s.sql(sql)).unwrap();
        let schema = block_on(df.schema()).unwrap();
        schema
            .fields()
            .iter()
            .map(|f| (f.name().clone(), f.is_nullable()))
            .collect()
    }

    /// The rows of `sql`: every cell as text (`NULL` for null).
    fn rows(s: &Session, sql: &str) -> Vec<Vec<String>> {
        use arrow_array::cast::AsArray as _;
        use arrow_schema::DataType as ArrowType;

        let df = block_on(s.sql(sql)).unwrap();
        let batches = block_on(df.collect()).unwrap();
        let mut out = Vec::new();
        for batch in &batches {
            for row in 0..batch.num_rows() {
                let mut rendered = Vec::new();
                for column in batch.columns() {
                    if column.is_null(row) {
                        rendered.push("NULL".to_owned());
                        continue;
                    }
                    match column.data_type() {
                        ArrowType::Int32 => rendered.push(
                            column
                                .as_primitive::<arrow_array::types::Int32Type>()
                                .value(row)
                                .to_string(),
                        ),
                        ArrowType::Int64 => rendered.push(
                            column
                                .as_primitive::<arrow_array::types::Int64Type>()
                                .value(row)
                                .to_string(),
                        ),
                        ArrowType::Utf8View => {
                            rendered.push(column.as_string_view().value(row).to_owned());
                        }
                        ArrowType::Utf8 => {
                            rendered.push(column.as_string::<i32>().value(row).to_owned());
                        }
                        ArrowType::LargeUtf8 => {
                            rendered.push(column.as_string::<i64>().value(row).to_owned());
                        }
                        other => panic!("unexpected type {other}"),
                    }
                }
                out.push(rendered);
            }
        }
        out
    }

    /// The one row of `SHOW CURRENT NAMESPACE`.
    fn current_namespace_row(s: &Session) -> Vec<String> {
        let df = block_on(s.sql("SHOW CURRENT NAMESPACE")).unwrap();
        let schema = block_on(df.schema()).unwrap();
        assert_eq!(
            schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect::<Vec<_>>(),
            ["catalog", "namespace"]
        );
        let batches = block_on(df.collect()).unwrap();
        assert_eq!(
            batches
                .iter()
                .map(arrow_array::RecordBatch::num_rows)
                .sum::<usize>(),
            1
        );
        rows(s, "SHOW CURRENT NAMESPACE").pop().unwrap()
    }

    #[test]
    fn create_and_use_namespace() {
        let s = session();
        assert_eq!(current_namespace_row(&s), ["spark_catalog", "default"]);
        run(&s, "CREATE DATABASE mydb1");
        assert_eq!(
            class(&s, "CREATE DATABASE mydb1"),
            ErrorClass::SchemaAlreadyExists
        );
        run(&s, "CREATE DATABASE IF NOT EXISTS mydb1");
        run(&s, "USE mydb1");
        assert_eq!(current_namespace_row(&s), ["spark_catalog", "mydb1"]);
        run(&s, "USE spark_catalog.default");
        assert_eq!(current_namespace_row(&s), ["spark_catalog", "default"]);
        run(&s, "USE mydb1");
        assert_eq!(class(&s, "USE missing"), ErrorClass::SchemaNotFound);
        run(&s, "USE default");
        assert_eq!(current_namespace_row(&s), ["spark_catalog", "default"]);
    }

    #[test]
    fn create_table_with_columns_reads_empty() {
        let s = session();
        run(
            &s,
            "CREATE TABLE t(a INT NOT NULL COMMENT 'id', b STRING DEFAULT 'x') USING parquet",
        );
        assert_eq!(
            schema_of(&s, "SELECT * FROM t"),
            [("a".to_owned(), false), ("b".to_owned(), true)]
        );
        assert!(rows(&s, "SELECT * FROM t").is_empty());
        let table = s.state.catalog.get_table("default", "t", false).unwrap();
        assert_eq!(table.provider, "parquet");
        assert_eq!(table.col_comments, [Some("id".to_owned()), None]);
        assert_eq!(table.col_defaults, [None, Some("'x'".to_owned())]);
        assert_eq!(table.owner, s.owner_name());
        assert!(!table.created_at.is_empty());
        run(&s, "DROP TABLE t");
        assert_eq!(
            class(&s, "SELECT * FROM t"),
            ErrorClass::TableOrViewNotFound
        );
        assert_eq!(class(&s, "DROP TABLE t"), ErrorClass::TableOrViewNotFound);
        run(&s, "DROP TABLE IF EXISTS t");
    }

    #[test]
    fn temp_views_shadow_store_tables_for_single_part_names() {
        let s = session();
        // A memory table and a temporary view over it.
        let schema =
            std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                "a",
                arrow_schema::DataType::Int32,
                false,
            )]));
        let batch = arrow_array::RecordBatch::try_new(
            std::sync::Arc::clone(&schema),
            vec![std::sync::Arc::new(arrow_array::Int32Array::from(vec![1]))],
        )
        .unwrap();
        s.register_batches("t", schema, vec![batch]).unwrap();
        block_on(s.table("t").create_or_replace_temp_view("v")).unwrap();
        // A store table of the same name: creating it through the
        // qualified name does not see the temporary view.
        run(&s, "CREATE TABLE default.v(a INT) USING parquet");
        // Single-part names resolve the temporary view first ...
        assert_eq!(rows(&s, "SELECT * FROM v"), [["1"]]);
        // ... while qualified names go to the store only.
        assert!(rows(&s, "SELECT * FROM default.v").is_empty());
        assert!(rows(&s, "SELECT * FROM spark_catalog.default.v").is_empty());
        // And creating through the single-part name sees the view.
        assert_eq!(
            class(&s, "CREATE TABLE v(a INT) USING parquet"),
            ErrorClass::TableOrViewAlreadyExists
        );
        run(&s, "CREATE TABLE IF NOT EXISTS v(a INT) USING parquet");
        run(&s, "DROP TABLE default.v PURGE");
    }

    #[test]
    fn ctas_csv_roundtrip() {
        let s = session();
        run(&s, "CREATE TABLE t USING csv AS SELECT 1 AS a, 'x' AS b");
        // Literals are non-nullable, as the query reports them.
        assert_eq!(
            schema_of(&s, "SELECT * FROM t"),
            [("a".to_owned(), false), ("b".to_owned(), false)]
        );
        assert_eq!(rows(&s, "SELECT * FROM t"), [["1", "x"]]);
        // CTAS to an existing location fails without touching the table.
        let warehouse = s
            .state
            .catalog
            .warehouse_root(&s.state.conf.snapshot(), s.cwd());
        std::fs::create_dir_all(warehouse.join("fresh")).unwrap();
        assert_eq!(
            class(
                &s,
                "CREATE TABLE fresh USING csv AS SELECT 2 AS a, 'y' AS b"
            ),
            ErrorClass::LocationAlreadyExists
        );
        assert!(
            s.state
                .catalog
                .get_table("default", "fresh", false)
                .is_none()
        );
        assert_eq!(rows(&s, "SELECT * FROM t"), [["1", "x"]]);
        run(&s, "DROP TABLE t");
    }

    #[test]
    fn ctas_or_replace_overwrites() {
        let s = session();
        run(&s, "CREATE TABLE t USING parquet AS SELECT 1 AS a");
        assert_eq!(rows(&s, "SELECT * FROM t"), [["1"]]);
        run(
            &s,
            "CREATE OR REPLACE TABLE t USING parquet AS SELECT 2 AS a",
        );
        assert_eq!(rows(&s, "SELECT * FROM t"), [["2"]]);
        run(&s, "DROP TABLE t");
    }

    #[test]
    fn failed_ctas_write_leaves_no_metadata_and_no_files() {
        let s = session();
        // Two columns cannot be written as `text`.
        let e = block_on(s.sql("CREATE TABLE t USING text AS SELECT 1 AS a, 2 AS b")).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::LegacyErrorTemp1290);
        assert!(s.state.catalog.get_table("default", "t", false).is_none());
        let warehouse = s
            .state
            .catalog
            .warehouse_root(&s.state.conf.snapshot(), s.cwd());
        assert!(!warehouse.join("t").exists());
        // `text` with one string column works.
        run(&s, "CREATE TABLE u(value STRING) USING text");
        assert!(rows(&s, "SELECT * FROM u").is_empty());
        run(&s, "DROP TABLE u");
    }

    #[test]
    fn dropped_table_metadata_is_gone_when_its_delete_fails() {
        let s = session();
        run(&s, "CREATE TABLE t(a INT) USING parquet");
        let location = s
            .state
            .catalog
            .get_table("default", "t", false)
            .unwrap()
            .location;
        // A file where the directory was makes the delete fail.
        std::fs::remove_dir_all(&location).unwrap();
        std::fs::write(&location, b"x").unwrap();
        assert!(block_on(s.sql("DROP TABLE t")).is_err());
        assert!(s.state.catalog.get_table("default", "t", false).is_none());
        std::fs::remove_file(&location).unwrap();
    }

    #[test]
    fn if_not_exists_or_replace_table() {
        let s = session();
        run(&s, "CREATE TABLE t(a INT) USING parquet");
        run(&s, "CREATE TABLE IF NOT EXISTS t(b STRING) USING parquet");
        assert_eq!(schema_of(&s, "SELECT * FROM t"), [("a".to_owned(), true)]);
        run(&s, "CREATE OR REPLACE TABLE t(b STRING) USING parquet");
        assert_eq!(schema_of(&s, "SELECT * FROM t"), [("b".to_owned(), true)]);
        assert_eq!(
            class(&s, "CREATE TABLE t(a INT) USING parquet"),
            ErrorClass::TableOrViewAlreadyExists
        );
        run(&s, "DROP TABLE t");
    }

    #[test]
    fn unknown_and_missing_connector_providers_are_not_found() {
        let s = session();
        assert_eq!(
            class(&s, "CREATE TABLE t(a INT) USING delta"),
            ErrorClass::DataSourceNotFound
        );
        assert_eq!(
            class(&s, "CREATE TABLE t(a INT) USING iceberg"),
            ErrorClass::DataSourceNotFound
        );
        assert_eq!(
            class(&s, "CREATE TABLE t(a INT) USING nope"),
            ErrorClass::DataSourceNotFound
        );
        // With no `USING`, the provider is `spark.sql.sources.default`.
        run(&s, "CREATE TABLE t(a INT)");
        assert_eq!(
            s.state
                .catalog
                .get_table("default", "t", false)
                .unwrap()
                .provider,
            "parquet"
        );
        run(&s, "DROP TABLE t");
        s.conf().set("spark.sql.sources.default", "csv").unwrap();
        run(&s, "CREATE TABLE u(a INT)");
        assert_eq!(
            s.state
                .catalog
                .get_table("default", "u", false)
                .unwrap()
                .provider,
            "csv"
        );
        run(&s, "DROP TABLE u");
        s.conf().set("spark.sql.sources.default", "delta").unwrap();
        assert_eq!(
            class(&s, "CREATE TABLE v(a INT)"),
            ErrorClass::DataSourceNotFound
        );
    }

    #[test]
    fn serde_and_partitioned_tables_are_unsupported() {
        let s = session();
        assert_eq!(
            class(&s, "CREATE TABLE t(a INT) STORED AS PARQUET"),
            ErrorClass::NotSupportedCommandWithoutHiveSupport
        );
        assert_eq!(
            class(
                &s,
                "CREATE TABLE t(a INT) ROW FORMAT DELIMITED FIELDS TERMINATED BY ','"
            ),
            ErrorClass::NotSupportedCommandWithoutHiveSupport
        );
        assert_eq!(
            class(&s, "CREATE TABLE t(a INT) USING parquet PARTITIONED BY (a)"),
            ErrorClass::KrishivUnsupportedFeature
        );
        assert_eq!(
            class(
                &s,
                "CREATE TABLE t(a INT) USING parquet CLUSTERED BY (a) INTO 4 BUCKETS"
            ),
            ErrorClass::KrishivUnsupportedFeature
        );
        assert_eq!(
            class(&s, "CREATE TEMPORARY TABLE t(a INT) USING parquet"),
            ErrorClass::KrishivUnsupportedFeature
        );
    }

    #[test]
    fn namespaces_drop_with_and_without_cascade() {
        let s = session();
        run(&s, "CREATE DATABASE mydb");
        run(&s, "CREATE TABLE mydb.t(a INT) USING parquet");
        assert_eq!(class(&s, "DROP DATABASE mydb"), ErrorClass::SchemaNotEmpty);
        assert_eq!(
            class(&s, "DROP DATABASE missing"),
            ErrorClass::SchemaNotFound
        );
        run(&s, "DROP DATABASE IF EXISTS missing");
        assert_eq!(
            class(&s, "DROP DATABASE default"),
            ErrorClass::KrishivUnsupportedFeature
        );
        assert_eq!(
            class(&s, "DROP DATABASE IF EXISTS default"),
            ErrorClass::KrishivUnsupportedFeature
        );
        let location = s
            .state
            .catalog
            .get_table("mydb", "t", false)
            .unwrap()
            .location;
        assert!(std::path::Path::new(&location).exists());
        run(&s, "DROP DATABASE mydb CASCADE");
        assert!(s.state.catalog.get_table("mydb", "t", false).is_none());
        assert!(!std::path::Path::new(&location).exists());
        assert!(!s.state.catalog.namespaces().contains(&"mydb".to_owned()));
        assert_eq!(class(&s, "USE mydb"), ErrorClass::SchemaNotFound);
    }

    #[test]
    fn current_namespace_switch_changes_resolution() {
        let s = session();
        run(&s, "CREATE DATABASE mydb");
        run(&s, "CREATE TABLE t USING parquet AS SELECT 1 AS a");
        run(&s, "CREATE TABLE mydb.t USING parquet AS SELECT 2 AS a");
        assert_eq!(rows(&s, "SELECT * FROM t"), [["1"]]);
        run(&s, "USE mydb");
        assert_eq!(rows(&s, "SELECT * FROM t"), [["2"]]);
        assert_eq!(rows(&s, "SELECT * FROM default.t"), [["1"]]);
        run(&s, "DROP TABLE mydb.t");
        run(&s, "USE default");
        run(&s, "DROP TABLE t");
        run(&s, "DROP DATABASE mydb");
    }

    #[test]
    fn epoch_bumps_on_every_ddl() {
        let s = session();
        let epoch = s.state.catalog.epoch();
        run(&s, "CREATE DATABASE mydb");
        let epoch = {
            assert!(s.state.catalog.epoch() > epoch);
            s.state.catalog.epoch()
        };
        run(&s, "USE mydb");
        assert!(s.state.catalog.epoch() > epoch);
        let epoch = s.state.catalog.epoch();
        run(&s, "CREATE TABLE t(a INT) USING parquet");
        assert!(s.state.catalog.epoch() > epoch);
        let epoch = s.state.catalog.epoch();
        run(&s, "DROP TABLE t");
        assert!(s.state.catalog.epoch() > epoch);
        let epoch = s.state.catalog.epoch();
        run(&s, "DROP DATABASE mydb CASCADE");
        assert!(s.state.catalog.epoch() > epoch);
    }

    #[test]
    fn warehouse_temp_dir_is_deleted_with_the_session() {
        let dir = {
            let s = session();
            let dir = s.state.catalog.warehouse_dir();
            assert!(dir.exists());
            dir
        };
        assert!(!dir.exists());
    }

    #[test]
    fn char_and_varchar_read_and_write_as_strings() {
        // The engine has no padded character types yet: `char`/`varchar`
        // columns are stored as strings.
        let s = session();
        run(&s, "CREATE TABLE t(c CHAR(5), v VARCHAR(6)) USING parquet");
        assert_eq!(
            schema_of(&s, "SELECT * FROM t"),
            [("c".to_owned(), true), ("v".to_owned(), true)]
        );
        assert!(rows(&s, "SELECT * FROM t").is_empty());
        run(&s, "CREATE TABLE u USING parquet AS SELECT * FROM t");
        assert!(rows(&s, "SELECT * FROM u").is_empty());
        run(&s, "DROP TABLE u");
        run(&s, "DROP TABLE t");
    }

    #[test]
    fn like_copies_schema_provider_and_options() {
        let s = session();
        run(
            &s,
            "CREATE TABLE src(a INT NOT NULL, b STRING) USING csv OPTIONS (header 'true') TBLPROPERTIES ('k' = 'v')",
        );
        run(&s, "CREATE TABLE dst LIKE src");
        let source = s.state.catalog.get_table("default", "src", false).unwrap();
        let liked = s.state.catalog.get_table("default", "dst", false).unwrap();
        assert_eq!(liked.columns, source.columns);
        assert_eq!(liked.provider, "csv");
        assert_eq!(liked.options, source.options);
        assert!(rows(&s, "SELECT * FROM dst").is_empty());
        run(&s, "CREATE TABLE IF NOT EXISTS dst LIKE src");
        assert_eq!(
            class(&s, "CREATE TABLE dst LIKE src"),
            ErrorClass::TableOrViewAlreadyExists
        );
        assert_eq!(
            class(&s, "CREATE TABLE dst2 LIKE missing"),
            ErrorClass::TableOrViewNotFound
        );
        run(&s, "CREATE TABLE json_dst LIKE src USING json");
        assert_eq!(
            s.state
                .catalog
                .get_table("default", "json_dst", false)
                .unwrap()
                .provider,
            "json"
        );
        assert_eq!(
            class(&s, "CREATE TABLE json_dst LIKE src USING json"),
            ErrorClass::TableOrViewAlreadyExists
        );
        run(&s, "DROP TABLE dst");
        run(&s, "DROP TABLE json_dst");
        run(&s, "DROP TABLE src");
    }

    #[test]
    fn external_table_directories_survive_drops() {
        let s = session();
        let dir = s.state.catalog.warehouse_dir().join("external");
        std::fs::create_dir_all(&dir).unwrap();
        let location = dir.to_string_lossy().into_owned();
        run(
            &s,
            &format!("CREATE EXTERNAL TABLE e(a INT) LOCATION '{location}'"),
        );
        assert_eq!(
            s.state
                .catalog
                .get_table("default", "e", false)
                .unwrap()
                .location,
            location
        );
        assert!(rows(&s, "SELECT * FROM e").is_empty());
        run(&s, "DROP TABLE e");
        assert!(std::path::Path::new(&location).exists());
        // `PURGE` is accepted.
        run(
            &s,
            &format!("CREATE EXTERNAL TABLE e(a INT) LOCATION '{location}'"),
        );
        run(&s, "DROP TABLE e PURGE");
        assert!(std::path::Path::new(&location).exists());
    }

    #[test]
    fn invalid_names_fail() {
        let s = session();
        assert_eq!(
            class(&s, "CREATE TABLE a.b.c.d(x INT) USING parquet"),
            ErrorClass::RequiresSinglePartNamespace
        );
        assert_eq!(
            class(&s, "CREATE TABLE spark_catalog(x INT) USING parquet"),
            ErrorClass::RequiresSinglePartNamespace
        );
        assert_eq!(
            class(&s, "CREATE TABLE spark_catalog.t(x INT) USING parquet"),
            ErrorClass::RequiresSinglePartNamespace
        );
        assert_eq!(
            class(&s, "DROP TABLE a.b.c.d"),
            ErrorClass::RequiresSinglePartNamespace
        );
        assert_eq!(
            class(&s, "USE a.b.c"),
            ErrorClass::RequiresSinglePartNamespace
        );
        assert_eq!(
            class(
                &s,
                "CREATE TABLE IDENTIFIER('a' || 'b')(x INT) USING parquet"
            ),
            ErrorClass::KrishivUnsupportedFeature
        );
        assert_eq!(
            class(&s, "USE IDENTIFIER('a' || 'b')"),
            ErrorClass::KrishivUnsupportedFeature
        );
        assert_eq!(
            class(&s, "CREATE TABLE missing.t(x INT) USING parquet"),
            ErrorClass::SchemaNotFound
        );
        assert_eq!(
            class(&s, "CREATE TABLE t(a INT, A STRING) USING parquet"),
            ErrorClass::ColumnAlreadyExists
        );
        assert_eq!(
            class(&s, "CREATE TABLE t(a INT) LOCATION ''"),
            ErrorClass::InvalidEmptyLocation
        );
        assert_eq!(
            class(
                &s,
                "CREATE TABLE t(a INT) USING parquet DEFAULT COLLATION UTF8_LCASE"
            ),
            ErrorClass::UnsupportedFeatureSchemaLevelCollations
        );
        assert_eq!(
            class(&s, "CREATE DATABASE d DEFAULT COLLATION UTF8_LCASE"),
            ErrorClass::UnsupportedFeatureSchemaLevelCollations
        );
        assert_eq!(
            class(&s, "SHOW TABLES"),
            ErrorClass::KrishivUnsupportedFeature
        );
    }

    #[test]
    fn table_options_and_properties_roundtrip() {
        let s = session();
        run(
            &s,
            "CREATE TABLE t(a INT) USING csv OPTIONS (header 'true', flag true, n 42, 'a b' = 'x', novalue, d DATE '2020-01-01', e 1 + 2) TBLPROPERTIES ('k' = 'v') COMMENT 'a table'",
        );
        let table = s.state.catalog.get_table("default", "t", false).unwrap();
        assert_eq!(
            table
                .options
                .iter()
                .map(|(k, v)| (k.clone(), v.to_string()))
                .collect::<Vec<_>>(),
            [
                ("a b".to_owned(), "x".to_owned()),
                ("d".to_owned(), "2020-01-01".to_owned()),
                ("e".to_owned(), "1 + 2".to_owned()),
                ("flag".to_owned(), "true".to_owned()),
                ("header".to_owned(), "true".to_owned()),
                ("n".to_owned(), "42".to_owned()),
                ("novalue".to_owned(), String::new()),
            ]
        );
        assert_eq!(
            table.properties,
            BTreeMap::from([("k".to_owned(), "v".to_owned())])
        );
        assert_eq!(table.comment.as_deref(), Some("a table"));
        run(&s, "DROP TABLE t");
    }

    #[test]
    fn configured_warehouse_dir_is_used() {
        let dir = std::env::temp_dir().join(format!("krishiv-test-wh-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let s = Session::builder()
            .config("spark.sql.warehouse.dir", dir.to_str().unwrap())
            .build()
            .unwrap();
        run(&s, "CREATE TABLE t(a INT) USING parquet");
        let location = s
            .state
            .catalog
            .get_table("default", "t", false)
            .unwrap()
            .location;
        assert!(location.starts_with(dir.to_str().unwrap()));
        run(&s, "DROP TABLE t");
        assert!(!std::path::Path::new(&location).exists());
        // The configured directory is the user's: dropping the session
        // leaves it behind.
        drop(s);
        assert!(dir.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn case_sensitive_names_match_exactly() {
        let s = Session::builder()
            .config("spark.sql.caseSensitive", "true")
            .build()
            .unwrap();
        run(&s, "CREATE DATABASE MyDb");
        run(&s, "USE MyDb");
        run(&s, "CREATE TABLE T(a INT) USING parquet");
        assert!(rows(&s, "SELECT * FROM T").is_empty());
        assert_eq!(
            class(&s, "SELECT * FROM t"),
            ErrorClass::TableOrViewNotFound
        );
        assert_eq!(
            class(&s, "DROP TABLE mydb.t"),
            ErrorClass::TableOrViewNotFound
        );
        assert_eq!(
            class(&s, "SELECT * FROM mydb.t"),
            ErrorClass::TableOrViewNotFound
        );
        run(&s, "USE default");
        run(&s, "DROP TABLE MyDb.T");
        run(&s, "DROP DATABASE MyDb");
    }

    #[test]
    fn constraints_are_accepted_and_ignored() {
        let s = session();
        run(
            &s,
            "CREATE TABLE t(a INT NOT NULL, b STRING, CONSTRAINT pk PRIMARY KEY (a), UNIQUE (b), CHECK (a > 0)) USING parquet",
        );
        assert_eq!(
            schema_of(&s, "SELECT * FROM t"),
            [("a".to_owned(), false), ("b".to_owned(), true)]
        );
        assert!(rows(&s, "SELECT * FROM t").is_empty());
        run(&s, "DROP TABLE t");
        assert_eq!(
            class(&s, "CREATE TABLE t(IDENTIFIER('a') INT) USING parquet"),
            ErrorClass::KrishivUnsupportedFeature
        );
    }

    #[test]
    fn ctas_reports_analysis_failures() {
        let s = session();
        assert_eq!(
            class(&s, "CREATE TABLE t USING parquet AS SELECT * FROM missing"),
            ErrorClass::TableOrViewNotFound
        );
        assert!(s.state.catalog.get_table("default", "t", false).is_none());
        run(&s, "CREATE TABLE t USING parquet AS SELECT 1 AS a");
        run(
            &s,
            "CREATE TABLE IF NOT EXISTS t USING parquet AS SELECT 2 AS a",
        );
        assert_eq!(rows(&s, "SELECT * FROM t"), [["1"]]);
        run(&s, "DROP TABLE t");
    }

    #[test]
    fn store_lists_and_alters() {
        let s = session();
        run(
            &s,
            "CREATE DATABASE mydb COMMENT 'test' WITH DBPROPERTIES ('k' 'v')",
        );
        assert!(s.state.catalog.namespaces().contains(&"mydb".to_owned()));
        run(&s, "CREATE TABLE mydb.t(a INT) USING parquet");
        assert_eq!(s.state.catalog.list_tables("mydb", false).unwrap(), ["t"]);
        assert_eq!(
            class(&s, "SELECT * FROM mydb.missing"),
            ErrorClass::TableOrViewNotFound
        );
        assert!(
            s.state
                .catalog
                .alter_store_table("mydb", "t", false, |t: &mut CatalogTable| {
                    t.comment = Some("updated".to_owned());
                })
        );
        assert_eq!(
            s.state
                .catalog
                .get_table("mydb", "t", false)
                .unwrap()
                .comment
                .as_deref(),
            Some("updated")
        );
        assert!(!s.state.catalog.alter_store_table(
            "mydb",
            "missing",
            false,
            |_: &mut CatalogTable| {}
        ));
        run(&s, "DROP DATABASE mydb CASCADE");
    }

    #[test]
    fn v2_write_stores_provider_options_and_properties() {
        use futures::executor::block_on;
        use krishiv_common::secret::OptionValue;

        let s = session();
        run(&s, "CREATE TABLE src USING csv AS SELECT 1 AS a");
        let df = s.table("src");
        block_on(
            df.write_to_table("t")
                .using("csv")
                .option("header", "true")
                .options([("sep", "|")])
                .table_property("k", "v")
                .create(),
        )
        .unwrap();
        let table = s.state.catalog.get_table("default", "t", false).unwrap();
        assert_eq!(table.provider, "csv");
        assert_eq!(
            table
                .options
                .iter()
                .map(|(k, v)| (k.clone(), v.to_string()))
                .collect::<Vec<_>>(),
            [
                ("header".to_owned(), "true".to_owned()),
                ("sep".to_owned(), "|".to_owned()),
            ]
        );
        // Options are typed as secrets where Spark redacts them: the
        // values never appear in errors.
        assert_eq!(
            table.options.get("header"),
            Some(&OptionValue::from_user("header", "true"))
        );
        assert_eq!(
            table.properties,
            BTreeMap::from([("k".to_owned(), "v".to_owned())])
        );
        // A secret option value is redacted from the write path too.
        block_on(
            s.table("src")
                .write_to_table("u")
                .option("password", "hunter2")
                .create(),
        )
        .unwrap();
        let table = s.state.catalog.get_table("default", "u", false).unwrap();
        assert_eq!(
            table.options.get("password"),
            Some(&OptionValue::from_user("password", "hunter2"))
        );
        run(&s, "DROP TABLE t");
        run(&s, "DROP TABLE u");
        run(&s, "DROP TABLE src");
    }

    #[test]
    fn v2_write_defaults_to_the_configured_provider() {
        use futures::executor::block_on;

        let s = session();
        run(&s, "CREATE TABLE src USING parquet AS SELECT 1 AS a");
        block_on(s.table("src").write_to_table("t").create()).unwrap();
        assert_eq!(
            s.state
                .catalog
                .get_table("default", "t", false)
                .unwrap()
                .provider,
            "parquet"
        );
        run(&s, "DROP TABLE t");
        s.conf().set("spark.sql.sources.default", "csv").unwrap();
        block_on(s.table("src").write_to_table("u").create()).unwrap();
        assert_eq!(
            s.state
                .catalog
                .get_table("default", "u", false)
                .unwrap()
                .provider,
            "csv"
        );
        run(&s, "DROP TABLE u");
        run(&s, "DROP TABLE src");
    }
}
