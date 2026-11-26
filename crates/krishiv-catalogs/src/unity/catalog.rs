// SPDX-License-Identifier: Apache-2.0
//! `UnityCatalog`: one Unity Catalog catalog as a read-only `krishiv_plan` [`Catalog`] (design P3-lakehouse §3, §5.6).
//!
//! A UC three-level name is `catalog.schema.table`; one Krishiv catalog serves one UC catalog, so a Krishiv table is
//! `<krishiv catalog>.<schema>.<table>` (one-part namespace). There is no write half: `as_write()` is `None` and
//! `capabilities().writes` is false, so the capability check raises `UNSUPPORTED_FEATURE.CATALOG_OPERATION` for any
//! write or DDL before a side effect.
//!
//! What a table exposes by `data_source_format`: `DELTA`, `PARQUET` and `CSV` give a [`UnityFormat`] and a storage
//! location (checked by [`check_location`], a refusal is a catalog fault naming no location); every other format,
//! Iceberg included, is `UNSUPPORTED_FEATURE.TABLE_OPERATION` (UC serves its Iceberg tables through its own Iceberg
//! REST endpoint, which is not built here). Columns come from `type_text` (Spark catalog type strings), ordered by
//! `position`, partition columns from `partition_index`. Each call runs on the blocking pool.

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_common::error::{
    KrishivCatalogFailure, KrishivCatalogLimitExceeded, KrishivError, KrishivUnsupportedFeature,
    UnsupportedFeatureTableOperation,
};
use krishiv_plan::TableVersion;
use krishiv_plan::analyzer::{
    BoxFuture, Catalog, FileMetadata, NameForm, TableColumn, TableIdent, TableKind, TableMeta,
};
use krishiv_types::{DataType, parse_data_type};

use super::client::{UcColumn, UcTable, UnityClient};
use super::config::UnityConfig;
use super::error::{quoted, shortened};
use crate::location::{LocationPolicy, check_location};

/// The longest `char(n)`/`varchar(n)` or column type string Krishiv parses.
const MAX_DECLARED_LEN: usize = 4096;

/// The file formats of a UC table Krishiv reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnityFormat {
    /// A Delta table at the storage location.
    Delta,
    /// Parquet files at the storage location.
    Parquet,
    /// CSV files at the storage location.
    Csv,
}

/// What a UC table is, read and checked: the planner's [`TableMeta`] plus what a reader of its files needs.
#[derive(Debug, Clone)]
pub struct UnityTable {
    /// The table as the analyzer sees it.
    pub meta: TableMeta,
    /// How its files are laid out.
    pub format: UnityFormat,
    /// Where its files are, after the location rules.
    pub location: String,
    /// The partition columns, in order of `partition_index`.
    pub partition_columns: Vec<String>,
    /// `MANAGED` or `EXTERNAL`, as UC says. A `MANAGED` Delta table of a v0.7.0 server may be catalog-managed: its
    /// log has commits UC ratified and has not backfilled, so a by-path read of `location` can be stale. T021 decides
    /// (design P3-lakehouse §3) before any reader opens one; until then nothing reads it.
    pub table_type: Option<String>,
    /// The table's string-valued properties.
    pub properties: BTreeMap<String, String>,
}

struct Inner {
    name: String,
    client: UnityClient,
    policy: Arc<dyn LocationPolicy + Send + Sync>,
}

/// A Unity Catalog catalog, read-only. Cheap to clone.
#[derive(Clone)]
pub struct UnityCatalog {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for UnityCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnityCatalog")
            .field("name", &self.inner.name)
            .field("uc_catalog", &self.inner.client.uc_catalog())
            .finish_non_exhaustive()
    }
}

impl UnityCatalog {
    /// A Krishiv catalog named `name` over the UC catalog `config` declares; `policy` decides which storage
    /// locations it may name. Nothing connects until the first call.
    ///
    /// # Errors
    /// An invalid declaration, or the HTTP client could not start.
    pub fn new(
        name: impl Into<String>,
        config: UnityConfig,
        policy: Arc<dyn LocationPolicy + Send + Sync>,
    ) -> Result<Self, KrishivError> {
        let name = name.into();
        let client = UnityClient::new(name.clone(), config)?;
        Ok(Self {
            inner: Arc::new(Inner {
                name,
                client,
                policy,
            }),
        })
    }

    /// The schemas of the UC catalog (blocking).
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE` or `KRISHIV_CATALOG_LIMIT_EXCEEDED`.
    pub fn list_schemas(&self) -> Result<Vec<String>, KrishivError> {
        self.inner.client.list_schemas()
    }

    /// The tables of `schema` (blocking).
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE` or `KRISHIV_CATALOG_LIMIT_EXCEEDED`.
    pub fn list_tables(&self, schema: &str) -> Result<Vec<String>, KrishivError> {
        self.inner.client.list_tables(schema)
    }

    /// `schema.table`, read and checked (blocking); `None` when UC does not know it.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE`, `KRISHIV_CATALOG_LIMIT_EXCEEDED`, `KRISHIV_CATALOG_LOCATION_REFUSED`,
    /// `UNSUPPORTED_FEATURE.TABLE_OPERATION` for a format other than Delta, Parquet and CSV.
    pub fn load(&self, schema: &str, table: &str) -> Result<Option<UnityTable>, KrishivError> {
        let Some(raw) = self.inner.client.get_table(schema, table)? else {
            return Ok(None);
        };
        let relation = format!(
            "{}.{}.{}",
            quoted(&self.inner.name),
            quoted(schema),
            quoted(table)
        );
        convert(
            &self.inner.name,
            &relation,
            schema,
            table,
            raw,
            self.inner.policy.as_ref(),
            self.inner.client.limits().max_columns,
        )
        .map(Some)
    }
}

impl Catalog for UnityCatalog {
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Only a change made through Krishiv bumps it, and this catalog has no write half: it stays 0.
    fn epoch(&self) -> u64 {
        0
    }

    fn single_part_namespace(&self) -> bool {
        true
    }

    /// A UC table has no version Krishiv can name: `version` must be `None`.
    fn load_table<'a>(
        &'a self,
        id: &'a TableIdent,
        version: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        let catalog = self.clone();
        let (schema, table) = match id.namespace.as_slice() {
            [schema] if version.is_none() && !form.temporary_only => {
                if form.case_sensitive {
                    (schema.clone(), id.name.clone())
                } else {
                    (schema.to_lowercase(), id.name.to_lowercase())
                }
            }
            _ => return Box::pin(async { Ok(None) }),
        };
        let relation = format!(
            "{}.{}.{}",
            quoted(&self.inner.name),
            quoted(&schema),
            quoted(&table)
        );
        Box::pin(async move {
            let name = catalog.inner.name.clone();
            match tokio::task::spawn_blocking(move || {
                catalog.load(&schema, &table).map(|t| t.map(|t| t.meta))
            })
            .await
            {
                Ok(loaded) => loaded,
                // The blocking task panicked or was cancelled: the panic text stays out of the user's error.
                Err(e) => {
                    tracing::warn!(
                        target: "krishiv::catalog",
                        catalog = name.as_str(),
                        table = relation.as_str(),
                        detail = shortened(&e.to_string()).as_str(),
                        "a Unity catalog call did not finish"
                    );
                    Err(KrishivCatalogFailure {
                        catalog: quoted(&name),
                        relation_name: relation,
                        reason: "the call did not finish; the server log has the detail (target `krishiv::catalog`)".to_owned(),
                    }
                    .into())
                }
            }
        })
    }
}

/// Only `MANAGED` and `EXTERNAL` tables are read. A `VIEW`, `METRIC_VIEW`, `MATERIALIZED_VIEW` or `STREAMING_TABLE`
/// has no storage location Krishiv can read as a table, and is not a catalog fault: it is refused before the location
/// check. A table that declares no type is left to the format and location rules.
fn check_table_type(relation: &str, table_type: Option<&str>) -> Result<(), KrishivError> {
    let Some(kind) = table_type else {
        return Ok(());
    };
    let kind = kind.to_ascii_uppercase();
    if matches!(kind.as_str(), "MANAGED" | "EXTERNAL") {
        return Ok(());
    }
    let shown = if kind.len() <= 32 && kind.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        kind.as_str()
    } else {
        "an unrecognised"
    };
    Err(UnsupportedFeatureTableOperation {
        table_name: relation.to_owned(),
        operation: format!("reading a Unity Catalog table of type {shown} (only MANAGED and EXTERNAL tables are read)"),
    }
    .into())
}

/// The format of a table, or `UNSUPPORTED_FEATURE.TABLE_OPERATION`. The server's format text reaches the error only
/// when it is a short identifier.
fn format_of(relation: &str, declared: Option<&str>) -> Result<UnityFormat, KrishivError> {
    let refused = |operation: String| {
        KrishivError::from(UnsupportedFeatureTableOperation {
            table_name: relation.to_owned(),
            operation,
        })
    };
    let Some(declared) = declared else {
        return Err(refused(
            "reading a Unity Catalog table that declares no data source format".to_owned(),
        ));
    };
    match declared.to_ascii_uppercase().as_str() {
        "DELTA" => Ok(UnityFormat::Delta),
        "PARQUET" => Ok(UnityFormat::Parquet),
        "CSV" => Ok(UnityFormat::Csv),
        "ICEBERG" => Err(refused(
            "reading a Unity Catalog Iceberg table (Unity Catalog serves Iceberg tables through its own Iceberg REST \
             endpoint, which Krishiv does not read yet)"
                .to_owned(),
        )),
        other => {
            let shown = if other.len() <= 32
                && other.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                other
            } else {
                "an unrecognised"
            };
            Err(refused(format!(
                "reading a Unity Catalog table in the {shown} format"
            )))
        }
    }
}

/// The fields nested under a type, so a table cannot hide its width in a struct.
fn nested_fields(t: &DataType) -> usize {
    match t {
        DataType::Struct(s) => s
            .fields()
            .iter()
            .map(|f| 1 + nested_fields(f.data_type()))
            .sum(),
        DataType::Array(a) => nested_fields(a.element_type()),
        DataType::Map(m) => nested_fields(m.key_type()) + nested_fields(m.value_type()),
        _ => 0,
    }
}

/// A declared `char(n)`/`varchar(n)` string, bounded in length and nesting.
fn bounded_declared(declared: &str) -> Option<String> {
    if declared.len() > MAX_DECLARED_LEN {
        return None;
    }
    let (mut depth, mut deepest) = (0_usize, 0_usize);
    for c in declared.chars() {
        match c {
            '<' => {
                depth += 1;
                deepest = deepest.max(depth);
            }
            '>' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    (deepest <= DataType::MAX_NESTING_DEPTH).then(|| declared.to_ascii_lowercase())
}

fn column(c: &UcColumn, catalog: &str, relation: &str) -> Result<TableColumn, KrishivError> {
    let declared = c.type_text.trim().to_ascii_lowercase();
    let fault = |detail: &str| {
        tracing::warn!(
            target: "krishiv::catalog",
            catalog,
            table = relation,
            detail,
            "a Unity catalog returned a column type Krishiv cannot use"
        );
        KrishivError::from(KrishivCatalogFailure {
            catalog: quoted(catalog),
            relation_name: relation.to_owned(),
            reason: "a column type Krishiv cannot use".to_owned(),
        })
    };
    if declared.starts_with("varchar(") || declared.starts_with("char(") {
        let mut col = TableColumn::new(&c.name, DataType::String, c.nullable);
        col.char_varchar =
            Some(bounded_declared(&declared).ok_or_else(|| {
                fault("a CHAR/VARCHAR type string is too long or too deeply nested")
            })?);
        return Ok(col);
    }
    if declared.len() > MAX_DECLARED_LEN || bounded_declared(&declared).is_none() {
        return Err(fault(
            "a column type string is too long or too deeply nested",
        ));
    }
    let data_type = parse_data_type(&c.type_text).map_err(|_| {
        // The server's type text goes to the log, cut short; the user's error names the table only.
        tracing::warn!(
            target: "krishiv::catalog",
            catalog,
            table = relation,
            detail = shortened(&c.type_text).as_str(),
            "a Unity catalog returned a column type Krishiv does not support"
        );
        KrishivError::from(KrishivUnsupportedFeature {
            feature: format!(
                "a Unity Catalog column type of {relation} that Krishiv does not support"
            ),
        })
    })?;
    Ok(TableColumn::new(&c.name, data_type, c.nullable))
}

fn convert(
    catalog: &str,
    relation: &str,
    schema: &str,
    table: &str,
    raw: UcTable,
    policy: &dyn LocationPolicy,
    max_columns: usize,
) -> Result<UnityTable, KrishivError> {
    check_table_type(relation, raw.table_type.as_deref())?;
    let format = format_of(relation, raw.data_source_format.as_deref())?;
    let location = raw.storage_location.clone().unwrap_or_default();
    check_location(catalog, relation, &location, policy)?;

    let mut ordered: Vec<&UcColumn> = raw.columns.iter().collect();
    ordered.sort_by_key(|c| c.position.unwrap_or(i64::MAX));
    let mut columns = Vec::with_capacity(ordered.len());
    let mut comments = BTreeMap::new();
    for c in &ordered {
        columns.push(column(c, catalog, relation)?);
        if let Some(comment) = c.comment.as_ref().filter(|c| !c.is_empty()) {
            comments.insert(c.name.clone(), comment.clone());
        }
    }
    let width: usize = columns
        .iter()
        .map(|c| 1 + nested_fields(&c.data_type))
        .sum();
    if width > max_columns {
        return Err(KrishivCatalogLimitExceeded {
            source: format!("The schema of {relation}"),
            limit: format!("{max_columns} columns"),
            value: format!("{width} columns"),
            config: "`krishiv.catalog.maxColumns`".to_owned(),
        }
        .into());
    }
    let mut partitions: Vec<(i64, &str)> = raw
        .columns
        .iter()
        .filter_map(|c| c.partition_index.map(|i| (i, c.name.as_str())))
        .collect();
    partitions.sort_by_key(|(i, _)| *i);
    let partition_columns = partitions.into_iter().map(|(_, n)| n.to_owned()).collect();
    Ok(UnityTable {
        meta: TableMeta {
            ident: TableIdent {
                catalog: catalog.to_owned(),
                namespace: vec![schema.to_owned()],
                name: table.to_owned(),
            },
            columns,
            version: None,
            kind: TableKind::Table,
            file_metadata: FileMetadata::None,
            column_comments: comments,
        },
        format,
        location,
        partition_columns,
        table_type: raw.table_type,
        properties: raw.properties,
    })
}
