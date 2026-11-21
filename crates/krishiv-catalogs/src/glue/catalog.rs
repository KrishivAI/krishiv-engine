// SPDX-License-Identifier: Apache-2.0
//! `GlueCatalog`: one AWS Glue Data Catalog as a read-only `krishiv_plan` [`Catalog`] (design P3-lakehouse §2.3, §5.6).
//!
//! A Glue table is `database.table`; one Krishiv catalog serves one Glue catalog, so a Krishiv table is
//! `<krishiv catalog>.<database>.<table>` (one-part namespace). Glue lower-cases every name, so lookups are
//! case-insensitive. There is no write half: `as_write()` is `None` and `capabilities().writes` is false, so the
//! capability check raises `UNSUPPORTED_FEATURE.CATALOG_OPERATION` for any write or DDL before a side effect.
//!
//! What a table exposes: the format comes from `Parameters` (`table_type=ICEBERG` with its `metadata_location`;
//! `spark.sql.sources.provider` of `delta`, `parquet` or `csv`), else from the Hive input format or serde
//! (Parquet, OpenCSV); anything else, and every view, is `UNSUPPORTED_FEATURE.TABLE_OPERATION`. The location (the
//! Iceberg metadata file for an Iceberg table) is checked by [`check_location`]; a refusal is a catalog fault naming no
//! location. Columns come from the storage descriptor's Hive type strings, partition keys last. Each call runs on the
//! blocking pool.

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

use super::client::{GlueClient, GlueColumn, GlueTableRaw};
use super::config::GlueConfig;
use super::error::{quoted, shortened};
use crate::location::{LocationPolicy, check_location};

/// The longest `char(n)`/`varchar(n)` or column type string Krishiv parses.
const MAX_DECLARED_LEN: usize = 4096;

/// The formats of a Glue table Krishiv reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlueFormat {
    /// An Iceberg table; the location is its current metadata file.
    Iceberg,
    /// A Delta table at the storage location.
    Delta,
    /// Parquet files at the storage location.
    Parquet,
    /// CSV files at the storage location.
    Csv,
}

/// What a Glue table is, read and checked: the planner's [`TableMeta`] plus what a reader of its files needs.
#[derive(Debug, Clone)]
pub struct GlueTable {
    /// The table as the analyzer sees it.
    pub meta: TableMeta,
    /// How its files are laid out.
    pub format: GlueFormat,
    /// Where its files (an Iceberg table's metadata file) are, after the location rules.
    pub location: String,
    /// The partition columns, in order of the table's partition keys.
    pub partition_columns: Vec<String>,
    /// `TableType` as Glue says (`EXTERNAL_TABLE`, `MANAGED_TABLE`), when it does.
    pub table_type: Option<String>,
    /// The table's string-valued properties.
    pub properties: BTreeMap<String, String>,
}

struct Inner {
    name: String,
    client: GlueClient,
    policy: Arc<dyn LocationPolicy + Send + Sync>,
}

/// An AWS Glue catalog, read-only. Cheap to clone.
#[derive(Clone)]
pub struct GlueCatalog {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for GlueCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlueCatalog")
            .field("name", &self.inner.name)
            .finish_non_exhaustive()
    }
}

impl GlueCatalog {
    /// A Krishiv catalog named `name` over the Glue catalog `config` declares; `policy` decides which storage
    /// locations it may name. Nothing connects until the first call.
    ///
    /// # Errors
    /// An invalid declaration, or the HTTP client could not start.
    pub fn new(
        name: impl Into<String>,
        config: GlueConfig,
        policy: Arc<dyn LocationPolicy + Send + Sync>,
    ) -> Result<Self, KrishivError> {
        let name = name.into();
        let client = GlueClient::new(name.clone(), config)?;
        Ok(Self {
            inner: Arc::new(Inner {
                name,
                client,
                policy,
            }),
        })
    }

    /// The databases of the Glue catalog (blocking).
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE` or `KRISHIV_CATALOG_LIMIT_EXCEEDED`.
    pub fn list_databases(&self) -> Result<Vec<String>, KrishivError> {
        self.inner.client.list_databases()
    }

    /// The tables of `database` (blocking).
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE` or `KRISHIV_CATALOG_LIMIT_EXCEEDED`.
    pub fn list_tables(&self, database: &str) -> Result<Vec<String>, KrishivError> {
        self.inner.client.list_tables(database)
    }

    /// `database.table`, read and checked (blocking); `None` when Glue does not know it.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE`, `KRISHIV_CATALOG_LIMIT_EXCEEDED`, `KRISHIV_CATALOG_LOCATION_REFUSED`,
    /// `UNSUPPORTED_FEATURE.TABLE_OPERATION` for a format other than Delta, Parquet and CSV.
    pub fn load(&self, database: &str, table: &str) -> Result<Option<GlueTable>, KrishivError> {
        let Some(raw) = self.inner.client.get_table(database, table)? else {
            return Ok(None);
        };
        let relation = format!(
            "{}.{}.{}",
            quoted(&self.inner.name),
            quoted(database),
            quoted(table)
        );
        convert(
            &self.inner.name,
            &relation,
            database,
            table,
            raw,
            self.inner.policy.as_ref(),
            self.inner.client.limits().max_columns,
        )
        .map(Some)
    }
}

impl Catalog for GlueCatalog {
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

    /// A Glue table has no version Krishiv can name: `version` must be `None`.
    fn load_table<'a>(
        &'a self,
        id: &'a TableIdent,
        version: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        let catalog = self.clone();
        let (database, table) = match id.namespace.as_slice() {
            [database] if version.is_none() && !form.temporary_only => {
                // Glue stores every name in lower case.
                (database.to_lowercase(), id.name.to_lowercase())
            }
            _ => return Box::pin(async { Ok(None) }),
        };
        let relation = format!(
            "{}.{}.{}",
            quoted(&self.inner.name),
            quoted(&database),
            quoted(&table)
        );
        Box::pin(async move {
            let name = catalog.inner.name.clone();
            match tokio::task::spawn_blocking(move || {
                catalog.load(&database, &table).map(|t| t.map(|t| t.meta))
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
                        "a Glue catalog call did not finish"
                    );
                    Err(KrishivCatalogFailure {
                        catalog: quoted(&name),
                        relation_name: relation,
                    }
                    .into())
                }
            }
        })
    }
}

/// Only tables are read: a `VIRTUAL_VIEW` or any other kind has no storage Krishiv can read as a table and is not a
/// catalog fault, so it is refused before the location check. A table that declares no type is left to the format
/// and location rules.
fn check_table_type(relation: &str, table_type: Option<&str>) -> Result<(), KrishivError> {
    let Some(kind) = table_type else {
        return Ok(());
    };
    let kind = kind.to_ascii_uppercase();
    if matches!(
        kind.as_str(),
        "EXTERNAL_TABLE" | "MANAGED_TABLE" | "EXTERNAL" | "MANAGED" | ""
    ) {
        return Ok(());
    }
    Err(UnsupportedFeatureTableOperation {
        table_name: relation.to_owned(),
        operation: format!(
            "reading an AWS Glue table of type {} (only tables are read)",
            identifier_or(&kind, "an unrecognised")
        ),
    }
    .into())
}

/// `text` when it is a short identifier, else `fallback`: a server's text reaches an error only in that form.
fn identifier_or<'a>(text: &'a str, fallback: &'a str) -> &'a str {
    if text.len() <= 32 && text.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        text
    } else {
        fallback
    }
}

/// The format of a table and where its data or metadata is, or `UNSUPPORTED_FEATURE.TABLE_OPERATION`.
fn format_of(relation: &str, raw: &GlueTableRaw) -> Result<(GlueFormat, String), KrishivError> {
    let refused = |operation: String| {
        KrishivError::from(UnsupportedFeatureTableOperation {
            table_name: relation.to_owned(),
            operation,
        })
    };
    let param = |key: &str| {
        raw.parameters
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.to_ascii_lowercase())
    };
    if param("table_type").as_deref() == Some("iceberg") {
        // The current metadata file is the entry point; the connector reads the table from it.
        let Some(location) = raw.parameters.iter().find_map(|(k, v)| {
            k.eq_ignore_ascii_case("metadata_location")
                .then(|| v.clone())
        }) else {
            return Err(refused(
                "reading an AWS Glue Iceberg table that has no metadata_location".to_owned(),
            ));
        };
        return Ok((GlueFormat::Iceberg, location));
    }
    // A Spark data-source table keeps its real schema in `spark.sql.sources.schema*`; the Glue column list is a
    // placeholder, so reading it would give the wrong columns. Refused until a golden pins the Spark encoding.
    if raw.parameters.iter().any(|(k, _)| {
        let k = k.to_ascii_lowercase();
        k == "spark.sql.sources.schema" || k.starts_with("spark.sql.sources.schema.")
    }) {
        return Err(refused(
            "reading an AWS Glue table whose schema Spark keeps in its table properties".to_owned(),
        ));
    }
    let hive = |needle: &str| {
        [raw.input_format.as_deref(), raw.serde.as_deref()]
            .into_iter()
            .flatten()
            .any(|t| t.to_ascii_lowercase().contains(needle))
    };
    let format = match param("spark.sql.sources.provider").as_deref() {
        Some("delta") => GlueFormat::Delta,
        Some("parquet") => GlueFormat::Parquet,
        Some("csv") => GlueFormat::Csv,
        Some(other) => {
            return Err(refused(format!(
                "reading an AWS Glue table in the {} format",
                identifier_or(other, "an unrecognised")
            )));
        }
        None if hive("parquet") => GlueFormat::Parquet,
        None if hive("opencsvserde") => GlueFormat::Csv,
        None => {
            return Err(refused(
                "reading an AWS Glue table whose format Krishiv does not read (Iceberg, Delta, Parquet and CSV are)"
                    .to_owned(),
            ));
        }
    };
    Ok((format, raw.location.clone().unwrap_or_default()))
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

fn column(c: &GlueColumn, catalog: &str, relation: &str) -> Result<TableColumn, KrishivError> {
    let declared = c.type_text.trim().to_ascii_lowercase();
    let fault = |detail: &str| {
        tracing::warn!(
            target: "krishiv::catalog",
            catalog,
            table = relation,
            detail,
            "a Glue catalog returned a column type Krishiv cannot use"
        );
        KrishivError::from(KrishivCatalogFailure {
            catalog: quoted(catalog),
            relation_name: relation.to_owned(),
        })
    };
    if declared.starts_with("varchar(") || declared.starts_with("char(") {
        let well_formed = declared
            .split_once('(')
            .and_then(|(_, rest)| rest.strip_suffix(')'))
            .is_some_and(|n| {
                !n.is_empty() && n.len() <= 10 && n.bytes().all(|b| b.is_ascii_digit())
            });
        if !well_formed {
            return Err(fault("a CHAR/VARCHAR type string has no valid length"));
        }
        let mut col = TableColumn::new(&c.name, DataType::String, true);
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
            "a Glue catalog returned a column type Krishiv does not support"
        );
        KrishivError::from(KrishivUnsupportedFeature {
            feature: format!("an AWS Glue column type of {relation} that Krishiv does not support"),
        })
    })?;
    Ok(TableColumn::new(&c.name, data_type, true))
}

fn convert(
    catalog: &str,
    relation: &str,
    database: &str,
    table: &str,
    raw: GlueTableRaw,
    policy: &dyn LocationPolicy,
    max_columns: usize,
) -> Result<GlueTable, KrishivError> {
    check_table_type(relation, raw.table_type.as_deref())?;
    let (format, location) = format_of(relation, &raw)?;
    check_location(catalog, relation, &location, policy)?;

    // Partition keys follow the data columns, as in Hive.
    let mut columns = Vec::with_capacity(raw.columns.len() + raw.partition_keys.len());
    let mut comments = BTreeMap::new();
    for c in raw.columns.iter().chain(&raw.partition_keys) {
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
    let partition_columns = raw.partition_keys.iter().map(|c| c.name.clone()).collect();
    Ok(GlueTable {
        meta: TableMeta {
            ident: TableIdent {
                catalog: catalog.to_owned(),
                namespace: vec![database.to_owned()],
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
        properties: raw.parameters,
    })
}
