// SPDX-License-Identifier: Apache-2.0
//! `HiveCatalog`: a Hive metastore as a read-only `krishiv_plan` [`Catalog`] (design P3-lakehouse §2.3, §5.6).
//!
//! A table's schema comes from where its writer put it: a table Spark wrote through a metastore keeps the real
//! schema as JSON in the table properties (`spark.sql.sources.schema`, or `.numParts` and `.part.N`), with a
//! placeholder column list in the metastore itself; any other table has its columns (and partition columns, last)
//! in the metastore's own lists, as Hive type strings. A Spark schema property that is damaged (a part missing, JSON
//! that does not parse or is not a struct, a partition column the schema does not have) is a catalog fault, never a
//! silent fall back to the placeholder columns. The storage locations (`sd.location`, and the serde's `path` property
//! where Spark records it) go through [`check_location`](crate::location::check_location) before they leave this
//! module, so a catalog that names `file:///etc` or an endpoint of its own is refused as a catalog fault, never read.
//!
//! The client is blocking: each call runs on the blocking pool, on one connection that is dropped on any failure so
//! the next call reconnects. A failure reaches the user as `KRISHIV_CATALOG_FAILURE` naming the catalog and the
//! table, and a limit as `KRISHIV_CATALOG_LIMIT_EXCEEDED`; the detail goes to the log, cut short.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use krishiv_common::error::{
    KrishivCatalogFailure, KrishivCatalogLimitExceeded, KrishivError, KrishivUnsupportedFeature,
};
use krishiv_plan::TableVersion;
use krishiv_plan::analyzer::{
    BoxFuture, Catalog, FileMetadata, NameForm, TableColumn, TableIdent, TableKind, TableMeta,
};
use krishiv_types::{DataType, parse_data_type};

use super::client::{HmsClient, HmsConfig, HmsError};
use super::limits::Limit;
use super::types::{FieldSchema, Table};
use crate::location::{LocationPolicy, check_location};

/// Spark's field-metadata key for a `CHAR(n)` / `VARCHAR(n)` column, whose type is `string`.
const CHAR_VARCHAR_KEY: &str = "__CHAR_VARCHAR_TYPE_STRING";

/// How much of a server's text reaches the log.
const LOGGED_DETAIL_CHARS: usize = 300;

/// What a Hive table is, read and checked: the planner's [`TableMeta`] plus what a reader of its files needs.
#[derive(Debug, Clone)]
pub struct HiveTable {
    /// The table as the analyzer sees it.
    pub meta: TableMeta,
    /// Where its files are, after the location rules.
    pub location: String,
    /// The partition columns, in order.
    pub partition_columns: Vec<String>,
    /// The Spark data source (`spark.sql.sources.provider`), for a table Spark wrote.
    pub provider: Option<String>,
    /// The Hadoop input format class, for a Hive table.
    pub input_format: Option<String>,
    /// The serde class, for a Hive table.
    pub serde: Option<String>,
    /// The table properties.
    pub properties: BTreeMap<String, String>,
}

struct Inner {
    name: String,
    config: HmsConfig,
    max_columns: usize,
    policy: Arc<dyn LocationPolicy + Send + Sync>,
    client: Mutex<Option<HmsClient>>,
    epoch: AtomicU64,
}

/// A Hive metastore as a catalog named `name`.
#[derive(Clone)]
pub struct HiveCatalog {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for HiveCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HiveCatalog")
            .field("name", &self.inner.name)
            .field("address", &self.inner.config.address)
            .finish_non_exhaustive()
    }
}

fn quoted(part: &str) -> String {
    format!("`{}`", part.replace('`', "``"))
}

/// A server's text, cut to what a log line should carry.
fn shortened(text: &str) -> String {
    let mut out: String = text.chars().take(LOGGED_DETAIL_CHARS).collect();
    if text.chars().count() > LOGGED_DETAIL_CHARS {
        out.push('…');
    }
    out
}

/// A catalog fault with the detail in the log only.
fn fault(catalog: &str, relation: &str, detail: &str) -> KrishivError {
    tracing::warn!(
        target: "krishiv::catalog",
        catalog,
        table = relation,
        detail = shortened(detail).as_str(),
        "a Hive metastore returned something Krishiv cannot use"
    );
    KrishivCatalogFailure {
        catalog: quoted(catalog),
        relation_name: relation.to_owned(),
    }
    .into()
}

impl HiveCatalog {
    /// A catalog named `name` over the metastore `config` reaches; `policy` decides which locations it may name and
    /// `max_columns` (`krishiv.catalog.maxColumns`) how wide a table may be. Nothing connects until the first call.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        config: HmsConfig,
        max_columns: usize,
        policy: Arc<dyn LocationPolicy + Send + Sync>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                name: name.into(),
                config,
                max_columns,
                policy,
                client: Mutex::new(None),
                epoch: AtomicU64::new(0),
            }),
        }
    }

    /// The error for a failed call: a limit is `KRISHIV_CATALOG_LIMIT_EXCEEDED`, anything else
    /// `KRISHIV_CATALOG_FAILURE` with the detail in the log.
    fn failure(&self, relation: &str, detail: &HmsError) -> KrishivError {
        let c = &self.inner.config;
        match detail {
            HmsError::Limit(limit) => {
                let (what, value, config) = match limit {
                    Limit::Bytes => (
                        "bytes in one reply",
                        c.max_message_bytes.to_string(),
                        "a higher `krishiv.catalog.maxMetadataBytes`",
                    ),
                    Limit::Elements => (
                        "elements in one reply",
                        c.max_elements.to_string(),
                        "a higher element limit for this catalog",
                    ),
                    Limit::Size => (
                        "the size of one string or list",
                        c.max_string_bytes.to_string(),
                        "a higher size limit for this catalog",
                    ),
                    Limit::Depth => (
                        "the nesting depth of a reply",
                        c.max_depth.to_string(),
                        "a higher depth limit for this catalog",
                    ),
                };
                KrishivCatalogLimitExceeded {
                    source: format!(
                        "The reply of the Hive metastore {} for {relation}",
                        quoted(&self.inner.name)
                    ),
                    limit: what.to_owned(),
                    value,
                    config: config.to_owned(),
                }
                .into()
            }
            HmsError::Meta(text) | HmsError::Failed(text) => {
                fault(&self.inner.name, relation, text)
            }
        }
    }

    /// Runs `call` on the one connection, connecting first if there is none. A failure drops the connection.
    fn with_client<T>(
        &self,
        call: impl FnOnce(&mut HmsClient) -> Result<T, HmsError>,
    ) -> Result<T, HmsError> {
        let mut guard = self
            .inner
            .client
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut client = match guard.take() {
            Some(client) => client,
            None => HmsClient::connect(&self.inner.config)?,
        };
        let result = call(&mut client);
        // Only a connection that worked is kept.
        if result.is_ok() {
            *guard = Some(client);
        }
        result
    }

    /// The names of the databases (Hive's databases are Krishiv's one-part namespaces).
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE` when the metastore cannot be reached or refuses the call.
    pub fn list_databases(&self) -> Result<Vec<String>, KrishivError> {
        self.with_client(HmsClient::get_all_databases)
            .map_err(|e| self.failure("`*`", &e))
    }

    /// The names of the tables and views of `database`.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE`.
    pub fn list_tables(&self, database: &str) -> Result<Vec<String>, KrishivError> {
        self.with_client(|c| c.get_all_tables(database))
            .map_err(|e| self.failure(&quoted(database), &e))
    }

    /// Loads `database.table`: `None` when it does not exist.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE` when the metastore fails or sends something damaged;
    /// `KRISHIV_CATALOG_LIMIT_EXCEEDED` when a reply or a schema is over a limit;
    /// `KRISHIV_CATALOG_LOCATION_REFUSED` when a location it names is refused; `KRISHIV_UNSUPPORTED_FEATURE` for a
    /// view, a table without storage or a column type that has no Krishiv type.
    pub fn load(&self, database: &str, table: &str) -> Result<Option<HiveTable>, KrishivError> {
        let relation = format!(
            "{}.{}.{}",
            quoted(&self.inner.name),
            quoted(database),
            quoted(table)
        );
        let found = self
            .with_client(|c| c.get_table(database, table))
            .map_err(|e| self.failure(&relation, &e))?;
        found
            .map(|t| {
                convert(
                    &self.inner.name,
                    &relation,
                    t,
                    &*self.inner.policy,
                    self.inner.max_columns,
                )
                .map_err(|e| match e {
                    Converted::Error(e) => e,
                    Converted::Columns(n) => self.columns_over_limit(&relation, n),
                })
            })
            .transpose()
    }

    fn columns_over_limit(&self, relation: &str, columns: usize) -> KrishivError {
        KrishivCatalogLimitExceeded {
            source: format!("The schema of {relation}"),
            limit: format!("{} columns", self.inner.max_columns),
            value: format!("{columns} columns"),
            config: "`krishiv.catalog.maxColumns`".to_owned(),
        }
        .into()
    }
}

impl Catalog for HiveCatalog {
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Only a change made through Krishiv bumps it, and this catalog has no write half: it stays 0.
    fn epoch(&self) -> u64 {
        self.inner.epoch.load(Ordering::Acquire)
    }

    fn single_part_namespace(&self) -> bool {
        true
    }

    /// A Hive table has no format version: `version` must be `None`, and any other asks for what does not exist.
    fn load_table<'a>(
        &'a self,
        id: &'a TableIdent,
        version: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        let catalog = self.clone();
        let (database, table) = match id.namespace.as_slice() {
            [database] if version.is_none() && !form.temporary_only => {
                if form.case_sensitive {
                    (database.clone(), id.name.clone())
                } else {
                    (database.to_lowercase(), id.name.to_lowercase())
                }
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
                Err(e) => Err(fault(
                    &name,
                    &relation,
                    &format!("the metastore call did not finish: {e}"),
                )),
            }
        })
    }
}

/// Why a conversion failed: an error to raise, or a schema over `krishiv.catalog.maxColumns`.
enum Converted {
    Error(KrishivError),
    Columns(usize),
}

impl From<KrishivError> for Converted {
    fn from(e: KrishivError) -> Self {
        Self::Error(e)
    }
}

/// The Spark schema property: the JSON in one value, or split in `numParts` parts. `Ok(None)` when the table has
/// none (then the metastore's columns are the schema); a damaged one is `Err`.
fn spark_schema_json(
    properties: &BTreeMap<String, String>,
) -> Result<Option<String>, &'static str> {
    if let Some(whole) = properties.get("spark.sql.sources.schema") {
        return Ok(Some(whole.clone()));
    }
    let Some(parts) = properties.get("spark.sql.sources.schema.numParts") else {
        return Ok(None);
    };
    let parts: usize = parts
        .parse()
        .map_err(|_| "the number of schema parts is not a number")?;
    let mut json = String::new();
    for i in 0..parts {
        json.push_str(
            properties
                .get(&format!("spark.sql.sources.schema.part.{i}"))
                .ok_or("a part of the schema property is missing")?,
        );
    }
    Ok(Some(json))
}

fn spark_partition_columns(
    properties: &BTreeMap<String, String>,
) -> Result<Vec<String>, &'static str> {
    let Some(n) = properties.get("spark.sql.sources.schema.numPartCols") else {
        return Ok(Vec::new());
    };
    let n: usize = n
        .parse()
        .map_err(|_| "the number of partition columns is not a number")?;
    (0..n)
        .map(|i| {
            properties
                .get(&format!("spark.sql.sources.schema.partCol.{i}"))
                .cloned()
                .ok_or("a partition column of the schema property is missing")
        })
        .collect()
}

/// A Hive column: its type string as a Krishiv type, `CHAR(n)`/`VARCHAR(n)` as a string that keeps its declared form.
fn hive_column(field: &FieldSchema, relation: &str) -> Result<TableColumn, KrishivError> {
    let declared = field.type_name.trim().to_ascii_lowercase();
    if declared.starts_with("varchar(") || declared.starts_with("char(") {
        let mut column = TableColumn::new(&field.name, DataType::String, true);
        column.char_varchar = Some(declared);
        return Ok(column);
    }
    let data_type = parse_data_type(&field.type_name).map_err(|_| {
        KrishivError::from(KrishivUnsupportedFeature {
            feature: format!(
                "the Hive column type `{}` of {relation}",
                field.type_name.replace('`', "")
            ),
        })
    })?;
    Ok(TableColumn::new(&field.name, data_type, true))
}

/// The schema of a table Spark wrote: its columns, their comments and its partition columns.
struct SparkSchema {
    columns: Vec<TableColumn>,
    comments: BTreeMap<String, String>,
    partitions: Vec<String>,
}

/// The schema of a table Spark wrote, from its schema property. `Ok(None)` when it has no schema property.
fn spark_table(
    catalog: &str,
    relation: &str,
    properties: &BTreeMap<String, String>,
) -> Result<Option<SparkSchema>, KrishivError> {
    let damaged = |why: &str| {
        fault(
            catalog,
            relation,
            &format!("the Spark schema property is damaged: {why}"),
        )
    };
    let Some(json) = spark_schema_json(properties).map_err(damaged)? else {
        return Ok(None);
    };
    let Ok(DataType::Struct(schema)) = DataType::from_json(&json) else {
        return Err(damaged("it is not the JSON of a struct type"));
    };
    let mut columns = Vec::new();
    let mut comments = BTreeMap::new();
    for f in schema.fields() {
        let mut column = TableColumn::new(f.name(), f.data_type().clone(), f.nullable());
        column.char_varchar = f
            .metadata()
            .as_map()
            .get(CHAR_VARCHAR_KEY)
            .and_then(|v| v.as_str())
            .map(str::to_ascii_lowercase);
        columns.push(column);
        if let Some(c) = f.metadata().comment() {
            comments.insert(f.name().to_owned(), c.to_owned());
        }
    }
    let partitions = spark_partition_columns(properties).map_err(damaged)?;
    for p in &partitions {
        if !columns.iter().any(|c| c.name.eq_ignore_ascii_case(p)) {
            return Err(damaged("a partition column is not a column of the schema"));
        }
    }
    Ok(Some(SparkSchema {
        columns,
        comments,
        partitions,
    }))
}

fn convert(
    catalog: &str,
    relation: &str,
    table: Table,
    policy: &dyn LocationPolicy,
    max_columns: usize,
) -> Result<HiveTable, Converted> {
    if matches!(
        table.table_type.as_deref(),
        Some("VIRTUAL_VIEW" | "MATERIALIZED_VIEW")
    ) {
        return Err(KrishivError::from(KrishivUnsupportedFeature {
            feature: format!("reading the Hive view {relation}"),
        })
        .into());
    }
    let Some(sd) = table.sd else {
        return Err(KrishivError::from(KrishivUnsupportedFeature {
            feature: format!("the Hive table {relation} without storage"),
        })
        .into());
    };
    let location = sd.location.clone().unwrap_or_default();
    check_location(catalog, relation, &location, policy)?;
    // Spark also records where a data source table reads from as the serde's `path`: it is a location a reader may
    // use, so it is checked the same way.
    if let Some(path) = sd
        .serde_info
        .as_ref()
        .and_then(|s| s.parameters.get("path"))
    {
        check_location(catalog, relation, path, policy)?;
    }

    let provider = table.parameters.get("spark.sql.sources.provider").cloned();
    let from_spark = if provider.is_some() {
        spark_table(catalog, relation, &table.parameters)?
    } else {
        None
    };
    let (columns, comments, partition_columns) = if let Some(spark) = from_spark {
        (spark.columns, spark.comments, spark.partitions)
    } else {
        let mut columns = Vec::new();
        let mut comments = BTreeMap::new();
        for f in sd.cols.iter().chain(&table.partition_keys) {
            columns.push(hive_column(f, relation)?);
            if let Some(c) = f.comment.as_ref().filter(|c| !c.is_empty()) {
                comments.insert(f.name.clone(), c.clone());
            }
        }
        let keys = table
            .partition_keys
            .iter()
            .map(|f| f.name.clone())
            .collect();
        (columns, comments, keys)
    };
    if columns.len() > max_columns {
        return Err(Converted::Columns(columns.len()));
    }
    let meta = TableMeta {
        ident: TableIdent {
            catalog: catalog.to_owned(),
            namespace: vec![table.db_name.to_lowercase()],
            name: table.table_name.to_lowercase(),
        },
        columns,
        version: None,
        kind: TableKind::Table,
        file_metadata: FileMetadata::None,
        column_comments: comments,
    };
    Ok(HiveTable {
        meta,
        location,
        partition_columns,
        provider,
        input_format: sd.input_format,
        serde: sd.serde_info.and_then(|s| s.serialization_lib),
        properties: table.parameters,
    })
}

#[cfg(test)]
mod tests {
    use krishiv_common::error::ErrorClass;
    use thrift::protocol::{TListIdentifier, TOutputProtocol, TStructIdentifier, TType};

    use super::super::test_server::fake::{
        cfg, exception, len32, map_field, reply_with, serve, serve_connections, string_field,
        strings_field, struct_field,
    };
    use super::*;
    use crate::location::RemoteLocationPolicy;

    /// A table as the metastore would send it.
    struct Fake {
        location: &'static str,
        table_type: &'static str,
        cols: Vec<(&'static str, &'static str, &'static str)>,
        partitions: Vec<(&'static str, &'static str)>,
        params: Vec<(String, String)>,
        serde_params: Vec<(&'static str, &'static str)>,
    }

    impl Fake {
        fn hive() -> Self {
            Self {
                location: "s3a://bucket/web/events",
                table_type: "EXTERNAL_TABLE",
                cols: vec![
                    ("id", "bigint", "the key"),
                    ("tags", "map<string,string>", ""),
                    ("name", "varchar(20)", ""),
                ],
                partitions: vec![("dt", "string")],
                params: vec![("EXTERNAL".to_owned(), "TRUE".to_owned())],
                serde_params: Vec::new(),
            }
        }
    }

    fn fields(o: &mut dyn TOutputProtocol, list: &[(&str, &str, &str)]) -> thrift::Result<()> {
        o.write_list_begin(&TListIdentifier::new(TType::Struct, len32(list.len())))?;
        for (name, ty, comment) in list {
            o.write_struct_begin(&TStructIdentifier::new("FieldSchema"))?;
            string_field(o, 1, name)?;
            string_field(o, 2, ty)?;
            if !comment.is_empty() {
                string_field(o, 3, comment)?;
            }
            o.write_field_stop()?;
            o.write_struct_end()?;
        }
        o.write_list_end()
    }

    fn write(o: &mut dyn TOutputProtocol, t: &Fake) -> thrift::Result<()> {
        string_field(o, 1, "events")?;
        string_field(o, 2, "web")?;
        struct_field(o, 7, |o| {
            o.write_field_begin(&thrift::protocol::TFieldIdentifier {
                name: None,
                field_type: TType::List,
                id: Some(1),
            })?;
            fields(o, &t.cols)?;
            o.write_field_end()?;
            string_field(o, 2, t.location)?;
            string_field(
                o,
                3,
                "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            )?;
            struct_field(o, 7, |o| {
                string_field(
                    o,
                    2,
                    "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
                )?;
                if t.serde_params.is_empty() {
                    Ok(())
                } else {
                    map_field(o, 3, &t.serde_params)
                }
            })
        })?;
        o.write_field_begin(&thrift::protocol::TFieldIdentifier {
            name: None,
            field_type: TType::List,
            id: Some(8),
        })?;
        fields(
            o,
            &t.partitions
                .iter()
                .map(|(n, ty)| (*n, *ty, ""))
                .collect::<Vec<_>>(),
        )?;
        o.write_field_end()?;
        let params: Vec<(&str, &str)> = t
            .params
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        map_field(o, 9, &params)?;
        string_field(o, 12, t.table_type)
    }

    fn catalog(address: &str) -> HiveCatalog {
        HiveCatalog::new(
            "hms",
            cfg(address),
            10_000,
            Arc::new(RemoteLocationPolicy::default()),
        )
    }

    fn serving(t: Fake) -> (String, std::thread::JoinHandle<()>) {
        serve(1, move |method, seq, o| {
            reply_with(o, method, seq, |o| struct_field(o, 0, |o| write(o, &t)))
        })
    }

    #[test]
    fn a_hive_table_has_its_columns_then_its_partition_columns_with_their_types_and_comments() {
        let (address, server) = serving(Fake::hive());
        let table = catalog(&address).load("web", "events").unwrap().unwrap();
        server.join().unwrap();
        let names: Vec<_> = table.meta.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["id", "tags", "name", "dt"]);
        assert_eq!(table.meta.columns[0].data_type, DataType::Long);
        assert!(matches!(table.meta.columns[1].data_type, DataType::Map(_)));
        assert_eq!(table.meta.columns[2].data_type, DataType::String);
        assert_eq!(
            table.meta.columns[2].char_varchar.as_deref(),
            Some("varchar(20)")
        );
        assert_eq!(table.meta.column_comments["id"], "the key");
        assert!(!table.meta.column_comments.contains_key("tags"));
        assert_eq!(table.partition_columns, ["dt"]);
        assert_eq!(table.location, "s3a://bucket/web/events");
        assert_eq!(table.meta.ident.catalog, "hms");
        assert_eq!(
            (
                table.meta.ident.namespace.as_slice(),
                table.meta.ident.name.as_str()
            ),
            (&["web".to_owned()][..], "events")
        );
        assert!(table.input_format.unwrap().contains("Parquet"));
        assert!(table.serde.unwrap().ends_with("ParquetHiveSerDe"));
        assert!(table.provider.is_none());
        assert_eq!(table.meta.version, None);
    }

    #[test]
    fn a_table_spark_wrote_takes_its_schema_from_the_properties_not_the_placeholder_columns() {
        let json = r#"{"type":"struct","fields":[{"name":"id","type":"long","nullable":false,"metadata":{"comment":"key"}},{"name":"name","type":"string","nullable":true,"metadata":{}},{"name":"dt","type":"string","nullable":true,"metadata":{}}]}"#;
        let (first, second) = json.split_at(60);
        let mut t = Fake::hive();
        t.cols = vec![("col", "array<string>", "from deserializer")];
        t.partitions = Vec::new();
        t.params = vec![
            (
                "spark.sql.sources.provider".to_owned(),
                "parquet".to_owned(),
            ),
            (
                "spark.sql.sources.schema.numParts".to_owned(),
                "2".to_owned(),
            ),
            (
                "spark.sql.sources.schema.part.0".to_owned(),
                first.to_owned(),
            ),
            (
                "spark.sql.sources.schema.part.1".to_owned(),
                second.to_owned(),
            ),
            (
                "spark.sql.sources.schema.numPartCols".to_owned(),
                "1".to_owned(),
            ),
            (
                "spark.sql.sources.schema.partCol.0".to_owned(),
                "dt".to_owned(),
            ),
        ];
        let (address, server) = serving(t);
        let table = catalog(&address).load("web", "events").unwrap().unwrap();
        server.join().unwrap();
        let columns: Vec<_> = table
            .meta
            .columns
            .iter()
            .map(|c| (c.name.as_str(), c.nullable))
            .collect();
        assert_eq!(columns, [("id", false), ("name", true), ("dt", true)]);
        assert_eq!(table.meta.columns[0].data_type, DataType::Long);
        assert_eq!(table.meta.column_comments["id"], "key");
        assert_eq!(table.partition_columns, ["dt"]);
        assert_eq!(table.provider.as_deref(), Some("parquet"));
    }

    #[test]
    fn a_location_the_rules_refuse_is_a_catalog_fault_that_never_names_it() {
        for bad in [
            "file:///etc/shadow",
            "http://169.254.169.254/x",
            "",
            "s3://b/../x",
        ] {
            let mut t = Fake::hive();
            t.location = bad;
            let (address, server) = serving(t);
            let e = catalog(&address).load("web", "events").unwrap_err();
            server.join().unwrap();
            assert_eq!(
                e.error_class(),
                ErrorClass::KrishivCatalogLocationRefused,
                "{bad:?}"
            );
            let text = e.to_string();
            assert!(text.contains("`hms`.`web`.`events`"), "{text}");
            assert!(
                !text.contains("shadow") && !text.contains("169.254"),
                "{text}"
            );
        }
    }

    #[test]
    fn a_missing_table_is_none() {
        let (address, server) = serve(1, |method, seq, o| {
            reply_with(o, method, seq, |o| exception(o, 2, "no such table"))
        });
        assert!(catalog(&address).load("web", "nope").unwrap().is_none());
        server.join().unwrap();
    }

    #[test]
    fn what_cannot_be_read_is_refused_by_name() {
        let cases: Vec<(Fake, &str)> = vec![
            (
                Fake {
                    table_type: "VIRTUAL_VIEW",
                    ..Fake::hive()
                },
                "view",
            ),
            (
                Fake {
                    cols: vec![("u", "uniontype<int,string>", "")],
                    ..Fake::hive()
                },
                "uniontype",
            ),
        ];
        for (t, what) in cases {
            let (address, server) = serving(t);
            let e = catalog(&address).load("web", "events").unwrap_err();
            server.join().unwrap();
            assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
            assert!(e.to_string().contains(what), "{e}");
        }
    }

    #[test]
    fn a_metastore_failure_names_the_catalog_and_the_table_and_keeps_the_detail_in_the_log() {
        let (address, server) = serve(1, |method, seq, o| {
            reply_with(o, method, seq, |o| {
                exception(o, 1, "jdbc password is s3cret")
            })
        });
        let e = catalog(&address).load("web", "events").unwrap_err();
        server.join().unwrap();
        assert_eq!(e.error_class(), ErrorClass::KrishivCatalogFailure);
        let text = e.to_string();
        assert!(
            text.contains("`hms`") && text.contains("`hms`.`web`.`events`"),
            "{text}"
        );
        assert!(!text.contains("s3cret"), "{text}");
    }

    #[test]
    fn a_connection_that_failed_is_replaced_by_the_next_call() {
        // The server closes each connection after one call: the second call finds its connection closed
        // and fails, the third connects again.
        let (address, server) = serve_connections(2, 1, |method, seq, o| {
            reply_with(o, method, seq, |o| strings_field(o, 0, &["default"]))
        });
        let c = catalog(&address);
        assert_eq!(c.list_databases().unwrap(), ["default"]);
        let e = c.list_databases().unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivCatalogFailure);
        assert_eq!(c.list_databases().unwrap(), ["default"]);
        server.join().unwrap();
    }

    #[test]
    fn it_is_a_catalog_whose_names_fold_and_that_has_no_versions_or_temporary_tables() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let (address, server) = serve(1, |method, seq, o| {
            reply_with(o, method, seq, |o| {
                struct_field(o, 0, |o| write(o, &Fake::hive()))
            })
        });
        let c = catalog(&address);
        assert_eq!(
            (c.name(), c.epoch(), c.single_part_namespace()),
            ("hms", 0, true)
        );
        assert!(
            c.as_write().is_none(),
            "a read-only catalog has no write half"
        );
        assert!(!c.capabilities().writes);
        let id = |ns: &[&str]| TableIdent {
            catalog: "hms".to_owned(),
            namespace: ns.iter().map(|s| (*s).to_owned()).collect(),
            name: "EVENTS".to_owned(),
        };
        let form = NameForm {
            case_sensitive: false,
            qualified: true,
            temporary_only: false,
        };
        let load =
            |id: &TableIdent, version, form| rt.block_on(c.load_table(id, version, form)).unwrap();
        // None of these reaches the metastore.
        assert!(load(&id(&["web"]), Some(TableVersion(3)), form).is_none());
        assert!(load(&id(&["a", "b"]), None, form).is_none());
        assert!(
            load(
                &id(&["web"]),
                None,
                NameForm {
                    temporary_only: true,
                    ..form
                }
            )
            .is_none()
        );
        // This does, and the name arrives folded.
        let meta = load(&id(&["WEB"]), None, form).unwrap();
        assert_eq!(meta.ident.name, "events");
        assert_eq!(meta.columns.len(), 4);
        server.join().unwrap();
    }

    #[test]
    fn databases_and_tables_are_listed() {
        let (address, server) = serve(2, |method, seq, o| {
            reply_with(o, method, seq, |o| match method {
                "get_all_databases" => strings_field(o, 0, &["default", "web"]),
                _ => strings_field(o, 0, &["events"]),
            })
        });
        let c = catalog(&address);
        assert_eq!(c.list_databases().unwrap(), ["default", "web"]);
        assert_eq!(c.list_tables("web").unwrap(), ["events"]);
        server.join().unwrap();
    }

    fn spark(props: &[(&str, &str)]) -> Fake {
        let mut t = Fake::hive();
        t.cols = vec![("col", "array<string>", "from deserializer")];
        t.partitions = Vec::new();
        t.params = std::iter::once(("spark.sql.sources.provider", "parquet"))
            .chain(props.iter().copied())
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect();
        t
    }

    const SCHEMA: &str = r#"{"type":"struct","fields":[{"name":"id","type":"long","nullable":false,"metadata":{}},{"name":"dt","type":"string","nullable":true,"metadata":{}}]}"#;

    /// A schema property that does not add up is a fault of the catalog, never the placeholder column list.
    #[test]
    fn a_damaged_spark_schema_property_is_a_catalog_fault_not_the_placeholder_schema() {
        let cases: Vec<(&str, Vec<(&str, &str)>)> = vec![
            (
                "a part is missing",
                vec![
                    ("spark.sql.sources.schema.numParts", "2"),
                    ("spark.sql.sources.schema.part.0", "{"),
                ],
            ),
            (
                "numParts is not a number",
                vec![("spark.sql.sources.schema.numParts", "two")],
            ),
            ("not JSON", vec![("spark.sql.sources.schema", "{not json")]),
            (
                "not a struct",
                vec![("spark.sql.sources.schema", "\"integer\"")],
            ),
            (
                "a partition column the schema lacks",
                vec![
                    ("spark.sql.sources.schema", SCHEMA),
                    ("spark.sql.sources.schema.numPartCols", "1"),
                    ("spark.sql.sources.schema.partCol.0", "nope"),
                ],
            ),
            (
                "a partition column entry is missing",
                vec![
                    ("spark.sql.sources.schema", SCHEMA),
                    ("spark.sql.sources.schema.numPartCols", "2"),
                    ("spark.sql.sources.schema.partCol.0", "dt"),
                ],
            ),
            (
                "numPartCols is not a number",
                vec![
                    ("spark.sql.sources.schema", SCHEMA),
                    ("spark.sql.sources.schema.numPartCols", "x"),
                ],
            ),
        ];
        for (why, props) in cases {
            let (address, server) = serving(spark(&props));
            let e = catalog(&address).load("web", "events").unwrap_err();
            server.join().unwrap();
            assert_eq!(e.error_class(), ErrorClass::KrishivCatalogFailure, "{why}");
            assert!(e.to_string().contains("`hms`.`web`.`events`"), "{why}: {e}");
            assert!(
                !e.to_string().contains("damaged"),
                "the detail stays in the log: {e}"
            );
        }
    }

    #[test]
    fn a_spark_table_without_a_schema_property_reads_the_metastore_columns() {
        let (address, server) = serving(spark(&[]));
        let table = catalog(&address).load("web", "events").unwrap().unwrap();
        server.join().unwrap();
        assert_eq!(table.meta.columns.len(), 1);
        assert_eq!(table.meta.columns[0].name, "col");
        assert_eq!(table.provider.as_deref(), Some("parquet"));
    }

    #[test]
    fn a_spark_char_or_varchar_column_keeps_its_declared_type() {
        let json = r#"{"type":"struct","fields":[{"name":"c","type":"string","nullable":true,"metadata":{"__CHAR_VARCHAR_TYPE_STRING":"VARCHAR(20)"}},{"name":"s","type":"string","nullable":true,"metadata":{}}]}"#;
        let (address, server) = serving(spark(&[("spark.sql.sources.schema", json)]));
        let table = catalog(&address).load("web", "events").unwrap().unwrap();
        server.join().unwrap();
        assert_eq!(table.meta.columns[0].data_type, DataType::String);
        assert_eq!(
            table.meta.columns[0].char_varchar.as_deref(),
            Some("varchar(20)")
        );
        assert_eq!(table.meta.columns[1].char_varchar, None);
    }

    /// Spark records a data source table's path in the serde too: it is a location a reader may use.
    #[test]
    fn the_serde_path_property_is_checked_like_the_location() {
        let mut t = Fake::hive();
        t.serde_params = vec![("path", "file:///etc/passwd")];
        let (address, server) = serving(t);
        let e = catalog(&address).load("web", "events").unwrap_err();
        server.join().unwrap();
        assert_eq!(e.error_class(), ErrorClass::KrishivCatalogLocationRefused);
        assert!(!e.to_string().contains("passwd"), "{e}");
        let mut ok = Fake::hive();
        ok.serde_params = vec![
            ("path", "s3a://bucket/web/events"),
            ("serialization.format", "1"),
        ];
        let (address, server) = serving(ok);
        assert!(catalog(&address).load("web", "events").unwrap().is_some());
        server.join().unwrap();
    }

    #[test]
    fn limits_are_the_limit_class_naming_the_limit_and_the_key() {
        let (address, server) = serving(Fake::hive());
        let tight = HmsConfig {
            max_message_bytes: 64,
            ..cfg(&address)
        };
        let c = HiveCatalog::new(
            "hms",
            tight,
            10_000,
            Arc::new(RemoteLocationPolicy::default()),
        );
        let e = c.load("web", "events").unwrap_err();
        drop(server);
        assert_eq!(e.error_class(), ErrorClass::KrishivCatalogLimitExceeded);
        let text = e.to_string();
        assert!(
            text.contains("64") && text.contains("krishiv.catalog.maxMetadataBytes"),
            "{text}"
        );
        assert!(text.contains("`hms`.`web`.`events`"), "{text}");
    }

    #[test]
    fn a_schema_wider_than_max_columns_is_the_limit_class() {
        let (address, server) = serving(Fake::hive());
        let c = HiveCatalog::new(
            "hms",
            cfg(&address),
            3,
            Arc::new(RemoteLocationPolicy::default()),
        );
        let e = c.load("web", "events").unwrap_err();
        server.join().unwrap();
        assert_eq!(e.error_class(), ErrorClass::KrishivCatalogLimitExceeded);
        let text = e.to_string();
        assert!(
            text.contains("4 columns") && text.contains("krishiv.catalog.maxColumns"),
            "{text}"
        );
    }
}
