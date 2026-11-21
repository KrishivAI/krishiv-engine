// SPDX-License-Identifier: Apache-2.0
//! Catalog metadata (design P3-T101 §5): `GetCatalogs`, `GetDbSchemas`, `GetTables`,
//! `GetTableTypes` and the four key commands.
//!
//! `GetFlightInfo` of a listing answers its fixed schema and a ticket that is the command itself.
//! Nothing runs there and no session is used. The listing runs at `DoGet`, in the call's session.
//! A cookie-less listing gets a transient session, released at the end of the call with no
//! `set-cookie`, because a listing changes no state (§3.2). Every listing reads the
//! `spark.catalog` API (`Session::catalog`), so it passes the access check of the `SHOW` it
//! stands for and shows no more than `SHOW` does. The whole listing runs under
//! `krishiv.flightSql.metadataTimeout`.
//!
//! What is listed:
//! - Catalogs: `spark_catalog` and the declared catalogs this principal sees (`SHOW CATALOGS`).
//! - Schemas and tables: those of `spark_catalog`, whatever the session's current catalog is.
//!   The namespaces and tables of a declared catalog are not answered through the catalog API
//!   yet (design P3-T067 PR-58a), so a declared catalog lists none.
//! - Temporary views: listed once with the empty catalog, the Flight SQL "without a catalog".
//!   Local ones have the empty schema, global ones `global_temp`. `GetDbSchemas` lists those two
//!   schemas under the empty catalog too (`global_temp` always, the empty one while the session
//!   has local temporary views), so a client that walks schemas, then tables, reaches them.
//! - A namespace a pattern selects (or no pattern) but whose `SHOW TABLES IN` the caller may not
//!   run is absent from `GetTables`, as a hidden object is; an exactly named one (a pattern with
//!   no `%` or `_`) is refused as `SHOW TABLES IN` refuses it.
//! - With `include_schema`, each table's schema comes from the metadata analysis
//!   (`DataFrame::schema_for_metadata`), which asks `Browse` as `DESCRIBE` does (ADR 0046 §3b).
//! - Table types: `TABLE`, `VIEW` and `TEMPORARY VIEW`.
//!
//! Patterns are Flight SQL's: `%` matches any run of characters, `_` matches one, `\` escapes the
//! next character. Matching ignores case, as identifiers here do. A catalog is named exactly
//! (ignoring case), and the empty catalog selects the rows without one.
//!
//! The key commands answer empty results with the schemas FlightSql.proto specifies. Krishiv has
//! no primary or foreign keys. The named table is looked up with `spark.catalog.tableExists`
//! (its access check included), so a refused lookup is refused; a missing table is an empty
//! result too, as a table with no keys is. A catalog named without a schema means the session's
//! current database in it.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::sql::metadata::{GetDbSchemasBuilder, GetTablesBuilder};
use arrow_flight::sql::{
    Any, CommandGetCatalogs, CommandGetCrossReference, CommandGetDbSchemas, CommandGetExportedKeys,
    CommandGetImportedKeys, CommandGetPrimaryKeys, CommandGetTableTypes, CommandGetTables,
    ProstMessageExt,
};
use arrow_flight::{FlightDescriptor, FlightEndpoint, FlightInfo, Ticket};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::stream::{self, StreamExt, TryStreamExt};
use krishiv_common::error::{
    ErrorClass, KrishivError, KrishivFlightSqlMetadataTimeout, KrishivFlightSqlSchemaTablesLimit,
};
use krishiv_session::Session;
use krishiv_session::auth::Authenticated;
use krishiv_session::spark_catalog::Table;
use prost::Message;
use tonic::metadata::MetadataMap;
use tonic::{Response, Status};

use crate::cookies::{SessionCookie, session_cookie};
use crate::metrics::Command;
use crate::server::FlightSqlServer;
use crate::statement::{DoGetStream, internal, spark_schema, time_zone};
use crate::status::flight_status;

/// The embedded catalog, the one whose schemas and tables are listed.
const EMBEDDED: &str = "spark_catalog";
/// The namespace of the global temporary views.
const GLOBAL_TEMP: &str = "global_temp";
/// `table_type` of a table.
const TABLE: &str = "TABLE";
/// `table_type` of a permanent view.
const VIEW: &str = "VIEW";
/// `table_type` of a temporary view, local or global.
const TEMPORARY_VIEW: &str = "TEMPORARY VIEW";
/// Every `table_type` there is, as `GetTableTypes` lists them.
pub(crate) const TABLE_TYPES: [&str; 3] = [TABLE, VIEW, TEMPORARY_VIEW];

/// One metadata command (design §5).
#[derive(Debug, Clone)]
pub(crate) enum Listing {
    /// `GetCatalogs`.
    Catalogs(CommandGetCatalogs),
    /// `GetDbSchemas`.
    DbSchemas(CommandGetDbSchemas),
    /// `GetTables`.
    Tables(CommandGetTables),
    /// `GetTableTypes`.
    TableTypes(CommandGetTableTypes),
    /// `GetPrimaryKeys`.
    PrimaryKeys(CommandGetPrimaryKeys),
    /// `GetExportedKeys`.
    ExportedKeys(CommandGetExportedKeys),
    /// `GetImportedKeys`.
    ImportedKeys(CommandGetImportedKeys),
    /// `GetCrossReference`.
    CrossReference(CommandGetCrossReference),
}

impl Listing {
    /// The metrics label of the command.
    pub(crate) const fn command(&self) -> Command {
        match self {
            Self::Catalogs(_) => Command::Catalogs,
            Self::DbSchemas(_) => Command::DbSchemas,
            Self::Tables(_) => Command::Tables,
            Self::TableTypes(_) => Command::TableTypes,
            Self::PrimaryKeys(_)
            | Self::ExportedKeys(_)
            | Self::ImportedKeys(_)
            | Self::CrossReference(_) => Command::Keys,
        }
    }

    /// The command's name, as the timeout error names it.
    const fn name(&self) -> &'static str {
        match self {
            Self::Catalogs(_) => "GetCatalogs",
            Self::DbSchemas(_) => "GetDbSchemas",
            Self::Tables(_) => "GetTables",
            Self::TableTypes(_) => "GetTableTypes",
            Self::PrimaryKeys(_) => "GetPrimaryKeys",
            Self::ExportedKeys(_) => "GetExportedKeys",
            Self::ImportedKeys(_) => "GetImportedKeys",
            Self::CrossReference(_) => "GetCrossReference",
        }
    }

    /// The command as the `Any` its ticket carries.
    fn as_any(&self) -> Any {
        match self {
            Self::Catalogs(c) => c.as_any(),
            Self::DbSchemas(c) => c.as_any(),
            Self::Tables(c) => c.as_any(),
            Self::TableTypes(c) => c.as_any(),
            Self::PrimaryKeys(c) => c.as_any(),
            Self::ExportedKeys(c) => c.as_any(),
            Self::ImportedKeys(c) => c.as_any(),
            Self::CrossReference(c) => c.as_any(),
        }
    }

    /// The schema of the command's result (fixed by the specification).
    fn schema(&self) -> SchemaRef {
        match self {
            Self::Catalogs(c) => (*c).into_builder().schema(),
            Self::DbSchemas(c) => c.clone().into_builder().schema(),
            Self::Tables(c) => c.clone().into_builder().schema(),
            Self::TableTypes(c) => (*c).into_builder().schema(),
            Self::PrimaryKeys(_) => primary_keys_schema(),
            Self::ExportedKeys(_) | Self::ImportedKeys(_) | Self::CrossReference(_) => {
                foreign_keys_schema()
            }
        }
    }
}

/// The `GetFlightInfo` answer of a command whose result has a fixed schema: one endpoint whose
/// ticket is the command itself (`GetSqlInfo` and the listings).
pub(crate) fn fixed_info(
    schema: &Schema,
    command: &Any,
    descriptor: FlightDescriptor,
) -> Result<FlightInfo, Status> {
    let ticket = Ticket::new(command.encode_to_vec());
    FlightInfo::new()
        .try_with_schema(schema)
        .map_err(|e| internal(format!("encoding a schema: {e}")))
        .map(|info| {
            info.with_endpoint(FlightEndpoint::new().with_ticket(ticket))
                .with_descriptor(descriptor)
                .with_total_records(-1)
                .with_total_bytes(-1)
        })
}

/// `GetPrimaryKeys`' result schema (FlightSql.proto).
fn primary_keys_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("catalog_name", DataType::Utf8, true),
        Field::new("db_schema_name", DataType::Utf8, true),
        Field::new("table_name", DataType::Utf8, false),
        Field::new("column_name", DataType::Utf8, false),
        Field::new("key_name", DataType::Utf8, true),
        Field::new("key_sequence", DataType::Int32, false),
    ]))
}

/// The result schema of `GetExportedKeys`, `GetImportedKeys` and `GetCrossReference`
/// (FlightSql.proto).
fn foreign_keys_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("pk_catalog_name", DataType::Utf8, true),
        Field::new("pk_db_schema_name", DataType::Utf8, true),
        Field::new("pk_table_name", DataType::Utf8, false),
        Field::new("pk_column_name", DataType::Utf8, false),
        Field::new("fk_catalog_name", DataType::Utf8, true),
        Field::new("fk_db_schema_name", DataType::Utf8, true),
        Field::new("fk_table_name", DataType::Utf8, false),
        Field::new("fk_column_name", DataType::Utf8, false),
        Field::new("key_sequence", DataType::Int32, false),
        Field::new("fk_key_name", DataType::Utf8, true),
        Field::new("pk_key_name", DataType::Utf8, true),
        Field::new("update_rule", DataType::UInt8, false),
        Field::new("delete_rule", DataType::UInt8, false),
    ]))
}

/// One piece of a Flight SQL pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Piece {
    /// `%`: any run of characters, the empty one too.
    Any,
    /// `_`: one character.
    One,
    /// A character to match, ignoring case.
    Char(char),
}

fn pieces(pattern: &str) -> Vec<Piece> {
    let mut out = Vec::new();
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        out.push(match c {
            '%' => Piece::Any,
            '_' => Piece::One,
            // A trailing `\` escapes nothing and stands for itself.
            '\\' => Piece::Char(chars.next().unwrap_or('\\')),
            c => Piece::Char(c),
        });
    }
    out
}

fn same_char(a: char, b: char) -> bool {
    a == b || a.to_lowercase().eq(b.to_lowercase())
}

/// Whether `value` matches the Flight SQL `pattern` (`%`, `_`, `\` escapes), ignoring case.
pub(crate) fn pattern_matches(pattern: &str, value: &str) -> bool {
    let p = pieces(pattern);
    let v: Vec<char> = value.chars().collect();
    // Glob matching with one backtrack point: the last `%` and where its run ended.
    let (mut i, mut j) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    while j < v.len() {
        match p.get(i) {
            Some(Piece::One) => {
                i += 1;
                j += 1;
            }
            Some(Piece::Char(c)) if same_char(*c, v[j]) => {
                i += 1;
                j += 1;
            }
            Some(Piece::Any) => {
                star = Some((i, j));
                i += 1;
            }
            _ => match star {
                Some((si, sj)) => {
                    i = si + 1;
                    j = sj + 1;
                    star = Some((si, sj + 1));
                }
                None => return false,
            },
        }
    }
    p[i.min(p.len())..].iter().all(|x| *x == Piece::Any)
}

/// Whether the optional pattern keeps `value` (no pattern keeps everything).
fn kept(pattern: Option<&str>, value: &str) -> bool {
    pattern.is_none_or(|p| pattern_matches(p, value))
}

/// Whether the optional catalog filter selects `catalog` (`""`: the rows without a catalog).
fn selects(filter: Option<&str>, catalog: &str) -> bool {
    filter.is_none_or(|f| f.to_lowercase() == catalog.to_lowercase())
}

/// Whether the optional pattern names one value exactly (no `%` or `_` outside an escape).
fn is_exact(pattern: Option<&str>) -> bool {
    pattern.is_some_and(|p| pieces(p).iter().all(|x| matches!(x, Piece::Char(_))))
}

/// Whether `e` is an access refusal (`KRISHIV_INSUFFICIENT_PERMISSIONS`).
fn is_refusal(e: &KrishivError) -> bool {
    e.error_class() == ErrorClass::KrishivInsufficientPermissions
}

/// The temporary views `SHOW TABLES IN global_temp` lists: the global ones and the session's
/// local ones.
async fn temporary_views(session: &Session) -> Result<Vec<Table>, KrishivError> {
    Ok(session
        .catalog()
        .list_tables(Some(GLOBAL_TEMP), None)
        .await?
        .into_iter()
        .filter(|t| t.is_temporary)
        .collect())
}

/// The temporary views a listing with the schema `pattern` shows: `None` when the caller may not
/// list them (`SHOW TABLES IN global_temp` refused) and the pattern does not name one of their
/// schemas exactly, so they are absent as a hidden namespace is; an exact `global_temp` (or the
/// empty schema) is refused as `SHOW TABLES IN global_temp` is.
async fn shown_temporary_views(
    session: &Session,
    pattern: Option<&str>,
) -> Result<Option<Vec<Table>>, KrishivError> {
    match temporary_views(session).await {
        Ok(views) => Ok(Some(views)),
        Err(e)
            if is_refusal(&e)
                && !(is_exact(pattern) && (kept(pattern, GLOBAL_TEMP) || kept(pattern, ""))) =>
        {
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

/// A temporary view's schema: `global_temp` or the empty one.
fn temporary_schema(t: &Table) -> String {
    t.namespace
        .as_ref()
        .and_then(|n| n.first().cloned())
        .unwrap_or_default()
}

/// One part of a name, backquoted for the catalog API's parser.
fn quote(part: &str) -> String {
    format!("`{}`", part.replace('`', "``"))
}

/// A row of `GetTables`, with the name its schema is loaded by.
struct TableRow {
    catalog: String,
    schema: String,
    name: String,
    kind: &'static str,
    qualified: String,
}

/// The namespaces of `spark_catalog` the command selects.
async fn embedded_schemas(
    session: &Session,
    catalog: Option<&str>,
    pattern: Option<&str>,
) -> Result<Vec<String>, KrishivError> {
    if !selects(catalog, EMBEDDED) {
        return Ok(Vec::new());
    }
    Ok(session
        .catalog()
        .list_embedded_databases()
        .await?
        .into_iter()
        .map(|d| d.name)
        .filter(|n| kept(pattern, n))
        .collect())
}

/// The rows of a `GetTables`, before the table types filter.
async fn table_rows(
    session: &Session,
    command: &CommandGetTables,
) -> Result<Vec<TableRow>, KrishivError> {
    let catalog = command.catalog.as_deref();
    let schema_pattern = command.db_schema_filter_pattern.as_deref();
    let name_pattern = command.table_name_filter_pattern.as_deref();
    let api = session.catalog();
    let mut rows = Vec::new();
    for db in embedded_schemas(session, catalog, schema_pattern).await? {
        let in_db = format!("{}.{}", quote(EMBEDDED), quote(&db));
        let listed = match api.list_tables(Some(&in_db), None).await {
            Ok(listed) => listed,
            // A namespace the caller may not browse is absent, unless it was named exactly.
            Err(e) if is_refusal(&e) && !is_exact(schema_pattern) => continue,
            Err(e) => return Err(e),
        };
        for t in listed {
            if t.is_temporary || !kept(name_pattern, &t.name) {
                continue;
            }
            rows.push(TableRow {
                qualified: format!("{in_db}.{}", quote(&t.name)),
                kind: if t.table_type == "VIEW" { VIEW } else { TABLE },
                catalog: EMBEDDED.to_owned(),
                schema: db.clone(),
                name: t.name,
            });
        }
    }
    if selects(catalog, "") {
        let views = shown_temporary_views(session, schema_pattern)
            .await?
            .unwrap_or_default();
        for t in views {
            let schema = temporary_schema(&t);
            if !kept(schema_pattern, &schema) || !kept(name_pattern, &t.name) {
                continue;
            }
            let qualified = if schema.is_empty() {
                quote(&t.name)
            } else {
                format!("{}.{}", quote(&schema), quote(&t.name))
            };
            rows.push(TableRow {
                catalog: String::new(),
                schema,
                name: t.name,
                kind: TEMPORARY_VIEW,
                qualified,
            });
        }
    }
    if !command.table_types.is_empty() {
        rows.retain(|r| {
            command
                .table_types
                .iter()
                .any(|t| t.eq_ignore_ascii_case(r.kind))
        });
    }
    Ok(rows)
}

/// `GetTables` (design §5): with `include_schema`, at most `max_schema_tables` tables, each
/// schema as the stream would send it; a table whose schema cannot be loaded is skipped and
/// logged.
async fn tables(
    session: &Session,
    command: &CommandGetTables,
    max_schema_tables: usize,
) -> Result<RecordBatch, KrishivError> {
    let rows = table_rows(session, command).await?;
    if command.include_schema && rows.len() > max_schema_tables {
        return Err(KrishivFlightSqlSchemaTablesLimit {
            count: rows.len().to_string(),
            limit: max_schema_tables.to_string(),
        }
        .into());
    }
    let mut builder = GetTablesBuilder::new(
        None::<String>,
        None::<String>,
        None::<String>,
        Vec::<String>::new(),
        command.include_schema,
    );
    let tz = time_zone(session);
    let empty = Schema::empty();
    for row in rows {
        let schema = if command.include_schema {
            let loaded = match session.table(&row.qualified).schema_for_metadata().await {
                Ok(s) => spark_schema(&s, tz.as_ref()),
                Err(e) => Err(e),
            };
            match loaded {
                Ok(s) => Some(s),
                Err(e) => {
                    tracing::warn!(
                        target: "krishiv::flight_sql",
                        table = %row.qualified,
                        error_class = e.error_class().name(),
                        "GetTables skipped a table whose schema cannot be loaded"
                    );
                    continue;
                }
            }
        } else {
            None
        };
        builder
            .append(
                &row.catalog,
                &row.schema,
                &row.name,
                row.kind,
                schema.as_deref().unwrap_or(&empty),
            )
            .map_err(|e| arrow_internal("the GetTables result", &e))?;
    }
    builder
        .build()
        .map_err(|e| arrow_internal("the GetTables result", &e))
}

/// `batch` as a `DoGet` stream (a listing's result is small and built whole).
pub(crate) fn batch_stream(batch: RecordBatch) -> DoGetStream {
    FlightDataEncoderBuilder::new()
        .with_schema(batch.schema())
        .build(stream::iter([Ok(batch)]))
        .map_err(Status::from)
        .boxed()
}

fn arrow_internal(what: &str, e: &dyn std::fmt::Display) -> KrishivError {
    krishiv_common::error::InternalError {
        message: format!("building {what}: {e}"),
    }
    .into()
}

/// The table a key command names, as the catalog API takes it. A catalog without a schema
/// means `current` (the session's current database), so the catalog is not read as a schema.
fn key_table(catalog: Option<&str>, schema: Option<&str>, table: &str, current: &str) -> String {
    let catalog = catalog.filter(|c| !c.is_empty());
    let schema = match schema.filter(|s| !s.is_empty()) {
        None if catalog.is_some() => Some(current),
        schema => schema,
    };
    [catalog, schema, Some(table)]
        .into_iter()
        .flatten()
        .filter(|p| !p.is_empty())
        .map(quote)
        .collect::<Vec<_>>()
        .join(".")
}

/// Runs `listing` in `session` (no time limit here; [`FlightSqlServer::listing`] sets it).
async fn run_listing(
    session: &Session,
    listing: &Listing,
    max_schema_tables: usize,
) -> Result<RecordBatch, KrishivError> {
    let api = session.catalog();
    let check = |name: String| async move { api.table_exists(&name).await.map(|_| ()) };
    let empty = || RecordBatch::new_empty(listing.schema());
    let current = api.current_database();
    match listing {
        Listing::Catalogs(c) => {
            let mut builder = (*c).into_builder();
            for name in session.catalog().list_catalogs() {
                builder.append(name);
            }
            builder
                .build()
                .map_err(|e| arrow_internal("the GetCatalogs result", &e))
        }
        Listing::DbSchemas(c) => {
            let mut builder = GetDbSchemasBuilder::new(None::<String>, None::<String>);
            let names = embedded_schemas(
                session,
                c.catalog.as_deref(),
                c.db_schema_filter_pattern.as_deref(),
            )
            .await?;
            for name in names {
                builder.append(EMBEDDED, name);
            }
            if selects(c.catalog.as_deref(), "") {
                let pattern = c.db_schema_filter_pattern.as_deref();
                // Hidden temporary views hide their schemas too.
                if let Some(views) = shown_temporary_views(session, pattern).await? {
                    let local = views.iter().any(|t| temporary_schema(t).is_empty());
                    if local && kept(pattern, "") {
                        builder.append("", "");
                    }
                    if kept(pattern, GLOBAL_TEMP) {
                        builder.append("", GLOBAL_TEMP);
                    }
                }
            }
            builder
                .build()
                .map_err(|e| arrow_internal("the GetDbSchemas result", &e))
        }
        Listing::Tables(c) => tables(session, c, max_schema_tables).await,
        Listing::TableTypes(c) => {
            let mut builder = (*c).into_builder();
            for t in TABLE_TYPES {
                builder.append(t);
            }
            builder
                .build()
                .map_err(|e| arrow_internal("the GetTableTypes result", &e))
        }
        Listing::PrimaryKeys(c) => {
            check(key_table(
                c.catalog.as_deref(),
                c.db_schema.as_deref(),
                &c.table,
                &current,
            ))
            .await?;
            Ok(empty())
        }
        Listing::ExportedKeys(c) => {
            check(key_table(
                c.catalog.as_deref(),
                c.db_schema.as_deref(),
                &c.table,
                &current,
            ))
            .await?;
            Ok(empty())
        }
        Listing::ImportedKeys(c) => {
            check(key_table(
                c.catalog.as_deref(),
                c.db_schema.as_deref(),
                &c.table,
                &current,
            ))
            .await?;
            Ok(empty())
        }
        Listing::CrossReference(c) => {
            check(key_table(
                c.pk_catalog.as_deref(),
                c.pk_db_schema.as_deref(),
                &c.pk_table,
                &current,
            ))
            .await?;
            check(key_table(
                c.fk_catalog.as_deref(),
                c.fk_db_schema.as_deref(),
                &c.fk_table,
                &current,
            ))
            .await?;
            Ok(empty())
        }
    }
}

impl FlightSqlServer {
    /// `GetFlightInfo` of a listing: its fixed schema and a ticket that is the command; no
    /// session.
    pub(crate) fn listing_info(
        listing: &Listing,
        descriptor: FlightDescriptor,
    ) -> Result<Response<FlightInfo>, Status> {
        fixed_info(&listing.schema(), &listing.as_any(), descriptor).map(Response::new)
    }

    /// `DoGet` of a listing (design §5): runs it in the call's session under
    /// `metadataTimeout`; a cookie-less one in a transient session (no `set-cookie`).
    pub(crate) async fn listing(
        &self,
        listing: Listing,
        metadata: &MetadataMap,
        caller: &Authenticated,
    ) -> Result<Response<DoGetStream>, Status> {
        let cookieless = matches!(session_cookie(metadata), SessionCookie::Absent);
        let mut call = self.call_session(metadata, caller)?;
        let entry = Arc::clone(&call.entry);
        let config = &self.inner.config;
        let limit = config.metadata_timeout;
        // The wait for the session's statement gate counts toward the limit too (§5).
        let run = async {
            let gate = entry.shared().await;
            let result = run_listing(entry.session(), &listing, config.max_schema_tables).await;
            drop(gate);
            result
        };
        let batch = match tokio::time::timeout(limit, run).await {
            Ok(result) => result.map_err(|e| flight_status(&e))?,
            Err(_) => {
                return Err(flight_status(
                    &KrishivFlightSqlMetadataTimeout {
                        command: listing.name().to_owned(),
                        limit: format!("{limit:?}"),
                    }
                    .into(),
                ));
            }
        };
        let mut response = Response::new(batch_stream(batch));
        if !cookieless {
            call.cookie_on(&mut response);
            call.kept();
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns_are_flight_sql_like_and_ignore_case() {
        let yes = [
            ("%", ""),
            ("%", "abc"),
            ("a%", "ABC"),
            ("%c", "abc"),
            ("a_c", "abc"),
            ("a%c", "ac"),
            ("a%b%c", "axxbyyc"),
            ("%b%", "abc"),
            ("t\\_1", "t_1"),
            ("t\\%", "t%"),
            ("ab\\", "ab\\"),
            ("ÄB", "äb"),
        ];
        for (p, v) in yes {
            assert!(pattern_matches(p, v), "{p} ~ {v}");
        }
        let no = [
            ("a", ""),
            ("_", ""),
            ("a_c", "ac"),
            ("t\\_1", "tx1"),
            ("t\\%", "tx"),
            ("a%c", "abd"),
            ("abc", "abcd"),
            ("%b", "abc"),
        ];
        for (p, v) in no {
            assert!(!pattern_matches(p, v), "{p} !~ {v}");
        }
    }

    #[test]
    fn a_catalog_filter_is_exact_and_empty_selects_no_catalog() {
        assert!(selects(None, EMBEDDED));
        assert!(selects(Some("SPARK_CATALOG"), EMBEDDED));
        assert!(!selects(Some("spark%"), EMBEDDED));
        assert!(!selects(Some(""), EMBEDDED));
        assert!(selects(Some(""), ""));
    }

    #[test]
    fn key_tables_quote_each_part() {
        assert_eq!(key_table(None, None, "t", "cur"), "`t`");
        assert_eq!(key_table(None, Some("d"), "t", "cur"), "`d`.`t`");
        assert_eq!(
            key_table(Some("spark_catalog"), Some("d.b"), "t`x", "cur"),
            "`spark_catalog`.`d.b`.`t``x`"
        );
        assert_eq!(key_table(Some(""), Some(""), "t", "cur"), "`t`");
        // A catalog alone is not read as a schema: the current database is named.
        assert_eq!(
            key_table(Some("spark_catalog"), None, "t", "cur"),
            "`spark_catalog`.`cur`.`t`"
        );
        assert_eq!(
            key_table(Some("spark_catalog"), Some(""), "t", "cur"),
            "`spark_catalog`.`cur`.`t`"
        );
    }

    #[test]
    fn exact_patterns_have_no_wildcard() {
        assert!(!is_exact(None));
        assert!(is_exact(Some("hr")));
        assert!(is_exact(Some(r"my\_db")));
        assert!(!is_exact(Some("h%")));
        assert!(!is_exact(Some("h_")));
    }

    #[test]
    fn key_schemas_follow_the_specification() {
        assert_eq!(primary_keys_schema().fields().len(), 6);
        let fk = foreign_keys_schema();
        assert_eq!(fk.fields().len(), 13);
        assert_eq!(fk.field(11).name(), "update_rule");
        assert_eq!(fk.field(11).data_type(), &DataType::UInt8);
    }
}
