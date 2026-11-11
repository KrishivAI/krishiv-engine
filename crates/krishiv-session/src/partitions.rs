// SPDX-License-Identifier: Apache-2.0
//! Partitioned and bucketed tables (T179 unit 5): the layout `CREATE TABLE` gives a table, the partitions its
//! catalog tracks, and the commands on them, as Spark 4.2 runs them for file-source tables in the session
//! catalog (`tracksPartitionsInCatalog`; checked against the goldens `describe`, `describe-part-after-analyze`
//! and `show-create-table`, and Spark's `PreprocessTableCreation`, `PartitioningUtils.normalizePartitionSpec`,
//! `AlterTableAddPartitionCommand`, `RepairTableCommand`, `AnalyzePartitionCommand` and `ShowPartitionsCommand`).
//!
//! - The partition columns are the table's last columns, in `PARTITIONED BY` order; a typed partition column
//!   (`PARTITIONED BY (p STRING)`) is added after the others.
//! - A partition is a spec (one string value per partition column, Spark's partition value text: the written
//!   value `CAST(... AS STRING)`) and a directory `<table>/<col>=<value>/...`, escaped as Spark escapes it.
//! - `INSERT` into a partitioned table registers the partitions found under the table's directory afterwards;
//!   a `CREATE TABLE ... AS SELECT` registers them with their file counts and sizes (`RepairTableCommand`).
//! - Bucketed tables keep their bucket specification as metadata; the engine writes no bucketed files, so a
//!   write to one fails with `KRISHIV_UNSUPPORTED_FEATURE` before anything is written.
//! - Reading a partitioned table's rows (`catalog.rs` `known_partitions`, `krishiv_exec::files::KnownPartitions`):
//!   its registered partitions at their default directories, values typed as declared (strings, integers and dates
//!   so far). A table with a partition registered elsewhere, a directory the catalog does not know, or another
//!   partition column type is `KRISHIV_UNSUPPORTED_FEATURE`, never read differently from Spark's
//!   `CatalogFileIndex`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use krishiv_common::error::{
    AllPartitionColumnsNotAllowed, ColumnNotDefinedInTable, ForbiddenOperation, InternalError,
    InvalidPartitionColumnDataType, InvalidPartitionOperationPartitionSchemaIsEmpty, KrishivError,
    KrishivUnsupportedFeature, LegacyErrorTemp1166, LegacyErrorTemp1167, LegacyErrorTemp1169,
    LegacyErrorTemp1232, LegacyErrorTemp1237, LegacyErrorTemp1251, LegacyErrorTemp2036,
    PartitionsAlreadyExist, PartitionsNotFound,
};
use krishiv_common::names::check_column_name_duplication;
use krishiv_plan::Literal;
use krishiv_plan::analyzer::{Privilege, Securable, authorize};
use krishiv_sql::StatementClass;
use krishiv_sql::ast::{AlterTableAction, BucketSpec, Expr, PartitionSpec, PartitionWithLocation};
use krishiv_types::DataType;

use crate::dataframe::DataFrame;
use crate::functions::col;
use crate::session::Session;
use crate::store::{
    BucketMeta, CatalogPartition, CatalogStore, CatalogTable, PartitionStats, StoreColumn,
    TableLayout,
};
use crate::table_commands::TableName;

fn unsupported(feature: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.to_owned(),
    }
    .into()
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// Spark's `quoteIdentifier`.
fn quote_identifier(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// Spark's `toSQLId` of a table's `quotedString` (`spark_catalog`.`ns`.`name`).
fn table_sql_id(t: &CatalogTable) -> String {
    format!(
        "{}.{}.{}",
        quote_identifier(crate::catalog::CATALOG),
        quote_identifier(&t.namespace),
        quote_identifier(&t.name)
    )
}

/// `PARTITION (`a` = 1, `b` = x)`, as `NoSuchPartitionException` and `PartitionsAlreadyExistException` print a
/// spec.
fn spec_text(spec: &[(String, String)]) -> String {
    format!(
        "PARTITION ({})",
        spec.iter()
            .map(|(k, v)| format!("{} = {v}", quote_identifier(k)))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// `db`.`table`, as the partition exceptions name a table.
fn db_table(t: &CatalogTable) -> String {
    format!(
        "{}.{}",
        quote_identifier(&t.namespace),
        quote_identifier(&t.name)
    )
}

/// Spark's `escapePathName` (the partition directory escaping).
pub(crate) fn escape_path_name(s: &str) -> String {
    krishiv_io::partition::escape_partition_value(s)
}

/// Spark's `unescapePathName`: `%XX` (two hex digits) becomes that character.
pub(crate) fn unescape_path_name(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '%'
            && i + 2 < chars.len()
            && let (Some(h), Some(l)) = (chars[i + 1].to_digit(16), chars[i + 2].to_digit(16))
            && let Some(c) = char::from_u32(h * 16 + l)
        {
            out.push(c);
            i += 3;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// The directory of the partition `spec` below the table directory `root` (`ExternalCatalogUtils
/// .generatePartitionPath`): `<col>=<value>` per column, NULL and empty values as the default partition.
pub(crate) fn partition_path(root: &Path, spec: &[(String, String)]) -> PathBuf {
    let mut path = root.to_path_buf();
    for (k, v) in spec {
        path.push(format!(
            "{}={}",
            escape_path_name(k),
            krishiv_io::partition::partition_segment(Some(v))
        ));
    }
    path
}

/// Moves a partition directory to `target`, creating its parents.
fn move_partition_dir(source: &Path, target: &Path) -> Result<(), KrishivError> {
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| internal(format!("cannot create {}: {e}", parent.display())))?;
    }
    std::fs::rename(source, target).map_err(|e| {
        internal(format!(
            "cannot rename {} to {}: {e}",
            source.display(),
            target.display()
        ))
    })
}

/// `col=value/...` of a partition (Spark's partition name, as `SHOW PARTITIONS` lists it).
fn partition_name(spec: &[(String, String)]) -> String {
    spec.iter()
        .map(|(k, v)| format!("{}={}", escape_path_name(k), escape_path_name(v)))
        .collect::<Vec<_>>()
        .join("/")
}

fn same_name(a: &str, b: &str, case_sensitive: bool) -> bool {
    if case_sensitive {
        a == b
    } else {
        a.eq_ignore_ascii_case(b)
    }
}

/// Spark's `canPartitionOn`: atomic types only.
fn can_partition_on(t: &DataType) -> bool {
    !matches!(
        t,
        DataType::Array(_) | DataType::Map(_) | DataType::Struct(_)
    )
}

/// `COLUMN_ALREADY_EXISTS` when `names` repeat one (`SchemaUtils.checkColumnNameDuplication`).
fn check_duplicates(names: &[String], case_sensitive: bool) -> Result<(), KrishivError> {
    check_column_name_duplication(names.iter().map(String::as_str), case_sensitive)
}

/// The table's layout from `PARTITIONED BY` names and `CLUSTERED BY`, checked against `columns`, which are
/// reordered so that the partition columns come last (`PreprocessTableCreation.normalizeCatalogTable`).
pub(crate) fn table_layout(
    table: &TableName,
    columns: &mut Vec<StoreColumn>,
    partition_by: &[String],
    bucket: Option<&BucketSpec>,
    case_sensitive: bool,
) -> Result<TableLayout, KrishivError> {
    let table_cols = || {
        columns
            .iter()
            .map(|c| quote_identifier(&c.name))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let normalize = |name: &str, kind: &str| -> Result<String, KrishivError> {
        columns
            .iter()
            .find(|c| same_name(&c.name, name, case_sensitive))
            .map(|c| c.name.clone())
            .ok_or_else(|| {
                ColumnNotDefinedInTable {
                    col_type: kind.to_owned(),
                    col_name: quote_identifier(name),
                    table_name: table.quoted(),
                    table_cols: table_cols(),
                }
                .into()
            })
    };
    let partition_columns = partition_by
        .iter()
        .map(|p| normalize(p, "partition"))
        .collect::<Result<Vec<_>, _>>()?;
    check_duplicates(&partition_columns, case_sensitive)?;
    if !columns.is_empty() && partition_columns.len() == columns.len() {
        return Err(AllPartitionColumnsNotAllowed {}.into());
    }
    for c in columns
        .iter()
        .filter(|c| partition_columns.contains(&c.name))
    {
        if !can_partition_on(&c.data_type) {
            return Err(InvalidPartitionColumnDataType {
                r#type: format!("\"{}\"", c.data_type.sql()),
            }
            .into());
        }
    }
    let bucket = match bucket {
        None => None,
        Some(spec) => {
            let bucket_columns = spec
                .clustered_by
                .iter()
                .map(|c| normalize(&c.value, "bucket"))
                .collect::<Result<Vec<_>, _>>()?;
            let sort_columns = spec
                .sorted_by
                .iter()
                .map(|c| normalize(&c.name.value, "sort"))
                .collect::<Result<Vec<_>, _>>()?;
            check_duplicates(&bucket_columns, case_sensitive)?;
            check_duplicates(&sort_columns, case_sensitive)?;
            if let Some(c) = bucket_columns
                .iter()
                .find(|c| partition_columns.contains(c))
            {
                return Err(LegacyErrorTemp1166 {
                    bucket_col: c.clone(),
                    normalized_part_cols: partition_columns.join(", "),
                }
                .into());
            }
            if let Some(c) = sort_columns.iter().find(|c| partition_columns.contains(c)) {
                return Err(LegacyErrorTemp1167 {
                    sort_col: c.clone(),
                    normalized_part_cols: partition_columns.join(", "),
                }
                .into());
            }
            Some(BucketMeta {
                num_buckets: spec.buckets,
                columns: bucket_columns,
                sort_columns,
            })
        }
    };
    let (mut data, mut parts): (Vec<StoreColumn>, Vec<StoreColumn>) = columns
        .drain(..)
        .partition(|c| !partition_columns.contains(&c.name));
    parts.sort_by_key(|c| partition_columns.iter().position(|p| *p == c.name));
    data.append(&mut parts);
    *columns = data;
    Ok(TableLayout {
        partition_columns,
        bucket,
        partitions: Vec::new(),
    })
}

/// One key of a written `PARTITION (...)` with its value text (`None` for a bare key).
pub(crate) struct WrittenPart {
    pub(crate) key: String,
    pub(crate) value: Option<String>,
}

/// Normalizes a written spec against the table's partition columns (`PartitioningUtils
/// .normalizePartitionSpec`): an unknown key is `PARTITIONS_NOT_FOUND`, a repeated one `COLUMN_ALREADY_EXISTS`.
/// Keys come back spelled as the table spells them, in written order.
fn normalize_spec(
    table: &CatalogTable,
    written: &[WrittenPart],
    case_sensitive: bool,
) -> Result<Vec<(String, Option<String>)>, KrishivError> {
    let mut out = Vec::with_capacity(written.len());
    for p in written {
        let Some(column) = table
            .layout
            .partition_columns
            .iter()
            .find(|c| same_name(c, &p.key, case_sensitive))
        else {
            return Err(PartitionsNotFound {
                partition_list: quote_identifier(&p.key),
                table_name: table_sql_id(table),
            }
            .into());
        };
        out.push((column.clone(), p.value.clone()));
    }
    let names: Vec<String> = out.iter().map(|(k, _)| k.clone()).collect();
    check_duplicates(&names, case_sensitive)?;
    Ok(out)
}

/// The spec `SessionCatalog.requireExactMatchedPartitionSpec` accepts: every partition column, no other
/// (`_LEGACY_ERROR_TEMP_1232`), in written order.
fn exact_spec(
    table: &CatalogTable,
    spec: &[(String, Option<String>)],
) -> Result<Vec<(String, String)>, KrishivError> {
    let complete = spec.len() == table.layout.partition_columns.len()
        && table
            .layout
            .partition_columns
            .iter()
            .all(|c| spec.iter().any(|(k, _)| k == c));
    let values: Option<Vec<(String, String)>> = spec
        .iter()
        .map(|(k, v)| v.clone().map(|v| (k.clone(), v)))
        .collect();
    match values {
        Some(values) if complete => Ok(values),
        _ => Err(LegacyErrorTemp1232 {
            spec_keys: spec
                .iter()
                .map(|(k, _)| k.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            partition_column_names: table.layout.partition_columns.join(", "),
            table_name: table_sql_id(table),
        }
        .into()),
    }
}

/// The `col=value/...` directories `depth` levels below `dir` (hidden and `_` entries are skipped, as the
/// partition listing of `RepairTableCommand` does), as paths relative to the table location.
fn list_partition_dirs(
    dir: &std::path::Path,
    depth: usize,
    prefix: &str,
    out: &mut Vec<String>,
) -> Result<(), KrishivError> {
    if depth == 0 {
        if !prefix.is_empty() {
            out.push(prefix.to_owned());
        }
        return Ok(());
    }
    let listing = |e: std::io::Error| internal(format!("cannot list {}: {e}", dir.display()));
    let mut names: Vec<(String, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(listing)? {
        let entry = entry.map_err(listing)?;
        // Spark lists the local file system through Hadoop, which follows a symbolic link to a directory.
        if !std::fs::metadata(entry.path()).map_err(listing)?.is_dir() {
            continue;
        }
        let Ok(n) = entry.file_name().into_string() else {
            continue;
        };
        if n.contains('=') && !n.starts_with(['.', '_']) {
            names.push((n, entry.path()));
        }
    }
    names.sort();
    for (n, path) in names {
        let next = if prefix.is_empty() {
            n
        } else {
            format!("{prefix}/{n}")
        };
        list_partition_dirs(&path, depth - 1, &next, out)?;
    }
    Ok(())
}

/// A partition `LOCATION` as the catalog stores it: qualified as the statement's access check qualified it
/// (`Session::qualify_path`: Spark's `makeQualified`, a relative path against the working directory), so the
/// directory the command probes is the one the policy was asked about. An empty location fails as
/// `CREATE TABLE ... LOCATION ''` does (Spark 4.2.0 throws Hadoop's unclassified `IllegalArgumentException`
/// "Can not create a Path from an empty string" here); a location outside the local file system is not supported.
fn qualified(session: &Session, location: &str) -> Result<PathBuf, KrishivError> {
    if location
        .strip_prefix("file:")
        .unwrap_or(location)
        .is_empty()
    {
        return Err(krishiv_common::error::InvalidEmptyLocation {
            location: location.to_owned(),
        }
        .into());
    }
    let display = session.qualify_path(location)?;
    match display.strip_prefix("file:") {
        Some(path) => Ok(PathBuf::from(path)),
        None => Err(unsupported(
            "partition locations outside the local file system",
        )),
    }
}

/// `spec` in partition-column order.
fn ordered(table: &CatalogTable, spec: &[(String, String)]) -> Vec<(String, String)> {
    table
        .layout
        .partition_columns
        .iter()
        .filter_map(|c| spec.iter().find(|(k, _)| k == c).cloned())
        .collect()
}

/// The partition of `table` with the exact `spec`, if the catalog has it.
fn find_partition<'a>(
    table: &'a CatalogTable,
    spec: &[(String, String)],
) -> Option<&'a CatalogPartition> {
    table
        .layout
        .partitions
        .iter()
        .find(|p| spec.iter().all(|kv| p.spec.contains(kv)))
}

/// What the catalog swap of a dropped or renamed partition did.
enum Swap {
    Done,
    /// The entry kept changing under it, or (a rename) its partition changed.
    Lost,
    /// The table was dropped, replaced or is no longer partitioned.
    Gone,
}

/// `_LEGACY_ERROR_TEMP_1251` unless the table's partitions are in the catalog (`DDLUtils
/// .verifyPartitionProviderIsHive`): only partitioned tables track them.
fn require_partition_provider(table: &CatalogTable, action: &str) -> Result<(), KrishivError> {
    if table.is_partitioned() {
        Ok(())
    } else {
        Err(LegacyErrorTemp1251 {
            action: action.to_owned(),
            table_name: table.name.clone(),
        }
        .into())
    }
}

/// Data files of a partition directory as `RepairTableCommand` counts them: not `_SUCCESS`, not
/// `_temporary`, not hidden.
fn listed_files(dir: &Path) -> Vec<(String, u64)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<(String, u64)> = entries
        .filter_map(Result::ok)
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if name == "_SUCCESS" || name == "_temporary" || name.starts_with('.') {
                return None;
            }
            let len = e.metadata().map(|m| m.len()).unwrap_or(0);
            Some((name, len))
        })
        .collect();
    out.sort();
    out
}

/// Whether `dir` or a directory below it holds a file, skipping names that start with `_` or `.` as
/// [`location_size`] does.
fn holds_files(dir: &Path) -> Result<bool, KrishivError> {
    let listing = |e: std::io::Error| internal(format!("cannot list {}: {e}", dir.display()));
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(listing(e)),
    };
    for entry in entries {
        let entry = entry.map_err(listing)?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('_') || name.starts_with('.') {
            continue;
        }
        if !entry.file_type().map_err(listing)?.is_dir() || holds_files(&entry.path())? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The size of the data under `dir` as `CommandUtils.calculateSingleLocationSize` sums it: files and
/// directories whose names start with `_` or `.` are skipped.
fn location_size(dir: &Path) -> i128 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            !name.starts_with('_') && !name.starts_with('.')
        })
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => location_size(&e.path()),
            _ => e.metadata().map_or(0, |m| i128::from(m.len())),
        })
        .sum()
}

/// The spec of the partition directory `dir` (`col=value/...` below the table directory, escaped), when its
/// segments name the partition `columns` in order.
fn spec_of_dir(
    dir: &str,
    columns: &[String],
    case_sensitive: bool,
) -> Option<Vec<(String, String)>> {
    let segments: Vec<&str> = dir.split('/').collect();
    if segments.len() != columns.len() {
        return None;
    }
    segments
        .iter()
        .zip(columns)
        .map(|(segment, column)| {
            let (name, value) = segment.split_once('=')?;
            same_name(&unescape_path_name(name), column, case_sensitive)
                .then(|| (column.clone(), unescape_path_name(value)))
        })
        .collect()
}

impl Session {
    /// The text of each value of a written `PARTITION (...)` (Spark's `visitStringConstant`:
    /// `CAST(value AS STRING)`; NULL stays NULL).
    pub(crate) async fn written_spec(
        &self,
        spec: &PartitionSpec,
        original: &str,
    ) -> Result<Vec<WrittenPart>, KrishivError> {
        let mut out = Vec::with_capacity(spec.parts.len());
        for p in &spec.parts {
            let value = match &p.value {
                None => None,
                Some(e) => Some(
                    self.partition_value(e, original)
                        .await?
                        .unwrap_or_else(|| krishiv_io::partition::DEFAULT_PARTITION.to_owned()),
                ),
            };
            out.push(WrittenPart {
                key: p.key.value.clone(),
                value,
            });
        }
        Ok(out)
    }

    async fn partition_value(
        &self,
        e: &Expr,
        original: &str,
    ) -> Result<Option<String>, KrishivError> {
        let text = original
            .get(e.span.start..e.span.end)
            .ok_or_else(|| internal("a partition value outside its statement".to_owned()))?;
        let sql = format!("SELECT CAST(({text}) AS STRING)");
        let analysis = self.analyze_variable_query(&sql).await?;
        let batches = self.run_analyzed_variable_query(&analysis).await?;
        let batch = batches
            .iter()
            .find(|b| b.num_rows() > 0)
            .ok_or_else(|| internal("a partition value returned no row".to_owned()))?;
        match crate::command::arrow_value(batch.column(0), 0)?.1 {
            Literal::String(s) => Ok(Some(s)),
            Literal::Null(_) => Ok(None),
            other => Err(internal(format!("a partition value read as {other:?}"))),
        }
    }

    /// The static partition values of `INSERT ... PARTITION (...)`, keyed by the table's partition columns;
    /// keys without a value are dynamic partition columns the query provides.
    pub(crate) async fn insert_partition_values(
        &self,
        table: &CatalogTable,
        spec: &PartitionSpec,
        original: &str,
    ) -> Result<Vec<(String, String)>, KrishivError> {
        let case_sensitive = self.query_env().analysis.conf.case_sensitive;
        let written = self.written_spec(spec, original).await?;
        let normalized = normalize_spec(table, &written, case_sensitive)?;
        // `PreprocessTableInsertion`: a written spec names every partition column (a dynamic one by key alone).
        if !normalized.is_empty() && normalized.len() != table.layout.partition_columns.len() {
            return Err(LegacyErrorTemp1169 {
                table_name: table_sql_id(table),
                normalized_part_spec: normalized
                    .iter()
                    .map(|(k, _)| k.as_str())
                    .collect::<Vec<_>>()
                    .join(","),
                part_col_names: table.layout.partition_columns.join(","),
            }
            .into());
        }
        let statics: Vec<(String, String)> = normalized
            .into_iter()
            .filter_map(|(k, v)| v.map(|v| (k, v)))
            .collect();
        // The static values are cast to their columns' types as an ADD PARTITION's are (Spark 4.2.0:
        // `PARTITION (p='007')` writes `p=7`, `'abc'` is `CAST_INVALID_INPUT`).
        self.typed_partition_spec(table, &statics).await
    }

    /// `MSCK REPAIR TABLE name [ADD | DROP | SYNC PARTITIONS]` (`RepairTableCommand`): `DROP` (and `SYNC`, first)
    /// forget the partitions whose directory is gone; `ADD` (the default, and `SYNC`) registers the
    /// `col=value/...` directories of the table's location that the catalog does not know, with their file
    /// statistics.
    pub(crate) async fn exec_repair(
        &self,
        r: &krishiv_sql::ast::Repair,
    ) -> Result<DataFrame, KrishivError> {
        use krishiv_sql::ast::RepairAction;
        let env = self.query_env();
        let name = self.table_name(&r.table, &env)?;
        let table = self.table_for_write(&name, "MSCK REPAIR TABLE", &env)?;
        // `RepairTableCommand` asks for partition columns before the partition provider.
        if table.layout.partition_columns.is_empty() {
            return Err(krishiv_common::error::NotAPartitionedTable {
                operation: "MSCK REPAIR TABLE".to_owned(),
                table_ident_with_d_b: table_sql_id(&table),
            }
            .into());
        }
        require_partition_provider(&table, "MSCK REPAIR TABLE")?;
        // Its log entries and registrations do not interleave with a rename's or a drop's.
        let _ddl = self
            .state
            .catalog
            .store
            .partition_ddl(&table.location)
            .await;
        let table = self.fresh_entry(&table, &name)?;
        if matches!(r.action, Some(RepairAction::Drop | RepairAction::Sync)) {
            // A logged root still lists the files of a directory that is gone: they leave the log too, or every
            // read of the table would see them missing.
            // The log changes first: a failure leaves the catalog as it was, and a rerun finds the same partitions.
            let gone: Vec<(PathBuf, Option<PathBuf>)> = table
                .layout
                .partitions
                .iter()
                .filter(|p| !p.location.exists())
                .map(|p| (p.location.clone(), None))
                .collect();
            self.relog_partitions(&table.location, &gone).await?;
            self.forget_missing_partitions(&name);
        }
        if matches!(r.action, Some(RepairAction::Drop)) {
            return self.empty_frame().await;
        }
        // Spark 4.2.0 fails on a missing table directory (an unclassified `FileNotFoundException`) rather than
        // registering nothing.
        if !table.location.is_dir() {
            return Err(krishiv_common::error::PathNotFound {
                path: format!("file:{}", table.location.display()),
            }
            .into());
        }
        let mut dirs = Vec::new();
        list_partition_dirs(
            &table.location,
            table.layout.partition_columns.len(),
            "",
            &mut dirs,
        )?;
        // Each partition is registered at its directory: the same check `ADD PARTITION` makes of a location.
        for dir in &dirs {
            self.authorize_dir(&table.location.join(dir)).await?;
        }
        self.register_partitions(&name, &dirs, true);
        self.empty_frame().await
    }

    /// The values of an `ALTER TABLE` partition spec: cast as [`Self::typed_partition_spec`] casts them, unless
    /// `spark.sql.legacy.skipTypeValidationOnAlterPartition` keeps them as written (Spark 4.2.0: `'007'` stays
    /// `007`, `'abc'` for an INT is accepted).
    async fn alter_partition_spec(
        &self,
        table: &CatalogTable,
        spec: &[(String, String)],
    ) -> Result<Vec<(String, String)>, KrishivError> {
        use krishiv_common::config::{ConfigValue, LEGACY_SKIP_TYPE_VALIDATION_ON_ALTER_PARTITION};
        let skip = self
            .query_env()
            .config
            .get(&LEGACY_SKIP_TYPE_VALIDATION_ON_ALTER_PARTITION)
            .and_then(ConfigValue::as_bool)
            .unwrap_or(false);
        if skip {
            Ok(spec.to_vec())
        } else {
            self.typed_partition_spec(table, spec).await
        }
    }

    /// Spark's `normalizePartitionSpec` for typed partition columns (`PartitioningUtils
    /// .normalizePartitionStringValue`, v4.2.0): the value is `CAST(CAST(v AS t) AS STRING)`, so `'007'` for an
    /// INT is `7`. The inner cast is ANSI unless `spark.sql.storeAssignmentPolicy` is `LEGACY`
    /// (`castPartitionSpec`), whatever `spark.sql.ansi.enabled` says: `'abc'` is `CAST_INVALID_INPUT`, and a value a
    /// LEGACY cast makes NULL is the default partition. String and CHAR/VARCHAR columns and
    /// `__HIVE_DEFAULT_PARTITION__` keep their text.
    async fn typed_partition_spec(
        &self,
        table: &CatalogTable,
        spec: &[(String, String)],
    ) -> Result<Vec<(String, String)>, KrishivError> {
        use krishiv_common::config::{ConfigValue, STORE_ASSIGNMENT_POLICY};
        let mut out = Vec::with_capacity(spec.len());
        for (key, value) in spec {
            let typed = table.columns.iter().find(|c| &c.name == key).filter(|c| {
                !matches!(c.data_type, DataType::String)
                    && c.char_varchar.is_none()
                    && value != krishiv_io::partition::DEFAULT_PARTITION
            });
            let Some(column) = typed else {
                out.push((key.clone(), value.clone()));
                continue;
            };
            let mut env = self.query_env();
            let ansi = !env
                .config
                .get(&STORE_ASSIGNMENT_POLICY)
                .and_then(ConfigValue::as_str)
                .is_some_and(|p| p.eq_ignore_ascii_case("LEGACY"));
            let mut config = (*env.config).clone();
            config.set(
                "spark.sql.ansi.enabled",
                if ansi { "true" } else { "false" },
            )?;
            env.config = Arc::new(config);
            env.analysis.conf.ansi = ansi;
            let sql = format!(
                "SELECT CAST(CAST('{}' AS {}) AS STRING)",
                value.replace('\\', "\\\\").replace('\'', "\\'"),
                column.data_type.simple_string()
            );
            let analysis = self.analyze_variable_query_in(&sql, env).await?;
            let batches = self.run_analyzed_variable_query(&analysis).await?;
            let batch = batches
                .iter()
                .find(|b| b.num_rows() > 0)
                .ok_or_else(|| internal("a partition value returned no row".to_owned()))?;
            let text = match crate::command::arrow_value(batch.column(0), 0)?.1 {
                Literal::String(s) => s,
                Literal::Null(_) => krishiv_io::partition::DEFAULT_PARTITION.to_owned(),
                other => return Err(internal(format!("a partition value read as {other:?}"))),
            };
            out.push((key.clone(), text));
        }
        Ok(out)
    }

    /// Follows partition directories of the table root `root` that move (`(from, Some(to))`) or are gone
    /// (`(from, None)`) in the root's commit log, when it has one: one overwrite entry removes the files below each
    /// `from` and adds them below its `to` with the new partition values. Paths are relative to the root, as a
    /// write records them; a directory outside the root is not in its log. An overwrite loses to any commit made
    /// after its snapshot (`KRISHIV_CONCURRENT_COMMIT`); the entry is derived from the snapshot, so it is derived
    /// again from a fresh one and retried. A move whose files the log no longer lists below `from` (a rerun after
    /// the log moved) commits nothing. Returns the records the committed entry removed (none without a log).
    async fn relog_partitions(
        &self,
        root: &Path,
        moves: &[(PathBuf, Option<PathBuf>)],
    ) -> Result<Vec<krishiv_io::commit_log::AddedFile>, KrishivError> {
        use krishiv_common::error::ErrorClass;
        use krishiv_io::commit_log::{AddedFile, CommitLog, CommitMode, LogEntry};
        const RACES: u32 = 32;
        let env = self.query_env();
        let locate = |p: &Path| {
            self.locate_write_path(&format!("file:{}", p.display()), &BTreeMap::new(), &env)
        };
        let at = locate(root)?;
        let below = |p: &Path| -> Result<Option<String>, KrishivError> {
            let located = locate(p)?;
            Ok(located.path.prefix_match(&at.path).map(|parts| {
                let mut rel = parts
                    .map(|part| part.as_ref().to_owned())
                    .collect::<Vec<_>>()
                    .join("/");
                rel.push('/');
                rel
            }))
        };
        let mut prefixes: Vec<(String, Option<String>)> = Vec::with_capacity(moves.len());
        for (from, to) in moves {
            let Some(old) = below(from)? else {
                continue;
            };
            let new = match to {
                Some(to) => below(to)?,
                None => None,
            };
            prefixes.push((old, new));
        }
        if prefixes.is_empty() {
            return Ok(Vec::new());
        }
        let log = CommitLog::new(
            &env.io,
            at.store.clone(),
            at.path.clone(),
            at.display.clone(),
        );
        if !log.exists().await? {
            return Ok(Vec::new());
        }
        let mut races = 0;
        loop {
            let snapshot = log.snapshot().await?;
            let mut remove = Vec::new();
            let mut removed = Vec::new();
            let mut add = Vec::new();
            for (old, new) in &prefixes {
                for (path, file) in snapshot.files.range(old.clone()..) {
                    let Some(rest) = path.strip_prefix(old.as_str()) else {
                        break;
                    };
                    remove.push(path.clone());
                    removed.push(file.clone());
                    if let Some(new) = new {
                        let moved = format!("{new}{rest}");
                        let mut dirs: Vec<&str> = moved.split('/').collect();
                        dirs.pop();
                        add.push(AddedFile {
                            partition: dirs
                                .into_iter()
                                .filter_map(krishiv_io::parse_segment)
                                .collect(),
                            path: moved.clone(),
                            ..file.clone()
                        });
                    }
                }
            }
            if remove.is_empty() {
                return Ok(Vec::new());
            }
            let job = crate::session::job_uuid();
            let entry = LogEntry {
                query_id: job.clone(),
                write_id: job.clone(),
                job_uuid: job,
                mode: CommitMode::Overwrite,
                add,
                remove,
            };
            match log.commit(snapshot.version, &entry).await {
                Ok(_) => return Ok(removed),
                Err(e)
                    if e.error_class() == ErrorClass::KrishivConcurrentCommit && races < RACES =>
                {
                    races += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Puts `files` (records [`Self::relog_partitions`] removed) back in the root's commit log in one append entry.
    async fn unremove_files(
        &self,
        root: &Path,
        files: Vec<krishiv_io::commit_log::AddedFile>,
    ) -> Result<(), KrishivError> {
        use krishiv_common::error::ErrorClass;
        use krishiv_io::commit_log::{CommitLog, CommitMode, LogEntry};
        const RACES: u32 = 32;
        if files.is_empty() {
            return Ok(());
        }
        let env = self.query_env();
        let at =
            self.locate_write_path(&format!("file:{}", root.display()), &BTreeMap::new(), &env)?;
        let log = CommitLog::new(
            &env.io,
            at.store.clone(),
            at.path.clone(),
            at.display.clone(),
        );
        let job = crate::session::job_uuid();
        let entry = LogEntry {
            query_id: job.clone(),
            write_id: job.clone(),
            job_uuid: job,
            mode: CommitMode::Append,
            add: files,
            remove: Vec::new(),
        };
        let mut races = 0;
        loop {
            let snapshot = log.snapshot().await?;
            match log.commit(snapshot.version, &entry).await {
                Ok(_) => return Ok(()),
                Err(e)
                    if e.error_class() == ErrorClass::KrishivConcurrentCommit && races < RACES =>
                {
                    races += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Drops from the catalog the partitions of `name` whose directory does not exist (`dropPartitions` with
    /// `retainData`: no file is touched). The swap can lose to a concurrent change: it is read and merged again.
    fn forget_missing_partitions(&self, name: &TableName) {
        let store = &self.state.catalog.store;
        loop {
            let Some(table) = store.table(&name.namespace, &name.name) else {
                return;
            };
            let mut changed = (*table).clone();
            changed.layout.partitions.retain(|p| p.location.exists());
            if changed.layout.partitions.len() == table.layout.partitions.len()
                || store.replace_table(&table, changed)
            {
                return;
            }
        }
    }

    /// `ALTER TABLE name PARTITION (...) SET LOCATION '...'`: only the catalog entry changes.
    pub(crate) async fn set_partition_location(
        &self,
        table: &Arc<CatalogTable>,
        partition: &PartitionSpec,
        location: &str,
        original: &str,
    ) -> Result<CatalogTable, KrishivError> {
        require_partition_provider(table, "ALTER TABLE SET LOCATION")?;
        let case_sensitive = self.query_env().analysis.conf.case_sensitive;
        let written = self.written_spec(partition, original).await?;
        // Cast as ADD and RENAME cast (`'007'` finds `p=7`), in the order written: `PARTITIONS_NOT_FOUND` lists
        // the spec as written (Spark 3.5.9-4.2.0 agree, krishiv-internal
        // `research/spark/partition-followups/probe4.py`, `probe5.py`).
        let spec = exact_spec(table, &normalize_spec(table, &written, case_sensitive)?)?;
        let spec = self.alter_partition_spec(table, &spec).await?;
        let mut changed = (**table).clone();
        let path = qualified(self, location)?;
        // Reads and writes of the table follow the table root, not a partition's own location, so a location
        // that already holds data would be reported as set and then ignored.
        if path.is_file() || holds_files(&path)? {
            return Err(unsupported(
                "ALTER TABLE ... PARTITION ... SET LOCATION onto existing files",
            ));
        }
        let Some(found) = changed
            .layout
            .partitions
            .iter_mut()
            .find(|p| spec.iter().all(|kv| p.spec.contains(kv)))
        else {
            return Err(PartitionsNotFound {
                partition_list: spec_text(&spec),
                table_name: db_table(table),
            }
            .into());
        };
        found.location = path;
        changed.column_stats.clear();
        Ok(changed)
    }

    /// `ALTER TABLE name PARTITION (...) RENAME TO PARTITION (...)` (`AlterTableRenamePartitionCommand`): both specs
    /// name every partition column (`_LEGACY_ERROR_TEMP_1232`); the old partition must exist and the new one must
    /// not. A managed table's partition directory moves to the new spec's path, as `InMemoryCatalog
    /// .renamePartitions` does, then the catalog entry is swapped against a fresh copy each try (a concurrent
    /// change of another part of the entry is kept); when the swap cannot happen, the directory and the log go
    /// back and the statement fails. Column statistics stay, as in Spark 4.2.0 (krishiv-internal
    /// `research/spark/partition-followups/probe6.py`).
    pub(crate) async fn rename_partition(
        &self,
        table: &Arc<CatalogTable>,
        name: &crate::table_commands::TableName,
        specs: [&PartitionSpec; 2],
        original: &str,
    ) -> Result<(), KrishivError> {
        // One partition command that moves or deletes directories at a time; the entry is read again under it.
        let _ddl = self
            .state
            .catalog
            .store
            .partition_ddl(&table.location)
            .await;
        let fresh = self.fresh_entry(table, name)?;
        let table = &fresh;
        require_partition_provider(table, "ALTER TABLE RENAME PARTITION")?;
        let case_sensitive = self.query_env().analysis.conf.case_sensitive;
        let mut exact = Vec::with_capacity(2);
        for spec in specs {
            let written = self.written_spec(spec, original).await?;
            // In the order written: both errors below list a spec as written (Spark 4.2.0,
            // krishiv-internal `research/spark/partition-followups/probe5.py`).
            let spec = exact_spec(table, &normalize_spec(table, &written, case_sensitive)?)?;
            exact.push(self.alter_partition_spec(table, &spec).await?);
        }
        let [from, to] = exact.as_slice() else {
            return Err(internal("two partition specs expected".to_owned()));
        };
        if find_partition(table, from).is_none() {
            return Err(PartitionsNotFound {
                partition_list: spec_text(from),
                table_name: db_table(table),
            }
            .into());
        }
        if find_partition(table, to).is_some() {
            return Err(PartitionsAlreadyExist {
                partition_list: spec_text(to),
                table_name: db_table(table),
            }
            .into());
        }
        // Reads follow the table root, so an external table's data could not be read under a new spec with its
        // directory left where it is (Spark keeps reading the old directory).
        if !table.managed {
            return Err(unsupported(
                "ALTER TABLE ... RENAME PARTITION of an external table",
            ));
        }
        // The catalog and the directory name keep the partition columns' order.
        let to = &ordered(table, to);
        let target = partition_path(&table.location, to);
        self.authorize_dir(&target).await?;
        let source = find_partition(table, from)
            .ok_or_else(|| internal("the partition was found above".to_owned()))?
            .location
            .clone();
        // The log moves first, then the directory: a lost race or a failed commit changes nothing, and a crash in
        // between leaves the log one rename ahead of the files, which a rerun of the statement completes (the
        // log then lists nothing below the old directory). A failed move puts the log back.
        self.relog_partitions(&table.location, &[(source.clone(), Some(target.clone()))])
            .await?;
        if let Err(e) = move_partition_dir(&source, &target) {
            if let Err(undo) = self
                .relog_partitions(&table.location, &[(target.clone(), Some(source.clone()))])
                .await
            {
                tracing::warn!(error = %undo, "the commit log of a failed partition rename could not be restored");
                return Err(undo);
            }
            return Err(e);
        }
        match self.swap_renamed(table, name, from, &source, to, &target) {
            Swap::Done => return Ok(()),
            // Dropped or replaced: its root (and any new table's log there) is left alone, as DROP PARTITION does.
            Swap::Gone => return Err(crate::table_commands::concurrent_change(name)),
            Swap::Lost => {}
        }
        // The partition changed under the rename (gone, moved or its new spec taken): the directory and the log go
        // back, so the catalog, the log and the files agree again.
        self.state
            .cache
            .invalidate_table(&format!("{}.{}", table.namespace, table.name));
        if let Err(e) = move_partition_dir(&target, &source) {
            tracing::warn!(error = %e, "a partition rename that lost its catalog swap could not move its directory back");
            return Err(e);
        }
        // A cancel or an error here leaves the log one rename ahead of the files; a rerun of the statement completes
        // it (as after a crash between the log and the move, above).
        self.relog_partitions(&table.location, &[(target, Some(source))])
            .await?;
        // Again after the undo, so no query that ran in between keeps the half-undone state cached.
        self.state
            .cache
            .invalidate_table(&format!("{}.{}", table.namespace, table.name));
        Err(crate::table_commands::concurrent_change(name))
    }

    /// `table`'s entry as the catalog holds it now; `concurrent_change` when it was dropped or replaced since
    /// the statement read it.
    fn fresh_entry(
        &self,
        table: &CatalogTable,
        name: &crate::table_commands::TableName,
    ) -> Result<Arc<CatalogTable>, KrishivError> {
        self.state
            .catalog
            .store
            .table(&name.namespace, &name.name)
            .filter(|t| t.created_ms == table.created_ms && t.location == table.location)
            .ok_or_else(|| crate::table_commands::concurrent_change(name))
    }

    /// Swaps `name`'s catalog entry for one where the partition `from` at `source` is `to` at `target`, against a
    /// fresh entry each try, as long as the entry is still `table`'s ([`Swap::Gone`] otherwise) and still holds
    /// `from` and not `to` ([`Swap::Lost`] otherwise).
    fn swap_renamed(
        &self,
        table: &CatalogTable,
        name: &crate::table_commands::TableName,
        from: &[(String, String)],
        source: &Path,
        to: &[(String, String)],
        target: &Path,
    ) -> Swap {
        const RACES: u32 = 32;
        let store = &self.state.catalog.store;
        for _ in 0..RACES {
            let Some(current) = store.table(&name.namespace, &name.name) else {
                return Swap::Gone;
            };
            if current.created_ms != table.created_ms || current.location != table.location {
                return Swap::Gone;
            }
            if find_partition(&current, to).is_some() {
                return Swap::Lost;
            }
            let mut changed = (*current).clone();
            let Some(found) = changed
                .layout
                .partitions
                .iter_mut()
                .find(|p| from.iter().all(|kv| p.spec.contains(kv)) && p.location == source)
            else {
                return Swap::Lost;
            };
            found.spec = to.to_vec();
            found.location = target.to_path_buf();
            if store.replace_table(&current, changed) {
                return Swap::Done;
            }
        }
        Swap::Lost
    }

    /// The partition actions of `ALTER TABLE`: the catalog entry to swap in.
    pub(crate) async fn alter_partitions(
        &self,
        table: &Arc<CatalogTable>,
        action: &AlterTableAction,
        original: &str,
    ) -> Result<CatalogTable, KrishivError> {
        match action {
            AlterTableAction::AddPartition {
                if_not_exists,
                parts,
                ..
            } => {
                self.add_partitions(table, parts, *if_not_exists, original)
                    .await
            }
            AlterTableAction::SetPartitionLocation {
                partition,
                location,
                ..
            } => {
                self.set_partition_location(table, partition, location, original)
                    .await
            }
            _ => Err(internal("not a partition action".to_owned())),
        }
    }

    /// `ALTER TABLE name DROP [IF EXISTS] PARTITION (...), ... [PURGE]` (`AlterTableDropPartitionCommand` over
    /// `InMemoryCatalog.dropPartitions`, Spark 4.2.0, krishiv-internal `research/spark/partition-followups/`): every
    /// spec is normalized and cast first (a bare key is refused by the validator), then the catalog check lists
    /// every spec it lacks (a partial one too) in one `PARTITIONS_NOT_FOUND` unless `IF EXISTS`; nothing is dropped
    /// when one is missing. Column statistics go. A managed table's partitions are dropped in three steps:
    /// 1. one commit-log entry removes their files (lost races are derived again);
    /// 2. the catalog swap, against a fresh entry each try; when the table is gone or no longer partitioned, the
    ///    removed files are put back in the log and the statement fails, with nothing deleted;
    /// 3. only the files that entry removed are deleted (a writer that committed into the partition after it keeps
    ///    its files), then the directories left empty. An I/O error there leaves files no read lists.
    ///
    /// `PURGE` changes nothing (no trash). Refused: an external table (Spark keeps its files and stops reading
    /// them; reads here refuse a directory the catalog does not know) and a partition registered away from its
    /// default directory (reads never list it, and its directory may hold anything).
    pub(crate) async fn drop_partitions(
        &self,
        table: &Arc<CatalogTable>,
        name: &crate::table_commands::TableName,
        specs: &[PartitionSpec],
        if_exists: bool,
        original: &str,
    ) -> Result<(), KrishivError> {
        // One partition command that moves or deletes directories at a time; the entry is read again under it.
        let _ddl = self
            .state
            .catalog
            .store
            .partition_ddl(&table.location)
            .await;
        let fresh = self.fresh_entry(table, name)?;
        let table = &fresh;
        require_partition_provider(table, "ALTER TABLE DROP PARTITION")?;
        let case_sensitive = self.query_env().analysis.conf.case_sensitive;
        let mut typed = Vec::with_capacity(specs.len());
        for spec in specs {
            let written = self.written_spec(spec, original).await?;
            let values: Vec<(String, String)> = normalize_spec(table, &written, case_sensitive)?
                .into_iter()
                .map(|(k, v)| {
                    let v =
                        v.unwrap_or_else(|| krishiv_io::partition::DEFAULT_PARTITION.to_owned());
                    (k, v)
                })
                .collect();
            typed.push(self.alter_partition_spec(table, &values).await?);
        }
        let exact = |spec: &[(String, String)]| {
            spec.len() == table.layout.partition_columns.len()
                && find_partition(table, spec).is_some()
        };
        let missing: Vec<&Vec<(String, String)>> = typed.iter().filter(|s| !exact(s)).collect();
        if !missing.is_empty() && !if_exists {
            return Err(PartitionsNotFound {
                partition_list: missing
                    .iter()
                    .map(|s| spec_text(s))
                    .collect::<Vec<_>>()
                    .join(", "),
                table_name: db_table(table),
            }
            .into());
        }
        let mut drop: Vec<Vec<(String, String)>> = Vec::with_capacity(typed.len());
        for spec in typed {
            if exact(&spec) && !drop.contains(&spec) {
                drop.push(spec);
            }
        }
        let hit =
            |p: &CatalogPartition| drop.iter().any(|s| s.iter().all(|kv| p.spec.contains(kv)));
        let gone: Vec<CatalogPartition> = table
            .layout
            .partitions
            .iter()
            .filter(|p| hit(p))
            .cloned()
            .collect();
        if gone.is_empty() {
            return Ok(());
        }
        if !table.managed {
            return Err(unsupported(
                "ALTER TABLE ... DROP PARTITION of an external table",
            ));
        }
        if let Some(p) = gone
            .iter()
            .find(|p| p.location != partition_path(&table.location, &p.spec))
        {
            return Err(unsupported(&format!(
                "ALTER TABLE ... DROP PARTITION of a partition at {} (not its default directory)",
                p.location.display()
            )));
        }
        for p in &gone {
            self.authorize_dir(&p.location).await?;
        }
        let moves: Vec<(PathBuf, Option<PathBuf>)> =
            gone.iter().map(|p| (p.location.clone(), None)).collect();
        // With a commit log, the files it lists are the table's: only those go. Without one, the directories go
        // whole, as Spark deletes a managed partition's location.
        let logged = table
            .location
            .join(krishiv_io::commit_log::LOG_DIR)
            .is_dir();
        let removed = self.relog_partitions(&table.location, &moves).await?;
        match self.swap_without(table, name, &hit) {
            Swap::Done => {}
            // The table is still this one: its log gets the files back. Dropped or replaced: its root (and any new
            // table's log there) is left alone.
            Swap::Lost => {
                self.unremove_files(&table.location, removed).await?;
                return Err(crate::table_commands::concurrent_change(name));
            }
            Swap::Gone => return Err(crate::table_commands::concurrent_change(name)),
        }
        let unable = |path: &Path| -> KrishivError {
            LegacyErrorTemp2036 {
                partition_path: path.display().to_string(),
            }
            .into()
        };
        if !logged {
            for p in &gone {
                match std::fs::remove_dir_all(&p.location) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(_) => return Err(unable(&p.location)),
                }
            }
        }
        for f in removed.iter().map(|f| table.location.join(&f.path)) {
            match std::fs::remove_file(&f) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(_) => return Err(unable(&f)),
            }
        }
        // A partition directory left empty goes; one still holding files stays, and so do its parents (Spark 4.2.0
        // keeps `p=1` after dropping `p=1/s=a`).
        for p in &gone {
            let _ = std::fs::remove_dir(&p.location);
        }
        Ok(())
    }

    /// Swaps `name`'s catalog entry for one without the partitions `hit` matches and without column statistics,
    /// against a fresh entry each try (a concurrent change of another part of the entry is kept), as long as the
    /// entry is still `table`'s (same creation time and location, still partitioned).
    fn swap_without(
        &self,
        table: &CatalogTable,
        name: &crate::table_commands::TableName,
        hit: &dyn Fn(&CatalogPartition) -> bool,
    ) -> Swap {
        const RACES: u32 = 32;
        let store = &self.state.catalog.store;
        for _ in 0..RACES {
            let Some(current) = store.table(&name.namespace, &name.name) else {
                return Swap::Gone;
            };
            if current.created_ms != table.created_ms
                || current.location != table.location
                || !current.is_partitioned()
            {
                return Swap::Gone;
            }
            let mut changed = (*current).clone();
            changed.layout.partitions.retain(|p| !hit(p));
            changed.column_stats.clear();
            if store.replace_table(&current, changed) {
                return Swap::Done;
            }
        }
        Swap::Lost
    }

    /// `ALTER TABLE name ADD [IF NOT EXISTS] PARTITION (...) [LOCATION '...'] ...`.
    pub(crate) async fn add_partitions(
        &self,
        table: &Arc<CatalogTable>,
        parts: &[PartitionWithLocation],
        if_not_exists: bool,
        original: &str,
    ) -> Result<CatalogTable, KrishivError> {
        require_partition_provider(table, "ALTER TABLE ADD PARTITION")?;
        let case_sensitive = self.query_env().analysis.conf.case_sensitive;
        let mut changed = (**table).clone();
        let mut added: Vec<CatalogPartition> = Vec::new();
        let mut existing: Vec<Vec<(String, String)>> = Vec::new();
        for part in parts {
            let written = self.written_spec(&part.spec, original).await?;
            let spec = exact_spec(table, &normalize_spec(table, &written, case_sensitive)?)?;
            let spec = self.alter_partition_spec(table, &spec).await?;
            let location = match &part.location {
                Some(l) => qualified(self, l)?,
                None => partition_path(&table.location, &ordered(table, &spec)),
            };
            if find_partition(&changed, &spec).is_some()
                || added
                    .iter()
                    .any(|p| spec.iter().all(|kv| p.spec.contains(kv)))
            {
                existing.push(spec);
                continue;
            }
            added.push(CatalogPartition {
                spec: ordered(table, &spec),
                location,
                parameters: BTreeMap::new(),
                created_ms: crate::table_commands::now_ms(),
                stats: None,
            });
        }
        if !existing.is_empty() && !if_not_exists {
            return Err(PartitionsAlreadyExist {
                partition_list: existing
                    .iter()
                    .map(|s| spec_text(s))
                    .collect::<Vec<_>>()
                    .join(", "),
                table_name: db_table(table),
            }
            .into());
        }
        // Every check first, then the directories: a failure removes the ones this command created.
        for p in &added {
            self.authorize_dir(&p.location).await?;
        }
        let mut created: Vec<&Path> = Vec::new();
        for p in &added {
            if p.location.is_dir() {
                continue;
            }
            if let Err(e) = std::fs::create_dir_all(&p.location) {
                for dir in created {
                    let _ = std::fs::remove_dir(dir);
                }
                return Err(internal(format!(
                    "cannot create {}: {e}",
                    p.location.display()
                )));
            }
            created.push(&p.location);
        }
        changed.layout.partitions.extend(added);
        changed.column_stats.clear();
        Ok(changed)
    }

    async fn authorize_dir(&self, path: &Path) -> Result<(), KrishivError> {
        let securable = Securable::Path(krishiv_common::secret::OptionValue::from_user(
            "path",
            path.display().to_string(),
        ));
        let r = authorize(
            &*self.state.policy,
            &self.state.principal,
            &securable,
            Privilege::Create,
        )
        .await;
        crate::access::audit(self, StatementClass::Ddl, r)
    }

    /// Registers the partitions a successful write committed files to (`dirs`, as
    /// [`DataFrameWriter`](crate::DataFrameWriter) reports them) that the catalog does not have yet
    /// (`AlterTableAddPartitionCommand` after an insert); with `file_stats` they carry `numFiles`,
    /// `totalSize` and `transient_lastDdlTime` (`RepairTableCommand` after a CTAS). Partitions are never
    /// dropped here: Spark keeps a catalog partition whose directory is gone.
    pub(crate) fn register_partitions(&self, name: &TableName, dirs: &[String], file_stats: bool) {
        let store = &self.state.catalog.store;
        // The swap can lose to a concurrent write or ALTER; read the table again and merge again (a partition
        // already registered is skipped, so a retry is harmless) until it lands or the table is gone.
        loop {
            let Some(table) = store.table(&name.namespace, &name.name) else {
                return;
            };
            if !table.is_partitioned() || dirs.is_empty() {
                return;
            }
            let case_sensitive = self.query_env().analysis.conf.case_sensitive;
            let mut changed = (*table).clone();
            let now = crate::table_commands::now_ms();
            for dir in dirs {
                let Some(spec) = spec_of_dir(dir, &table.layout.partition_columns, case_sensitive)
                else {
                    continue;
                };
                if find_partition(&changed, &spec).is_some() {
                    continue;
                }
                let location = dir
                    .split('/')
                    .fold(table.location.clone(), |p, s| p.join(s));
                let mut parameters = BTreeMap::new();
                if file_stats {
                    let files = listed_files(&location);
                    parameters.insert("numFiles".to_owned(), files.len().to_string());
                    parameters.insert(
                        "totalSize".to_owned(),
                        files.iter().map(|(_, n)| n).sum::<u64>().to_string(),
                    );
                    parameters.insert("transient_lastDdlTime".to_owned(), (now / 1000).to_string());
                }
                changed.layout.partitions.push(CatalogPartition {
                    spec,
                    location,
                    parameters,
                    created_ms: now,
                    stats: None,
                });
            }
            if changed == *table || store.replace_table(&table, changed) {
                return;
            }
        }
    }

    /// `ANALYZE TABLE name PARTITION (...) COMPUTE STATISTICS [NOSCAN]` (`AnalyzePartitionCommand`): each
    /// matching partition gets its size and, unless `NOSCAN`, its row count.
    pub(crate) async fn analyze_partitions(
        &self,
        name: &TableName,
        table: &Arc<CatalogTable>,
        spec: &PartitionSpec,
        noscan: bool,
        original: &str,
    ) -> Result<(), KrishivError> {
        let case_sensitive = self.query_env().analysis.conf.case_sensitive;
        let written = self.written_spec(spec, original).await?;
        let spec = normalize_spec(table, &written, case_sensitive)?;
        let not_specified: Vec<bool> = table
            .layout
            .partition_columns
            .iter()
            .map(|c| !spec.iter().any(|(k, v)| k == c && v.is_some()))
            .collect();
        if not_specified.windows(2).any(|w| w[0] && !w[1]) {
            return Err(LegacyErrorTemp1237 {
                table: table.name.clone(),
                database: table.namespace.clone(),
                schema_columns: table.layout.partition_columns.join(","),
                spec_columns: spec
                    .iter()
                    .map(|(k, _)| k.as_str())
                    .collect::<Vec<_>>()
                    .join(","),
            }
            .into());
        }
        let filter: Vec<(String, String)> = spec
            .into_iter()
            .filter_map(|(k, v)| v.map(|v| (k, v)))
            .collect();
        let matching: Vec<CatalogPartition> = table
            .layout
            .partitions
            .iter()
            .filter(|p| filter.iter().all(|kv| p.spec.contains(kv)))
            .cloned()
            .collect();
        if matching.is_empty() {
            if filter.is_empty() {
                return Ok(());
            }
            return Err(PartitionsNotFound {
                partition_list: spec_text(&filter),
                table_name: db_table(table),
            }
            .into());
        }
        let mut updated = Vec::with_capacity(matching.len());
        for p in matching {
            let size = location_size(&p.location);
            let rows = if noscan {
                None
            } else {
                self.count_partition_rows(table, &p.location).await?
            };
            let old_size = p.stats.map_or(-1, |s| s.size_in_bytes);
            let old_rows = p.stats.and_then(|s| s.row_count);
            let mut stats = None;
            if size >= 0 && size != old_size {
                stats = Some(PartitionStats {
                    size_in_bytes: size,
                    row_count: None,
                });
            }
            if let Some(n) = rows
                && Some(n) != old_rows
            {
                stats = Some(PartitionStats {
                    size_in_bytes: stats.map_or(old_size, |s| s.size_in_bytes),
                    row_count: Some(n),
                });
            }
            if let Some(stats) = stats {
                updated.push(CatalogPartition {
                    stats: Some(stats),
                    ..p
                });
            }
        }
        if updated.is_empty() {
            return Ok(());
        }
        let mut changed = (**table).clone();
        for p in &mut changed.layout.partitions {
            if let Some(u) = updated.iter().find(|u| u.spec == p.spec) {
                *p = u.clone();
            }
        }
        if !self.state.catalog.store.replace_table(table, changed) {
            return Err(internal(format!(
                "{} was changed by another statement while this one ran; retry",
                name.quoted()
            )));
        }
        Ok(())
    }

    /// The rows in a partition's directory; `None` when it holds no data file (Spark's per-partition count
    /// has no row for an empty partition).
    async fn count_partition_rows(
        &self,
        table: &CatalogTable,
        dir: &Path,
    ) -> Result<Option<u64>, KrishivError> {
        if listed_files(dir).is_empty() {
            return Ok(None);
        }
        let mut reader = self.read().format(&table.provider);
        for (k, v) in &table.options {
            reader = reader.option(k, v);
        }
        let n = reader.load([dir.display().to_string()]).count().await?;
        Ok((n > 0).then_some(n))
    }

    /// `SHOW PARTITIONS name [PARTITION (...)]`.
    pub(crate) async fn show_partitions(
        &self,
        table: &CatalogTable,
        spec: Option<&PartitionSpec>,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        if !table.is_partitioned() {
            return Err(InvalidPartitionOperationPartitionSchemaIsEmpty {
                name: table_sql_id(table),
            }
            .into());
        }
        let case_sensitive = self.query_env().analysis.conf.case_sensitive;
        let filter: Vec<(String, String)> = match spec {
            None => Vec::new(),
            Some(spec) => {
                let written = self.written_spec(spec, original).await?;
                normalize_spec(table, &written, case_sensitive)?
                    .into_iter()
                    .filter_map(|(k, v)| v.map(|v| (k, v)))
                    .collect()
            }
        };
        let mut names: Vec<String> = table
            .layout
            .partitions
            .iter()
            .filter(|p| filter.iter().all(|kv| p.spec.contains(kv)))
            .map(|p| partition_name(&ordered(table, &p.spec)))
            .collect();
        names.sort();
        let rows = names.into_iter().map(|n| vec![Some(n)]).collect();
        // A union of rows keeps no order, so sort the frame itself.
        Ok(self
            .string_frame(&["partition"], rows)
            .await?
            .order_by([col("partition")]))
    }

    /// The partition a `DESCRIBE ... PARTITION (...)` names (`DescribeTableCommand
    /// .describeDetailedPartitionInfo`): views refuse it, the table must be partitioned, the spec must name
    /// every partition column and the partition must exist.
    /// `statement` names the command in errors (`DESC PARTITION`, `SHOW TABLE EXTENDED`).
    pub(crate) async fn described_partition(
        &self,
        name: &TableName,
        temp: bool,
        spec: &PartitionSpec,
        original: &str,
        statement: &str,
    ) -> Result<CatalogPartition, KrishivError> {
        let written = self.written_spec(spec, original).await?;
        let forbidden = |object_type: &str| -> KrishivError {
            ForbiddenOperation {
                statement: statement.to_owned(),
                object_type: object_type.to_owned(),
                object_name: quote_identifier(&name.name),
            }
            .into()
        };
        if temp {
            return Err(forbidden("TEMPORARY VIEW"));
        }
        let table = self
            .state
            .catalog
            .store
            .table(&name.namespace, &name.name)
            .ok_or_else(|| name.not_found())?;
        if table.view.is_some() {
            return Err(forbidden("VIEW"));
        }
        require_partition_provider(&table, statement)?;
        let case_sensitive = self.query_env().analysis.conf.case_sensitive;
        let spec = exact_spec(&table, &normalize_spec(&table, &written, case_sensitive)?)?;
        find_partition(&table, &spec).cloned().ok_or_else(|| {
            PartitionsNotFound {
                partition_list: spec_text(&spec),
                table_name: db_table(&table),
            }
            .into()
        })
    }
}

/// Spark's `CatalogStatistics.simpleString`: `<bytes> bytes[, <rows> rows]`.
pub(crate) fn stats_text(s: PartitionStats) -> String {
    match s.row_count {
        Some(rows) => format!("{} bytes, {rows} rows", s.size_in_bytes),
        None => format!("{} bytes", s.size_in_bytes),
    }
}

impl crate::Session {
    /// [`Self::analyze_variable_query`] in `env` (a query environment the caller adjusted, such as a cast mode
    /// Spark fixes whatever the session says).
    pub(crate) async fn analyze_variable_query_in(
        &self,
        sql: &str,
        env: crate::session::QueryEnv,
    ) -> Result<Arc<crate::dataframe::Analysis>, KrishivError> {
        let redacted = krishiv_sql::redact_sql(sql);
        let query = krishiv_sql::parse_query(sql, &env.parser).map_err(|e| {
            crate::session::sql_context(e.error, &redacted, e.span.start, e.span.end)
        })?;
        let plan = krishiv_sql::lower(&query, krishiv_plan::SourceId(0)).map_err(|e| {
            crate::session::sql_context(e.error, &redacted, e.span.start, e.span.end)
        })?;
        let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
        df.analyzed_in_with_text(env, Some(&redacted)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(s: &Session, sql: &str) {
        futures::executor::block_on(async { s.sql(sql).await?.collect().await })
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
    }

    fn name() -> crate::table_commands::TableName {
        crate::table_commands::TableName {
            namespace: "default".to_owned(),
            name: "r".to_owned(),
            written: "`r`".to_owned(),
            bare: true,
        }
    }

    fn spec(p: &str) -> Vec<(String, String)> {
        vec![("p".to_owned(), p.to_owned())]
    }

    /// A RENAME PARTITION waits for a write into its table that holds the table's partition lock (an INSERT holds
    /// it from its write to its partition registration), then runs; a write into another table does not hold it.
    #[test]
    fn a_rename_waits_for_a_write_in_progress() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let s = Session::new().unwrap_or_else(|e| panic!("{e}"));
        s.conf()
            .set(
                "spark.sql.warehouse.dir",
                dir.path().to_str().unwrap_or_default(),
            )
            .unwrap_or_else(|e| panic!("{e}"));
        run(
            &s,
            "create table r (a int, p int) using parquet partitioned by (p)",
        );
        run(&s, "insert into r values (1, 1)");
        run(
            &s,
            "create table o (a int, p int) using parquet partitioned by (p)",
        );
        run(&s, "insert into o values (1, 1)");
        let store = Arc::clone(&s.state.catalog.store);
        let location = |t: &str| {
            store
                .table("default", t)
                .unwrap_or_else(|| panic!("no table {t}"))
                .location
                .clone()
        };
        let renamed = |p: &str| {
            store
                .table("default", "r")
                .is_some_and(|t| find_partition(&t, &spec(p)).is_some())
        };
        // A write held on another table never holds a rename back.
        let other = futures::executor::block_on(store.partition_write(&location("o")));
        run(
            &s,
            "alter table r partition (p=1) rename to partition (p=2)",
        );
        drop(other);
        run(
            &s,
            "alter table r partition (p=2) rename to partition (p=1)",
        );
        let write = futures::executor::block_on(store.partition_write(&location("r")));
        std::thread::scope(|scope| {
            let rename = scope.spawn(|| {
                run(
                    &s,
                    "alter table r partition (p=1) rename to partition (p=5)",
                );
            });
            // Without the lock the rename is done well inside this (milliseconds); with it, it waits.
            std::thread::sleep(std::time::Duration::from_millis(300));
            assert!(
                renamed("1") && !renamed("5"),
                "the rename ran during the write"
            );
            drop(write);
            rename
                .join()
                .unwrap_or_else(|_| panic!("the rename panicked"));
        });
        assert!(renamed("5") && !renamed("1"));
    }

    /// A rename's catalog swap goes against the entry as it is now: a concurrent change of another part of it is
    /// kept, and a dropped table or a partition taken meanwhile refuses the swap (the rename then moves back).
    #[test]
    fn a_rename_swaps_against_a_fresh_entry() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let s = Session::new().unwrap_or_else(|e| panic!("{e}"));
        s.conf()
            .set(
                "spark.sql.warehouse.dir",
                dir.path().to_str().unwrap_or_default(),
            )
            .unwrap_or_else(|e| panic!("{e}"));
        run(
            &s,
            "create table r (a int, p int) using parquet partitioned by (p)",
        );
        run(&s, "insert into r values (1, 1)");
        let store = &s.state.catalog.store;
        let old = store
            .table("default", "r")
            .unwrap_or_else(|| panic!("no table"));
        let source = find_partition(&old, &spec("1"))
            .unwrap_or_else(|| panic!("no partition"))
            .location
            .clone();
        let target = partition_path(&old.location, &spec("5"));
        // A concurrent SET TBLPROPERTIES replaced the entry after the rename read it.
        run(&s, "alter table r set tblproperties ('k' = 'v')");
        assert!(matches!(
            s.swap_renamed(&old, &name(), &spec("1"), &source, &spec("5"), &target),
            Swap::Done
        ));
        let now = store
            .table("default", "r")
            .unwrap_or_else(|| panic!("no table"));
        assert_eq!(now.properties.get("k").map(String::as_str), Some("v"));
        assert!(find_partition(&now, &spec("5")).is_some_and(|p| p.location == target));
        assert!(find_partition(&now, &spec("1")).is_none());
        // The partition is gone under the old spec: nothing to swap.
        assert!(matches!(
            s.swap_renamed(&old, &name(), &spec("1"), &source, &spec("6"), &target),
            Swap::Lost
        ));
        // A partition taken meanwhile.
        run(&s, "alter table r add partition (p=7)");
        let t5 = partition_path(&old.location, &spec("5"));
        assert!(matches!(
            s.swap_renamed(&now, &name(), &spec("5"), &t5, &spec("7"), &target),
            Swap::Lost
        ));
        // A dropped table, then one made again under the name: neither is the statement's table.
        run(&s, "drop table r");
        assert!(matches!(
            s.swap_renamed(&now, &name(), &spec("5"), &t5, &spec("8"), &target),
            Swap::Gone
        ));
        assert!(s.fresh_entry(&now, &name()).is_err());
        run(
            &s,
            "create table r (a int, p int) using parquet partitioned by (p)",
        );
        let again = store
            .table("default", "r")
            .unwrap_or_else(|| panic!("no table"));
        assert!(s.fresh_entry(&again, &name()).is_ok());
        if again.created_ms != now.created_ms {
            assert!(s.fresh_entry(&now, &name()).is_err());
        }
    }

    #[test]
    fn path_names_round_trip() {
        assert_eq!(
            unescape_path_name("2018-11-17 13%3A33%3A33"),
            "2018-11-17 13:33:33"
        );
        assert_eq!(unescape_path_name("a%2"), "a%2");
        assert_eq!(unescape_path_name("%zz"), "%zz");
        assert_eq!(escape_path_name("a:b"), "a%3Ab");
        assert_eq!(
            partition_name(&[
                ("c".to_owned(), "U/s".to_owned()),
                ("d".to_owned(), "1".to_owned())
            ]),
            "c=U%2Fs/d=1"
        );
        assert_eq!(
            spec_text(&[
                ("c".to_owned(), "Us".to_owned()),
                ("d".to_owned(), "2".to_owned())
            ]),
            "PARTITION (`c` = Us, `d` = 2)"
        );
    }
}
