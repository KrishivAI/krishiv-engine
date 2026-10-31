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
//!   write to one fails with `KRISHIV_UNSUPPORTED_FEATURE` before anything is written. Reading a partitioned
//!   table's rows needs partition discovery, which file reads refuse (`KRISHIV_UNSUPPORTED_FEATURE`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use krishiv_common::error::{
    AllPartitionColumnsNotAllowed, ColumnAlreadyExists, ColumnNotDefinedInTable,
    ForbiddenOperation, InternalError, InvalidPartitionColumnDataType,
    InvalidPartitionOperationPartitionSchemaIsEmpty, KrishivError, KrishivUnsupportedFeature,
    LegacyErrorTemp1166, LegacyErrorTemp1167, LegacyErrorTemp1232, LegacyErrorTemp1237,
    LegacyErrorTemp1251, PartitionsAlreadyExist, PartitionsNotFound,
};
use krishiv_plan::Literal;
use krishiv_plan::analyzer::{Privilege, Securable, authorize};
use krishiv_sql::StatementClass;
use krishiv_sql::ast::{BucketSpec, Expr, PartitionSpec, PartitionWithLocation};
use krishiv_types::DataType;

use crate::dataframe::DataFrame;
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
    for (i, a) in names.iter().enumerate() {
        if names[..i].iter().any(|b| same_name(a, b, case_sensitive)) {
            let shown = if case_sensitive {
                a.clone()
            } else {
                a.to_lowercase()
            };
            return Err(ColumnAlreadyExists {
                column_name: quote_identifier(&shown),
            }
            .into());
        }
    }
    Ok(())
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

/// The partition directories below `root` for `columns` (`RepairTableCommand.scanPartitions`): `<col>=<value>`
/// directories, the column matched by name, the value unescaped.
fn scan_partitions(
    root: &Path,
    columns: &[String],
    prefix: &mut Vec<(String, String)>,
    case_sensitive: bool,
    out: &mut Vec<(Vec<(String, String)>, PathBuf)>,
) {
    let Some((head, rest)) = columns.split_first() else {
        out.push((prefix.clone(), root.to_path_buf()));
        return;
    };
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    let mut dirs: Vec<(String, PathBuf)> = entries
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
        .filter(|(name, _)| name != "_temporary" && !name.starts_with('.'))
        .collect();
    dirs.sort();
    for (name, path) in dirs {
        let Some((column, value)) = name.split_once('=') else {
            continue;
        };
        if !same_name(&unescape_path_name(column), head, case_sensitive) {
            continue;
        }
        prefix.push((head.clone(), unescape_path_name(value)));
        scan_partitions(&path, rest, prefix, case_sensitive, out);
        prefix.pop();
    }
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
        let text = original.get(e.span.start..e.span.end).unwrap_or_default();
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
        Ok(normalize_spec(table, &written, case_sensitive)?
            .into_iter()
            .filter_map(|(k, v)| v.map(|v| (k, v)))
            .collect())
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
            let location = match &part.location {
                Some(l) => {
                    let path = PathBuf::from(l.strip_prefix("file:").unwrap_or(l));
                    if !path.is_absolute() {
                        return Err(unsupported("a relative partition LOCATION"));
                    }
                    path
                }
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
        for p in &added {
            self.authorize_dir(&p.location).await?;
            std::fs::create_dir_all(&p.location)
                .map_err(|e| internal(format!("cannot create {}: {e}", p.location.display())))?;
        }
        changed.layout.partitions.extend(added);
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

    /// Registers the partition directories found under the table's directory after a write, and forgets
    /// the partitions whose directory the write removed (`INSERT OVERWRITE`); with `file_stats` the new ones
    /// carry `numFiles`, `totalSize` and `transient_lastDdlTime` (`RepairTableCommand` after a CTAS).
    pub(crate) fn sync_partitions(&self, name: &TableName, file_stats: bool) {
        let store = &self.state.catalog.store;
        let Some(table) = store.table(&name.namespace, &name.name) else {
            return;
        };
        if !table.is_partitioned() {
            return;
        }
        let case_sensitive = self.query_env().analysis.conf.case_sensitive;
        let mut found = Vec::new();
        scan_partitions(
            &table.location,
            &table.layout.partition_columns,
            &mut Vec::new(),
            case_sensitive,
            &mut found,
        );
        let mut changed = (*table).clone();
        let root = &table.location;
        changed
            .layout
            .partitions
            .retain(|p| !p.location.starts_with(root) || p.location.is_dir());
        let now = crate::table_commands::now_ms();
        for (spec, dir) in found {
            if find_partition(&changed, &spec).is_some() {
                continue;
            }
            let mut parameters = BTreeMap::new();
            if file_stats {
                let files = listed_files(&dir);
                parameters.insert("numFiles".to_owned(), files.len().to_string());
                parameters.insert(
                    "totalSize".to_owned(),
                    files.iter().map(|(_, n)| n).sum::<u64>().to_string(),
                );
                parameters.insert("transient_lastDdlTime".to_owned(), (now / 1000).to_string());
            }
            changed.layout.partitions.push(CatalogPartition {
                spec,
                location: dir,
                parameters,
                created_ms: now,
                stats: None,
            });
        }
        if changed != *table {
            // A concurrent change wins: the partitions are found again by the next write.
            let _ = store.replace_table(&table, changed);
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
        self.string_frame(&["partition"], rows).await
    }

    /// The partition a `DESCRIBE ... PARTITION (...)` names (`DescribeTableCommand
    /// .describeDetailedPartitionInfo`): views refuse it, the table must be partitioned, the spec must name
    /// every partition column and the partition must exist.
    pub(crate) async fn described_partition(
        &self,
        name: &TableName,
        temp: bool,
        spec: &PartitionSpec,
        original: &str,
    ) -> Result<CatalogPartition, KrishivError> {
        let written = self.written_spec(spec, original).await?;
        let forbidden = |object_type: &str| -> KrishivError {
            ForbiddenOperation {
                statement: "DESC PARTITION".to_owned(),
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
        require_partition_provider(&table, "DESC PARTITION")?;
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

#[cfg(test)]
mod tests {
    use super::*;

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
