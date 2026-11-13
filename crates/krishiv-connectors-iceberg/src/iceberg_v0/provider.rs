// SPDX-License-Identifier: Apache-2.0
//! [`IcebergTableProvider`]: an Iceberg snapshot as a table the executor reads
//! (P3-T026, PR-25a; design note §3.1). It adds a provider and one hook, not a
//! reader: the snapshot's data files are the file list of the existing
//! [`FileScan`] (Parquet, Spark's split planning), so morsels, prefetch, row
//! groups and runtime filters work as for any Parquet read.
//!
//! - **Columns by field id.** The table schema carries every Iceberg id
//!   ([`krishiv_io::FIELD_ID_KEY`]); files map by id
//!   ([`FieldMapping::FieldIdOr`]). A file with no id anywhere gets top-level
//!   positional ids 1..n, as Java's `ParquetSchemaUtil.addFallbackIds` (FC S4);
//!   with a name mapping (`schema.name-mapping.default`) such a file is refused
//!   until PR-25c reads the mapping.
//! - **Identity partitions** are per-file constants from the manifest
//!   ([`FileConstants`]): they win over the file's column and fill a column the
//!   file lacks (FC S4). A column is constant when every data file's spec has an
//!   identity field on it (a top-level column); otherwise it is read from the
//!   files.
//! - **Deletes** are not applied yet (PR-25b): a snapshot with a live delete
//!   file is refused, never read without them.
//! - **Filters** prune data files by partition values and metrics through
//!   [`Pruner`] when the scan is built; every filter is still evaluated on the
//!   rows ([`FilterPushdown::Inexact`]).

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use arrow_schema::{FieldRef, SchemaRef};
use krishiv_common::config::Config;
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::planner::TableProvider;
use krishiv_exec::{FileScan, FilterPushdown, PushedFilter, ScanPushdown, ScanPushdowns, Source};
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::split::{FilePartition, SplitConfig};
use krishiv_io::{
    FieldMapping, FileConstants, FileFormat, FileRef, FormatOptions, IoContext, PushdownExpr,
    ScalarValue, ScanOptions, VersionToken, WithoutFieldIds,
};

use super::table::IcebergTable;
use super::types::arrow_schema;
use crate::plan::{
    DataFileEntry, FileFormatKind, ManifestFilter, PartitionValue, PruneOptions, Pruner, ScanPlan,
    TableSnapshot, Transform, VersionSpec, WriterMetrics, decode,
};

/// The table property naming a name mapping for files without field ids.
const NAME_MAPPING: &str = "schema.name-mapping.default";

fn unsupported(feature: String) -> KrishivError {
    KrishivUnsupportedFeature { feature }.into()
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// An Iceberg table pinned at one snapshot, read by the executor (see the
/// module docs).
#[derive(Debug, Clone)]
pub struct IcebergTableProvider {
    scan: Arc<IcebergScan>,
}

impl IcebergTableProvider {
    /// The provider of `table` at `version`: resolves the snapshot, plans its
    /// data files (reading the manifest list and manifests through the
    /// table's storage) and builds the scan. Splits follow `config`
    /// (`spark.sql.files.maxPartitionBytes`, ...) over `worker_slots` drivers.
    ///
    /// # Errors
    /// The time-travel errors of [`IcebergTable::snapshot`]; the planning
    /// errors of [`IcebergTable::plan`]; `KRISHIV_UNSUPPORTED_FEATURE` for a
    /// snapshot with delete files (naming the first), a data file that is not
    /// Parquet, or a file that lacks a column its spec partitions by identity
    /// when other files' specs do not (see [`refuse_missing_mixed_identity`]);
    /// `INTERNAL_ERROR` for a partition value that does not decode.
    pub async fn open(
        table: &IcebergTable,
        version: &VersionSpec,
        config: &Config,
        worker_slots: u64,
    ) -> Result<Self, KrishivError> {
        let snapshot = table.snapshot(version)?;
        let plan = table
            .plan(
                &snapshot,
                None,
                ManifestFilter::All,
                PruneOptions::default(),
            )
            .await?;
        let schema = table
            .table
            .metadata()
            .schema_by_id(snapshot.schema_id)
            .ok_or_else(|| {
                internal(format!(
                    "the table metadata has no schema {}",
                    snapshot.schema_id
                ))
            })?;
        let arrow = arrow_schema(schema)?;
        let name_mapping = table
            .table
            .metadata()
            .properties()
            .contains_key(NAME_MAPPING);
        let reg = &table.registered;
        let locate = |path: &str| -> Result<FileRef, KrishivError> {
            let at = reg.paths.locate(path)?;
            Ok(FileRef {
                store: at.store,
                path: at.path,
                size: 0,
                version: VersionToken::Immutable,
                modified_micros: None,
            })
        };
        let metrics = table.writer_metrics(&snapshot);
        let scan = IcebergScan::new(
            snapshot,
            plan,
            arrow,
            name_mapping,
            reg.io.clone(),
            &locate,
            config,
            worker_slots,
            metrics,
        )?;
        refuse_missing_mixed_identity(&scan).await?;
        Ok(Self {
            scan: Arc::new(scan),
        })
    }

    /// The snapshot read and its schema.
    #[must_use]
    pub fn snapshot(&self) -> &TableSnapshot {
        &self.scan.snapshot
    }

    /// The data files read, in plan order.
    #[must_use]
    pub fn files(&self) -> &[FileRef] {
        &self.scan.files
    }

    /// The columns read from the manifests' identity-partition values instead
    /// of the files.
    #[must_use]
    pub fn constant_columns(&self) -> Vec<String> {
        self.scan
            .parts
            .constant_columns
            .iter()
            .map(|f| f.name().clone())
            .collect()
    }
}

#[cfg(test)]
impl IcebergTableProvider {
    /// How many files a scan with the pushed filter `expr` opens.
    pub(super) fn files_read(&self, expr: PushdownExpr) -> usize {
        let predicate: Arc<dyn krishiv_exec::Predicate> =
            Arc::new(krishiv_exec::FnPredicate(|b: &arrow_array::RecordBatch| {
                Ok(arrow_array::BooleanArray::from(vec![true; b.num_rows()]))
            }));
        self.scan
            .pruned(&[PushedFilter::typed(predicate, expr)])
            .files()
            .len()
    }
}

impl TableProvider for IcebergTableProvider {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        Ok(Arc::clone(&self.scan) as Arc<dyn ScanPushdown>)
    }
}

/// The scan factory of an [`IcebergTableProvider`]: a [`FileScan`] over the
/// snapshot's data files that prunes them by the pushed filters when built.
#[derive(Clone)]
struct IcebergScan {
    snapshot: Arc<TableSnapshot>,
    /// The files, aligned with `entries`.
    files: Arc<[FileRef]>,
    entries: Arc<[Arc<DataFileEntry>]>,
    mapping: FieldMapping,
    /// The writers' metrics modes, for pruning by file metrics.
    metrics: Arc<WriterMetrics>,
    parts: ScanParts,
    /// The scan over every file (or a distributed task's given splits).
    base: FileScan,
    /// Whether `build` prunes files (not for given splits).
    prune: bool,
}

impl fmt::Debug for IcebergScan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IcebergScan")
            .field("snapshot_id", &self.snapshot.snapshot_id)
            .field("files", &self.files.len())
            .finish_non_exhaustive()
    }
}

impl IcebergScan {
    #[expect(
        clippy::too_many_arguments,
        reason = "one private constructor of the provider's parts"
    )]
    fn new(
        snapshot: TableSnapshot,
        plan: ScanPlan,
        table: SchemaRef,
        name_mapping: bool,
        io: IoContext,
        locate: &dyn Fn(&str) -> Result<FileRef, KrishivError>,
        config: &Config,
        worker_slots: u64,
        metrics: WriterMetrics,
    ) -> Result<Self, KrishivError> {
        if let Some(d) = plan.delete_files.first() {
            return Err(unsupported(format!(
                "reading Iceberg tables with delete files (snapshot {} has {})",
                plan.snapshot_id.unwrap_or_default(),
                d.entry.file.path
            )));
        }
        let entries: Vec<Arc<DataFileEntry>> = plan.tasks.into_iter().map(|t| t.file).collect();
        let mut files = Vec::with_capacity(entries.len());
        for e in &entries {
            if e.file.format != FileFormatKind::Parquet {
                return Err(unsupported(format!(
                    "reading Iceberg {:?} data files ({})",
                    e.file.format, e.file.path
                )));
            }
            let mut f = locate(&e.file.path)?;
            f.size = e.file.size;
            files.push(f);
        }
        let (constant_columns, constants) = constants(&snapshot, &table, &entries, &files)?;
        let mapping = FieldMapping::FieldIdOr(if name_mapping {
            WithoutFieldIds::Refuse
        } else {
            WithoutFieldIds::Position
        });
        let parts = ScanParts {
            table,
            constant_columns,
            constants: Arc::new(constants),
            io,
            format_options: FormatOptions::defaults(ParquetFormat.options()),
            scan_options: ScanOptions::from_config(config),
            splits: SplitConfig::from_config(config, worker_slots),
        };
        Ok(Self {
            snapshot: Arc::new(snapshot),
            base: parts.file_scan(files.clone()),
            files: files.into(),
            entries: entries.into(),
            mapping,
            metrics: Arc::new(metrics),
            parts,
            prune: true,
        })
    }
}

/// What a [`FileScan`] of some of the table's files is built from.
#[derive(Clone)]
struct ScanParts {
    table: SchemaRef,
    constant_columns: Vec<FieldRef>,
    constants: Arc<FileConstants>,
    io: IoContext,
    format_options: FormatOptions,
    scan_options: ScanOptions,
    splits: SplitConfig,
}

impl ScanParts {
    /// The file scan of `files` (with Spark's split planning).
    fn file_scan(&self, files: Vec<FileRef>) -> FileScan {
        let scan = FileScan::new(
            Arc::new(ParquetFormat) as Arc<dyn FileFormat>,
            self.io.clone(),
            Arc::clone(&self.table),
            files,
            self.format_options.clone(),
            self.scan_options,
            self.splits,
        );
        if self.constant_columns.is_empty() {
            scan
        } else {
            scan.with_file_constants(self.constant_columns.clone(), Arc::clone(&self.constants))
        }
    }
}

impl IcebergScan {
    /// `p` with the table's field mapping.
    fn mapped(&self, mut p: ScanPushdowns) -> ScanPushdowns {
        p.mapping = self.mapping.clone();
        p
    }

    /// The file scan of the files that may hold rows passing `filters`.
    fn pruned(&self, filters: &[PushedFilter]) -> FileScan {
        let exprs: Vec<PushdownExpr> = filters.iter().filter_map(|f| f.expr.clone()).collect();
        if !self.prune || exprs.is_empty() {
            return self.base.clone();
        }
        let expr = PushdownExpr::And(exprs);
        let pruner = Pruner::new(
            &expr,
            &self.snapshot.fields,
            &self.snapshot.partition_specs,
            PruneOptions::default(),
        )
        .with_metrics(&self.metrics);
        let kept: Vec<FileRef> = self
            .entries
            .iter()
            .zip(self.files.iter())
            .filter(|(e, _)| pruner.file_may_match(&e.file))
            .map(|(_, f)| f.clone())
            .collect();
        if kept.len() == self.files.len() {
            self.base.clone()
        } else {
            self.parts.file_scan(kept)
        }
    }
}

impl ScanPushdown for IcebergScan {
    fn table_schema(&self) -> SchemaRef {
        Arc::clone(&self.parts.table)
    }

    fn filter_pushdown(&self, filter: &PushedFilter) -> FilterPushdown {
        // Files are pruned by it (a superset of the matching rows is kept);
        // the filter above the scan stays.
        if filter.expr.is_some() {
            FilterPushdown::Inexact
        } else {
            FilterPushdown::Unsupported
        }
    }

    fn build(&self, pushdowns: ScanPushdowns) -> krishiv_exec::Result<Arc<dyn Source>> {
        let p = self.mapped(pushdowns);
        self.pruned(&p.filters).build(p)
    }

    fn with_file_splits(
        &self,
        files: Vec<FileRef>,
        partitions: Vec<FilePartition>,
    ) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        let mut scan = self.clone();
        scan.base = scan.base.with_file_splits(files, partitions);
        scan.prune = false;
        Ok(Arc::new(scan))
    }

    fn output_schema(&self, pushdowns: &ScanPushdowns) -> krishiv_exec::Result<SchemaRef> {
        self.base.output_schema(&self.mapped(pushdowns.clone()))
    }
}

/// The identity-partition columns constant in every data file (top-level
/// columns of the read schema, in table order) and each file's values.
fn constants(
    snapshot: &TableSnapshot,
    table: &SchemaRef,
    entries: &[Arc<DataFileEntry>],
    files: &[FileRef],
) -> Result<(Vec<FieldRef>, FileConstants), KrishivError> {
    // For each file: the identity partition fields of its spec, by source id.
    let identity = |e: &DataFileEntry| -> BTreeSet<i32> {
        snapshot
            .partition_specs
            .get(&e.file.spec_id)
            .map(|spec| {
                spec.fields
                    .iter()
                    .filter(|f| f.transform == Transform::Identity)
                    .map(|f| f.source_id)
                    .collect()
            })
            .unwrap_or_default()
    };
    let columns: Vec<_> = snapshot
        .fields
        .iter()
        .filter(|f| f.path.0.len() == 1)
        .filter(|f| !entries.is_empty() && entries.iter().all(|e| identity(e).contains(&f.id)))
        .collect();
    let mut out = FileConstants::new();
    if columns.is_empty() {
        return Ok((Vec::new(), out));
    }
    let fields = columns
        .iter()
        .map(|c| {
            table
                .field_with_name(c.path.root())
                .map(|f| Arc::new(f.clone()))
                .map_err(|e| internal(format!("Iceberg column {}: {e}", c.path)))
        })
        .collect::<Result<Vec<FieldRef>, _>>()?;
    for (e, f) in entries.iter().zip(files) {
        let spec = snapshot
            .partition_specs
            .get(&e.file.spec_id)
            .ok_or_else(|| internal(format!("no partition spec {}", e.file.spec_id)))?;
        let mut values = Vec::with_capacity(columns.len());
        for c in &columns {
            let i = spec
                .fields
                .iter()
                .position(|p| p.transform == Transform::Identity && p.source_id == c.id)
                .ok_or_else(|| internal(format!("no identity field on {}", c.path)))?;
            let bad = || {
                internal(format!(
                    "the partition value of {} in {} does not decode",
                    c.path, e.file.path
                ))
            };
            values.push(match e.file.partition.get(i) {
                Some(PartitionValue::Null) => ScalarValue::Null,
                Some(PartitionValue::Value(b)) => decode(b, &c.ty).ok_or_else(bad)?,
                Some(PartitionValue::Unreadable) | None => return Err(bad()),
            });
        }
        out.insert(f.path.clone(), values);
    }
    Ok((fields, out))
}

/// Mixed specs (review of PR-25a): a top-level column that is an identity
/// partition source in some data files' specs but not in others is read from
/// the files, not from the manifests. That is right while every file whose spec
/// partitions by it also stores it (conforming writers do); a file that lacks it
/// would read NULL where Java reads the manifest's value, so such a table is
/// refused instead. Reads the footers of those files only.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` naming the column and the file; the error of
/// a footer read.
async fn refuse_missing_mixed_identity(scan: &IcebergScan) -> Result<(), KrishivError> {
    let snapshot = &scan.snapshot;
    let identity_of = |e: &DataFileEntry, id: i32| {
        snapshot
            .partition_specs
            .get(&e.file.spec_id)
            .is_some_and(|spec| {
                spec.fields
                    .iter()
                    .any(|f| f.transform == Transform::Identity && f.source_id == id)
            })
    };
    let constant: BTreeSet<&str> = scan
        .parts
        .constant_columns
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    let mixed: Vec<_> = snapshot
        .fields
        .iter()
        .filter(|f| f.path.0.len() == 1 && !constant.contains(f.path.root()))
        .filter(|f| scan.entries.iter().any(|e| identity_of(e, f.id)))
        .collect();
    if mixed.is_empty() {
        return Ok(());
    }
    let format = ParquetFormat;
    for (e, file) in scan.entries.iter().zip(scan.files.iter()) {
        let wanted: Vec<_> = mixed.iter().filter(|c| identity_of(e, c.id)).collect();
        if wanted.is_empty() {
            continue;
        }
        let meta = scan
            .parts
            .io
            .read_metadata(
                &format,
                file,
                krishiv_io::MetadataNeeds::default(),
                &scan.parts.format_options,
            )
            .await?;
        let schema = meta.schema();
        let has_ids = schema
            .fields()
            .iter()
            .any(|f| f.metadata().contains_key(krishiv_io::FIELD_ID_KEY));
        for c in wanted {
            let present = if has_ids {
                let id = c.id.to_string();
                schema
                    .fields()
                    .iter()
                    .any(|f| f.metadata().get(krishiv_io::FIELD_ID_KEY) == Some(&id))
            } else {
                // Positional fallback ids (FC S4).
                usize::try_from(c.id).is_ok_and(|k| k >= 1 && k <= schema.fields().len())
            };
            if !present {
                return Err(unsupported(format!(
                    "reading the Iceberg identity partition column {} from the manifest for \
                     some data files but not others: the file {} lacks the column",
                    c.path.0.join("."),
                    e.file.path
                )));
            }
        }
    }
    Ok(())
}
