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
//!   with a name mapping (`schema.name-mapping.default`, PR-25c) such a file
//!   takes the ids the mapping gives its columns' names
//!   ([`WithoutFieldIds::Names`]), and an unmapped column is not read.
//! - **Identity partitions** are per-file constants from the manifest
//!   ([`FileConstants`]): they win over the file's column and fill a column the
//!   file lacks (FC S4). A column is constant when every data file's spec has an
//!   identity field on it (a top-level column); otherwise it is read from the
//!   files.
//! - **Deletes** (design note §3.2–3.3, PR-25b): position delete files and
//!   deletion vectors are read when the provider opens and become each data
//!   file's row selection ([`super::deletes`]), so deleted rows are never
//!   decoded. Equality delete files (PR-25c) are read into key sets when the
//!   provider opens, and the scan's source drops the rows whose key is in a
//!   set of their data file ([`super::eq_deletes`]).
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
    ColumnPath, FieldMapping, FileConstants, FileFormat, FileRef, FormatOptions, IoContext,
    NameMapping, PushdownExpr, ScalarValue, ScanOptions, VersionToken, WithoutFieldIds,
};

use super::deletes::DeleteSelections;
use super::eq_deletes::{self, EqDeleteSource, EqualityDeletes, FILE_ORDINAL};
use super::table::IcebergTable;
use super::types::arrow_schema;
use crate::plan::{
    DataFileEntry, DeleteFileEntry, FileFormatKind, ManifestFilter, PartitionValue, PruneOptions,
    Pruner, ScanPlan, TableSnapshot, TaskDeletes, Transform, VersionSpec, WriterMetrics, decode,
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
    /// errors of [`IcebergTable::plan`] (`KRISHIV_ICEBERG_INVALID_DELETES` for
    /// deletes that cannot be assigned); the errors of reading the delete
    /// files (`deletes::load`, `eq_deletes::load`); `KRISHIV_UNSUPPORTED_FEATURE` for
    /// an equality delete that cannot be applied, a data file that is not
    /// Parquet, or a file that lacks a column its spec partitions by identity
    /// when other files' specs do not (see `refuse_missing_mixed_identity`);
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
            .get(NAME_MAPPING)
            .map(|text| super::types::parse_name_mapping(text))
            .transpose()?;
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
        let mut scan = scan;
        let tasks: Vec<_> = scan
            .entries
            .iter()
            .zip(scan.files.iter())
            .zip(scan.deletes.iter())
            .filter(|(_, d)| !d.is_empty())
            .map(|((e, f), d)| (e.as_ref(), f, d.as_ref()))
            .collect();
        if !tasks.is_empty() {
            let selections = super::deletes::load(&reg.io, &locate, &tasks).await?;
            if !selections.is_empty() {
                scan.selections = Some(Arc::new(selections));
            }
        }
        if scan.parts.ordinal.is_some() {
            let per_file: Vec<&[Arc<DeleteFileEntry>]> = scan
                .deletes
                .iter()
                .map(|d| d.equality_deletes.as_slice())
                .collect();
            let eq = eq_deletes::load(
                &reg.io,
                &locate,
                &scan.parts.table,
                // Java's delete loader reads delete files without the name
                // mapping (writers give them ids): a file without ids takes
                // the positional fallback.
                &FieldMapping::FieldIdOr(WithoutFieldIds::Position),
                &scan.snapshot.fields,
                &per_file,
            )
            .await?;
            scan.equality = Some(Arc::new(eq));
        }
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

#[cfg(test)]
impl IcebergTableProvider {
    /// The pushdowns the inner file scan of an equality-delete scan gets for
    /// `p` (`None`: the scan has no equality deletes).
    pub(super) fn equality_inner_pushdowns(&self, p: ScanPushdowns) -> Option<ScanPushdowns> {
        let eq = self.scan.equality.as_ref()?;
        Some(self.scan.with_equality_columns(self.scan.mapped(p), eq).0)
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
    /// The deletes of each file, aligned with `entries`.
    deletes: Arc<[Arc<TaskDeletes>]>,
    /// The rows of each data file with deletes (`None`: no file has any).
    selections: Option<Arc<DeleteSelections>>,
    /// The equality deletes, by file index (`None`: no file has any).
    equality: Option<Arc<EqualityDeletes>>,
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
        name_mapping: Option<NameMapping>,
        io: IoContext,
        locate: &dyn Fn(&str) -> Result<FileRef, KrishivError>,
        config: &Config,
        worker_slots: u64,
        metrics: WriterMetrics,
    ) -> Result<Self, KrishivError> {
        let any_equality = plan
            .tasks
            .iter()
            .any(|t| !t.deletes.equality_deletes.is_empty());
        let (entries, deletes): (Vec<Arc<DataFileEntry>>, Vec<Arc<TaskDeletes>>) =
            plan.tasks.into_iter().map(|t| (t.file, t.deletes)).unzip();
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
        let (constant_columns, mut constants) = constants(&snapshot, &table, &entries, &files)?;
        // With equality deletes every file also carries its index in the
        // scan, the key the deletes' source finds its sets by.
        let ordinal = if any_equality {
            let mut with_ordinal = FileConstants::new();
            for (i, f) in files.iter().enumerate() {
                let i = i32::try_from(i)
                    .map_err(|_| unsupported("more than 2^31 data files in a scan".to_owned()))?;
                let mut values = constants
                    .get(&f.path)
                    .map(<[_]>::to_vec)
                    .unwrap_or_default();
                values.push(ScalarValue::Int32(i));
                with_ordinal.insert(f.path.clone(), values);
            }
            constants = with_ordinal;
            Some(eq_deletes::with_ordinal(&table))
        } else {
            None
        };
        // Files without field ids: by the table's name mapping, else by
        // Java's positional fallback ids.
        let mapping = FieldMapping::FieldIdOr(match name_mapping {
            Some(m) => WithoutFieldIds::Names(Arc::new(m)),
            None => WithoutFieldIds::Position,
        });
        let parts = ScanParts {
            table,
            constant_columns,
            constants: Arc::new(constants),
            ordinal,
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
            deletes: deletes.into(),
            selections: None,
            equality: None,
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
    /// The values of `constant_columns` and, with `ordinal`, the file
    /// ordinal last.
    constants: Arc<FileConstants>,
    /// With equality deletes: the table with [`FILE_ORDINAL`] appended, the
    /// schema the file scan reads.
    ordinal: Option<SchemaRef>,
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
            Arc::clone(self.ordinal.as_ref().unwrap_or(&self.table)),
            files,
            self.format_options.clone(),
            self.scan_options,
            self.splits,
        );
        let mut columns = self.constant_columns.clone();
        if self.ordinal.is_some() {
            columns.push(Arc::new(eq_deletes::ordinal_field()));
        }
        if columns.is_empty() {
            scan
        } else {
            scan.with_file_constants(columns, Arc::clone(&self.constants))
        }
    }
}

impl IcebergScan {
    /// `p` with the table's field mapping and the files' delete selections.
    fn mapped(&self, mut p: ScanPushdowns) -> ScanPushdowns {
        p.mapping = self.mapping.clone();
        if let Some(s) = &self.selections {
            p.selection = Some(Arc::clone(s) as Arc<dyn krishiv_exec::FileSelections>);
        }
        p
    }

    /// The pushdowns of the inner scan of an [`EqDeleteSource`] for `p`: the
    /// equality columns and [`FILE_ORDINAL`] are read too, and no limit is
    /// pushed (deleted rows would count towards it). Also the paths of the
    /// output in the inner batches: `p`'s columns, then `_metadata`.
    fn with_equality_columns(
        &self,
        mut p: ScanPushdowns,
        eq: &EqualityDeletes,
    ) -> (ScanPushdowns, Vec<ColumnPath>) {
        let wanted: Vec<ColumnPath> = p.projection.take().unwrap_or_else(|| {
            self.parts
                .table
                .fields()
                .iter()
                .map(|f| ColumnPath::new(f.name()))
                .collect()
        });
        let mut read = wanted.clone();
        read.extend(eq.columns().iter().cloned());
        read.push(ColumnPath::new(FILE_ORDINAL));
        let mut output = wanted;
        if eq_deletes::any_metadata(&p.metadata) {
            output.push(ColumnPath::new(krishiv_exec::METADATA_COLUMN));
        }
        p.projection = Some(read);
        p.limit = None;
        (p, output)
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

    fn supports_selection(&self) -> bool {
        // Deletes are pushed as selections; Parquet honours them.
        self.base.supports_selection()
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
        let scan = self.pruned(&p.filters);
        match &self.equality {
            None => scan.build(p),
            Some(eq) => {
                let (inner, output) = self.with_equality_columns(p, eq);
                let source = scan.build(inner)?;
                Ok(Arc::new(EqDeleteSource::new(
                    source,
                    Arc::clone(eq),
                    output,
                )?))
            }
        }
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
        let p = self.mapped(pushdowns.clone());
        match &self.equality {
            None => self.base.output_schema(&p),
            Some(eq) => {
                let (inner, output) = self.with_equality_columns(p, eq);
                let schema = self.base.output_schema(&inner)?;
                krishiv_io::project_schema(&schema, &output)
            }
        }
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
    // One footer read per file whose spec makes a mixed column an identity
    // partition, one after another: only tables whose specs disagree on a
    // column pay it (rare; spec evolution after an import). Bounded
    // concurrency belongs with 25d's planning hook.
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
        let mut schema = meta.schema();
        // Any id at any depth counts, as `SchemaMapper` decides (FC S4); a
        // file without ids takes the name mapping's, as the read does.
        let mut has_ids = krishiv_io::has_field_ids(schema.fields());
        if !has_ids && let FieldMapping::FieldIdOr(WithoutFieldIds::Names(m)) = &scan.mapping {
            schema = m.apply(&schema);
            has_ids = true;
        }
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
