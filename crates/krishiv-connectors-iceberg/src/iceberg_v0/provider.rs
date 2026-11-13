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
//! - **Deletes** (design note §3.2–3.3, PR-25b): assigned to the data files
//!   a built scan reads, after filter pruning, and read when its first batch
//!   is asked for ([`super::lazy`], PR-25d). Position delete files and
//!   deletion vectors become each data file's row selection
//!   ([`super::deletes`]), so deleted rows are never decoded. Equality delete
//!   files (PR-25c) are read into key sets, and the scan's source drops the
//!   rows whose key is in a set of their data file ([`super::eq_deletes`]).
//! - **Filters** prune data files by partition values and metrics through
//!   [`Pruner`] when the scan is built; every filter is still evaluated on the
//!   rows ([`FilterPushdown::Inexact`]).

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::sync::{Arc, OnceLock};

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
use krishiv_optimizer::TableStats;

use super::eq_deletes::{self, EqProjection, FILE_ORDINAL};
use super::lazy::{self, DeleteEnv, DeleteSource, DeleteState, LazySelections, Locate, ScanTask};
use super::table::IcebergTable;
use super::types::arrow_schema;
use crate::plan::deletes::DeleteIndex;
use crate::plan::{
    DataFileEntry, FileFormatKind, PartitionValue, PruneOptions, Pruner, ScanPlan, TableSnapshot,
    Transform, VersionSpec, WriterMetrics, decode,
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
    /// two DVs of one data file); `KRISHIV_UNSUPPORTED_FEATURE` for a data file
    /// that is not Parquet; `INTERNAL_ERROR` for a partition value that does
    /// not decode. Deletes are assigned and read, and the deletes and
    /// mixed-spec identity columns that cannot be applied refused, by the
    /// scans that read the files they concern (see the module docs).
    pub async fn open(
        table: &IcebergTable,
        version: &VersionSpec,
        config: &Config,
        worker_slots: u64,
    ) -> Result<Self, KrishivError> {
        let snapshot = table.snapshot(version)?;
        let (plan, index) = table.plan_files(&snapshot).await?;
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
        let reg = table.registered.clone();
        let locate: Locate = Arc::new(move |path: &str| -> Result<FileRef, KrishivError> {
            let at = reg.paths.locate(path)?;
            Ok(FileRef {
                store: at.store,
                path: at.path,
                size: 0,
                version: VersionToken::Immutable,
                modified_micros: None,
            })
        });
        let metrics = table.writer_metrics(&snapshot);
        let scan = IcebergScan::new(
            snapshot,
            plan,
            index,
            arrow,
            name_mapping,
            table.registered.io.clone(),
            locate,
            config,
            worker_slots,
            metrics,
        )?;
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
        self.scan.parts.ordinal.as_ref()?;
        let columns = eq_deletes::equality_columns(
            &self.scan.snapshot.fields,
            self.scan.index.equality_files(),
        )
        .ok()?;
        Some(self.scan.with_equality_columns(p, &columns).0)
    }
}

#[cfg(test)]
impl IcebergTableProvider {
    /// The delete state of a scan of every file, loading into `pool`.
    pub(super) fn delete_state(
        &self,
        pool: &Arc<dyn krishiv_memory::MemoryPool>,
    ) -> Option<Arc<DeleteState>> {
        let (state, _) = self.scan.deletes_of(&self.scan.base).ok()??;
        state.set_pool(pool);
        Some(Arc::new(state))
    }
}

impl TableProvider for IcebergTableProvider {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        Ok(Arc::clone(&self.scan) as Arc<dyn ScanPushdown>)
    }

    /// From the manifests ([`super::stats`]), worked out once.
    fn statistics(&self) -> Option<TableStats> {
        let s = &self.scan;
        Some(
            s.stats
                .get_or_init(|| {
                    super::stats::table_stats(&s.entries, &s.index, &s.snapshot.fields, &s.metrics)
                })
                .clone(),
        )
    }

    fn scan_units(&self) -> Option<u64> {
        Some(super::stats::scan_units(&self.scan.entries))
    }
}

/// The scan factory of an [`IcebergTableProvider`]: a [`FileScan`] over the
/// snapshot's data files that prunes them by the pushed filters when built,
/// and assigns the deletes of the files it reads ([`super::lazy`]).
#[derive(Clone)]
struct IcebergScan {
    snapshot: Arc<TableSnapshot>,
    /// The files, aligned with `entries`.
    files: Arc<[FileRef]>,
    entries: Arc<[Arc<DataFileEntry>]>,
    /// Each file's index in `files`, by store and path.
    ordinals: Arc<HashMap<(String, String), usize>>,
    /// The snapshot's delete files, assigned when a scan is built.
    index: Arc<DeleteIndex>,
    /// What the deletes of a built scan are loaded with.
    env: Arc<DeleteEnv>,
    mapping: FieldMapping,
    /// The writers' metrics modes, for pruning by file metrics.
    metrics: Arc<WriterMetrics>,
    parts: ScanParts,
    /// The scan over every file (or a distributed task's given splits).
    base: FileScan,
    /// Whether `build` prunes files (not for given splits).
    prune: bool,
    /// The table's statistics, once asked for.
    stats: Arc<OnceLock<TableStats>>,
}

impl fmt::Debug for IcebergScan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IcebergScan")
            .field("snapshot_id", &self.snapshot.snapshot_id)
            .field("files", &self.files.len())
            .finish_non_exhaustive()
    }
}

/// A file's key in [`IcebergScan::ordinals`].
fn file_key(f: &FileRef) -> (String, String) {
    (f.store.key().to_owned(), f.path.as_ref().to_owned())
}

impl IcebergScan {
    #[expect(
        clippy::too_many_arguments,
        reason = "one private constructor of the provider's parts"
    )]
    fn new(
        snapshot: TableSnapshot,
        plan: ScanPlan,
        index: DeleteIndex,
        table: SchemaRef,
        name_mapping: Option<NameMapping>,
        io: IoContext,
        locate: Locate,
        config: &Config,
        worker_slots: u64,
        metrics: WriterMetrics,
    ) -> Result<Self, KrishivError> {
        let any_equality = index.has_equality_deletes();
        let entries: Vec<Arc<DataFileEntry>> = plan.tasks.into_iter().map(|t| t.file).collect();
        let mut files = Vec::with_capacity(entries.len());
        let mut ordinals = HashMap::with_capacity(entries.len());
        for (i, e) in entries.iter().enumerate() {
            if e.file.format != FileFormatKind::Parquet {
                return Err(unsupported(format!(
                    "reading Iceberg {:?} data files ({})",
                    e.file.format, e.file.path
                )));
            }
            let mut f = locate(&e.file.path)?;
            f.size = e.file.size;
            ordinals.insert(file_key(&f), i);
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
        let constant_names: BTreeSet<&str> =
            constant_columns.iter().map(|f| f.name().as_str()).collect();
        let mixed = lazy::mixed_columns(&snapshot, &constant_names, &entries);
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
        let snapshot = Arc::new(snapshot);
        let env = DeleteEnv {
            io: parts.io.clone(),
            locate,
            table: Arc::clone(&parts.table),
            snapshot: Arc::clone(&snapshot),
            files: files.len(),
            mixed,
            mapping: mapping.clone(),
            format_options: parts.format_options.clone(),
        };
        Ok(Self {
            snapshot,
            base: parts.file_scan(files.clone()),
            files: files.into(),
            entries: entries.into(),
            ordinals: Arc::new(ordinals),
            index: Arc::new(index),
            env: Arc::new(env),
            mapping,
            metrics: Arc::new(metrics),
            parts,
            prune: true,
            stats: Arc::default(),
        })
    }

    /// The deletes of the files `scan` reads, and the equality columns they
    /// need read (`None`: no file of `scan` has deletes or needs the
    /// mixed-spec check, and the table has no equality deletes).
    ///
    /// # Errors
    /// The errors of [`DeleteIndex::for_data_file`] for a file `scan` reads;
    /// the equality-id errors of [`eq_deletes::equality_columns`];
    /// `INTERNAL_ERROR` for a file the provider does not have.
    fn deletes_of(
        &self,
        scan: &FileScan,
    ) -> Result<Option<(DeleteState, Vec<ColumnPath>)>, KrishivError> {
        let mut tasks = Vec::new();
        for f in scan.files() {
            let ordinal = *self
                .ordinals
                .get(&file_key(f))
                .ok_or_else(|| lazy::unknown_file(f))?;
            let entry = &self.entries[ordinal];
            let deletes = self.index.for_data_file(entry)?;
            if !deletes.is_empty() || self.env.needs_mixed_check(entry) {
                tasks.push(ScanTask {
                    ordinal,
                    entry: Arc::clone(entry),
                    file: f.clone(),
                    deletes,
                });
            }
        }
        if tasks.is_empty() && self.parts.ordinal.is_none() {
            return Ok(None);
        }
        let columns = eq_deletes::equality_columns(
            &self.snapshot.fields,
            tasks.iter().flat_map(|t| t.deletes.equality_deletes.iter()),
        )?;
        Ok(Some((
            DeleteState::new(Arc::clone(&self.env), tasks),
            columns,
        )))
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
    /// `p` with the table's field mapping.
    fn mapped(&self, mut p: ScanPushdowns) -> ScanPushdowns {
        p.mapping = self.mapping.clone();
        p
    }

    /// The pushdowns of the inner scan of an [`EqDeleteSource`] for `p`: the
    /// equality columns and [`FILE_ORDINAL`] are read too, and no limit is
    /// pushed (deleted rows would count towards it). Also the paths of the
    /// output in the inner batches: `p`'s columns, then `_metadata`.
    fn with_equality_columns(
        &self,
        mut p: ScanPushdowns,
        columns: &[ColumnPath],
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
        read.extend(columns.iter().cloned());
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
        let mut p = self.mapped(pushdowns);
        let scan = self.pruned(&p.filters);
        let Some((state, columns)) = self.deletes_of(&scan)? else {
            return scan.build(p);
        };
        let state = Arc::new(state);
        if state.has_positions() {
            p.selection = Some(Arc::new(LazySelections(Arc::clone(&state))));
        }
        if self.parts.ordinal.is_none() {
            let source = scan.build(p)?;
            return Ok(Arc::new(DeleteSource::new(source, state, None)));
        }
        let (inner, output) = self.with_equality_columns(p, &columns);
        let source = scan.build(inner)?;
        let eq = EqProjection::new(&source.schema(), output)?;
        Ok(Arc::new(DeleteSource::new(source, state, Some(eq))))
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
        if self.parts.ordinal.is_none() {
            return self.base.output_schema(&p);
        }
        // The equality columns are projected away: the output does not
        // depend on them.
        let (inner, output) = self.with_equality_columns(p, &[]);
        let schema = self.base.output_schema(&inner)?;
        krishiv_io::project_schema(&schema, &output)
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
