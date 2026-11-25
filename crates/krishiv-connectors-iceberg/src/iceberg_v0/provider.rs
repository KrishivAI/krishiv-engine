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

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use arrow_schema::{FieldRef, SchemaRef};
use krishiv_common::config::{
    Config, ConfigValue, ICEBERG_COMMIT_MAX_INTERVENING, ICEBERG_WRITE_MAX_OPEN_FILES,
    ICEBERG_WRITE_STATISTICS_ENABLED,
};
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::planner::{
    RowLevelOperation, RowLevelTable, TableCapabilities, TableProvider, WriteMode, WriteRequest,
};
use krishiv_exec::{
    FileScan, FilterPushdown, PushedFilter, ROW_FILE_COLUMN, ROW_POSITION_COLUMN, ScanPushdown,
    ScanPushdowns, Sink, Source, TakeRows,
};
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::split::{FilePartition, SplitConfig};
use krishiv_io::{
    ColumnPath, FieldMapping, FileConstants, FileFormat, FileRef, FormatOptions, IoContext,
    NameMapping, PushdownExpr, RowFile, ScalarValue, ScanOptions, TakeRequest, TakeStream,
    VersionToken, WithoutFieldIds,
};
use krishiv_optimizer::TableStats;

use super::eq_deletes::{self, EqProjection, FILE_ORDINAL};
use super::lazy::{self, DeleteEnv, DeleteSource, DeleteState, LazySelections, Locate, ScanTask};
use super::lineage::{self, LineageSource, LineageSplit};
use super::sink::{IcebergWriteSink, WriteTarget};
use super::statistics_file::enabled as statistics_enabled;
use super::table::IcebergTable;
use super::types::arrow_schema;
use super::write::DEFAULT_MAX_OPEN_FILES;
use crate::plan::deletes::DeleteIndex;
use crate::plan::{
    DataFileEntry, FileFormatKind, PartitionValue, PruneOptions, Pruner, ScanPlan, TableSnapshot,
    Transform, VersionSpec, WriterMetrics, decode,
};
use futures::future::BoxFuture;

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
    /// Where an append goes: set when the table has a committer and the
    /// provider reads its current state ([`super::sink`]).
    write: Option<Arc<WriteTarget>>,
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
        let mut provider = Self::from_plan(table, snapshot, plan, index, config, worker_slots)?;
        // A time-travelled read is not a write target.
        if let (Some(committer), VersionSpec::Current) = (&table.committer, version) {
            let max_open_files = config
                .get(&ICEBERG_WRITE_MAX_OPEN_FILES)
                .and_then(ConfigValue::as_int)
                .and_then(|n| usize::try_from(n).ok())
                .unwrap_or(DEFAULT_MAX_OPEN_FILES);
            let statistics = config
                .get(&ICEBERG_WRITE_STATISTICS_ENABLED)
                .and_then(ConfigValue::as_bool)
                .or_else(|| {
                    ICEBERG_WRITE_STATISTICS_ENABLED
                        .default
                        .as_ref()
                        .and_then(ConfigValue::as_bool)
                })
                .unwrap_or(false);
            let base = table.table.metadata();
            provider.write = Some(Arc::new(WriteTarget {
                committer: Arc::clone(committer),
                statistics: statistics_enabled(statistics, base.properties()),
                base: base.clone(),
                registered: table.registered.clone(),
                file_io: table.table.file_io().clone(),
                max_metadata_bytes: table.max_metadata_bytes,
                max_open_files,
                max_intervening: config
                    .get(&ICEBERG_COMMIT_MAX_INTERVENING)
                    .and_then(ConfigValue::as_int)
                    .and_then(|n| usize::try_from(n).ok())
                    .unwrap_or(super::validate::DEFAULT_MAX_INTERVENING),
            }));
        }
        Ok(provider)
    }

    /// The provider of `snapshot`'s planned files.
    fn from_plan(
        table: &IcebergTable,
        snapshot: TableSnapshot,
        plan: ScanPlan,
        index: DeleteIndex,
        config: &Config,
        worker_slots: u64,
    ) -> Result<Self, KrishivError> {
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
        let ndv = snapshot.snapshot_id.map_or_else(HashMap::new, |id| {
            super::puffin_stats::ndv_by_field(table.table.metadata(), id)
        });
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
            ndv,
        )?;
        Ok(Self {
            scan: Arc::new(scan),
            write: None,
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

    /// The table's scan with the v3 row lineage columns `_row_id` and
    /// `_last_updated_sequence_number` after the table's columns
    /// ([`super::lineage`]): what a copy-on-write rewrite reads, so the rows
    /// it moves keep their lineage (design note §8 item 3). Deletes apply
    /// before the values are derived. A scan that asks for neither column
    /// reads as [`TableProvider::scan`] does.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` on a format v1 or v2 table and when the
    /// table has a column named like either lineage column.
    #[cfg(test)]
    pub(crate) fn lineage_scan(&self) -> Result<Arc<dyn ScanPushdown>, KrishivError> {
        lineage::check_table(self.scan.snapshot.format_version, &self.scan.parts.table)?;
        Ok(Arc::new(self.scan.with_lineage()) as Arc<dyn ScanPushdown>)
    }

    /// The scan a row-level statement reads (`row_level.rs`): the table's
    /// scan, or with `lineage` its lineage scan; restricted to `files` (values
    /// of `_krishiv_row_file`) when given.
    ///
    /// # Errors
    /// [`Self::lineage_scan`]'s; `INTERNAL_ERROR` for a file value that does
    /// not decode or a file the snapshot does not hold.
    pub(super) fn row_level_scan(
        &self,
        lineage: bool,
        files: Option<&[Vec<u8>]>,
    ) -> Result<Arc<dyn ScanPushdown>, KrishivError> {
        let scan = if lineage {
            lineage::check_table(self.scan.snapshot.format_version, &self.scan.parts.table)?;
            self.scan.with_lineage()
        } else {
            (*self.scan).clone()
        };
        let scan = match files {
            None => scan,
            Some(files) => scan.restricted(&self.ordinals_of(files)?),
        };
        Ok(Arc::new(scan))
    }

    /// The snapshot's entries of `files` (values of `_krishiv_row_file`), in
    /// plan order, each with the deletes that apply to it.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a value that does not decode or a file the
    /// snapshot does not hold; the delete assignment's errors.
    pub(super) fn touched(
        &self,
        files: &[Vec<u8>],
    ) -> Result<Vec<(Arc<DataFileEntry>, crate::plan::TaskDeletes)>, KrishivError> {
        self.ordinals_of(files)?
            .into_iter()
            .map(|i| {
                let e = Arc::clone(&self.scan.entries[i]);
                let d = self.scan.index.for_data_file(&e)?;
                Ok((e, d))
            })
            .collect()
    }

    /// Every live data file of the pinned snapshot as a `rewrite_data_files` candidate, with the
    /// `_krishiv_row_file` value that restricts a [`Self::row_level_scan`] to it. `default_spec`
    /// is the table's default partition spec id.
    ///
    /// # Errors
    /// The delete assignment's errors.
    #[expect(dead_code, reason = "group execution (next commit of 36e-2a) calls it")]
    pub(super) fn rewrite_candidates(
        &self,
        default_spec: i32,
    ) -> Result<Vec<(super::maintenance::rewrite_plan::CandidateFile, Vec<u8>)>, KrishivError> {
        use crate::plan::PartitionValue;
        let mut out = Vec::with_capacity(self.scan.entries.len());
        for (e, f) in self.scan.entries.iter().zip(self.scan.files.iter()) {
            let d = self.scan.index.for_data_file(e)?;
            let file_scoped =
                |x: &Arc<crate::plan::DeleteFileEntry>| x.entry.file.referenced_data_file.is_some();
            let dv = d.deletion_vector.iter();
            let scoped: u64 = dv
                .chain(d.position_deletes.iter().filter(|x| file_scoped(x)))
                .map(|x| x.entry.file.record_count)
                .sum();
            let partition = e
                .file
                .partition
                .iter()
                .map(|v| match v {
                    PartitionValue::Null => "null".to_owned(),
                    PartitionValue::Value(b) => format!("v{b:?}"),
                    PartitionValue::Unreadable => "unreadable".to_owned(),
                })
                .collect();
            let saturate = |n: u64| i64::try_from(n).unwrap_or(i64::MAX);
            out.push((
                super::maintenance::rewrite_plan::CandidateFile {
                    path: e.file.path.clone(),
                    partition,
                    default_spec: e.file.spec_id == default_spec,
                    length: saturate(e.file.size),
                    record_count: saturate(e.file.record_count),
                    delete_files: d.position_deletes.len()
                        + d.equality_deletes.len()
                        + usize::from(d.deletion_vector.is_some()),
                    file_scoped_deleted_records: saturate(scoped),
                },
                RowFile::encode(f),
            ));
        }
        Ok(out)
    }

    /// The scan's ordinals of `files`, sorted, each once.
    fn ordinals_of(&self, files: &[Vec<u8>]) -> Result<Vec<usize>, KrishivError> {
        let mut out = Vec::with_capacity(files.len());
        for bytes in files {
            let f = RowFile::decode(bytes)?;
            let i = *self
                .scan
                .ordinals
                .get(&file_key(&f))
                .ok_or_else(|| lazy::unknown_file(&f))?;
            out.push(i);
        }
        out.sort_unstable();
        out.dedup();
        Ok(out)
    }

    /// The I/O and file lookup the scan's deletes are read with (a
    /// merge-on-read write reads the deletes it absorbs through them).
    pub(super) fn delete_io(&self) -> (krishiv_io::IoContext, super::lazy::Locate) {
        (self.scan.env.io.clone(), Arc::clone(&self.scan.env.locate))
    }

    /// The write target (a committer and the current version).
    pub(super) fn write_target(&self) -> Option<&Arc<WriteTarget>> {
        self.write.as_ref()
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
    /// [`Self::open`] with the snapshot and its planned files changed by
    /// `edit` first (states iceberg-rust's writers cannot produce); the
    /// delete index is built from the edited plan.
    pub(super) async fn open_edited(
        table: &IcebergTable,
        edit: impl FnOnce(&mut TableSnapshot, &mut ScanPlan),
    ) -> Result<Self, KrishivError> {
        let mut snapshot = table.snapshot(&VersionSpec::Current)?;
        let (mut plan, _) = table.plan_files(&snapshot).await?;
        edit(&mut snapshot, &mut plan);
        let metrics = table.writer_metrics(&snapshot);
        let index = DeleteIndex::new(&plan.delete_files, &snapshot.partition_specs)?
            .with_fields(&snapshot.fields, &metrics);
        Self::from_plan(table, snapshot, plan, index, &Config::new(), 4)
    }

    /// How many files a scan with the pushed filter `expr` opens.
    pub(super) fn files_read(&self, expr: PushdownExpr) -> usize {
        self.scan.pruned(&[expr]).files().len()
    }
}

#[cfg(test)]
impl IcebergTableProvider {
    /// How many splits the file scan of the lineage scan plans.
    pub(super) fn lineage_splits(&self) -> usize {
        self.scan
            .with_lineage()
            .base
            .source(ScanPushdowns::default())
            .map_or(0, |s| s.partitions().len())
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
        let planned = self.scan.planned(&[]).ok()?;
        let state = DeleteState::new(Arc::clone(&self.scan.env), planned.tasks.clone());
        state.set_pool(pool);
        Some(Arc::new(state))
    }
}

impl TableProvider for IcebergTableProvider {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        Ok(Arc::clone(&self.scan) as Arc<dyn ScanPushdown>)
    }

    /// From the manifests (`stats.rs`), worked out once.
    fn statistics(&self) -> Option<TableStats> {
        let s = &self.scan;
        Some(
            s.stats
                .get_or_init(|| {
                    super::stats::table_stats(
                        &s.entries,
                        &s.index,
                        &s.snapshot.fields,
                        &s.metrics,
                        &s.ndv,
                    )
                })
                .clone(),
        )
    }

    fn scan_units(&self) -> Option<u64> {
        Some(super::stats::scan_units(&self.scan.entries))
    }

    /// `BATCH_WRITE`, `TRUNCATE`, `OVERWRITE_BY_FILTER` and
    /// `OVERWRITE_DYNAMIC` (as Iceberg's `SparkTable`) when the table has a
    /// committer and the provider reads its current state; else none.
    fn capabilities(&self) -> TableCapabilities {
        let mut caps = TableCapabilities::default();
        let writes = self.write.is_some();
        caps.batch_write = writes;
        caps.truncate = writes;
        caps.overwrite_by_filter = writes;
        caps.overwrite_dynamic = writes;
        caps
    }

    /// A copy-on-write row-level table (`row_level.rs`) when the provider
    /// can write (a committer, the current version).
    fn row_level(&self, op: RowLevelOperation) -> Option<Arc<dyn RowLevelTable>> {
        self.write.as_ref()?;
        Some(Arc::new(super::row_level::IcebergRowLevel::new(
            self.clone(),
            op,
        )))
    }

    /// The append sink (`sink.rs`) or an overwrite's (`overwrite.rs`),
    /// committing when it finalizes.
    fn writer(&self, request: &WriteRequest) -> krishiv_exec::Result<Arc<dyn Sink>> {
        self.capabilities()
            .check(&request.mode, &request.table_name)?;
        let target = self
            .write
            .as_ref()
            .ok_or_else(|| internal("a write capability without a target".to_owned()))?;
        if request.mode == WriteMode::Append {
            return Ok(Arc::new(IcebergWriteSink::try_new(
                Arc::clone(target),
                request,
            )?));
        }
        Ok(Arc::new(super::overwrite::sink(
            Arc::clone(target),
            request,
        )?))
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
    /// The distinct count of each column with a Theta blob in the pinned snapshot's statistics file.
    ndv: Arc<HashMap<i32, u64>>,
    /// The table's statistics, once asked for.
    stats: Arc<OnceLock<TableStats>>,
    /// The last filter list planned and its plan ([`Planned`]): `explain`
    /// and `build` of one scan share it, as do the scans of later queries
    /// with the same filters.
    planned: Arc<Mutex<Option<PlanMemo>>>,
}

/// A filter list and its [`Planned`].
type PlanMemo = (Vec<PushdownExpr>, Arc<Planned>);

/// A scan's files after pruning by its pushed filters, with the deletes that
/// apply to them (review 25d S2: worked out once per filter list).
struct Planned {
    scan: FileScan,
    /// The files with deletes or that need the mixed-spec check.
    tasks: Vec<ScanTask>,
    /// The equality columns `tasks` need read.
    columns: Vec<ColumnPath>,
    /// The delete files of `tasks` by kind (position, DV, equality), each
    /// counted once.
    counts: [usize; 3],
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
        ndv: HashMap<i32, u64>,
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
                    .get(&f.store, &f.path)
                    .map(<[_]>::to_vec)
                    .unwrap_or_default();
                values.push(ScalarValue::Int32(i));
                with_ordinal.insert(f.store.clone(), f.path.clone(), values);
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
            lineage: false,
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
            ndv: Arc::new(ndv),
            stats: Arc::default(),
            planned: Arc::default(),
        })
    }

    /// The files `filters` leave and the deletes that apply to them
    /// ([`Planned`]), from the memo when the filters are the last ones
    /// planned. A failed plan is not kept.
    ///
    /// # Errors
    /// The errors of [`DeleteIndex::for_data_file`] for a file the scan reads;
    /// the equality-id errors of [`eq_deletes::equality_columns`];
    /// `INTERNAL_ERROR` for a file the provider does not have.
    fn planned(&self, filters: &[PushedFilter]) -> Result<Arc<Planned>, KrishivError> {
        let exprs: Vec<PushdownExpr> = filters.iter().filter_map(|f| f.expr.clone()).collect();
        if let Some((k, p)) = &*self.planned.lock().unwrap_or_else(PoisonError::into_inner)
            && *k == exprs
        {
            return Ok(Arc::clone(p));
        }
        let scan = self.pruned(&exprs);
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
        let columns = eq_deletes::equality_columns(
            &self.snapshot.fields,
            tasks.iter().flat_map(|t| t.deletes.equality_deletes.iter()),
        )?;
        let mut seen: HashSet<&str> = HashSet::new();
        let mut counts = [0_usize; 3];
        for t in &tasks {
            let d = &t.deletes;
            for (n, files) in [
                (0, d.position_deletes.as_slice()),
                (1, d.deletion_vector.as_slice()),
                (2, d.equality_deletes.as_slice()),
            ] {
                for f in files {
                    if seen.insert(f.entry.file.path.as_str()) {
                        counts[n] += 1;
                    }
                }
            }
        }
        drop(seen);
        let planned = Arc::new(Planned {
            scan,
            tasks,
            columns,
            counts,
        });
        *self.planned.lock().unwrap_or_else(PoisonError::into_inner) =
            Some((exprs, Arc::clone(&planned)));
        Ok(planned)
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
    /// A lineage scan ([`super::lineage`]): the file scan also reads the
    /// stored lineage values and the per-file `first_row_id` and data
    /// sequence number (after the ordinal in `constants`).
    lineage: bool,
    io: IoContext,
    format_options: FormatOptions,
    scan_options: ScanOptions,
    splits: SplitConfig,
}

impl ScanParts {
    /// The file scan of `files` (with Spark's split planning).
    fn file_scan(&self, files: Vec<FileRef>) -> FileScan {
        let mut read = Arc::clone(self.ordinal.as_ref().unwrap_or(&self.table));
        if self.lineage {
            read = lineage::with_inputs(&read);
        }
        let scan = FileScan::new(
            Arc::new(ParquetFormat) as Arc<dyn FileFormat>,
            self.io.clone(),
            read,
            files,
            self.format_options.clone(),
            self.scan_options,
            self.splits,
        );
        let mut columns = self.constant_columns.clone();
        if self.ordinal.is_some() {
            columns.push(Arc::new(eq_deletes::ordinal_field()));
        }
        if self.lineage {
            columns.extend(lineage::constant_fields());
        }
        if columns.is_empty() {
            scan
        } else {
            scan.with_file_constants(columns, Arc::clone(&self.constants))
        }
    }
}

impl IcebergScan {
    /// This scan with the lineage columns (see
    /// [`IcebergTableProvider::lineage_scan`]): every file's constants gain
    /// its `first_row_id` and data sequence number.
    fn with_lineage(&self) -> Self {
        if self.parts.lineage {
            return self.clone();
        }
        let mut scan = self.clone();
        let mut constants = FileConstants::new();
        for (e, f) in self.entries.iter().zip(self.files.iter()) {
            let mut values = self
                .parts
                .constants
                .get(&f.store, &f.path)
                .map(<[_]>::to_vec)
                .unwrap_or_default();
            values.extend(lineage::constants_of(e));
            constants.insert(f.store.clone(), f.path.clone(), values);
        }
        scan.parts.constants = Arc::new(constants);
        scan.parts.lineage = true;
        scan.base = scan.parts.file_scan(self.files.to_vec());
        scan.planned = Arc::default();
        scan
    }

    /// This scan reading only the files at `ordinals` (sorted): filters
    /// prune nothing more (a row-level scan pushes none).
    fn restricted(&self, ordinals: &[usize]) -> Self {
        let mut scan = self.clone();
        let files: Vec<FileRef> = ordinals
            .iter()
            .filter_map(|i| self.files.get(*i).cloned())
            .collect();
        scan.base = scan.parts.file_scan(files);
        scan.prune = false;
        scan.planned = Arc::default();
        scan
    }

    /// Whether `column` is read from the files by [`ScanPushdown::take`]: not
    /// an identity-partition constant, the equality deletes' file ordinal, a
    /// lineage column or `_metadata`.
    fn takes(&self, column: &str) -> bool {
        !self
            .parts
            .constant_columns
            .iter()
            .any(|f| f.name() == column)
            && column != FILE_ORDINAL
            && column != lineage::ROW_ID
            && column != lineage::LAST_UPDATED
            && column != krishiv_exec::METADATA_COLUMN
    }

    /// The table's columns and, for a lineage scan, the two lineage columns.
    fn outward_schema(&self) -> SchemaRef {
        let table = &self.parts.table;
        if !self.parts.lineage {
            return Arc::clone(table);
        }
        let mut fields: Vec<FieldRef> = table.fields().iter().cloned().collect();
        fields.extend(lineage::lineage_fields());
        Arc::new(arrow_schema::Schema::new_with_metadata(
            fields,
            table.metadata().clone(),
        ))
    }

    /// For a lineage scan: the inner pushdowns and the split of the caller's
    /// columns (`None`: the scan reads no lineage column, and `p` is
    /// returned with an explicit projection of the table's columns).
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for lineage columns with `_metadata`
    /// columns; [`LineageSplit::new`]'s.
    fn lineage_pushdowns(
        &self,
        mut p: ScanPushdowns,
    ) -> Result<(ScanPushdowns, Option<LineageSplit>), KrishivError> {
        let split = LineageSplit::new(&self.parts.table, p.projection.as_deref(), p.row_id)?;
        // Filters naming a lineage column are not pushed
        // (`filter_pushdown`); any that arrive stay with the filter above.
        p.filters.retain(|f| !names_lineage(f));
        p.projection = Some(split.inner.clone());
        // The row position is read as the file scan's row id, which does not
        // combine with `_metadata` (review nit 6: refused here, not as the
        // file scan's internal error).
        if (split.wanted || p.row_id) && eq_deletes::any_metadata(&p.metadata) {
            return Err(unsupported(
                "the Iceberg row lineage or row id columns together with _metadata columns"
                    .to_owned(),
            ));
        }
        if !split.wanted {
            return Ok((p, None));
        }
        p.row_id = true;
        Ok((p, Some(split)))
    }

    /// `p` with the table's field mapping.
    fn mapped(&self, mut p: ScanPushdowns) -> ScanPushdowns {
        p.mapping = self.mapping.clone();
        p
    }

    /// The pushdowns of the inner scan of an [`EqDeleteSource`] for `p`: the
    /// equality columns and [`FILE_ORDINAL`] are read too, and no limit is
    /// pushed (deleted rows would count towards it). Also the paths of the
    /// output in the inner batches: `p`'s columns, then `_metadata` or the
    /// row id.
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
        // The row id the file scan appends (a lineage scan's row positions).
        if p.row_id {
            output.push(ColumnPath::new(ROW_FILE_COLUMN));
            output.push(ColumnPath::new(ROW_POSITION_COLUMN));
        }
        p.projection = Some(read);
        p.limit = None;
        (p, output)
    }

    /// The file scan of the files that may hold rows passing `filters`.
    fn pruned(&self, exprs: &[PushdownExpr]) -> FileScan {
        if !self.prune || exprs.is_empty() {
            return self.base.clone();
        }
        let expr = PushdownExpr::And(exprs.to_vec());
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
        self.outward_schema()
    }

    fn supports_selection(&self) -> bool {
        // Deletes are pushed as selections; Parquet honours them.
        self.base.supports_selection()
    }

    fn filter_pushdown(&self, filter: &PushedFilter) -> FilterPushdown {
        // Files are pruned by it (a superset of the matching rows is kept);
        // the filter above the scan stays. A lineage column is not in the
        // files' columns the pruner and the file scan know.
        if self.parts.lineage && names_lineage(filter) {
            return FilterPushdown::Unsupported;
        }
        if filter.expr.is_some() {
            FilterPushdown::Inexact
        } else {
            FilterPushdown::Unsupported
        }
    }

    fn build(&self, pushdowns: ScanPushdowns) -> krishiv_exec::Result<Arc<dyn Source>> {
        if !self.parts.lineage {
            return self.build_inner(pushdowns);
        }
        let (inner, split) = self.lineage_pushdowns(pushdowns)?;
        let source = self.build_inner(inner)?;
        let Some(split) = split else {
            return Ok(source);
        };
        let projection = split.projection(&source.schema())?;
        Ok(Arc::new(LineageSource::new(source, projection)))
    }

    /// `Snapshot: <id>, DataFiles: <read>/<total>, DeleteFiles: [position=..,
    /// dv=.., equality=..]`: the delete files that apply to the data files
    /// the scan reads, each counted once. Read from the plan `build` uses
    /// ([`Planned`], worked out once per filter list); `None` when that plan
    /// fails (assigning a file's deletes), which `build` then reports.
    fn explain(&self, pushdowns: &ScanPushdowns) -> Option<String> {
        let planned = self.planned(&pushdowns.filters).ok()?;
        let snapshot = self
            .snapshot
            .snapshot_id
            .map_or_else(|| "none".to_owned(), |id| id.to_string());
        let [position, dv, equality] = planned.counts;
        Some(format!(
            "Snapshot: {snapshot}, DataFiles: {}/{}, DeleteFiles: [position={position}, \
             dv={dv}, equality={equality}]",
            planned.scan.files().len(),
            self.files.len(),
        ))
    }

    /// As the file scan's: rows of a built scan are read again by position.
    /// The positions a scan returned are live (its deletes applied), so a take
    /// applies no delete.
    fn supports_take(&self) -> bool {
        self.base.supports_take()
    }

    // `in_files` stays false (the default): a late fetch over Iceberg is not
    // planned yet, so only row-level statements take rows.

    /// The file scan's take with the table's field mapping. Identity-partition
    /// constants, lineage columns and `_metadata` are not in the files and are
    /// refused (`KRISHIV_UNSUPPORTED_FEATURE`).
    fn take(&self, take: TakeRows) -> BoxFuture<'_, krishiv_exec::Result<TakeStream>> {
        Box::pin(async move {
            let TakeRows {
                file,
                rows,
                projection,
                mem,
                gate,
            } = take;
            if let Some(c) = projection.iter().find(|c| !self.takes(c.root())) {
                return Err(unsupported(format!(
                    "reading the Iceberg column {} by row position",
                    c.root()
                )));
            }
            let request = TakeRequest::new(
                file,
                Arc::clone(&self.parts.table),
                projection,
                self.mapping.clone(),
                rows,
            );
            let stream = krishiv_io::take(
                &ParquetFormat,
                &self.parts.io,
                request,
                &self.parts.format_options,
                mem,
                self.parts.scan_options,
            )
            .await?;
            Ok(stream.with_gate(gate))
        })
    }

    fn with_file_splits(
        &self,
        files: Vec<FileRef>,
        partitions: Vec<FilePartition>,
    ) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        let mut scan = self.clone();
        scan.base = scan.base.with_file_splits(files, partitions);
        scan.prune = false;
        scan.planned = Arc::default();
        Ok(Arc::new(scan))
    }

    fn output_schema(&self, pushdowns: &ScanPushdowns) -> krishiv_exec::Result<SchemaRef> {
        if !self.parts.lineage {
            return self.inner_output_schema(pushdowns);
        }
        let (inner, split) = self.lineage_pushdowns(pushdowns.clone())?;
        let schema = self.inner_output_schema(&inner)?;
        match split {
            None => Ok(schema),
            Some(split) => Ok(split.projection(&schema)?.schema()),
        }
    }
}

/// Whether `f` reads a lineage column.
fn names_lineage(f: &PushedFilter) -> bool {
    f.expr.as_ref().is_some_and(|e| {
        e.columns()
            .iter()
            .any(|c| c.root() == lineage::ROW_ID || c.root() == lineage::LAST_UPDATED)
    })
}

impl IcebergScan {
    /// The output schema of the delete-aware scan for `pushdowns`.
    fn inner_output_schema(&self, pushdowns: &ScanPushdowns) -> krishiv_exec::Result<SchemaRef> {
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

impl IcebergScan {
    /// The delete-aware source for `pushdowns` (the scan of a plain
    /// provider, the inner scan of a lineage scan).
    fn build_inner(&self, pushdowns: ScanPushdowns) -> krishiv_exec::Result<Arc<dyn Source>> {
        let mut p = self.mapped(pushdowns);
        let planned = self.planned(&p.filters)?;
        let scan = planned.scan.clone();
        if planned.tasks.is_empty() && self.parts.ordinal.is_none() {
            return scan.build(p);
        }
        let columns = &planned.columns;
        let state = Arc::new(DeleteState::new(
            Arc::clone(&self.env),
            planned.tasks.clone(),
        ));
        if state.has_positions() {
            p.selection = Some(Arc::new(LazySelections(Arc::clone(&state))));
        }
        if self.parts.ordinal.is_none() {
            let source = scan.build(p)?;
            return Ok(Arc::new(DeleteSource::new(source, state, None)));
        }
        let (inner, output) = self.with_equality_columns(p, columns);
        let source = scan.build(inner)?;
        let eq = EqProjection::new(&source.schema(), output)?;
        Ok(Arc::new(DeleteSource::new(source, state, Some(eq))))
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
        out.insert(f.store.clone(), f.path.clone(), values);
    }
    Ok((fields, out))
}
