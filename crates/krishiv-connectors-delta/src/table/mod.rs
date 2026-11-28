// SPDX-License-Identifier: Apache-2.0
//! [`DeltaTable`], the provider of one Delta table at one version for one query (design note
//! P3-T021 section 3.4): the kernel reads the log, Krishiv reads the data.
//!
//! [`DeltaTable::open`] builds the kernel snapshot on the log pool, checks Krishiv's reader
//! allow-list ([`crate::features`]) before anything else is read, maps the logical schema to
//! the engine's Arrow forms (column-mapping metadata kept on every field) and lists the live
//! files once (each a [`ScanUnit`]): that listing gives the optimizer its row count
//! ([`TableProvider::statistics`]) and the scan its units. A scan with a pushed filter lists
//! again through the kernel with the translated predicate ([`predicate`]), which prunes by
//! partition values and statistics.
//!
//! # Snapshot cache
//! [`DeltaTable::open_with`] reads through a [`SnapshotCache`] when it is given one (ADR 0037):
//! the latest snapshot of a table is kept across queries under its root, store and credential
//! identity, brought up to date by each new query (only the log after the cached version is
//! listed) and its file listing reused while the version is unchanged. Before that, one `head`
//! of the cached last commit file confirms the table at the root is still the one cached (same
//! size and modification time); a mismatch, a missing file or any error on the cached path
//! drops the entry and the table is built afresh. Table authorization runs before the lookup,
//! in the session's analysis.
//!
//! The stamp misses two narrow cases. A store whose modification times have 1-second
//! precision (S3's `LastModified`) does not notice a table recreated within the same second
//! with a last commit of the same size; an ETag would close this, but the kernel's `FileMeta`
//! carries none. A table recreated between that `head` and the listing of the newer log is
//! combined with the cached segment.

pub mod cache;
pub mod predicate;
pub mod scan;

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{DataType as ArrowType, Field, FieldRef, Schema, SchemaRef};
use delta_kernel::expressions::Scalar;
use delta_kernel::scan::state::DvInfo;
use delta_kernel::schema::{DataType, StructField};
use delta_kernel::table_features::ColumnMappingMode;
use delta_kernel::{KernelError, Snapshot, SnapshotRef};
use krishiv_common::config::Config;
use krishiv_common::error::{DeltaTableNotFound, InternalError, KrishivError, PathNotFound};
use krishiv_exec::ScanPushdown;
use krishiv_exec::planner::TableProvider;
use krishiv_io::{FieldMapping, ScanOptions};
use krishiv_optimizer::TableStats;

use crate::engine::DeltaEngine;
use crate::engine::schema::{scalar_to_array, to_arrow_field};
use crate::error::from_kernel;
use crate::features::{ReadProtocol, check_read, snapshot_error};

pub use self::cache::{CacheKey, SnapshotCache, listing_bytes};
pub use self::scan::{DeltaScan, DeltaScanSource};

/// How [`DeltaTable::open_with`] opens a table.
#[derive(Debug, Clone, Default)]
pub struct OpenOptions {
    /// The version to read; the latest when `None`.
    pub version: Option<u64>,
    /// The table as errors name it (delta-spark's `tableNameOrPath`): a catalog table's
    /// `namespace.table`; the Hadoop form of the root ([`hadoop_path`]) when `None`.
    pub name: Option<String>,
    /// The cross-query snapshot cache and this table's key in it. Used for the latest version
    /// only.
    pub cache: Option<(Arc<SnapshotCache>, CacheKey)>,
}

/// The Hadoop `Path` form of the table root `url`, as delta-spark names a table read by path:
/// `file:/abs/dir` for a local root (no authority, decoded, no trailing `/`), the URL without its
/// trailing `/` otherwise (`s3://bucket/key`).
#[must_use]
pub fn hadoop_path(url: &url::Url) -> String {
    if url.scheme() == "file" && url.host_str().is_none_or(str::is_empty) {
        // The URL's own path, decoded, not `Url::to_file_path`: on Windows that is `C:\x\` (and
        // fails without a drive), while Hadoop's form is `file:/C:/x` there too.
        let text = percent_decoded(url.path());
        let trimmed = text.trim_end_matches('/');
        return format!("file:{}", if trimmed.is_empty() { "/" } else { trimmed });
    }
    url.as_str().trim_end_matches('/').to_owned()
}

/// `text` with its `%XX` escapes decoded; a `%` not followed by two hex digits stays as written.
fn percent_decoded(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = |b: u8| char::from(b).to_digit(16);
        if bytes[i] == b'%'
            && let (Some(&h), Some(&l)) = (bytes.get(i + 1), bytes.get(i + 2))
            && let (Some(h), Some(l)) = (hex(h), hex(l))
        {
            // Two hex digits: at most 0xFF, so the cast is exact.
            out.push(u8::try_from(h * 16 + l).unwrap_or(b'%'));
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// What a snapshot that could not be built says about a root with no table (R1-Q-1), as delta-spark's
/// `DeltaTableV2` path check does: a root that does not exist is `PATH_NOT_FOUND`; one with no
/// versioned `_delta_log/` file is `DELTA_TABLE_NOT_FOUND` (a path table is named `` delta.`<path>` ``).
/// `None`: the root holds a log, so the kernel's own error stands (or the listing itself failed).
fn missing_table_error(
    e: &DeltaEngine,
    root: &str,
    path_based: bool,
    label: &str,
) -> Option<KrishivError> {
    use delta_kernel::StorageHandler as _;
    let url = url::Url::parse(root).ok()?;
    if url.scheme() == "file" && url.to_file_path().is_ok_and(|p| !p.exists()) {
        return Some(
            PathNotFound {
                path: hadoop_path(&url),
            }
            .into(),
        );
    }
    let log = url.join("_delta_log/").ok()?;
    let has_log = e.storage().list_from(&log).ok()?.any(|m| {
        m.ok().is_some_and(|m| {
            let name = m.location.path().rsplit('/').next().unwrap_or("");
            let v = name.split('.').next().unwrap_or("");
            !v.is_empty()
                && v.bytes().all(|b| b.is_ascii_digit())
                && (name.ends_with(".json") || name.ends_with(".parquet"))
        })
    });
    if has_log {
        return None;
    }
    let table_name = if path_based {
        format!("delta.`{label}`")
    } else {
        label.to_owned()
    };
    Some(DeltaTableNotFound { table_name }.into())
}

/// Whether `engine`'s root holds a Delta table: a commit or checkpoint file under `_delta_log/`.
///
/// # Errors
/// Storage errors listing the log; `OPERATION_CANCELED`.
pub async fn has_delta_log(engine: &DeltaEngine) -> Result<bool, KrishivError> {
    let log = engine
        .table_root()
        .join("_delta_log/")
        .map_err(|e| internal(format!("the log of {}: {e}", engine.table_root())))?;
    let root = engine.table_root().as_str().to_owned();
    engine
        .delta_blocking(move |e| -> Result<bool, KrishivError> {
            use delta_kernel::StorageHandler as _;
            let listed = e
                .storage()
                .list_from(&log)
                .map_err(|err| from_kernel(&err, &root))?;
            for meta in listed {
                let meta = meta.map_err(|err| from_kernel(&err, &root))?;
                let name = meta.location.path().rsplit('/').next().unwrap_or("");
                let versioned = name
                    .split('.')
                    .next()
                    .is_some_and(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()));
                if versioned && (name.ends_with(".json") || name.ends_with(".parquet")) {
                    return Ok(true);
                }
            }
            Ok(false)
        })
        .await?
}

/// Whether `engine`'s root holds any object at all (delta-spark's `tableExists` for a path: a
/// Parquet directory with no log is "already exists" too, an empty or missing root is not).
///
/// # Errors
/// Storage errors listing the root; `OPERATION_CANCELED`.
pub async fn root_holds_files(engine: &DeltaEngine) -> Result<bool, KrishivError> {
    let at = engine.table_root().clone();
    let root = at.as_str().to_owned();
    engine
        .delta_blocking(move |e| -> Result<bool, KrishivError> {
            use delta_kernel::StorageHandler as _;
            let mut listed = e
                .storage()
                .list_from(&at)
                .map_err(|err| from_kernel(&err, &root))?;
            match listed.next() {
                Some(Err(err)) => Err(from_kernel(&err, &root)),
                Some(Ok(_)) => Ok(true),
                None => Ok(false),
            }
        })
        .await?
}

/// One live data file of the snapshot: one scan unit.
#[derive(Debug, Clone)]
pub struct ScanUnit {
    /// The path the log names, relative to the table root (URL-encoded) or absolute.
    pub path: String,
    /// The file's size in bytes.
    pub size: u64,
    /// The file's `numRecords`, when its statistics have it (rows before its deletion vector).
    pub records: Option<u64>,
    /// The `add` action's modification time, ms since the epoch.
    pub modification_time: i64,
    /// The raw partition values, by physical column name.
    pub partition_values: HashMap<String, String>,
    /// Its deletion vector, if any.
    pub dv: DvInfo,
}

/// A partition column: the table field and the name its values are stored under.
#[derive(Debug, Clone)]
pub(crate) struct PartitionColumn {
    /// The table field (the engine's Arrow form).
    pub(crate) field: FieldRef,
    /// The kernel type values are parsed as.
    pub(crate) kernel_type: DataType,
    /// The physical name (the key of `partitionValues`).
    pub(crate) physical: String,
}

/// What a table and its scans share.
#[derive(Debug)]
pub(crate) struct Inner {
    pub(crate) engine: DeltaEngine,
    pub(crate) name: String,
    pub(crate) snapshot: SnapshotRef,
    pub(crate) schema: SchemaRef,
    pub(crate) partitions: Vec<PartitionColumn>,
    pub(crate) mapping: FieldMapping,
    pub(crate) units: Arc<[ScanUnit]>,
    pub(crate) scan_options: ScanOptions,
    pub(crate) limits: crate::commit::RetryLimits,
    /// delta-spark's `spark.databricks.delta.{delete,update,merge}.deletionVectors.persistent`
    /// (internal, default true; `DeltaSQLConf.scala:2949-2968`) as the opening query's
    /// configuration sets them.
    pub(crate) persistent_dvs: PersistentDvs,
}

/// Whether `DELETE`, `UPDATE` and `MERGE` mark rows in deletion vectors where the table allows
/// it (delta-spark's per-command `...deletionVectors.persistent` confs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PersistentDvs {
    pub(crate) delete: bool,
    pub(crate) update: bool,
    pub(crate) merge: bool,
}

impl PersistentDvs {
    /// The confs as `config` sets them (unset: true; only `false` turns one off: another
    /// value, which Spark refuses at `SET` for these boolean confs, keeps the default).
    pub(crate) fn from_config(config: &Config) -> Self {
        let flag = |op: &str| {
            config
                .get_unregistered(&format!(
                    "spark.databricks.delta.{op}.deletionVectors.persistent"
                ))
                .is_none_or(|v| !v.trim().eq_ignore_ascii_case("false"))
        };
        Self {
            delete: flag("delete"),
            update: flag("update"),
            merge: flag("merge"),
        }
    }
}

/// A Delta table at one version for one query (see the module docs).
#[derive(Debug, Clone)]
pub struct DeltaTable {
    inner: Arc<Inner>,
}

impl DeltaTable {
    /// Opens the table at `engine`'s root at `version` (the latest when `None`) and lists its
    /// live files. Scan settings follow `config`. As [`Self::open_with`] without a cache, the
    /// table named by its root's Hadoop form.
    ///
    /// # Errors
    /// As [`Self::open_with`].
    pub async fn open(
        engine: DeltaEngine,
        version: Option<u64>,
        config: &Config,
    ) -> Result<Self, KrishivError> {
        let options = OpenOptions {
            version,
            ..OpenOptions::default()
        };
        Self::open_with(engine, options, config).await
    }

    /// Opens the table at `engine`'s root as `options` say and lists its live files (or reuses
    /// the cached listing of an unchanged version). Scan settings follow `config`.
    ///
    /// # Errors
    /// The allow-list's refusals ([`crate::features`]), before any file of the table's data is
    /// read; `DELTA_INVALID_PROTOCOL_VERSION`; the kernel's errors otherwise
    /// ([`crate::error`]), each naming the table as `options.name` (or the root's Hadoop form).
    pub async fn open_with(
        engine: DeltaEngine,
        options: OpenOptions,
        config: &Config,
    ) -> Result<Self, KrishivError> {
        let OpenOptions {
            version,
            name,
            cache,
        } = options;
        let path_based = name.is_none();
        let name = name.unwrap_or_else(|| hadoop_path(engine.table_root()));
        let root = engine.table_root().as_str().to_owned();
        let cache = cache.filter(|_| version.is_none());
        let cached = cache.as_ref().and_then(|(c, k)| c.lookup(k));
        let label = name.clone();
        let entry = cache.clone();
        let opened = engine
            .delta_blocking(move |e| -> Result<Opened, KrishivError> {
                if let Some(c) = cached {
                    if let Some(opened) = refresh(e, &c, &label) {
                        return Ok(opened);
                    }
                    // A refresh cut short by the query's cancellation says nothing about the
                    // entry: it stays, and the query ends with its cancellation.
                    if e.kernel_cancellation().is_cancelled() {
                        return Err(from_kernel(&KernelError::Cancelled, &label));
                    }
                    // Stale or broken: dropped (unless another query has meanwhile stored a
                    // newer entry), and the table is built afresh (a lower version than the
                    // dropped one may then be cached).
                    if let Some((cache, key)) = &entry {
                        cache.invalidate_if(key, c.snapshot.version(), c.last);
                    }
                }
                let mut builder =
                    Snapshot::builder_for(&root).with_cancellation_token(e.kernel_cancellation());
                if let Some(v) = version {
                    builder = builder.at_version(v);
                }
                let snapshot = match builder.build(e) {
                    Ok(snapshot) => snapshot,
                    Err(err) => {
                        return Err(missing_table_error(e, &root, path_based, &label)
                            .unwrap_or_else(|| snapshot_error(&err, &label)));
                    }
                };
                // The allow-list before the file listing: a refused table reads no more.
                check_read(&label, &ReadProtocol::of(&snapshot))?;
                let units: Arc<[ScanUnit]> = list_units(e, &snapshot, None)
                    .map_err(|err| from_kernel(&err, &label))?
                    .into();
                let stamp = entry.as_ref().and_then(|_| listed_commit(e, &snapshot));
                Ok(Opened {
                    snapshot,
                    units,
                    stamp,
                })
            })
            .await??;
        let Opened {
            snapshot,
            units,
            stamp,
        } = opened;
        if let Some((cache, key)) = cache {
            match stamp {
                Some(last) => cache.store(
                    key,
                    cache::Cached {
                        snapshot: Arc::clone(&snapshot),
                        units: Some(Arc::clone(&units)),
                        last,
                    },
                ),
                // A version whose commit file cannot be stamped is not cached: a later query
                // could not tell it is still current.
                None => cache.invalidate(&key),
            }
        }
        let tc = snapshot.table_configuration();
        let mode = tc.column_mapping_mode();
        let logical = snapshot.schema();
        let fields = logical
            .fields()
            .map(|f| engine_field(f, &name))
            .collect::<Result<Vec<_>, _>>()?;
        let schema: SchemaRef = Arc::new(Schema::new(fields));
        let partitions = tc
            .metadata()
            .partition_columns()
            .iter()
            .map(|p| {
                let (i, kf) = logical
                    .fields()
                    .enumerate()
                    .find(|(_, f)| f.name() == p)
                    .ok_or_else(|| {
                        internal(format!("partition column {p} is not in the schema"))
                    })?;
                Ok(PartitionColumn {
                    field: Arc::clone(&schema.fields()[i]),
                    kernel_type: kf.data_type().clone(),
                    physical: kf.physical_name(mode).to_owned(),
                })
            })
            .collect::<Result<Vec<_>, KrishivError>>()?;
        let mapping = match mode {
            ColumnMappingMode::None => FieldMapping::ByName {
                case_sensitive: true,
            },
            ColumnMappingMode::Id => FieldMapping::DeltaColumnMapping { by_id: true },
            ColumnMappingMode::Name => FieldMapping::DeltaColumnMapping { by_id: false },
        };
        Ok(Self {
            inner: Arc::new(Inner {
                engine,
                name,
                snapshot,
                schema,
                partitions,
                mapping,
                units,
                scan_options: ScanOptions::from_config(config),
                limits: crate::commit::RetryLimits::from_config(config),
                persistent_dvs: PersistentDvs::from_config(config),
            }),
        })
    }

    /// The snapshot's version.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.inner.snapshot.version()
    }

    /// The table's columns in the engine's Arrow forms.
    #[must_use]
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.inner.schema)
    }

    /// Whether this version already records the write key `key` (`krishiv.write.idempotencyKey`):
    /// a `txn` action under [`crate::commit::idempotency_app_id`].
    ///
    /// # Errors
    /// The kernel's errors reading the log, naming the table; `OPERATION_CANCELED`.
    pub async fn has_idempotency_key(&self, key: &str) -> Result<bool, KrishivError> {
        let (snapshot, app) = (
            Arc::clone(&self.inner.snapshot),
            crate::commit::idempotency_app_id(key),
        );
        let root = self.inner.engine.table_root().as_str().to_owned();
        self.inner
            .engine
            .delta_blocking(move |e| -> Result<bool, KrishivError> {
                snapshot
                    .get_app_id_version(&app, e)
                    .map(|v| v.is_some())
                    .map_err(|err| from_kernel(&err, &root))
            })
            .await?
    }

    /// The kernel snapshot.
    #[must_use]
    pub fn snapshot(&self) -> &SnapshotRef {
        &self.inner.snapshot
    }

    /// The partition columns, in the log's order (`metaData.partitionColumns`).
    #[must_use]
    pub fn partition_columns(&self) -> Vec<String> {
        self.inner
            .partitions
            .iter()
            .map(|p| p.field.name().clone())
            .collect()
    }

    /// The live files, in the kernel's order.
    #[must_use]
    pub fn units(&self) -> &[ScanUnit] {
        &self.inner.units
    }

    /// The table as errors name it.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// The engine the table was opened with (bound to the planning cancellation).
    #[must_use]
    pub fn engine(&self) -> &DeltaEngine {
        &self.inner.engine
    }

    /// The writer of this table with the commit loop bounded by `limits` (design section 4.1;
    /// [`TableProvider::writer`] uses the default limits).
    ///
    /// # Errors
    /// See [`crate::write`]'s refusals.
    pub fn writer_with(
        &self,
        request: &krishiv_exec::planner::WriteRequest,
        limits: crate::commit::RetryLimits,
    ) -> Result<Arc<dyn krishiv_exec::Sink>, KrishivError> {
        crate::write::sink::writer(self, request, limits)
    }

    /// The scan, typed.
    #[must_use]
    pub fn delta_scan(&self) -> DeltaScan {
        DeltaScan::new(Arc::clone(&self.inner))
    }

    /// The scan of a row-level statement (it reads row ids), of every live file or of the
    /// files at `paths` (as the log names them), in the listing's order.
    pub(crate) fn row_level_scan(
        &self,
        paths: Option<&std::collections::HashSet<String>>,
    ) -> Result<DeltaScan, KrishivError> {
        let Some(paths) = paths else {
            return Ok(DeltaScan::with_row_ids(Arc::clone(&self.inner)));
        };
        let units: Arc<[ScanUnit]> = self
            .inner
            .units
            .iter()
            .filter(|u| paths.contains(u.path.as_str()))
            .cloned()
            .collect();
        let i = &self.inner;
        Ok(DeltaScan::with_row_ids(Arc::new(Inner {
            engine: i.engine.clone(),
            name: i.name.clone(),
            snapshot: Arc::clone(&i.snapshot),
            schema: Arc::clone(&i.schema),
            partitions: i.partitions.clone(),
            mapping: i.mapping.clone(),
            units,
            scan_options: i.scan_options,
            limits: i.limits,
            persistent_dvs: i.persistent_dvs,
        })))
    }

    /// The commit loop's bounds from the table's configuration.
    pub(crate) fn retry_limits(&self) -> crate::commit::RetryLimits {
        self.inner.limits
    }

    /// The per-command persistent deletion vector confs of the opening query.
    pub(crate) fn persistent_dvs(&self) -> PersistentDvs {
        self.inner.persistent_dvs
    }

    /// Whether type widening is enabled: `delta.enableTypeWidening = true` (delta-spark's
    /// `TypeWidening.isEnabled`, `TypeWidening.scala:43-52`), so a write's schema merge would
    /// widen a column's type.
    #[must_use]
    pub fn type_widening_enabled(&self) -> bool {
        self.inner.snapshot.table_properties().enable_type_widening == Some(true)
    }

    /// Whether the table takes new deletion vectors: the feature in its protocol and
    /// `delta.enableDeletionVectors = true` (delta-spark's `deletionVectorsWritable`,
    /// `DeletionVectorUtils.scala:72-74`).
    pub(crate) fn dv_writable(&self) -> bool {
        self.inner
            .snapshot
            .table_configuration()
            .is_feature_enabled(&delta_kernel::table_features::TableFeature::DeletionVectors)
    }
}

impl TableProvider for DeltaTable {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        Ok(Arc::new(self.delta_scan()))
    }

    /// The live rows: every file's `numRecords` less its deletion vector's cardinality, when
    /// every file has the statistic. Column statistics are not exposed in P3 (T024).
    fn statistics(&self) -> Option<TableStats> {
        let mut rows = 0_u64;
        for u in self.inner.units.iter() {
            let deleted = u.dv.cardinality().ok()?.unwrap_or(0);
            rows = rows.saturating_add(u.records?.saturating_sub(deleted));
        }
        Some(TableStats {
            rows,
            ..TableStats::default()
        })
    }

    /// One unit per live file.
    fn scan_units(&self) -> Option<u64> {
        u64::try_from(self.inner.units.len()).ok()
    }

    /// Append, truncate, overwrite by filter and dynamic overwrite (design section 4.1).
    fn capabilities(&self) -> krishiv_exec::planner::TableCapabilities {
        crate::write::sink::capabilities()
    }

    /// The Delta write sink ([`crate::write`]); its commit loop is bounded by
    /// [`crate::commit::RetryLimits::from_config`] of the table's configuration.
    fn writer(
        &self,
        request: &krishiv_exec::planner::WriteRequest,
    ) -> krishiv_exec::Result<Arc<dyn krishiv_exec::Sink>> {
        self.writer_with(request, self.inner.limits)
    }

    /// One snapshot at one version: every scan reads the same rows.
    fn pinned(&self) -> bool {
        true
    }

    /// Copy-on-write `DELETE`, `UPDATE` ([`crate::write::dml`]) and `MERGE` ([`crate::write::merge`]).
    fn row_level(
        &self,
        op: krishiv_exec::planner::RowLevelOperation,
    ) -> Option<Arc<dyn krishiv_exec::planner::RowLevelTable>> {
        crate::write::dml::DeltaRowLevel::new(self.clone(), op)
            .map(|t| Arc::new(t) as Arc<dyn krishiv_exec::planner::RowLevelTable>)
    }
}

/// What an open read on the log pool.
struct Opened {
    snapshot: SnapshotRef,
    units: Arc<[ScanUnit]>,
    /// The snapshot's last commit file, when it is to be cached.
    stamp: Option<cache::CommitStamp>,
}

/// The snapshot's last commit file (`_delta_log/<version>.json`) as the store reports it now,
/// if it has one. Log cleanup never removes the latest commit, so a cached snapshot whose last
/// commit no longer matches belongs to a table that was removed or recreated.
fn last_commit(engine: &DeltaEngine, snapshot: &SnapshotRef) -> Option<cache::CommitStamp> {
    use delta_kernel::StorageHandler as _;
    let url = engine
        .table_root()
        .join(&format!("_delta_log/{:020}.json", snapshot.version()))
        .ok()?;
    let meta = engine.storage().head(&url).ok()?;
    Some(cache::CommitStamp {
        size: meta.size,
        last_modified: meta.last_modified,
    })
}

/// The snapshot's last commit file as the listing that built the snapshot returned it: the
/// stamp a cache entry is stored with, bound to the files the snapshot read (a `head` after the
/// build could see a table recreated at the root meanwhile). `None` when that listing did not
/// return it (the entry is then not stored).
fn listed_commit(engine: &DeltaEngine, snapshot: &SnapshotRef) -> Option<cache::CommitStamp> {
    let url = engine
        .table_root()
        .join(&format!("_delta_log/{:020}.json", snapshot.version()))
        .ok()?;
    let (size, last_modified) = engine.listed_commit(&url)?;
    Some(cache::CommitStamp {
        size,
        last_modified,
    })
}

/// The cached table `c` brought up to date (ADR 0037 `per_query`): `None` when its last commit
/// is gone or rewritten (the table was removed or recreated at its root) or when anything on
/// the way fails, so the caller builds the table afresh.
fn refresh(engine: &DeltaEngine, c: &cache::Cached, label: &str) -> Option<Opened> {
    if last_commit(engine, &c.snapshot) != Some(c.last) {
        return None;
    }
    // Only the log after the cached version is listed.
    let snapshot = Snapshot::builder_from(Arc::clone(&c.snapshot))
        .with_cancellation_token(engine.kernel_cancellation())
        .build(engine)
        .ok()?;
    check_read(label, &ReadProtocol::of(&snapshot)).ok()?;
    if snapshot.version() == c.snapshot.version()
        && let Some(units) = &c.units
    {
        return Some(Opened {
            snapshot,
            units: Arc::clone(units),
            stamp: Some(c.last),
        });
    }
    // The cached commit checked again after the newer log was read: a table removed and
    // recreated at the root meanwhile no longer has it, and its commits must not be replayed
    // onto the cached snapshot.
    if last_commit(engine, &c.snapshot) != Some(c.last) {
        return None;
    }
    let units: Arc<[ScanUnit]> = list_units(engine, &snapshot, None).ok()?.into();
    let stamp = listed_commit(engine, &snapshot);
    Some(Opened {
        snapshot,
        units,
        stamp,
    })
}

/// The live files of `snapshot`, pruned by `predicate`. Drains the kernel's lazy iterator, so
/// it runs inside a `delta_blocking` call.
pub(crate) fn list_units(
    engine: &DeltaEngine,
    snapshot: &SnapshotRef,
    predicate: Option<delta_kernel::expressions::PredicateRef>,
) -> Result<Vec<ScanUnit>, KernelError> {
    let scan = Arc::clone(snapshot)
        .scan_builder()
        .with_predicate(predicate)
        .with_cancellation_token(engine.kernel_cancellation())
        .build()?;
    let mut files: Vec<delta_kernel::scan::state::ScanFile> = Vec::new();
    for metadata in scan.scan_metadata(engine)? {
        files = metadata?.visit_scan_files(files, Vec::push)?;
    }
    files
        .into_iter()
        .map(|f| {
            let size = u64::try_from(f.size).map_err(|_| {
                KernelError::generic(format!(
                    "the Delta log gives the data file {} a negative size ({})",
                    f.path, f.size
                ))
            })?;
            Ok(ScanUnit {
                path: f.path,
                size,
                records: f.stats.map(|s| s.num_records),
                modification_time: f.modification_time,
                partition_values: f.partition_values,
                dv: f.dv_info,
            })
        })
        .collect()
}

/// The value of a partition column stored as `raw`, as the kernel's scan parses it: an
/// absent value is NULL, an empty one is the empty string or bytes (NULL for other types).
pub(crate) fn partition_scalar(raw: Option<&String>, t: &DataType) -> Result<Scalar, KernelError> {
    match (raw, t.as_primitive_opt()) {
        (Some(v), Some(p)) if v.is_empty() => Ok(match p {
            delta_kernel::schema::PrimitiveType::String => Scalar::String(String::new()),
            delta_kernel::schema::PrimitiveType::Binary => Scalar::Binary(Vec::new()),
            _ => Scalar::Null(t.clone()),
        }),
        (Some(v), Some(p)) => p.parse_scalar(v),
        (Some(_), None) => Err(KernelError::generic(format!(
            "unexpected partition column type {t:?}"
        ))),
        (None, _) => Ok(Scalar::Null(t.clone())),
    }
}

/// The one-row column of `column`'s value `raw`, in the table field's type.
pub(crate) fn partition_array(
    column: &PartitionColumn,
    raw: Option<&String>,
    table: &str,
) -> Result<arrow_array::ArrayRef, KrishivError> {
    let scalar = partition_scalar(raw, &column.kernel_type).map_err(|e| from_kernel(&e, table))?;
    let array = scalar_to_array(&scalar, 1).map_err(|e| from_kernel(&e, table))?;
    arrow_cast::cast(&array, column.field.data_type())
        .map_err(|e| internal(format!("partition value of {}: {e}", column.field.name())))
}

/// The engine's Arrow field of the kernel field `field`: the kernel's form (with its metadata
/// and field ids) in the engine's canonical types (`Utf8View`, `BinaryView`, `Decimal64` up to
/// precision 18, the engine's Variant and map entries).
pub(crate) fn engine_field(field: &StructField, table: &str) -> Result<Field, KrishivError> {
    let arrow = to_arrow_field(field).map_err(|e| from_kernel(&e, table))?;
    canonical(field.data_type(), &arrow)
}

fn canonical(kernel: &DataType, arrow: &Field) -> Result<Field, KrishivError> {
    let t = match (kernel, arrow.data_type()) {
        (DataType::Primitive(_), leaf) => krishiv_types::DataType::from_arrow(leaf)?.to_arrow(),
        (DataType::Variant(_), _) => krishiv_types::DataType::Variant.to_arrow(),
        (DataType::Struct(k), ArrowType::Struct(a)) => ArrowType::Struct(
            k.fields()
                .zip(a.iter())
                .map(|(kf, af)| canonical(kf.data_type(), af).map(Arc::new))
                .collect::<Result<Vec<_>, _>>()?
                .into(),
        ),
        (DataType::Array(k), ArrowType::List(e)) => {
            ArrowType::List(Arc::new(canonical(k.element_type(), e)?))
        }
        (DataType::Map(k), ArrowType::Map(entries, sorted)) => {
            let ArrowType::Struct(kv) = entries.data_type() else {
                return Err(internal(format!("map {} without entries", arrow.name())));
            };
            let (Some(key), Some(value)) = (kv.first(), kv.get(1)) else {
                return Err(internal(format!(
                    "map {} without key and value",
                    arrow.name()
                )));
            };
            let kv = vec![
                canonical(k.key_type(), key)?,
                canonical(k.value_type(), value)?,
            ];
            ArrowType::Map(
                Arc::new(Field::new("entries", ArrowType::Struct(kv.into()), false)),
                *sorted,
            )
        }
        _ => {
            return Err(internal(format!(
                "kernel type {kernel:?} has Arrow form {:?}",
                arrow.data_type()
            )));
        }
    };
    let field = arrow.clone().with_data_type(t);
    if matches!(kernel, DataType::Variant(_)) {
        // The engine's own Variant field carries its type tags (the Arrow extension name), so
        // a scan's column meets one the engine made (a CTAS's new table) in a union.
        let mut metadata = field.metadata().clone();
        metadata.extend(
            krishiv_types::DataType::Variant
                .to_arrow_field("", true)
                .metadata()
                .clone(),
        );
        return Ok(field.with_metadata(metadata));
    }
    Ok(field)
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

#[cfg(test)]
mod cache_tests;
#[cfg(test)]
mod oracle_tests;
#[cfg(test)]
mod roundtrip_tests;
#[cfg(test)]
mod tests;
