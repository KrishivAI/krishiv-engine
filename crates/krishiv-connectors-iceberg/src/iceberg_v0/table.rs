// SPDX-License-Identifier: Apache-2.0
//! An Iceberg table loaded from its metadata file through Krishiv's I/O: its
//! schema, `inspect` tables, the snapshot a read sees and its scan plan
//! (P3-T026; commits are T027).

use std::fmt;
use std::sync::Arc;

use arrow_array::RecordBatch;
use futures::TryStreamExt;
use iceberg::TableIdent;
use iceberg::spec::TableMetadata;
use iceberg::table::Table;
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_types::StructType;

use super::commit::{CommitClock, CommitTable};
use super::rest_commit::RestCommitTable;
use super::scan;
use super::storage::{Registered, StorageRegistration, from_iceberg};
use super::types::spark_schema;
use crate::plan::deletes::DeleteIndex;
use crate::plan::{
    ManifestFilter, PruneOptions, ScanPlan, TableSnapshot, VersionSpec, WriterMetrics,
};
use crate::rest_commit::RestCommitTransport;
use krishiv_common::config::{
    CATALOG_MAX_METADATA_BYTES, Config, ConfigValue, ICEBERG_MANIFEST_CACHE_BYTES,
};

/// `krishiv.catalog.maxMetadataBytes`'s default (256 MiB).
const DEFAULT_MAX_METADATA_BYTES: u64 = 256 << 20;
/// The default of `krishiv.iceberg.manifestCache.bytes`.
const DEFAULT_MANIFEST_CACHE_BYTES: u64 = 64 << 20;
use krishiv_io::PushdownExpr;

/// `krishiv.catalog.maxMetadataBytes` of `config`, else its default.
fn max_metadata_bytes(config: &Config) -> u64 {
    config
        .get(&CATALOG_MAX_METADATA_BYTES)
        .and_then(ConfigValue::as_size)
        .unwrap_or(DEFAULT_MAX_METADATA_BYTES)
}

/// An Iceberg table's metadata document as a catalog returned it (the
/// `metadata` of an Iceberg REST `loadTable` response), parsed: its schema and
/// snapshots are known before any file of the table is read.
#[derive(Clone)]
pub struct IcebergMetadata(Arc<TableMetadata>);

impl fmt::Debug for IcebergMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IcebergMetadata")
            .field("table_uuid", &self.0.uuid())
            .field("current_snapshot", &self.current_snapshot_id())
            .finish_non_exhaustive()
    }
}

impl IcebergMetadata {
    /// The metadata document `value` (Iceberg's table metadata JSON, format 1 to 3).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` naming iceberg-rust's message for a document it cannot
    /// parse or that breaks the spec.
    pub fn from_json(value: serde_json::Value) -> Result<Self, KrishivError> {
        serde_json::from_value::<TableMetadata>(value)
            .map(|m| Self(Arc::new(m)))
            .map_err(|e| {
                krishiv_common::error::InternalError {
                    message: format!("Iceberg table metadata that does not parse: {e}"),
                }
                .into()
            })
    }

    /// The snapshot `version` reads and its schema, as [`IcebergTable::snapshot`].
    ///
    /// # Errors
    /// As [`IcebergTable::snapshot`].
    pub fn snapshot(&self, version: &VersionSpec) -> Result<TableSnapshot, KrishivError> {
        scan::table_snapshot(&self.0, version)
    }

    /// The current snapshot's id; `None` for a table without one.
    #[must_use]
    pub fn current_snapshot_id(&self) -> Option<i64> {
        self.0.current_snapshot_id()
    }

    /// The table's root location (`location`).
    #[must_use]
    pub fn location(&self) -> &str {
        self.0.location()
    }
}

/// An Iceberg table's metadata, read through a [`StorageRegistration`].
pub struct IcebergTable {
    pub(super) table: Table,
    /// The I/O and path placement data files are read through.
    pub(super) registered: Registered,
    /// The catalog parse limit on each metadata file, manifest list and
    /// manifest (`krishiv.catalog.maxMetadataBytes`).
    pub(super) max_metadata_bytes: u64,
    /// `krishiv.iceberg.manifestCache.bytes` (0: no manifest cache).
    pub(super) manifest_cache_bytes: u64,
    /// The catalog's commit endpoint, when the table was loaded through one
    /// that can commit: providers opened over it accept appends.
    pub(super) committer: Option<Arc<dyn CommitTable>>,
}

impl fmt::Debug for IcebergTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IcebergTable")
            .field("identifier", self.table.identifier())
            .field("metadata_location", &self.table.metadata_location())
            .finish_non_exhaustive()
    }
}

/// What a maintenance procedure works through: the commit endpoint, the file access, the clock.
pub(crate) type MaintenanceParts = (
    Arc<dyn CommitTable>,
    iceberg::io::FileIO,
    Arc<dyn CommitClock>,
);

impl IcebergTable {
    /// The table whose metadata file is `metadata_location`, named `namespace.name`.
    /// The file and every manifest read later go through `storage`. The
    /// default configuration's limits apply ([`Self::load_with_config`]).
    ///
    /// # Errors
    /// As [`Self::load_with_config`].
    pub async fn load(
        storage: &StorageRegistration,
        metadata_location: &str,
        namespace: &str,
        name: &str,
    ) -> Result<Self, KrishivError> {
        Self::load_with_config(storage, metadata_location, namespace, name, &Config::new()).await
    }

    /// As [`Self::load`], with the catalog parse limit of `config`
    /// (`krishiv.catalog.maxMetadataBytes`) on the metadata file and, later, on
    /// every manifest list and manifest a plan reads.
    ///
    /// # Errors
    /// The I/O error of the read (its Krishiv class kept);
    /// `KRISHIV_CATALOG_LIMIT_EXCEEDED` for a metadata file over the limit; an
    /// internal error naming iceberg-rust's message for metadata it cannot parse.
    pub async fn load_with_config(
        storage: &StorageRegistration,
        metadata_location: &str,
        namespace: &str,
        name: &str,
        config: &Config,
    ) -> Result<Self, KrishivError> {
        let max_metadata_bytes = max_metadata_bytes(config);
        let file_io = storage.file_io();
        let size = file_io
            .new_input(metadata_location)
            .map_err(from_iceberg)?
            .metadata()
            .await
            .map_err(from_iceberg)?
            .size;
        scan::check_size("metadata file", metadata_location, size, max_metadata_bytes)?;
        let metadata = TableMetadata::read_from(&file_io, metadata_location)
            .await
            .map_err(from_iceberg)?;
        let ident = TableIdent::from_strs([namespace, name]).map_err(from_iceberg)?;
        Self::build(
            storage,
            file_io,
            Arc::new(metadata),
            metadata_location,
            ident,
            config,
        )
    }

    /// The table whose metadata a catalog returned (`metadata`, read from its
    /// `metadata_location` by the catalog, as Java's REST catalog takes it from
    /// the `loadTable` response), named `namespace.name`; no metadata file is
    /// read here. Manifests read later go through `storage` under the parse
    /// limit of `config`, as for [`Self::load_with_config`].
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a namespace or name iceberg-rust refuses, or a
    /// registration that is gone.
    pub fn from_metadata(
        storage: &StorageRegistration,
        metadata: &IcebergMetadata,
        metadata_location: &str,
        namespace: &[String],
        name: &str,
        config: &Config,
    ) -> Result<Self, KrishivError> {
        let ident = TableIdent::from_strs(
            namespace
                .iter()
                .map(String::as_str)
                .chain(std::iter::once(name)),
        )
        .map_err(from_iceberg)?;
        Self::build(
            storage,
            storage.file_io(),
            Arc::clone(&metadata.0),
            metadata_location,
            ident,
            config,
        )
    }

    /// The table over `metadata`, read through `file_io` (resolved by the
    /// registration's name on first use; a `FileIO` that already read through it
    /// keeps working after the registration is dropped).
    fn build(
        storage: &StorageRegistration,
        file_io: iceberg::io::FileIO,
        metadata: Arc<TableMetadata>,
        metadata_location: &str,
        ident: TableIdent,
        config: &Config,
    ) -> Result<Self, KrishivError> {
        let manifest_cache_bytes = config
            .get(&ICEBERG_MANIFEST_CACHE_BYTES)
            .and_then(ConfigValue::as_size)
            .unwrap_or(DEFAULT_MANIFEST_CACHE_BYTES);
        let table = Table::builder()
            .metadata(metadata)
            .metadata_location(metadata_location)
            .identifier(ident)
            .file_io(file_io)
            .runtime(storage.runtime()?)
            .readonly(true)
            .build()
            .map_err(from_iceberg)?;
        let registered = storage.registered()?;
        Ok(Self {
            table,
            registered,
            max_metadata_bytes: max_metadata_bytes(config),
            manifest_cache_bytes,
            committer: None,
        })
    }

    /// The same table committing through `committer`: providers opened over
    /// it declare `BATCH_WRITE` and append ([`super::sink`]).
    #[must_use]
    pub(crate) fn with_committer(mut self, committer: Arc<dyn CommitTable>) -> Self {
        self.committer = Some(committer);
        self
    }

    /// The same table committing through an Iceberg REST catalog's commit
    /// endpoint (`transport`, [`crate::rest_commit`]): providers opened over
    /// it at the current version declare `BATCH_WRITE`, `TRUNCATE`,
    /// `OVERWRITE_BY_FILTER` and `OVERWRITE_DYNAMIC` and offer copy-on-write
    /// row-level changes; each commit is the spec's `CommitTableRequest` for
    /// this table's identifier, its answers mapped as design note §7.2 says.
    #[must_use]
    pub fn with_rest_commit(self, transport: Arc<dyn RestCommitTransport>) -> Self {
        let ident = self.table.identifier().clone();
        self.with_committer(Arc::new(RestCommitTable::new(transport, ident)))
    }

    /// What a maintenance procedure commits and deletes through: the catalog's commit endpoint,
    /// the table's file access and the commit clock (`make_clock` builds it from the I/O
    /// registration).
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` when the table has no commit endpoint (a read-only catalog
    /// table or a table at an older version); the clock's failure.
    pub(crate) fn maintenance_parts(
        &self,
        make_clock: fn(&Registered) -> Result<Arc<dyn CommitClock>, KrishivError>,
    ) -> Result<MaintenanceParts, KrishivError> {
        let committer = self.committer.clone().ok_or_else(|| {
            KrishivError::from(KrishivUnsupportedFeature {
                feature: format!(
                    "an Iceberg procedure on {}: the catalog serves it read-only",
                    self.table.identifier().name()
                ),
            })
        })?;
        Ok((
            committer,
            self.table.file_io().clone(),
            make_clock(&self.registered)?,
        ))
    }

    /// Every object under the directory `location` (a table location or a path below it), at any
    /// depth, as `(uri, last-modified in ms, key relative to `location`)`; the URI is `location`
    /// with the key appended, so it keeps the scheme and authority the table was registered with.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a location no store serves; the listing's failure.
    pub(crate) async fn list_under(
        &self,
        location: &str,
    ) -> Result<Vec<(String, i64, String)>, KrishivError> {
        let at = self.registered.paths.locate(location)?;
        let prefix: &str = AsRef::<str>::as_ref(&at.path);
        let base = location.trim_end_matches('/');
        let listed = self.registered.io.list(&at.store, &at.path).await?;
        let mut out = Vec::with_capacity(listed.len());
        for meta in listed {
            let key: &str = AsRef::<str>::as_ref(&meta.location);
            let rel = key
                .strip_prefix(prefix)
                .unwrap_or(key)
                .trim_start_matches('/')
                .to_owned();
            out.push((
                format!("{base}/{rel}"),
                meta.last_modified.timestamp_millis(),
                rel,
            ));
        }
        Ok(out)
    }

    /// `krishiv.catalog.maxMetadataBytes`, the parse limit on what a procedure reads.
    pub(crate) fn max_metadata_bytes(&self) -> u64 {
        self.max_metadata_bytes
    }

    /// The current schema as Spark's Iceberg integration shows it.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a column type Spark refuses (see
    /// `iceberg_v0::types`).
    pub fn schema(&self) -> Result<StructType, KrishivError> {
        spark_schema(self.table.metadata().current_schema())
    }

    /// The snapshot `version` reads and the schema it reads with (see
    /// [`crate::plan::VersionSpec`] for the rules and Iceberg's error texts).
    ///
    /// # Errors
    /// An unknown snapshot id, ref or a timestamp before the first snapshot, with
    /// Iceberg's message; `KRISHIV_UNSUPPORTED_FEATURE` for a column type Spark
    /// refuses.
    pub fn snapshot(&self, version: &VersionSpec) -> Result<TableSnapshot, KrishivError> {
        scan::table_snapshot(self.table.metadata(), version)
    }

    /// The data files of `snapshot` that may hold rows matching `predicate`
    /// (in the snapshot schema's names and types), and its live delete files.
    /// Reads the manifest list and the surviving manifests (at most 8 at once)
    /// through the storage; run it on the I/O side.
    ///
    /// Files are pruned by their metrics under the table's metrics modes
    /// ([`Self::writer_metrics`]).
    ///
    /// # Errors
    /// The I/O error of a read (its class kept); `KRISHIV_CATALOG_LIMIT_EXCEEDED`
    /// for a manifest list or manifest over the size limit; `INTERNAL_ERROR`
    /// for one that cannot be parsed or breaks the spec;
    /// `KRISHIV_UNSUPPORTED_FEATURE` for an encrypted manifest.
    pub async fn plan(
        &self,
        snapshot: &TableSnapshot,
        predicate: Option<&PushdownExpr>,
        filter: ManifestFilter,
        options: PruneOptions,
    ) -> Result<ScanPlan, KrishivError> {
        let metrics = self.writer_metrics(snapshot);
        let env = scan::PlanEnv {
            max_metadata_bytes: self.max_metadata_bytes,
            metrics: &metrics,
            cache: Some((&self.registered.manifests, self.manifest_cache_bytes)),
        };
        scan::plan(
            self.table.file_io(),
            snapshot,
            predicate,
            filter,
            options,
            env,
        )
        .await
    }

    /// The data files of `snapshot` and the index of its delete files,
    /// unassigned ([`scan::plan_files`]): the provider assigns deletes to the
    /// files a scan reads.
    ///
    /// # Errors
    /// As [`Self::plan`], except the errors of assigning deletes to one file.
    pub(super) async fn plan_files(
        &self,
        snapshot: &TableSnapshot,
    ) -> Result<(ScanPlan, DeleteIndex), KrishivError> {
        let metrics = self.writer_metrics(snapshot);
        let env = scan::PlanEnv {
            max_metadata_bytes: self.max_metadata_bytes,
            metrics: &metrics,
            cache: Some((&self.registered.manifests, self.manifest_cache_bytes)),
        };
        scan::plan_files(
            self.table.file_io(),
            snapshot,
            None,
            ManifestFilter::All,
            PruneOptions::default(),
            env,
        )
        .await
    }

    /// The writers' metrics modes of `snapshot`'s columns, from the table
    /// properties `write.metadata.metrics.*` (review F3).
    #[must_use]
    pub fn writer_metrics(&self, snapshot: &TableSnapshot) -> WriterMetrics {
        WriterMetrics::from_properties(
            self.table
                .metadata()
                .properties()
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str())),
            &snapshot.fields,
        )
    }

    /// The `snapshots` metadata table (Spark's `<table>.snapshots`), Arrow 60.
    ///
    /// # Errors
    /// As [`Self::load`]; an internal error if the bridge refuses a batch.
    pub async fn snapshots(&self) -> Result<Vec<RecordBatch>, KrishivError> {
        let stream = self
            .table
            .inspect()
            .snapshots()
            .scan()
            .await
            .map_err(from_iceberg)?;
        bridge(stream).await
    }

    /// The `manifests` metadata table of the current snapshot, Arrow 60 (reads the
    /// manifest list and manifests through the storage).
    ///
    /// # Errors
    /// As [`Self::snapshots`].
    pub async fn manifests(&self) -> Result<Vec<RecordBatch>, KrishivError> {
        let stream = self
            .table
            .inspect()
            .manifests()
            .scan()
            .await
            .map_err(from_iceberg)?;
        bridge(stream).await
    }
}

async fn bridge(
    stream: iceberg::scan::ArrowRecordBatchStream,
) -> Result<Vec<RecordBatch>, KrishivError> {
    let batches: Vec<_> = stream.try_collect().await.map_err(from_iceberg)?;
    batches
        .iter()
        .map(krishiv_arrow_bridge::record_batch_58_to_60)
        .collect()
}
