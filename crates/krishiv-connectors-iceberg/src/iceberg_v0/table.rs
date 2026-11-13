// SPDX-License-Identifier: Apache-2.0
//! An Iceberg table loaded from its metadata file through Krishiv's I/O: its
//! schema, `inspect` tables, the snapshot a read sees and its scan plan
//! (P3-T026; commits are T027).

use std::fmt;

use arrow_array::RecordBatch;
use futures::TryStreamExt;
use iceberg::TableIdent;
use iceberg::spec::TableMetadata;
use iceberg::table::Table;
use krishiv_common::error::KrishivError;
use krishiv_types::StructType;

use super::scan;
use super::storage::{Registered, StorageRegistration, from_iceberg};
use super::types::spark_schema;
use crate::plan::{ManifestFilter, PruneOptions, ScanPlan, TableSnapshot, VersionSpec};
use krishiv_io::PushdownExpr;

/// An Iceberg table's metadata, read through a [`StorageRegistration`].
pub struct IcebergTable {
    pub(super) table: Table,
    /// The I/O and path placement data files are read through.
    pub(super) registered: Registered,
}

impl fmt::Debug for IcebergTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IcebergTable")
            .field("identifier", self.table.identifier())
            .field("metadata_location", &self.table.metadata_location())
            .finish_non_exhaustive()
    }
}

impl IcebergTable {
    /// The table whose metadata file is `metadata_location`, named `namespace.name`.
    /// The file and every manifest read later go through `storage`.
    ///
    /// # Errors
    /// The I/O error of the read (its Krishiv class kept); an internal error naming
    /// iceberg-rust's message for metadata it cannot parse.
    pub async fn load(
        storage: &StorageRegistration,
        metadata_location: &str,
        namespace: &str,
        name: &str,
    ) -> Result<Self, KrishivError> {
        let file_io = storage.file_io();
        let metadata = TableMetadata::read_from(&file_io, metadata_location)
            .await
            .map_err(from_iceberg)?;
        let ident = TableIdent::from_strs([namespace, name]).map_err(from_iceberg)?;
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
        Ok(Self { table, registered })
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
    /// # Errors
    /// The I/O error of a read (its class kept); `INTERNAL_ERROR` for a manifest
    /// list or manifest that cannot be parsed or breaks the spec;
    /// `KRISHIV_UNSUPPORTED_FEATURE` for an encrypted manifest.
    pub async fn plan(
        &self,
        snapshot: &TableSnapshot,
        predicate: Option<&PushdownExpr>,
        filter: ManifestFilter,
        options: PruneOptions,
    ) -> Result<ScanPlan, KrishivError> {
        scan::plan(self.table.file_io(), snapshot, predicate, filter, options).await
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
