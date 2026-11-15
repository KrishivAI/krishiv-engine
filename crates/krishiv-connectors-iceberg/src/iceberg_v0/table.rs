// SPDX-License-Identifier: Apache-2.0
//! An Iceberg table loaded from its metadata file through Krishiv's I/O
//! (metadata only: no scan, no commit; those are P3-T026/T027).

use std::fmt;

use arrow_array::RecordBatch;
use futures::TryStreamExt;
use iceberg::TableIdent;
use iceberg::spec::TableMetadata;
use iceberg::table::Table;
use krishiv_common::error::KrishivError;
use krishiv_types::StructType;

use super::storage::{StorageRegistration, from_iceberg};
use super::types::spark_schema;

/// An Iceberg table's metadata, read through a [`StorageRegistration`].
pub struct IcebergTable {
    pub(super) table: Table,
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
        Ok(Self { table })
    }

    /// The current schema as Spark's Iceberg integration shows it.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a column type Spark refuses (see
    /// `iceberg_v0::types`).
    pub fn schema(&self) -> Result<StructType, KrishivError> {
        spark_schema(self.table.metadata().current_schema())
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
