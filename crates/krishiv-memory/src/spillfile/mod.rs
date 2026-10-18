// SPDX-License-Identifier: Apache-2.0
//! Spill files (ADR 0007, spec 03 §3.2): batches written to local disk as
//! Arrow IPC, compressed with LZ4 or Zstandard, and read back one batch at a
//! time.
//!
//! This lives in `krishiv-memory` (L1) next to the [`Spillable`](crate::Spillable)
//! protocol it serves: the spill cap, the scratch memory of every write and
//! read, and the cancel path all meet the memory pool here, and operators in
//! higher layers (hash join, aggregate, sort, window) only call this API. The
//! crate depends on `krishiv-types` (same layer) for [`Batch`](krishiv_types::batch::Batch).
//!
//! # Flow
//! ```
//! use std::sync::Arc;
//! use arrow_array::{Int64Array, RecordBatch};
//! use arrow_schema::{DataType, Field, Schema};
//! use krishiv_memory::{
//!     MemoryConfig, MemoryPool, Reservation, SpillConfig, SpillManager, TreePool,
//! };
//! use krishiv_types::batch::{Batch, Weights};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let dir = std::env::temp_dir().join(format!("krishiv-doc-{}", std::process::id()));
//! let manager = SpillManager::open(&SpillConfig { dir: dir.clone(), ..SpillConfig::default() })?;
//! let scope = manager.scope()?; // one per query; `scope.cancel()` on query cancel
//!
//! let pool = TreePool::new(&MemoryConfig::default()).child("query-1", None);
//! let id = pool.register("sort", None);
//! let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
//! let record = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(vec![3, 1, 2]))])?;
//! let batch = Batch::with_weights(record, Weights::Const(2))?;
//!
//! let mut writer = scope.writer(schema, Reservation::new(Arc::clone(&pool), id))?;
//! writer.write(&batch)?;
//! let file = writer.finish()?; // atomic: temp file, then rename
//!
//! let mut reader = file.reader(Reservation::new(Arc::clone(&pool), id))?;
//! assert_eq!(reader.next().transpose()?, Some(batch));
//! assert_eq!(reader.next().transpose()?, None);
//! drop((reader, file, scope, manager)); // everything is deleted
//! assert!(std::fs::read_dir(&dir)?.next().is_none());
//! # let _ = std::fs::remove_dir_all(&dir);
//! # Ok(())
//! # }
//! ```
//!
//! # Guarantees
//! - **Weights survive**: `Const` and `Column` round-trip exactly, per batch.
//! - **Atomic files**: a finished `.spill` file is complete; see [`SpillWriter`].
//! - **No leaks**: files go with their [`SpillFile`], a query's directory with
//!   its last handle or on [`SpillScope::cancel`], a session with the
//!   [`SpillManager`], and a crash's leftovers at the next [`SpillManager::open`].
//! - **A cap** over all spill bytes, and **bounded memory**: a reader holds one
//!   frame (one batch), accounted in its [`Reservation`](crate::Reservation).
//! - **Never wrong data, never a panic**: frames are checksummed and
//!   length-checked and the footer pins the counts ([`SpillError::Corrupt`]);
//!   a version header (ADR 0029) makes readers refuse other versions.
//!
//! # Config keys
//! `krishiv.spill.dir`, `krishiv.spill.maxBytes`, `krishiv.spill.compression`
//! (`lz4` | `zstd` | `none`) and `krishiv.spill.zstd.level`, read by
//! [`SpillConfig::from_config`].

use arrow_ipc::CompressionType;
use arrow_ipc::writer::IpcWriteOptions;
use arrow_schema::ArrowError;

mod dir;
mod error;
mod format;
mod reader;
mod writer;

#[cfg(test)]
#[cfg(not(loom))]
mod tests;

pub use dir::{SpillConfig, SpillManager, SpillScope};
pub use error::SpillError;
pub use reader::{SpillFile, SpillReader};
pub use writer::SpillWriter;

/// How the buffers of a spill file are compressed
/// (`krishiv.spill.compression`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SpillCodec {
    /// Uncompressed: fastest, largest files.
    None,
    /// LZ4 frames: the default (ADR 0007).
    #[default]
    Lz4,
    /// Zstandard at `krishiv.spill.zstd.level`: smaller files, more CPU.
    Zstd,
}

impl SpillCodec {
    /// Parses a `krishiv.spill.compression` value.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "none" => Some(Self::None),
            "lz4" => Some(Self::Lz4),
            "zstd" => Some(Self::Zstd),
            _ => None,
        }
    }

    /// The Arrow IPC write options for this codec.
    pub(crate) fn write_options(self, zstd_level: i64) -> Result<IpcWriteOptions, ArrowError> {
        let options = IpcWriteOptions::default();
        match self {
            Self::None => Ok(options),
            Self::Lz4 => options.try_with_compression(Some(CompressionType::LZ4_FRAME)),
            Self::Zstd => options
                .try_with_compression(Some(CompressionType::ZSTD))?
                .try_with_compression_level(Some(i32::try_from(zstd_level).map_err(|_| {
                    ArrowError::InvalidArgumentError(format!(
                        "ZSTD compression level {zstd_level} is out of range"
                    ))
                })?)),
        }
    }
}
