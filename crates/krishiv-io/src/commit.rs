// SPDX-License-Identifier: Apache-2.0
//! Writes to a plain directory (P2-T082a): Spark's save modes and the commit
//! of the files a write produced.
//!
//! - **Modes** (`DataFrameWriter.mode`): `error` / `errorifexists` /
//!   `default` fail with `PATH_ALREADY_EXISTS` when the path exists (a file,
//!   anything under it, or an empty directory); `ignore` then writes
//!   nothing; `append` adds files; `overwrite` replaces everything under
//!   the path (or the file at it).
//! - **Files are written under their final names**, unique per write
//!   (`part-<task>-<job uuid>-c<n>`), and never renamed (ADR 0014). The
//!   file sink deletes the files of a failed or cancelled write.
//! - **A failed overwrite keeps the old data.** The objects under the path
//!   are listed when the write begins and deleted only once every new file
//!   is complete (Spark deletes them before writing). A file at the path
//!   itself is deleted first, as it would block the directory.
//! - **Commit**, as Spark's: an output without any file gets one empty file
//!   holding the schema, and an empty `_SUCCESS` marker is written last.
//! - Readers listing the directory during the write can see new files
//!   before the commit, and old and new files between the two steps of an
//!   overwrite (as with Spark on object stores); atomic publication is the
//!   manifest commit's (P2-T082b).

use std::sync::Arc;

use arrow_schema::SchemaRef;
use bytes::{Bytes, BytesMut};
use futures::{StreamExt, TryStreamExt};
use krishiv_common::error::{InvalidSaveMode, PathAlreadyExists};
use object_store::path::Path;

use crate::error::Result;
use crate::format::FileFormat;
use crate::io::{IoContext, StoreId};
use crate::options::FormatOptions;
use crate::write::{WriteOptions, WrittenFileInfo, spark_file_name};

/// Name of the marker a committed write leaves (Hadoop's committers).
pub const SUCCESS_MARKER: &str = "_SUCCESS";

/// Deletes in flight while an overwrite removes the old objects.
const PARALLEL_DELETES: usize = 16;

/// Spark's `SaveMode` for a path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveMode {
    /// `error`, `errorifexists`, `default`: fail if the path exists.
    ErrorIfExists,
    /// `append`: add files.
    Append,
    /// `overwrite`: replace what is there.
    Overwrite,
    /// `ignore`: write nothing if the path exists.
    Ignore,
}

impl SaveMode {
    /// The mode named `mode` (ASCII case-insensitive, as Spark).
    ///
    /// # Errors
    /// `INVALID_SAVE_MODE` for any other name.
    pub fn parse(mode: &str) -> Result<Self> {
        match mode.to_ascii_lowercase().as_str() {
            "error" | "errorifexists" | "default" => Ok(Self::ErrorIfExists),
            "append" => Ok(Self::Append),
            "overwrite" => Ok(Self::Overwrite),
            "ignore" => Ok(Self::Ignore),
            _ => Err(InvalidSaveMode {
                mode: format!("\"{mode}\""),
            }
            .into()),
        }
    }
}

/// One write to a directory, from [`Self::begin`] to [`Self::commit`].
#[derive(Debug)]
pub struct DirectoryWrite {
    store: StoreId,
    dir: Path,
    job_uuid: String,
    /// The objects an overwrite replaces, listed at the start.
    replaced: Vec<Path>,
}

impl DirectoryWrite {
    /// Starts a write of `dir` in `store` with `mode`; `job_uuid` names its
    /// files. `None`: `ignore` found the path, so nothing is to be written.
    ///
    /// # Errors
    /// `PATH_ALREADY_EXISTS` (`error` mode on an existing path); store
    /// failures.
    pub async fn begin(
        io: &IoContext,
        store: StoreId,
        dir: Path,
        mode: SaveMode,
        job_uuid: impl Into<String>,
    ) -> Result<Option<Self>> {
        let file = io.head(&store, &dir).await?;
        // An append needs no listing: only a file at the path blocks it.
        let listed = match mode {
            SaveMode::Append => Vec::new(),
            _ => io.list(&store, &dir).await?,
        };
        let exists = file.is_some()
            || !listed.is_empty()
            || (matches!(mode, SaveMode::ErrorIfExists | SaveMode::Ignore)
                && io.is_directory(&store, &dir).await?);
        let replaced = match mode {
            SaveMode::ErrorIfExists if exists => {
                return Err(PathAlreadyExists {
                    output_path: format!("{store}/{dir}"),
                }
                .into());
            }
            SaveMode::Ignore if exists => return Ok(None),
            // A file at the path cannot hold files: Spark rejects an
            // append to it, and overwriting replaces it.
            SaveMode::Append if file.is_some() => {
                return Err(PathAlreadyExists {
                    output_path: format!("{store}/{dir}"),
                }
                .into());
            }
            SaveMode::Overwrite => {
                if let Some(f) = file {
                    // A file system cannot hold a directory where a file
                    // is: the file goes first (it cannot survive a failed
                    // write, as in Spark).
                    io.delete(&store, &f.location).await?;
                }
                listed.into_iter().map(|m| m.location).collect()
            }
            _ => Vec::new(),
        };
        Ok(Some(Self {
            store,
            dir,
            job_uuid: job_uuid.into(),
            replaced,
        }))
    }

    /// The writer options of task `task` (no `maxRecordsPerFile`, no
    /// target size: set them on the result).
    #[must_use]
    pub fn write_options(&self, task: u32) -> WriteOptions {
        WriteOptions {
            store: self.store.clone(),
            dir: self.dir.clone(),
            task,
            job_uuid: self.job_uuid.clone(),
            max_records_per_file: 0,
            target_file_bytes: 0,
        }
    }

    /// Commits the write of `files` (all of them, from every task): an
    /// empty file of `schema` if there are none, then the old objects of an
    /// overwrite are deleted, then `_SUCCESS` is written. Returns the files
    /// of the output.
    ///
    /// # Errors
    /// Encode and store failures. A failure while deleting the old objects
    /// of an overwrite leaves some of them next to the new files.
    pub async fn commit(
        self,
        io: &IoContext,
        format: &dyn FileFormat,
        schema: SchemaRef,
        o: &FormatOptions,
        files: Vec<WrittenFileInfo>,
    ) -> Result<Vec<WrittenFileInfo>> {
        let mut files = files;
        if files.is_empty() {
            files.push(self.empty_file(io, format, schema, o).await?);
        }
        // Never a file of this write, even if its job UUID was reused.
        let new: std::collections::HashSet<&Path> = files.iter().map(|f| &f.path).collect();
        // The (lazy) delete futures are built first: a stream mapping
        // through a borrowing closure makes this future unprovably `Send`
        // for callers that spawn it.
        let deletes: Vec<_> = self
            .replaced
            .iter()
            .filter(|p| !new.contains(p))
            .map(|p| io.delete(&self.store, p))
            .collect();
        futures::stream::iter(deletes)
            .buffer_unordered(PARALLEL_DELETES)
            .try_collect::<Vec<()>>()
            .await?;
        io.put_object(
            &self.store,
            &self.dir.clone().join(SUCCESS_MARKER),
            Bytes::new(),
        )
        .await?;
        Ok(files)
    }

    /// One file of `schema` without rows: Spark writes it so that the
    /// output keeps its schema.
    async fn empty_file(
        &self,
        io: &IoContext,
        format: &dyn FileFormat,
        schema: SchemaRef,
        o: &FormatOptions,
    ) -> Result<WrittenFileInfo> {
        let name = spark_file_name(0, &self.job_uuid, 0, &format.file_suffix(o));
        let path = self.dir.clone().join(name);
        let display = format!("{}/{path}", self.store);
        let seed = format!("{}/0/0", self.job_uuid);
        let mut encoder = format
            .writer(io, Arc::clone(&schema), &o.clone().with_write_seed(seed))
            .await
            .map_err(|e| e.into_error(&display))?;
        let mut bytes = BytesMut::new();
        while let Some(b) = encoder.take_bytes() {
            bytes.extend_from_slice(&b);
        }
        let done = encoder.finish().map_err(|e| e.into_error(&display))?;
        bytes.extend_from_slice(&done.tail);
        let size = bytes.len() as u64;
        io.put_object(&self.store, &path, bytes.freeze()).await?;
        Ok(WrittenFileInfo {
            path,
            size,
            rows: 0,
            statistics: done.statistics,
        })
    }
}

#[cfg(test)]
mod tests;
