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
//!   is complete (Spark deletes them before writing), except with a dynamic
//!   partition overwrite, which lists only the partition directories the
//!   write produced at commit time. A file at the path itself is deleted
//!   first, as it would block the directory.
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
    /// Whether an overwrite replaces only the partition directories the
    /// write produced (`DYNAMIC`) rather than the whole directory.
    dynamic: bool,
    /// The objects an overwrite replaces, listed at the start (empty for
    /// a `DYNAMIC` overwrite, which lists what it wrote at commit time).
    replaced: Vec<Path>,
}

impl DirectoryWrite {
    /// Starts a write of `dir` in `store` with `mode`; `job_uuid` names its
    /// files. `None`: `ignore` found the path, so nothing is to be written.
    /// With `dynamic_overwrite`, an overwrite lists nothing now: at commit
    /// it deletes only under the partition directories the write produced.
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
        dynamic_overwrite: bool,
    ) -> Result<Option<Self>> {
        let file = io.head(&store, &dir).await?;
        // An append needs no listing: only a file at the path blocks it. A
        // dynamic overwrite lists what it wrote at commit time instead.
        let listed = match mode {
            SaveMode::Append => Vec::new(),
            SaveMode::Overwrite if dynamic_overwrite => Vec::new(),
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
            dynamic: dynamic_overwrite && matches!(mode, SaveMode::Overwrite),
            replaced,
        }))
    }

    /// The writer options of task `task` (no `maxRecordsPerFile`, no
    /// target size, unpartitioned: set them on the result).
    #[must_use]
    pub fn write_options(&self, task: u32) -> WriteOptions {
        WriteOptions {
            store: self.store.clone(),
            dir: self.dir.clone(),
            task,
            job_uuid: self.job_uuid.clone(),
            max_records_per_file: 0,
            target_file_bytes: 0,
            partition_by: Vec::new(),
            partition_names: Vec::new(),
        }
    }

    /// Commits the write of `files` (all of them, from every task): an
    /// empty file of `schema` if there are none, then the old objects of an
    /// overwrite are deleted, then `_SUCCESS` is written. Returns the files
    /// of the output. Files listed twice (a retried task attempt rewriting
    /// the same names) commit once.
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
        files = dedupe_by_path(files);
        // Never a file of this write, even if its job UUID was reused.
        let new: std::collections::HashSet<&Path> = files.iter().map(|f| &f.path).collect();
        let replaced = if self.dynamic {
            self.dynamic_replaced(io, &files).await?
        } else {
            self.replaced.clone()
        };
        // The (lazy) delete futures are built first: a stream mapping
        // through a borrowing closure makes this future unprovably `Send`
        // for callers that spawn it.
        let deletes: Vec<_> = replaced
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

    /// The old objects a dynamic overwrite replaces: everything already
    /// under the partition directories `files` were written to (never the
    /// new files themselves). Computed from the task outputs at commit
    /// time, never by listing before the write.
    async fn dynamic_replaced(
        &self,
        io: &IoContext,
        files: &[WrittenFileInfo],
    ) -> Result<Vec<Path>> {
        use std::collections::BTreeSet;
        // Every ancestor directory of a new file, strictly below the
        // write's root (a file at the root has no partition directory).
        let mut dirs: BTreeSet<Path> = BTreeSet::new();
        for f in files {
            let mut dir = f.path.parent();
            while let Some(d) = dir {
                if d == self.dir || (!self.dir.as_ref().is_empty() && !d.prefix_matches(&self.dir))
                {
                    break;
                }
                dir = d.parent();
                dirs.insert(d);
            }
        }
        let mut out = Vec::new();
        for dir in &dirs {
            // A partition directory holds data files only; the write's own
            // `_SUCCESS` (at the root) is never listed here.
            let listed = io.list(&self.store, dir).await?;
            out.extend(listed.into_iter().map(|m| m.location));
        }
        Ok(out)
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
        let mut encoder = format
            .writer(io, Arc::clone(&schema), o)
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

/// Files with repeated paths (a retried task attempt rewriting the same
/// file names) commit once, in first-seen order.
fn dedupe_by_path(files: Vec<WrittenFileInfo>) -> Vec<WrittenFileInfo> {
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    files
        .into_iter()
        .filter(|f| seen.insert(f.path.clone()))
        .collect()
}
