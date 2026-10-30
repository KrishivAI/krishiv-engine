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
use std::sync::atomic::{AtomicU32, Ordering};

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
    /// The highest task attempt observed, for the commit's orphan sweep.
    max_attempt: AtomicU32,
    /// Whether the output is partitioned: an empty one then has no data
    /// file (Spark writes only `_SUCCESS`), since a file at the root next
    /// to the partition directories breaks partition discovery.
    partitioned: bool,
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
            max_attempt: AtomicU32::new(0),
            partitioned: false,
        }))
    }

    /// Records that a task attempt ran: the commit lists for orphans only
    /// when an attempt above 0 ran. Single-process writes (attempt 0 only)
    /// then skip the commit-time listing.
    pub fn note_attempt(&self, attempt: u32) {
        self.max_attempt.fetch_max(attempt, Ordering::Relaxed);
    }

    /// This write with a partitioned layout: an empty result writes no data
    /// file.
    #[must_use]
    pub fn partitioned(mut self, partitioned: bool) -> Self {
        self.partitioned = partitioned;
        self
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
            zone: Some(0),
        }
    }

    /// Commits the write of `files` (all of them, from every committed
    /// task): an empty file of `schema` if there are none (none for a
    /// partitioned output, which Spark leaves without data files), then every
    /// object of this write no committed task listed (a failed or late
    /// attempt's files, or orphans) is deleted, then the old objects of an
    /// overwrite are deleted, then `_SUCCESS` is written. Returns the files
    /// of the output. Files listed twice (one task result
    /// reported twice) commit once. Retried attempts write under their own
    /// names (the attempt number is part of the job segment), so their files
    /// are swept as orphans unless a committed task lists them.
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
        let max_attempt = self.max_attempt.load(Ordering::Relaxed);
        self.commit_with_attempts(io, format, schema, o, files, max_attempt)
            .await
    }

    /// [`Self::commit`], sweeping orphans only when `max_attempt > 0`: with
    /// no retried attempt no late or failed attempt file can exist on the
    /// success path (a failed task fails the write before the commit), so an
    /// append into many files skips the commit-time listing. Callers that
    /// ran tasks pass the highest attempt they started.
    pub async fn commit_with_attempts(
        self,
        io: &IoContext,
        format: &dyn FileFormat,
        schema: SchemaRef,
        o: &FormatOptions,
        files: Vec<WrittenFileInfo>,
        max_attempt: u32,
    ) -> Result<Vec<WrittenFileInfo>> {
        let mut files = files;
        if files.is_empty() && !self.partitioned {
            files.push(self.empty_file(io, format, schema, o).await?);
        }
        files = dedupe_by_path(files);
        // Never a file of this write, even if its job UUID was reused.
        let new: std::collections::HashSet<&Path> = files.iter().map(|f| &f.path).collect();
        let observed = self.max_attempt.load(Ordering::Relaxed).max(max_attempt);
        if observed > 0 {
            self.delete_orphans(io, &new).await?;
        }
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

    /// Deletes every object under the write's directory that looks like a
    /// file of this job but no committed task listed: the files of a failed
    /// or late attempt (attempt `k > 0` appends `_<k>` to the job-id
    /// segment, so they still contain the job UUID), or orphans. Files of
    /// other writes (other job UUIDs) and `_SUCCESS` are kept.
    async fn delete_orphans(
        &self,
        io: &IoContext,
        committed: &std::collections::HashSet<&Path>,
    ) -> Result<()> {
        let listed = io.list(&self.store, &self.dir).await?;
        let mut orphans = Vec::new();
        for meta in &listed {
            if committed.contains(&meta.location) {
                continue;
            }
            let Some(name) = meta.location.filename() else {
                continue;
            };
            // Data files of this write (`part-<task>-<job>-c<n>...`, or
            // `part-<task>-<job>_<attempt>-c<n>...` for a retry); `_SUCCESS`
            // and other writes' files never carry this job in that slot.
            // A bare `contains(job)` over-matches single-letter test jobs
            // (`part-` and `-c000` contain `a` and `c`), so match the job
            // segment anchored to its layout: after `part-<task>-` the name
            // starts with `<job>-c` or `<job>_<attempt>-c`.
            if is_job_file(name, &self.job_uuid) {
                orphans.push(meta.location.clone());
            }
        }
        let deletes: Vec<_> = orphans.iter().map(|p| io.delete(&self.store, p)).collect();
        futures::stream::iter(deletes)
            .buffer_unordered(PARALLEL_DELETES)
            .try_collect::<Vec<()>>()
            .await?;
        Ok(())
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
        // The directory of each new file (its leaf partition) when it is
        // strictly below the write's root. Only leaves are replaced: with
        // `partitionBy(a, b)` a write of `a=1/b=2` leaves `a=1/b=3` alone,
        // and the listing of a leaf holds its data files only.
        let mut dirs: BTreeSet<Path> = BTreeSet::new();
        for f in files {
            if let Some(d) = f.path.parent()
                && d != self.dir
                && (self.dir.as_ref().is_empty() || d.prefix_matches(&self.dir))
            {
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

/// Whether `name` is a data file of the write with job id `job`: anchored
/// to `part-<task>-<job>-c<n>` or `part-<task>-<job>_<attempt>-c<n>` (and
/// `.c<n>` for the files of a partitioned write), so a job id that is a
/// suffix of another write's (`"1"` vs `"job-1"`) never matches the other's
/// files.
fn is_job_file(name: &str, job: &str) -> bool {
    let Some(rest) = name.strip_prefix("part-") else {
        return false;
    };
    let Some(dash) = rest.find('-') else {
        return false;
    };
    let Some(after_job) = rest[dash + 1..].strip_prefix(job) else {
        return false;
    };
    // An optional `_<attempt>` (digits), then `-c<n>` or `.c<n>`; the
    // suffix check stays lenient so future suffixes still match.
    let after_attempt = match after_job.strip_prefix('_') {
        Some(after_underscore) => {
            let digits = after_underscore
                .bytes()
                .take_while(u8::is_ascii_digit)
                .count();
            if digits == 0 {
                return false;
            }
            &after_underscore[digits..]
        }
        None => after_job,
    };
    after_attempt
        .strip_prefix("-c")
        .or_else(|| after_attempt.strip_prefix(".c"))
        .and_then(|after_c| after_c.chars().next())
        .is_some_and(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod tests;

/// Files with repeated paths (one task result reported twice) commit once, in first-seen order.
fn dedupe_by_path(files: Vec<WrittenFileInfo>) -> Vec<WrittenFileInfo> {
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    files
        .into_iter()
        .filter(|f| seen.insert(f.path.clone()))
        .collect()
}
