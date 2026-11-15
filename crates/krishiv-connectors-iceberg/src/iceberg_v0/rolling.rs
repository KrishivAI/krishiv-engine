// SPDX-License-Identifier: Apache-2.0
//! The new manifests of an append, laid out as Java's `SnapshotProducer`
//! writes them (P3-T027, PR-36a; owner decision 2026-11-15; apache-iceberg
//! 1.12.0 `SnapshotProducer.writeDataManifests`/`writeDataFileGroup`/
//! `manifestWriterCount`, `ManifestFiles.writeParallel`/`divide`,
//! `RollingManifestWriter`).
//!
//! - **Groups** ([`manifest_writer_count`], [`divide`]): one spec's files are
//!   split, in order, into `max(1, min(pool, round_half_up(n / 10,000)))`
//!   groups of `ceil(n / groups)` (the last smaller; fewer groups when the
//!   last would be empty, as `Lists.partition`). Each group is rolled apart
//!   and the outputs are concatenated in group order. Java's `pool` is
//!   `ThreadPools.WORKER_THREAD_POOL_SIZE` (`iceberg.worker.num-threads`,
//!   default the JVM's processor count), **so Java's layout above 15,000 files
//!   of one spec depends on the host**; Krishiv fixes it at [`WRITER_POOL`]
//!   (4), Java's layout on a 4-core driver. Java writes the groups in parallel
//!   and numbers each manifest when it opens it, so its `-m<n>` order across
//!   groups is not deterministic; Krishiv writes the groups one after another
//!   and numbers in list order.
//! - **Rolling** (`RollingManifestWriter`): the first manifest of a group is
//!   opened lazily (no files, no manifest); before each further entry, when
//!   the open manifest holds a multiple of 250 entries and its length is at
//!   least `commit.manifest.target-size-bytes`, it is closed and a new one
//!   opened. Merged manifests (`merge.rs`) are written by a plain writer, as
//!   Java's `ManifestMergeManager.createManifest`, and never roll.
//!
//! **The length** (a divergence in where the split falls, not in its rule):
//! iceberg-rust 0.11's `ManifestWriter` keeps its entries in memory and
//! encodes the whole file in `write_manifest_file`, so it has no running
//! length. Krishiv's length of a manifest of `k` entries is the exact size of
//! the file iceberg-rust writes for those `k` entries (header, uncompressed
//! Avro blocks of 16,000 bytes, sync markers), found by encoding prefixes in
//! memory (`FileIO::new_with_memory`; scratch bytes, no store is touched):
//! an exponential then binary search over the multiples of 250, so a
//! manifest of `k` entries costs O(k log k) encoding. The first multiple of
//! 250 whose prefix reaches the target closes the manifest, exactly what
//! Java's loop does with this length (`rolling_tests` checks it against a
//! transcription of the loop). Java's `ManifestWriter.length()` is instead
//! the bytes its Avro `DataFileWriter` has flushed so far (whole blocks,
//! compressed with the codec `manifestWriterProperties` passes; the default
//! codec is in `TableProperties`, not read), while iceberg-rust writes
//! manifests uncompressed. So for the same files Java's manifests hold more
//! entries than Krishiv's at the same target (by the codec's ratio), and
//! Java may roll up to one Avro block later; Krishiv's manifests do reach the
//! target size on disk.

use std::sync::Arc;

use iceberg::io::FileIO;
use iceberg::spec::{
    DataFile, FormatVersion, ManifestFile, ManifestWriter, ManifestWriterBuilder, PartitionSpec,
    SchemaRef, UNASSIGNED_SEQUENCE_NUMBER,
};
use krishiv_common::error::KrishivError;

use super::storage::from_iceberg;

/// `RollingManifestWriter.ROWS_DIVISOR`: the roll check runs every 250 entries.
pub(crate) const ROWS_DIVISOR: usize = 250;

/// `SnapshotProducer.MIN_FILE_GROUP_SIZE`.
pub(crate) const MIN_FILE_GROUP_SIZE: usize = 10_000;

/// The worker pool size [`manifest_writer_count`] is given: Java's
/// `ThreadPools.WORKER_THREAD_POOL_SIZE` on a 4-core driver (module docs).
pub(crate) const WRITER_POOL: usize = 4;

/// Java's `SnapshotProducer.manifestWriterCount`:
/// `max(1, min(pool, round_half_up(files / 10,000)))`.
pub(crate) fn manifest_writer_count(pool: usize, files: usize) -> usize {
    let (q, r) = (files / MIN_FILE_GROUP_SIZE, files % MIN_FILE_GROUP_SIZE);
    let limit = if r * 2 >= MIN_FILE_GROUP_SIZE {
        q + 1
    } else {
        q
    };
    limit.min(pool).max(1)
}

/// Java's `ManifestFiles.divide`: `items` in order, in chunks of
/// `ceil(len / groups)` (`Lists.partition`). No items, no chunk.
pub(crate) fn divide<T>(items: Vec<T>, groups: usize) -> Vec<Vec<T>> {
    let n = items.len();
    if n == 0 {
        return Vec::new();
    }
    let size = n.div_ceil(groups.max(1));
    let mut out = Vec::with_capacity(n.div_ceil(size));
    let mut rest = items;
    while rest.len() > size {
        let tail = rest.split_off(size);
        out.push(rest);
        rest = tail;
    }
    out.push(rest);
    out
}

/// What one spec's new data manifests are written for.
pub(crate) struct SpecManifests<'a> {
    /// The table's file access.
    pub(crate) io: &'a FileIO,
    /// The table's format version.
    pub(crate) version: FormatVersion,
    /// The table's current schema.
    pub(crate) schema: &'a SchemaRef,
    /// The files' partition spec.
    pub(crate) spec: &'a PartitionSpec,
    /// The write's snapshot id.
    pub(crate) snapshot_id: i64,
    /// `commit.manifest.target-size-bytes`.
    pub(crate) target_size_bytes: i64,
}

/// The names of a write's manifests and the paths written so far.
pub(crate) struct ManifestNames<'a> {
    /// `<metadata dir>/<commit uuid>-m`.
    pub(crate) prefix: String,
    /// The number of the next manifest.
    pub(crate) next: usize,
    /// Every path written, pushed before its file is created, so a failed or
    /// dropped write's partial files are cleaned up with the rest.
    pub(crate) written: &'a mut Vec<String>,
}

impl SpecManifests<'_> {
    /// A writer of a manifest at `output`.
    fn writer(&self, output: iceberg::io::OutputFile) -> ManifestWriter {
        let builder = ManifestWriterBuilder::new(
            output,
            Some(self.snapshot_id),
            Arc::clone(self.schema),
            self.spec.clone(),
        );
        match self.version {
            FormatVersion::V1 => builder.build_v1(),
            FormatVersion::V2 => builder.build_v2_data(),
            FormatVersion::V3 => builder.build_v3_data(),
        }
    }

    /// Java's `writeDataManifests` over `files` (module docs): the manifests
    /// in list order.
    ///
    /// # Errors
    /// iceberg-rust's write failures, with their Krishiv class when the
    /// storage raised one.
    pub(crate) async fn write(
        &self,
        names: &mut ManifestNames<'_>,
        files: Vec<DataFile>,
    ) -> Result<Vec<ManifestFile>, KrishivError> {
        let groups = manifest_writer_count(WRITER_POOL, files.len());
        let scratch = FileIO::new_with_memory();
        let mut out = Vec::new();
        for group in divide(files, groups) {
            let mut rest = group;
            while !rest.is_empty() {
                let k = self.split(&scratch, &rest).await?;
                let tail = rest.split_off(k);
                out.push(self.write_one(names, rest).await?);
                rest = tail;
            }
        }
        Ok(out)
    }

    /// The number of `files` (not empty) the next manifest holds: the first
    /// multiple of 250 below `files.len()` whose prefix reaches the target,
    /// else all of them.
    async fn split(&self, scratch: &FileIO, files: &[DataFile]) -> Result<usize, KrishivError> {
        let n = files.len();
        // A roll needs an entry after the 250 * j-th: 250 * j < n.
        let max_j = n.saturating_sub(1) / ROWS_DIVISOR;
        if max_j == 0 {
            return Ok(n);
        }
        // Exponential search for a reaching j, then binary search below it:
        // `lo` does not reach (0 trivially), `hi` does.
        let mut lo = 0;
        let mut j = 1;
        let hi = loop {
            if self.reaches(scratch, &files[..j * ROWS_DIVISOR]).await? {
                break j;
            }
            if j == max_j {
                return Ok(n);
            }
            lo = j;
            j = (j * 2).min(max_j);
        };
        let mut hi = hi;
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if self.reaches(scratch, &files[..mid * ROWS_DIVISOR]).await? {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Ok(hi * ROWS_DIVISOR)
    }

    /// Whether a manifest of `files` is at least the target size.
    async fn reaches(&self, scratch: &FileIO, files: &[DataFile]) -> Result<bool, KrishivError> {
        const PROBE: &str = "memory:///krishiv-manifest-probe.avro";
        let mut writer = self.writer(scratch.new_output(PROBE).map_err(from_iceberg)?);
        for f in files {
            writer
                .add_file(f.clone(), UNASSIGNED_SEQUENCE_NUMBER)
                .map_err(from_iceberg)?;
        }
        let written = writer.write_manifest_file().await.map_err(from_iceberg);
        let _ = scratch.delete(PROBE).await;
        Ok(written?.manifest_length >= self.target_size_bytes)
    }

    /// One manifest of `files`, named next.
    async fn write_one(
        &self,
        names: &mut ManifestNames<'_>,
        files: Vec<DataFile>,
    ) -> Result<ManifestFile, KrishivError> {
        let path = format!("{}{}.avro", names.prefix, names.next);
        names.next += 1;
        names.written.push(path.clone());
        let mut writer = self.writer(self.io.new_output(&path).map_err(from_iceberg)?);
        for f in files {
            writer
                .add_file(f, UNASSIGNED_SEQUENCE_NUMBER)
                .map_err(from_iceberg)?;
        }
        writer.write_manifest_file().await.map_err(from_iceberg)
    }
}
