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
//! Avro blocks flushed once at least 16,000 bytes are buffered, sync
//! markers), found by encoding prefixes in memory (`FileIO::new_with_memory`;
//! scratch bytes, no store is touched, the one exception to "every byte
//! through `IoContext`"). The search (`SpecManifests::split`) probes 250
//! entries, then where the line through the known lengths crosses the target
//! (halving when that did not halve the bracket), so with entries of similar
//! size a manifest costs a probe at or just above its count and one 250
//! below; a no-roll remainder is probed whole. The probe that settled a
//! manifest is written as it (the bytes hold no path), so measured on a
//! 10,000-file group (debug build, review of fd17eb77d): about 1.1x one
//! plain encode without a roll, 2.9x with one near 8,000 (was 3.3x and
//! 5.8x). The task yields after each probe, as encoding is CPU work on the
//! I/O runtime. The first multiple of
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

use bytes::Bytes;
use iceberg::io::FileIO;
use iceberg::spec::{
    DataFile, FormatVersion, ManifestContentType, ManifestFile, ManifestWriter,
    ManifestWriterBuilder, PartitionSpec, SchemaRef, UNASSIGNED_SEQUENCE_NUMBER,
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
    /// Data or delete manifests (a row delta's delete files, PR-36c, rolled
    /// as Java's `writeDeleteManifests` rolls them).
    pub(crate) content: ManifestContentType,
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
        match (self.version, self.content) {
            (FormatVersion::V1, _) => builder.build_v1(),
            (FormatVersion::V2, ManifestContentType::Data) => builder.build_v2_data(),
            (FormatVersion::V2, ManifestContentType::Deletes) => builder.build_v2_deletes(),
            (FormatVersion::V3, ManifestContentType::Data) => builder.build_v3_data(),
            (FormatVersion::V3, ManifestContentType::Deletes) => builder.build_v3_deletes(),
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
                let (k, probe) = self.split(&scratch, &rest).await?;
                let tail = rest.split_off(k);
                let path = names.take();
                let file = match probe {
                    Some(p) => self.write_probe(&path, p).await?,
                    None => self.write_one(&path, rest).await?,
                };
                out.push(file);
                rest = tail;
            }
        }
        Ok(out)
    }

    /// The number of `files` (not empty) the next manifest holds, and the
    /// probe of exactly those entries when one was encoded: the first
    /// multiple of 250 below `files.len()` whose prefix reaches the target,
    /// else all of them.
    ///
    /// The candidates are `c_j = 250 * j` for `1 <= j <= max_j` and
    /// `c_{max_j + 1} = n` (taken when no smaller one reaches); the length
    /// grows with the prefix, so the answer is the smallest candidate that
    /// reaches. The search keeps `lo` (known not to reach) and `hi` (reaches,
    /// or is the last candidate) and probes the candidate where the line
    /// through the known lengths crosses the target, or the middle when that
    /// did not halve the bracket: entries have similar sizes, so it usually
    /// takes the 250-entry probe, one probe just at or above the answer and
    /// one just below. Every answer is a probed fact, never an estimate.
    async fn split(
        &self,
        scratch: &FileIO,
        files: &[DataFile],
    ) -> Result<(usize, Option<Probe>), KrishivError> {
        let n = files.len();
        let max_j = n.saturating_sub(1) / ROWS_DIVISOR;
        if max_j == 0 {
            return Ok((n, None));
        }
        let at = |j: usize| if j > max_j { n } else { j * ROWS_DIVISOR };
        let target = self.target_size_bytes;
        let empty = self.probe(scratch, &[]).await?;
        let first = self.probe(scratch, &files[..ROWS_DIVISOR]).await?;
        if first.length >= target {
            return Ok((ROWS_DIVISOR, Some(first)));
        }
        // (candidate index, its length); `lo` does not reach.
        let mut lo = (1, first.length);
        // The last candidate (`n`) counts as reaching without a probe.
        let mut hi: (usize, Option<Probe>) = (max_j + 1, None);
        let mut halved = true;
        while hi.0 - lo.0 > 1 {
            let width = hi.0 - lo.0;
            // The crossing on the line through `lo` and `hi` (or, with `hi`
            // not probed, through the empty manifest and `lo`), in entries.
            let (k0, l0) = match &hi.1 {
                Some(p) => (at(hi.0), p.length),
                None => (0, empty.length),
            };
            let guess = if halved {
                crossing(at(lo.0), lo.1, k0, l0, target).map_or(lo.0 + width / 2, |k| {
                    k.div_ceil(ROWS_DIVISOR).min(max_j + 1)
                })
            } else {
                lo.0 + width / 2
            };
            let j = guess.clamp(lo.0 + 1, hi.0);
            let j = if j == hi.0 && hi.1.is_some() {
                hi.0 - 1
            } else {
                j
            };
            let p = self.probe(scratch, &files[..at(j)]).await?;
            if j > max_j {
                // All of `files`: the answer unless a smaller one reaches.
                if p.length < target {
                    return Ok((n, Some(p)));
                }
                hi = (j, Some(p));
                halved = true;
                continue;
            } else if p.length >= target {
                hi = (j, Some(p));
            } else {
                lo = (j, p.length);
            }
            halved = (hi.0 - lo.0) * 2 <= width;
        }
        Ok((at(hi.0), hi.1))
    }

    /// A manifest of `files` encoded in memory: its length, the manifest
    /// iceberg-rust described and the bytes.
    async fn probe(&self, scratch: &FileIO, files: &[DataFile]) -> Result<Probe, KrishivError> {
        const PROBE: &str = "memory:///krishiv-manifest-probe.avro";
        let mut writer = self.writer(scratch.new_output(PROBE).map_err(from_iceberg)?);
        for f in files {
            writer
                .add_file(f.clone(), UNASSIGNED_SEQUENCE_NUMBER)
                .map_err(from_iceberg)?;
        }
        let written = writer.write_manifest_file().await.map_err(from_iceberg);
        let bytes = match &written {
            Ok(_) => match scratch.new_input(PROBE) {
                Ok(input) => input.read().await.map_err(from_iceberg),
                Err(e) => Err(from_iceberg(e)),
            },
            Err(_) => Ok(Bytes::new()),
        };
        let _ = scratch.delete(PROBE).await;
        // Encoding is CPU work on the I/O runtime: let other tasks run.
        tokio::task::yield_now().await;
        let file = written?;
        let bytes = bytes?;
        Ok(Probe {
            length: file.manifest_length,
            file,
            bytes,
        })
    }

    /// The probe's bytes at `path`: the manifest holds no path of its own, so
    /// they are the manifest iceberg-rust would write there.
    async fn write_probe(&self, path: &str, probe: Probe) -> Result<ManifestFile, KrishivError> {
        let output = self.io.new_output(path).map_err(from_iceberg)?;
        output.write(probe.bytes).await.map_err(from_iceberg)?;
        let mut file = probe.file;
        file.manifest_path = path.to_owned();
        Ok(file)
    }

    /// One manifest of `files` at `path`.
    async fn write_one(
        &self,
        path: &str,
        files: Vec<DataFile>,
    ) -> Result<ManifestFile, KrishivError> {
        let mut writer = self.writer(self.io.new_output(path).map_err(from_iceberg)?);
        for f in files {
            writer
                .add_file(f, UNASSIGNED_SEQUENCE_NUMBER)
                .map_err(from_iceberg)?;
        }
        writer.write_manifest_file().await.map_err(from_iceberg)
    }
}

impl ManifestNames<'_> {
    /// The next manifest's path, recorded in `written` before any file is
    /// created at it.
    fn take(&mut self) -> String {
        let path = format!("{}{}.avro", self.prefix, self.next);
        self.next += 1;
        self.written.push(path.clone());
        path
    }
}

/// A manifest encoded in memory (`SpecManifests::probe`).
struct Probe {
    /// Its length in bytes.
    length: i64,
    /// What iceberg-rust's writer returned (the path is the probe's).
    file: ManifestFile,
    /// The encoded manifest.
    bytes: Bytes,
}

/// The entry count where the line through `(k1, l1)` and `(k0, l0)` reaches
/// `target`, rounded up; `None` when the line does not rise.
fn crossing(k1: usize, l1: i64, k0: usize, l0: i64, target: i64) -> Option<usize> {
    let (k1, k0) = (i128::try_from(k1).ok()?, i128::try_from(k0).ok()?);
    let (l1, l0, target) = (i128::from(l1), i128::from(l0), i128::from(target));
    let (dk, mut dl) = (k1 - k0, l1 - l0);
    if dk == 0 || dl == 0 || (dl > 0) != (dk > 0) {
        return None;
    }
    // k = k0 + (target - l0) * dk / dl, rounded up (the divisor made
    // positive, so `div_euclid` floors).
    let mut num = (target - l0) * dk;
    if dl < 0 {
        num = -num;
        dl = -dl;
    }
    let k = k0 + num.div_euclid(dl) + i128::from(num.rem_euclid(dl) != 0);
    usize::try_from(k.max(0)).ok()
}
