// SPDX-License-Identifier: Apache-2.0
//! Merge-on-read delete writers (P3-T027, PR-36c unit 1; design note §8 item
//! 4): one write task's deleted positions, collected per data file
//! ([`PositionDeletes`]), written as v3 deletion vectors or v2 position
//! delete files, each described by an iceberg `DataFile` for a `RowDelta`
//! (the commit is a later unit), with the delete files they supersede.
//!
//! Java references are apache-iceberg 1.12.0 (`ij:`) and its Spark 4.1
//! module (`spark:`), from the fetched sources.
//!
//! - **The kind of delete is fixed by the format version**
//!   (`ij:MergingSnapshotProducer.java:302-309`): v3 writes deletion vectors
//!   only, v2 position delete files only, v1 is refused
//!   (`KRISHIV_UNSUPPORTED_FEATURE`: row-level deletes need v2). On v3 the
//!   delete format is Puffin whatever `write.delete.format.default` says
//!   (`spark:SparkWriteConf.java:245-257`); on v2 that property, else
//!   `write.format.default`, must be `parquet`.
//! - **Granularity** (`write.delete.granularity`, case-insensitive `file` or
//!   `partition`; otherwise "Unknown delete granularity: <v>", Java's
//!   `DeleteGranularity.fromString` text, as `KRISHIV_UNSUPPORTED_FEATURE`)
//!   defaults to `file`, Spark's `SparkWriteConf.deleteGranularity`
//!   (`spark:SparkWriteConf.java:727-733`), not `TableProperties`'
//!   `partition` (`ij:TableProperties.java:429-430`). Spark parses it for
//!   every row-level write (its `Context`), so a bad value fails v3 writes
//!   too; so does it here.
//! - **Previous deletes** ([`load_previous`], Java's `rewritableDeletes` /
//!   `shouldRewrite`, `spark:SparkRuntimeFilterableScan.java:148-173`, and
//!   `shouldRewriteDeletes`, `spark:SparkPositionDeltaWrite.java:215-228`):
//!   with DVs every non-equality delete of the data file (its DV, or the v2
//!   position delete files that apply to it after an upgrade); on v2 with
//!   granularity `file` the file-scoped position delete files only; on v2
//!   with `partition` none. Positions are read by `deletes`' readers and
//!   charged as a scan's are.
//! - **Deletion vectors** (`ij:deletes/BaseDVFileWriter.java:113-212`): one
//!   Puffin file per writer, written only when there is a deleted position,
//!   holding one `deletion-vector-v1` blob per data file (in path order;
//!   Java's follows a hash map): the union of the new and the previous
//!   positions ([`encode_dv`]), fields `[_pos]` (`i32::MAX - 2`), snapshot
//!   id and sequence number -1 (inherited), uncompressed, properties
//!   `referenced-data-file` and `cardinality`; the file property
//!   `created-by` is `Krishiv <version>` (Java writes
//!   `IcebergBuild.fullVersion()`); the footer is uncompressed. One
//!   `DataFile` per blob shares the Puffin path and size, with the blob's
//!   offset and length (computed: iceberg-rust's `PuffinWriter` does not
//!   return them; the 4-byte magic, then the blobs back to back) and the
//!   data file's spec and partition. Superseded (`rewritten`): the DVs and
//!   file-scoped position delete files among the previous deletes
//!   (`ContentFileUtil.isFileScoped`, `ij:ContentFileUtil.java:63-89,138-140`);
//!   partition-scoped position delete files are merged into the DV but stay.
//! - **Position delete files** (`ij:deletes/SortingPositionOnlyDeleteWriter.java`,
//!   `ij:deletes/PositionDeleteWriter.java`): Parquet with `file_path`
//!   (string, required, id `i32::MAX - 101`) and `pos` (long, required, id
//!   `i32::MAX - 102`), rows sorted by path then position; granularity
//!   `file` one file (or more, rolling) per data path, `partition` one per
//!   spec and partition tuple with its paths sorted. Previous (file-scoped)
//!   deletes are merged first and returned as rewritten (only when they hold
//!   a position, as Java's `isNotEmpty` check; anything not file-scoped is
//!   an internal error, Java's "Previous deletes must be file-scoped").
//!   Metrics as `PositionDeleteWriter.metrics` with the two columns in
//!   `full` mode (Java's `MetricsConfig.forPositionDelete`, not in the
//!   fetched sources): the record count; when the file names exactly one
//!   data path, untruncated lower and upper bounds of both columns (so the
//!   `file_path` bounds are equal and the file is file-scoped) and no
//!   counts; with more paths no counts and no bounds. **`referenced_data_file`
//!   is not set**: Java's writer never sets it, file scope comes from the
//!   equal bounds (design note §8 item 4 says it is set; Java wins).
//!   Rolling as Java's `RollingFileWriter` (a new file once the current one
//!   holds a multiple of 1,000 rows and has reached
//!   `write.delete.target-file-size-bytes`, 64 MiB by default). Parquet
//!   codec: `write.delete.parquet.compression-codec`, else
//!   `write.parquet.compression-codec`, else `gzip`
//!   (`spark:SparkWriteConf.java:570-588`); row-group and page sizes from
//!   `write.parquet.row-group-size-bytes` / `page-size-bytes`: the
//!   `write.delete.parquet.*` variants exist (`ij:TableProperties.java:129-134`)
//!   but their fallback lives in `Parquet.WriteBuilder`, which was not
//!   fetched, so they are not read (an open item). No split offsets.
//! - **Names and locations** (`ij:io/OutputFileFactory.java:93-121`):
//!   `%05d-%d-%s-%05d-deletes.parquet` / `.puffin` (partition id, attempt id,
//!   write id, a counter of this writer from 1; the suffix `deletes` as
//!   `spark:SparkPositionDeltaWrite.java:448-453`). Position delete files go
//!   where a data file of their spec and partition goes; a Puffin file goes
//!   where an unpartitioned data file goes (`BaseDVFileWriter` takes
//!   `fileFactory::newOutputFile`, the no-partition overload). One Puffin
//!   per task whatever partitions its paths span follows Java's
//!   `PartitioningDVWriter` holding one `BaseDVFileWriter` (that class was
//!   not fetched: from its use in `spark:SparkPositionDeltaWrite.java:548-549`).
//! - **Failure:** any error deletes what the writer wrote (finished files
//!   deleted, open uploads aborted); a write future dropped midway deletes
//!   its finished files in the background, as `write.rs`'s writer does.
//!
//! The collected positions themselves are not charged to memory (the
//! caller holds them); encoder and upload buffers and a blob being written
//! are charged to the writer's `Reservation`.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "called by the merge-on-read DML path (PR-36c unit 2, after PR-37b's RowLevelTable)"
    )
)]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use iceberg::io::FileIO;
use iceberg::puffin::{Blob, CompressionCodec, DELETION_VECTOR_V1, PuffinWriter};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, FormatVersion,
    PartitionSpec, PartitionSpecRef, SchemaRef, Struct, TableMetadata,
};
use krishiv_common::config::Config;
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_common::secret::OptionValue;
use krishiv_exec::files::{LocatedPath, PathLocator};
use krishiv_io::io::upload::Upload;
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{Encoder, FileFormat as _, FileRef, FormatOptions, IoContext, PinnedStore};
use krishiv_memory::{ConsumerHandle, Reservation};
use roaring::RoaringTreemap;

use super::data_file::{DataLocations, check_partition};
use super::deletes::{read_dv, read_position_file};
use super::storage::{Registered, from_iceberg};
use super::write::TaskIds;
use crate::plan::deletes::referenced_data_file;
use crate::plan::{
    DELETE_FILE_PATH_ID, DELETE_FILE_POS_ID, DeleteFileEntry, DeleteKind, TaskDeletes, encode_dv,
};

/// `write.delete.granularity` (`ij:TableProperties.java:429`).
pub(crate) const DELETE_GRANULARITY: &str = "write.delete.granularity";
/// `write.delete.format.default` (`ij:TableProperties.java:125`).
pub(crate) const DELETE_FORMAT: &str = "write.delete.format.default";
/// `write.format.default`, the delete format when the above is unset.
const FILE_FORMAT: &str = "write.format.default";
/// `write.delete.target-file-size-bytes` and its default, 64 MiB
/// (`ij:TableProperties.java:351-352`).
pub(crate) const DELETE_TARGET_FILE_SIZE: &str = "write.delete.target-file-size-bytes";
const DELETE_TARGET_FILE_SIZE_DEFAULT: i64 = 64 * 1024 * 1024;
/// `write.delete.parquet.compression-codec` (`ij:TableProperties.java:152`).
pub(crate) const DELETE_PARQUET_CODEC: &str = "write.delete.parquet.compression-codec";
/// `write.parquet.compression-codec` and Java's fallback `gzip`.
const PARQUET_CODEC: &str = "write.parquet.compression-codec";
const PARQUET_CODEC_DEFAULT: &str = "gzip";
/// `write.parquet.row-group-size-bytes` and its default (128 MiB).
const ROW_GROUP_SIZE: &str = "write.parquet.row-group-size-bytes";
const ROW_GROUP_SIZE_DEFAULT: i32 = 128 * 1024 * 1024;
/// `write.parquet.page-size-bytes` and its default (1 MiB).
const PAGE_SIZE: &str = "write.parquet.page-size-bytes";
const PAGE_SIZE_DEFAULT: i32 = 1024 * 1024;

/// `_pos`, the field a deletion vector is computed for (`MetadataColumns.ROW_POSITION`).
pub(crate) const ROW_POSITION_ID: i32 = i32::MAX - 2;
/// A DV blob's property naming its data file.
pub(crate) const REFERENCED_DATA_FILE_KEY: &str = "referenced-data-file";
/// A DV blob's property holding its position count.
pub(crate) const CARDINALITY_KEY: &str = "cardinality";
/// The Puffin header magic's length: the first blob starts after it.
const PUFFIN_MAGIC_LEN: u64 = 4;
/// Rows between two checks of the target file size (`RollingFileWriter.ROWS_DIVISOR`).
const ROWS_DIVISOR: u64 = 1000;

/// How the v2 position deletes of a task are split into files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Granularity {
    /// One delete file per data file (file-scoped).
    File,
    /// One delete file per spec and partition tuple.
    Partition,
}

impl Granularity {
    /// `write.delete.granularity` of a table with `props` (module docs).
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` with Java's text for another value.
    pub(crate) fn from_table(props: &HashMap<String, String>) -> Result<Self, KrishivError> {
        match props.get(DELETE_GRANULARITY) {
            None => Ok(Self::File),
            Some(v) if v.eq_ignore_ascii_case("file") => Ok(Self::File),
            Some(v) if v.eq_ignore_ascii_case("partition") => Ok(Self::Partition),
            Some(v) => Err(unsupported(format!(
                "writing Iceberg deletes ({DELETE_GRANULARITY}): Unknown delete granularity: {v}"
            ))),
        }
    }
}

/// What a table's row-level deletes are written as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeleteFormat {
    /// v3: deletion vectors in a Puffin file.
    DeletionVectors,
    /// v2: Parquet position delete files.
    PositionFiles(Granularity),
}

impl DeleteFormat {
    /// The delete format of the table `metadata` describes (module docs).
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a v1 table, a bad granularity, or a
    /// v2 delete format other than Parquet.
    pub(crate) fn for_table(metadata: &TableMetadata) -> Result<Self, KrishivError> {
        let props = metadata.properties();
        match metadata.format_version() {
            FormatVersion::V1 => Err(unsupported(
                "row-level deletes on an Iceberg v1 table (they need format version 2)".to_owned(),
            )),
            FormatVersion::V2 => {
                let granularity = Granularity::from_table(props)?;
                let format = props
                    .get(DELETE_FORMAT)
                    .or_else(|| props.get(FILE_FORMAT))
                    .map_or("parquet", String::as_str);
                if !format.eq_ignore_ascii_case("parquet") {
                    return Err(unsupported(format!(
                        "writing Iceberg {format} position delete files ({DELETE_FORMAT})"
                    )));
                }
                Ok(Self::PositionFiles(granularity))
            }
            FormatVersion::V3 => {
                Granularity::from_table(props)?;
                Ok(Self::DeletionVectors)
            }
        }
    }

    /// Whether the previous delete `d` of a data file is loaded and merged
    /// (Java's `shouldRewrite`; v2 `partition` loads nothing).
    fn loads(self, d: &DeleteFileEntry) -> bool {
        match self {
            Self::DeletionVectors => d.kind != DeleteKind::EqualityDelete,
            Self::PositionFiles(Granularity::File) => is_file_scoped(d),
            Self::PositionFiles(Granularity::Partition) => false,
        }
    }
}

/// Whether `d` applies to one data file only (`ContentFileUtil.isFileScoped`:
/// a DV, or a position delete file with `referenced_data_file` or equal
/// `file_path` bounds; never an equality delete).
pub(crate) fn is_file_scoped(d: &DeleteFileEntry) -> bool {
    d.kind != DeleteKind::EqualityDelete && referenced_data_file(d).is_some()
}

/// The deleted positions of one data file.
#[derive(Debug, Clone)]
struct PathDeletes {
    spec_id: i32,
    partition: Struct,
    positions: RoaringTreemap,
}

/// One task's deleted positions per data file path, with each file's spec
/// id and partition tuple.
#[derive(Debug, Clone, Default)]
pub(crate) struct PositionDeletes {
    by_path: BTreeMap<String, PathDeletes>,
}

impl PositionDeletes {
    /// Deletes row `pos` of the data file `path` (of spec `spec_id`,
    /// partition `partition`).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when `path` was given another spec or partition.
    pub(crate) fn delete(
        &mut self,
        path: &str,
        pos: u64,
        spec_id: i32,
        partition: &Struct,
    ) -> Result<(), KrishivError> {
        self.entry(path, spec_id, partition)?.positions.insert(pos);
        Ok(())
    }

    /// Deletes every row of `positions` of the data file `path`.
    ///
    /// # Errors
    /// As [`Self::delete`].
    pub(crate) fn delete_all(
        &mut self,
        path: &str,
        positions: &RoaringTreemap,
        spec_id: i32,
        partition: &Struct,
    ) -> Result<(), KrishivError> {
        self.entry(path, spec_id, partition)?.positions |= positions;
        Ok(())
    }

    /// Whether no position was deleted.
    pub(crate) fn is_empty(&self) -> bool {
        self.by_path.is_empty()
    }

    /// The data paths with deletes, in order.
    pub(crate) fn paths(&self) -> impl Iterator<Item = &str> {
        self.by_path.keys().map(String::as_str)
    }

    fn entry(
        &mut self,
        path: &str,
        spec_id: i32,
        partition: &Struct,
    ) -> Result<&mut PathDeletes, KrishivError> {
        let d = self
            .by_path
            .entry(path.to_owned())
            .or_insert_with(|| PathDeletes {
                spec_id,
                partition: partition.clone(),
                positions: RoaringTreemap::new(),
            });
        if d.spec_id != spec_id || d.partition != *partition {
            return Err(internal(format!(
                "the data file {path} was given spec {spec_id} and partition {partition:?}, \
                 earlier spec {} and partition {:?}",
                d.spec_id, d.partition
            )));
        }
        Ok(d)
    }
}

/// The previous deletes of one data file a write merges (Java's
/// `PositionDeleteIndex` with its `deleteFiles`).
#[derive(Debug)]
pub(crate) struct Previous {
    /// Their positions, unioned.
    pub(crate) positions: RoaringTreemap,
    /// The delete files they were read from.
    pub(crate) files: Vec<Arc<DeleteFileEntry>>,
    /// The memory holding the positions.
    _held: Vec<Reservation>,
}

/// The previous deletes of the data file `data_path` among `deletes` (its
/// task's deletes) that a write of `format` merges (module docs), read
/// through `io` and charged to `mem`; `None` when none is loaded.
///
/// # Errors
/// `deletes::read_position_file`'s and `read_dv`'s: the store's, a corrupt
/// DV, an unreadable format, memory.
pub(crate) async fn load_previous(
    io: &IoContext,
    locate: &(dyn Fn(&str) -> Result<FileRef, KrishivError> + Send + Sync),
    mem: &Arc<ConsumerHandle>,
    format: DeleteFormat,
    data_path: &str,
    deletes: &TaskDeletes,
) -> Result<Option<Previous>, KrishivError> {
    let files: Vec<Arc<DeleteFileEntry>> = deletes
        .deletion_vector
        .iter()
        .chain(&deletes.position_deletes)
        .filter(|d| format.loads(d))
        .cloned()
        .collect();
    if files.is_empty() {
        return Ok(None);
    }
    let paths = BTreeSet::from([data_path.to_owned()]);
    let mut positions = RoaringTreemap::new();
    let mut held = Vec::with_capacity(files.len());
    for d in &files {
        if d.kind == DeleteKind::DeletionVector {
            let (p, charge) = read_dv(io, locate, mem, d).await?;
            positions |= p;
            held.push(charge);
        } else {
            let (mut by_path, charge) = read_position_file(io, locate, mem, d, &paths).await?;
            if let Some(p) = by_path.remove(data_path) {
                positions.extend(p);
            }
            held.push(charge);
        }
    }
    Ok(Some(Previous {
        positions,
        files,
        _held: held,
    }))
}

/// What a delete write produced (Java's `DeleteWriteResult`).
#[derive(Debug, Default)]
pub(crate) struct DeleteWriteResult {
    /// The new delete files.
    pub(crate) delete_files: Vec<DataFile>,
    /// The data files they reference.
    pub(crate) referenced_data_files: BTreeSet<String>,
    /// The previous delete files they supersede (to remove in the same commit).
    pub(crate) rewritten: Vec<Arc<DeleteFileEntry>>,
}

/// A file being written, or one written, that a failure deletes.
#[derive(Default)]
struct Written {
    parquet: Vec<(PinnedStore, LocatedPath)>,
    puffin: Vec<String>,
}

/// Deletes what a failed write wrote; on drop (a write future dropped
/// midway) in a task on the I/O runtime.
struct Cleanup {
    io: IoContext,
    file_io: FileIO,
    written: Written,
    armed: bool,
}

impl Cleanup {
    /// Cancellation-safe: a file leaves the list only once its delete ran.
    async fn run(&mut self) {
        while let Some((store, at)) = self.written.parquet.last() {
            let _ = self.io.delete_pinned(store, &at.path).await;
            self.written.parquet.pop();
        }
        while let Some(location) = self.written.puffin.last() {
            let _ = self.file_io.delete(location).await;
            self.written.puffin.pop();
        }
        self.armed = false;
    }
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let written = std::mem::take(&mut self.written);
        if written.parquet.is_empty() && written.puffin.is_empty() {
            return;
        }
        let (io, file_io) = (self.io.clone(), self.file_io.clone());
        let delete = async move {
            for (store, at) in written.parquet {
                let _ = io.delete_pinned(&store, &at.path).await;
            }
            for location in written.puffin {
                let _ = file_io.delete(&location).await;
            }
        };
        let handle = self
            .io
            .scheduler()
            .runtime()
            .cloned()
            .or_else(|| tokio::runtime::Handle::try_current().ok());
        if let Some(h) = handle {
            h.spawn(delete);
        }
    }
}

/// An open position delete file.
struct OpenFile {
    spec_id: i32,
    partition: Struct,
    location: String,
    at: LocatedPath,
    encoder: Box<dyn Encoder>,
    upload: Upload,
    rows: u64,
    /// The data paths it names.
    paths: BTreeSet<String>,
    min_pos: i64,
    max_pos: i64,
}

/// One task's delete writer (module docs).
pub(crate) struct DeleteFileWriter {
    io: IoContext,
    paths: Arc<dyn PathLocator>,
    file_io: FileIO,
    schema: SchemaRef,
    specs: HashMap<i32, PartitionSpecRef>,
    locations: DataLocations,
    format: DeleteFormat,
    /// Parquet options (v2 only).
    options: Option<FormatOptions>,
    target_file_bytes: u64,
    arrow: arrow_schema::SchemaRef,
    partition_id: u32,
    attempt_id: u64,
    write_id: String,
    /// Files opened so far (the last file's number).
    files: u32,
    mem: Reservation,
}

impl std::fmt::Debug for DeleteFileWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeleteFileWriter")
            .field("format", &self.format)
            .field("write_id", &self.write_id)
            .finish_non_exhaustive()
    }
}

impl DeleteFileWriter {
    /// A delete writer of the table `metadata` describes, writing through
    /// `registered` (Parquet) and `file_io` (Puffin), holding buffers in
    /// `mem`.
    ///
    /// # Errors
    /// [`DeleteFormat::for_table`]'s; `KRISHIV_UNSUPPORTED_FEATURE` for a
    /// codec Krishiv cannot write or a size property that is not a number;
    /// [`DataLocations::for_table`]'s.
    pub(crate) fn new(
        metadata: &TableMetadata,
        registered: Registered,
        file_io: FileIO,
        ids: TaskIds<'_>,
        mem: Reservation,
    ) -> Result<Self, KrishivError> {
        let format = DeleteFormat::for_table(metadata)?;
        let props = metadata.properties();
        let options = match format {
            DeleteFormat::DeletionVectors => None,
            DeleteFormat::PositionFiles(_) => Some(parquet_options(props)?),
        };
        Ok(Self {
            io: registered.io,
            paths: registered.paths,
            file_io,
            schema: Arc::clone(metadata.current_schema()),
            specs: metadata
                .partition_specs_iter()
                .map(|s| (s.spec_id(), Arc::clone(s)))
                .collect(),
            locations: DataLocations::for_table(metadata)?,
            format,
            options,
            target_file_bytes: target_file_bytes(props)?,
            arrow: position_schema(),
            partition_id: ids.partition_id,
            attempt_id: ids.attempt_id,
            write_id: ids.write_id.to_owned(),
            files: 0,
            mem,
        })
    }

    /// What this writer writes (and so which previous deletes to load).
    pub(crate) fn format(&self) -> DeleteFormat {
        self.format
    }

    /// Writes `deletes`, each data path merged with its `previous` deletes
    /// (by data path, from [`load_previous`] with [`Self::format`]).
    ///
    /// # Errors
    /// Encode, upload and memory errors; `INTERNAL_ERROR` for a spec id the
    /// table does not have, a partition tuple that does not fit its spec, a
    /// position iceberg cannot hold. Every file written is then deleted.
    pub(crate) async fn write(
        mut self,
        deletes: &PositionDeletes,
        previous: &HashMap<String, Previous>,
    ) -> Result<DeleteWriteResult, KrishivError> {
        let mut cleanup = Cleanup {
            io: self.io.clone(),
            file_io: self.file_io.clone(),
            written: Written::default(),
            armed: true,
        };
        let result = match self.format {
            DeleteFormat::DeletionVectors => {
                self.write_dvs(deletes, previous, &mut cleanup.written)
                    .await
            }
            DeleteFormat::PositionFiles(g) => {
                self.write_positions(g, deletes, previous, &mut cleanup.written)
                    .await
            }
        };
        let held = self.mem.size();
        self.mem.shrink(held);
        match result {
            Ok(r) => {
                cleanup.armed = false;
                Ok(r)
            }
            Err(e) => {
                cleanup.run().await;
                Err(e)
            }
        }
    }

    /// The next file's name with `extension` (module docs).
    fn next_name(&mut self, extension: &str) -> Result<String, KrishivError> {
        self.files = self
            .files
            .checked_add(1)
            .ok_or_else(|| internal("too many delete files in one task".to_owned()))?;
        Ok(delete_file_name(
            self.partition_id,
            self.attempt_id,
            &self.write_id,
            self.files,
            extension,
        ))
    }

    fn spec(&self, spec_id: i32) -> Result<&PartitionSpecRef, KrishivError> {
        self.specs.get(&spec_id).ok_or_else(|| {
            internal(format!(
                "a data file of partition spec {spec_id}, which the table does not have"
            ))
        })
    }

    async fn write_dvs(
        &mut self,
        deletes: &PositionDeletes,
        previous: &HashMap<String, Previous>,
        written: &mut Written,
    ) -> Result<DeleteWriteResult, KrishivError> {
        let mut out = DeleteWriteResult::default();
        if deletes.is_empty() {
            return Ok(out);
        }
        for d in deletes.by_path.values() {
            check_partition(self.spec(d.spec_id)?, &self.schema, &d.partition)?;
        }
        let name = self.next_name("puffin")?;
        let location = self.locations.location(
            &PartitionSpec::unpartition_spec(),
            &self.schema,
            &Struct::empty(),
            &name,
        )?;
        let output = self.file_io.new_output(&location).map_err(from_iceberg)?;
        written.puffin.push(location.clone());
        let created_by = HashMap::from([(
            iceberg::puffin::CREATED_BY_PROPERTY.to_owned(),
            format!("Krishiv {}", env!("CARGO_PKG_VERSION")),
        )]);
        let mut writer = PuffinWriter::new(&output, created_by, false)
            .await
            .map_err(from_iceberg)?;
        let mut offset = PUFFIN_MAGIC_LEN;
        let mut blobs = Vec::with_capacity(deletes.by_path.len());
        for (path, d) in &deletes.by_path {
            let mut positions = d.positions.clone();
            if let Some(prev) = previous.get(path) {
                positions |= &prev.positions;
                out.rewritten.extend(
                    prev.files
                        .iter()
                        .filter(|f| is_file_scoped(f))
                        .map(Arc::clone),
                );
            }
            let data = encode_dv(&positions)?;
            let len = data.len() as u64;
            self.mem
                .grow(usize::try_from(len).unwrap_or(usize::MAX))
                .await?;
            let cardinality = positions.len();
            let blob = Blob::builder()
                .r#type(DELETION_VECTOR_V1.to_owned())
                .fields(vec![ROW_POSITION_ID])
                .snapshot_id(-1)
                .sequence_number(-1)
                .data(data)
                .properties(HashMap::from([
                    (REFERENCED_DATA_FILE_KEY.to_owned(), path.clone()),
                    (CARDINALITY_KEY.to_owned(), cardinality.to_string()),
                ]))
                .build();
            writer
                .add(blob, CompressionCodec::None)
                .await
                .map_err(from_iceberg)?;
            let held = self.mem.size();
            self.mem.shrink(held);
            blobs.push((path, d, offset, len, cardinality));
            offset = offset.saturating_add(len);
            out.referenced_data_files.insert(path.clone());
        }
        let size = writer.close().await.map_err(from_iceberg)?.size;
        for (path, d, offset, len, cardinality) in blobs {
            let mut b = DataFileBuilder::default();
            b.content(DataContentType::PositionDeletes)
                .file_path(location.clone())
                .file_format(DataFileFormat::Puffin)
                .partition(d.partition.clone())
                .partition_spec_id(d.spec_id)
                .record_count(cardinality)
                .file_size_in_bytes(size)
                .referenced_data_file(Some(path.clone()))
                .content_offset(Some(to_i64(offset)?))
                .content_size_in_bytes(Some(to_i64(len)?));
            out.delete_files.push(
                b.build()
                    .map_err(|e| internal(format!("deletion vector of {path}: {e}")))?,
            );
        }
        Ok(out)
    }

    async fn write_positions(
        &mut self,
        granularity: Granularity,
        deletes: &PositionDeletes,
        previous: &HashMap<String, Previous>,
        written: &mut Written,
    ) -> Result<DeleteWriteResult, KrishivError> {
        let mut out = DeleteWriteResult::default();
        // Data paths by spec and partition, groups in their first path's order.
        let mut index: HashMap<(i32, &Struct), usize> = HashMap::new();
        let mut groups: Vec<Vec<(&String, &PathDeletes)>> = Vec::new();
        for (path, d) in &deletes.by_path {
            let at = *index.entry((d.spec_id, &d.partition)).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[at].push((path, d));
        }
        for group in groups {
            let units: Vec<Vec<(&String, &PathDeletes)>> = match granularity {
                Granularity::File => group.into_iter().map(|p| vec![p]).collect(),
                Granularity::Partition => vec![group],
            };
            for unit in units {
                let mut open: Option<OpenFile> = None;
                let r = self
                    .write_unit(&unit, previous, &mut open, written, &mut out)
                    .await;
                if let Err(e) = r {
                    if let Some(f) = open {
                        f.upload.abort().await;
                    }
                    return Err(e);
                }
            }
        }
        Ok(out)
    }

    /// Writes the paths of `unit` (sorted, one spec and partition) into
    /// delete files, rolling over; the last file is closed.
    async fn write_unit(
        &mut self,
        unit: &[(&String, &PathDeletes)],
        previous: &HashMap<String, Previous>,
        open: &mut Option<OpenFile>,
        written: &mut Written,
        out: &mut DeleteWriteResult,
    ) -> Result<(), KrishivError> {
        for (path, d) in unit {
            let mut positions = d.positions.clone();
            if let Some(prev) = previous.get(*path)
                && !prev.positions.is_empty()
            {
                if !prev.files.iter().all(|f| is_file_scoped(f)) {
                    return Err(internal(format!(
                        "Previous deletes must be file-scoped (data file {path})"
                    )));
                }
                positions |= &prev.positions;
                out.rewritten.extend(prev.files.iter().map(Arc::clone));
            }
            out.referenced_data_files.insert((*path).clone());
            let mut iter = positions.iter().peekable();
            while iter.peek().is_some() {
                if open.is_none() {
                    *open = Some(self.open_file(d).await?);
                }
                let Some(f) = open.as_mut() else {
                    return Err(internal("no open delete file".to_owned()));
                };
                let room = ROWS_DIVISOR - f.rows % ROWS_DIVISOR;
                let chunk = iter
                    .by_ref()
                    .take(usize::try_from(room).unwrap_or(usize::MAX))
                    .map(|p| {
                        i64::try_from(p).map_err(|_| internal(format!("position {p} past a long")))
                    })
                    .collect::<Result<Vec<i64>, _>>()?;
                let n = chunk.len();
                if let (Some(lo), Some(hi)) = (chunk.first(), chunk.last()) {
                    if f.paths.is_empty() {
                        (f.min_pos, f.max_pos) = (*lo, *hi);
                    } else {
                        f.min_pos = f.min_pos.min(*lo);
                        f.max_pos = f.max_pos.max(*hi);
                    }
                }
                f.paths.insert((*path).clone());
                let batch = RecordBatch::try_new(
                    Arc::clone(&self.arrow),
                    vec![
                        Arc::new(StringArray::from(vec![path.as_str(); n])) as ArrayRef,
                        Arc::new(Int64Array::from(chunk)),
                    ],
                )
                .map_err(|e| internal(format!("a position delete batch: {e}")))?;
                f.encoder
                    .write(&batch)
                    .map_err(|e| e.into_error(&f.at.display))?;
                f.rows += n as u64;
                drain(f).await?;
                self.account(open).await?;
                if let Some(f) = open.as_ref()
                    && f.rows.is_multiple_of(ROWS_DIVISOR)
                    && f.encoder.written_bytes() >= self.target_file_bytes
                    && let Some(f) = open.take()
                {
                    self.close(f, written, out).await?;
                }
            }
        }
        if let Some(f) = open.take() {
            self.close(f, written, out).await?;
        }
        Ok(())
    }

    /// A new position delete file in the spec and partition of `d`.
    async fn open_file(&mut self, d: &PathDeletes) -> Result<OpenFile, KrishivError> {
        let name = self.next_name("parquet")?;
        let spec = Arc::clone(self.spec(d.spec_id)?);
        let location = self
            .locations
            .location(&spec, &self.schema, &d.partition, &name)?;
        let at = self.paths.locate(&location)?;
        let store = self.io.pin(&at.store)?;
        let seed = format!(
            "{}/{}/{}/deletes/{}",
            self.write_id, self.partition_id, self.attempt_id, self.files
        );
        let options = self
            .options
            .clone()
            .ok_or_else(|| internal("position deletes without Parquet options".to_owned()))?;
        let encoder = ParquetFormat
            .writer(
                &self.io,
                Arc::clone(&self.arrow),
                &options.with_write_seed(seed),
            )
            .await
            .map_err(|e| e.into_error(&at.display))?;
        let upload = self.io.upload_pinned(&store, at.path.clone())?;
        Ok(OpenFile {
            spec_id: d.spec_id,
            partition: d.partition.clone(),
            location,
            at,
            encoder,
            upload,
            rows: 0,
            paths: BTreeSet::new(),
            min_pos: 0,
            max_pos: 0,
        })
    }

    /// Ends `f` and describes it (module docs: metrics).
    async fn close(
        &mut self,
        mut f: OpenFile,
        written: &mut Written,
        out: &mut DeleteWriteResult,
    ) -> Result<(), KrishivError> {
        drain(&mut f).await?;
        let OpenFile {
            spec_id,
            partition,
            location,
            at,
            encoder,
            mut upload,
            rows,
            paths,
            min_pos,
            max_pos,
        } = f;
        let store = self.io.pin(&at.store)?;
        let done = match encoder.finish() {
            Ok(done) => done,
            Err(e) => {
                upload.abort().await;
                return Err(e.into_error(&at.display));
            }
        };
        if let Err(e) = upload.write(done.tail).await {
            upload.abort().await;
            return Err(e);
        }
        // From here the file may exist: a failure deletes it.
        written.parquet.push((store, at));
        let size = upload.finish().await?;
        let mut b = DataFileBuilder::default();
        b.content(DataContentType::PositionDeletes)
            .file_path(location.clone())
            .file_format(DataFileFormat::Parquet)
            .partition(partition)
            .partition_spec_id(spec_id)
            .record_count(rows)
            .file_size_in_bytes(size);
        if let (1, Some(path)) = (paths.len(), paths.first()) {
            b.lower_bounds(HashMap::from([
                (DELETE_FILE_PATH_ID, Datum::string(path)),
                (DELETE_FILE_POS_ID, Datum::long(min_pos)),
            ]))
            .upper_bounds(HashMap::from([
                (DELETE_FILE_PATH_ID, Datum::string(path)),
                (DELETE_FILE_POS_ID, Datum::long(max_pos)),
            ]));
        }
        out.delete_files.push(
            b.build()
                .map_err(|e| internal(format!("position delete file {location}: {e}")))?,
        );
        Ok(())
    }

    /// Charges what the open file holds; under pressure ships what it
    /// buffers first (as `write.rs`).
    async fn account(&mut self, open: &mut Option<OpenFile>) -> Result<(), KrishivError> {
        let held = open.as_ref().map_or(0, open_bytes);
        let size = self.mem.size();
        if held <= size {
            self.mem.shrink(size - held);
            return Ok(());
        }
        if self.mem.try_grow(held - size).is_ok() {
            return Ok(());
        }
        if let Some(f) = open.as_mut() {
            f.encoder.flush().map_err(|e| e.into_error(&f.at.display))?;
            drain(f).await?;
            f.upload.wait_idle().await?;
        }
        let grow = open
            .as_ref()
            .map_or(0, open_bytes)
            .saturating_sub(self.mem.size());
        if grow > 0 {
            self.mem.grow(grow).await?;
        }
        Ok(())
    }
}

fn open_bytes(f: &OpenFile) -> usize {
    f.encoder
        .buffered_bytes()
        .saturating_add(usize::try_from(f.upload.held_bytes()).unwrap_or(usize::MAX))
}

async fn drain(f: &mut OpenFile) -> Result<(), KrishivError> {
    while let Some(bytes) = f.encoder.take_bytes() {
        f.upload.write(bytes).await?;
    }
    Ok(())
}

/// The name of the `n`th delete file (from 1) with `extension` that task
/// attempt `task` of partition `partition` of the write `write_id` wrote
/// (`OutputFileFactory.generateFilename` with the suffix `deletes`).
pub(crate) fn delete_file_name(
    partition: u32,
    task: u64,
    write_id: &str,
    n: u32,
    extension: &str,
) -> String {
    format!("{partition:05}-{task}-{write_id}-{n:05}-deletes.{extension}")
}

/// The Arrow schema of a position delete file (module docs).
fn position_schema() -> arrow_schema::SchemaRef {
    let field = |name: &str, ty: DataType, id: i32| {
        Field::new(name, ty, false).with_metadata(HashMap::from([(
            krishiv_io::FIELD_ID_KEY.to_owned(),
            id.to_string(),
        )]))
    };
    Arc::new(Schema::new(vec![
        field("file_path", DataType::Utf8, DELETE_FILE_PATH_ID),
        field("pos", DataType::Int64, DELETE_FILE_POS_ID),
    ]))
}

/// The Parquet options of the position delete files of a table with `props`
/// (module docs).
fn parquet_options(props: &HashMap<String, String>) -> Result<FormatOptions, KrishivError> {
    let codec = props
        .get(DELETE_PARQUET_CODEC)
        .or_else(|| props.get(PARQUET_CODEC))
        .map_or(PARQUET_CODEC_DEFAULT, String::as_str);
    // Java's `CompressionCodecName.valueOf(codec.toUpperCase(Locale.ENGLISH))`.
    let name = codec.to_ascii_lowercase();
    match name.as_str() {
        "uncompressed" | "snappy" | "gzip" | "brotli" | "lz4" | "lz4_raw" | "zstd" => {}
        "lzo" => return Err(unsupported("writing the lzo Parquet codec".to_owned())),
        _ => {
            return Err(unsupported(format!(
                "writing Iceberg delete files: Unsupported compression codec: {codec}"
            )));
        }
    }
    let row_group = int_property(props, ROW_GROUP_SIZE, ROW_GROUP_SIZE_DEFAULT)?;
    let page = int_property(props, PAGE_SIZE, PAGE_SIZE_DEFAULT)?;
    let user = [
        ("compression", name),
        ("parquet.block.size", row_group.to_string()),
        ("parquet.page.size", page.to_string()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), OptionValue::Plain(v)))
    .collect();
    let (options, _) = ParquetFormat.options().parse(&user, &Config::new())?;
    Ok(options)
}

fn int_property(
    props: &HashMap<String, String>,
    key: &str,
    default: i32,
) -> Result<i32, KrishivError> {
    match props.get(key) {
        None => Ok(default),
        Some(v) => v
            .parse::<i32>()
            .ok()
            .filter(|n| *n > 0)
            .ok_or_else(|| bad_property(key, v)),
    }
}

/// `write.delete.target-file-size-bytes` (Java's `propertyAsLong`); below 0
/// rolls as 0 does.
fn target_file_bytes(props: &HashMap<String, String>) -> Result<u64, KrishivError> {
    let n = match props.get(DELETE_TARGET_FILE_SIZE) {
        None => DELETE_TARGET_FILE_SIZE_DEFAULT,
        Some(v) => v
            .parse::<i64>()
            .map_err(|_| bad_property(DELETE_TARGET_FILE_SIZE, v))?,
    };
    Ok(u64::try_from(n).unwrap_or(0))
}

fn bad_property(key: &str, value: &str) -> KrishivError {
    unsupported(format!(
        "writing an Iceberg table whose {key} is {value:?}, not a positive number"
    ))
}

fn to_i64(n: u64) -> Result<i64, KrishivError> {
    i64::try_from(n).map_err(|_| internal(format!("{n} past a long")))
}

fn unsupported(feature: String) -> KrishivError {
    KrishivUnsupportedFeature { feature }.into()
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}
