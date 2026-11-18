// SPDX-License-Identifier: Apache-2.0
//! Statistics on write (P3-T024, PR-54b; design note `P3-T024-statistics-on-write.md` §2 rules 1-3): a Theta
//! sketch of each sketched column per write task ([`TaskSketches`], fed by `write.rs`), their union per write
//! ([`union_tasks`], `sink.rs`), and per commit attempt a Puffin statistics file of the snapshot the attempt adds
//! ([`attempt_statistics`], `append.rs`), committed with `set-statistics` in the same request.
//!
//! - **Columns** as Java's `compute_table_stats` default (`NDVSketchUtil`, apache-iceberg 1.12.0): every top-level
//!   primitive column, except `uuid` (review S1: Spark sketches its text, the spec its 16 bytes), the first
//!   [`MAX_SKETCHED_COLUMNS`] in schema order (as `write.metadata.metrics.max-inferred-column-defaults` caps
//!   metrics); the columns past the cap get no statistics.
//! - **Sketches** are the `datasketches` crate's QuickSelect update sketches at its default `k` (4,096) and seed
//!   (9001), compacted ordered through a union (trimmed to `k`). The Puffin spec names an Alpha-family sketch,
//!   which the crate lacks (review B1): the compact form records no family, so Java reads and unions these blobs
//!   as its own, but above 4,096 distinct values the retained hashes and theta differ from what Java would write.
//! - **Whole snapshot or nothing** (rule 2): a column's blob describes every row of the snapshot. It is the
//!   write's own sketch when the parent has no live rows (no parent, or `total-records` 0: a CTAS, a first
//!   insert), else the union with the parent snapshot's blob of that field, read from its Puffin file; a column
//!   whose parent has no such blob, or whose type differs from the parent's schema or the current one (review
//!   N6: an `int` blob unioned with `long` values would count every value twice), is left out, and with no
//!   column left there is no file. So a table whose current snapshot has no statistics (created before this,
//!   last written by another engine) never gets any from appends (review S3): a full-scan compute restarts it.
//! - **Never fails a write** (rule 3): a refused memory reservation, an unsupported column, a parent blob that
//!   cannot be read or a Puffin write that fails drops the statistics with a warning and counts one in
//!   [`statistics_skipped_total`] (`krishiv_iceberg_write_statistics_skipped_total`); the data commits.
//! - **File** `<metadata dir>/<snapshot id>-<commit uuid>-<attempt>.stats` (Java: `<snapshot id>-<uuid>.stats`;
//!   one per attempt, as the parent and sequence number are the attempt's), blobs uncompressed (Java writes
//!   zstd; readers take either, and the footer size is then known without reading the file back), blob
//!   property `ndv` the estimate truncated to an integer (Java's `(long) sketch.getEstimate()`).
//! - **Memory**: each task's sketches are charged to a reservation of the write's consumer before the first
//!   row ([`SKETCH_BYTES`] per column, the most a `k` = 4,096 update sketch holds), the write's unions to one
//!   reservation of the finalize's consumer; a refusal drops the statistics, never the write.
//!
//! Deletes do not lower a unioned count (an upper bound; Java computes the snapshot's own count from a scan,
//! review S5).

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::RecordBatch;
use datasketches::theta::{CompactThetaSketch, ThetaSketch, ThetaUnion, ThetaUnionBuilder};
use iceberg::io::FileIO;
use iceberg::puffin::{
    APACHE_DATASKETCHES_THETA_V1, Blob, CREATED_BY_PROPERTY, CompressionCodec, PuffinWriter,
};
use iceberg::spec::{BlobMetadata, Schema, Snapshot, StatisticsFile, TableMetadata, Type};
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_memory::Reservation;

use super::puffin_stats::read_sketch;
use super::scan::iceberg_type;
use super::storage::from_iceberg;
use crate::plan::theta_value::{new_sketch, sketched, update_column};
use crate::plan::types::IcebergType;

/// The table property that turns statistics on write off for one table (`false`; Java's `parseBoolean`: any
/// value other than `true`, case aside, is off).
pub(crate) const STATISTICS_PROPERTY: &str = "krishiv.write.statistics.enabled";

/// Columns sketched at most (Java's `write.metadata.metrics.max-inferred-column-defaults` default).
pub(crate) const MAX_SKETCHED_COLUMNS: usize = 100;

/// Bytes charged per sketched column and task: a `k` = 4,096 update sketch's table at its largest (2^13 slots
/// of 8 bytes).
pub(crate) const SKETCH_BYTES: usize = 64 * 1024;

/// The blob property of the estimate (Java's `NDVSketchUtil`).
const NDV_PROPERTY: &str = "ndv";

/// The summary key of a snapshot's live rows.
const TOTAL_RECORDS: &str = "total-records";

/// Writes that dropped their statistics on a failure (module docs).
static SKIPPED: AtomicU64 = AtomicU64::new(0);

/// `krishiv_iceberg_write_statistics_skipped_total`: writes of this process that dropped their statistics on a
/// failure (memory, sketching, the parent's blob, the Puffin write, the catalog). Not counted: the columns rule 2
/// leaves out.
#[must_use]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "read by the engine metrics registry once its catalogue names it (a follow-up)"
    )
)]
pub(crate) fn statistics_skipped_total() -> u64 {
    SKIPPED.load(Ordering::Relaxed)
}

/// Counts and logs one write's statistics dropped because of `why`.
pub(crate) fn skipped(why: &str) {
    SKIPPED.fetch_add(1, Ordering::Relaxed);
    tracing::warn!(
        target: "krishiv::iceberg::statistics",
        "an Iceberg write commits without statistics: {why}"
    );
}

/// Whether a write writes statistics: `config` (`krishiv.iceberg.write.statistics.enabled`) and the table's
/// [`STATISTICS_PROPERTY`].
#[must_use]
pub(crate) fn enabled(config: bool, properties: &HashMap<String, String>) -> bool {
    config
        && properties
            .get(STATISTICS_PROPERTY)
            .is_none_or(|v| v.eq_ignore_ascii_case("true"))
}

/// One sketched column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SketchColumn {
    /// Its position among the table's top-level columns (and the write's batches).
    pub(crate) index: usize,
    /// Its field id.
    pub(crate) field_id: i32,
    /// Its type.
    pub(crate) ty: IcebergType,
}

/// The columns of `schema` a write sketches (module docs).
#[must_use]
pub(crate) fn sketch_columns(schema: &Schema) -> Vec<SketchColumn> {
    schema
        .as_struct()
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(index, f)| match f.field_type.as_ref() {
            Type::Primitive(p) => Some(SketchColumn {
                index,
                field_id: f.id,
                ty: iceberg_type(p),
            }),
            _ => None,
        })
        .filter(|c| sketched(&c.ty))
        .take(MAX_SKETCHED_COLUMNS)
        .collect()
}

/// One write task's sketches, their memory held until they are dropped.
pub(crate) struct TaskSketches {
    columns: Arc<[SketchColumn]>,
    sketches: Vec<ThetaSketch>,
    _mem: Reservation,
}

impl fmt::Debug for TaskSketches {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskSketches")
            .field("columns", &self.columns.len())
            .finish_non_exhaustive()
    }
}

impl TaskSketches {
    /// Empty sketches of `columns`, charged to `mem`; `None` (counted as skipped) when the reservation is refused.
    #[must_use]
    pub(crate) fn new(columns: Arc<[SketchColumn]>, mut mem: Reservation) -> Option<Self> {
        if mem
            .try_grow(columns.len().saturating_mul(SKETCH_BYTES))
            .is_err()
        {
            skipped("the memory for a task's sketches was refused");
            return None;
        }
        let sketches = columns
            .iter()
            .map(|_| new_sketch())
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            columns,
            sketches,
            _mem: mem,
        })
    }

    /// Adds the rows of `batch` (of the table's top-level columns); `false` when a column is not of a type its
    /// Iceberg type is carried in (the caller drops the sketches).
    pub(crate) fn update(&mut self, batch: &RecordBatch) -> bool {
        self.columns.iter().zip(&mut self.sketches).all(|(c, s)| {
            batch
                .columns()
                .get(c.index)
                .is_some_and(|a| update_column(s, a.as_ref(), &c.ty))
        })
    }
}

/// The sketches of a whole write, one compact sketch per column, their memory held until the commit ends.
pub(crate) struct WriteSketches {
    /// Field id, type, sketch.
    pub(crate) columns: Vec<(i32, IcebergType, CompactThetaSketch)>,
    _mem: Option<Reservation>,
}

impl fmt::Debug for WriteSketches {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WriteSketches")
            .field("columns", &self.columns.len())
            .finish_non_exhaustive()
    }
}

impl WriteSketches {
    /// The sketches as they are, uncharged (tests and foreign writers in the simulations).
    #[cfg(test)]
    pub(crate) fn of(columns: Vec<(i32, IcebergType, CompactThetaSketch)>) -> Self {
        Self {
            columns,
            _mem: None,
        }
    }
}

fn union() -> Result<ThetaUnion, KrishivError> {
    ThetaUnionBuilder::default().build().map_err(sketch_error)
}

fn sketch_error(e: impl fmt::Display) -> KrishivError {
    InternalError {
        message: format!("a Theta sketch union: {e}"),
    }
    .into()
}

/// The union of every task's sketches of `columns`, charged to `mem`; `None` (counted as skipped) when a task has
/// none (its memory was refused or a column could not be sketched), the reservation is refused or a union fails.
/// No task (a write of no rows) gives empty sketches.
#[must_use]
pub(crate) fn union_tasks(
    columns: &[SketchColumn],
    tasks: Vec<Option<TaskSketches>>,
    mut mem: Reservation,
) -> Option<WriteSketches> {
    let Some(tasks) = tasks.into_iter().collect::<Option<Vec<_>>>() else {
        skipped("a write task has no sketches");
        return None;
    };
    if mem
        .try_grow(columns.len().saturating_mul(SKETCH_BYTES))
        .is_err()
    {
        skipped("the memory for the write's sketch unions was refused");
        return None;
    }
    let mut out = Vec::with_capacity(columns.len());
    for (i, c) in columns.iter().enumerate() {
        let merged = union().and_then(|mut u| {
            for t in &tasks {
                if let Some(s) = t.sketches.get(i) {
                    u.update(s).map_err(sketch_error)?;
                }
            }
            Ok(u.to_sketch(true))
        });
        match merged {
            Ok(s) => out.push((c.field_id, c.ty.clone(), s)),
            Err(e) => {
                skipped(&e.to_string());
                return None;
            }
        }
    }
    Some(WriteSketches {
        columns: out,
        _mem: Some(mem),
    })
}

/// What one attempt's statistics file describes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttemptSnapshot<'a> {
    /// The table the attempt is based on.
    pub(crate) metadata: &'a TableMetadata,
    /// The attempt's parent snapshot.
    pub(crate) parent: Option<&'a Snapshot>,
    /// The snapshot the attempt adds.
    pub(crate) snapshot_id: i64,
    /// Its sequence number.
    pub(crate) sequence_number: i64,
}

/// The statistics file of an attempt adding `at.snapshot_id`, written at `path` (rule 2, module docs); `None`, with
/// nothing written, when no column describes the whole snapshot.
///
/// # Errors
/// Reading the parent's blobs, a union, or writing the Puffin file; the caller drops the statistics (rule 3) and
/// deletes `path` with the attempt's files.
pub(crate) async fn attempt_statistics(
    io: &FileIO,
    at: AttemptSnapshot<'_>,
    sketches: &WriteSketches,
    path: &str,
) -> Result<Option<StatisticsFile>, KrishivError> {
    let blobs = whole_snapshot(io, at, sketches).await?;
    if blobs.is_empty() {
        return Ok(None);
    }
    let output = io.new_output(path).map_err(from_iceberg)?;
    let created_by = HashMap::from([(
        CREATED_BY_PROPERTY.to_owned(),
        format!("krishiv {}", env!("CARGO_PKG_VERSION")),
    )]);
    let mut writer = PuffinWriter::new(&output, created_by, false)
        .await
        .map_err(from_iceberg)?;
    let mut metadata = Vec::with_capacity(blobs.len());
    let mut blob_bytes: u64 = 0;
    for (field, sketch) in blobs {
        let data = sketch.serialize();
        blob_bytes += data.len() as u64;
        let properties = HashMap::from([(NDV_PROPERTY.to_owned(), ndv(&sketch).to_string())]);
        metadata.push(BlobMetadata {
            r#type: APACHE_DATASKETCHES_THETA_V1.to_owned(),
            snapshot_id: at.snapshot_id,
            sequence_number: at.sequence_number,
            fields: vec![field],
            properties: properties.clone(),
        });
        let blob = Blob::builder()
            .r#type(APACHE_DATASKETCHES_THETA_V1.to_owned())
            .fields(vec![field])
            .snapshot_id(at.snapshot_id)
            .sequence_number(at.sequence_number)
            .data(data)
            .properties(properties)
            .build();
        writer
            .add(blob, CompressionCodec::None)
            .await
            .map_err(from_iceberg)?;
    }
    let size = writer.close().await.map_err(from_iceberg)?.size;
    // The file is the magic, the uncompressed blobs, then the footer.
    let footer = size.saturating_sub(4 + blob_bytes);
    let int = |v: u64| i64::try_from(v).unwrap_or(i64::MAX);
    Ok(Some(StatisticsFile {
        snapshot_id: at.snapshot_id,
        statistics_path: path.to_owned(),
        file_size_in_bytes: int(size),
        file_footer_size_in_bytes: int(footer),
        key_metadata: None,
        blob_metadata: metadata,
    }))
}

/// The estimate truncated toward zero, as Java's `(long) sketch.getEstimate()` (never negative).
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a saturating float-to-integer cast is Java's truncation of a non-negative estimate"
)]
fn ndv(sketch: &CompactThetaSketch) -> u64 {
    sketch.estimate() as u64
}

/// The sketches that describe every row of the attempt's snapshot (module docs, rule 2).
async fn whole_snapshot(
    io: &FileIO,
    at: AttemptSnapshot<'_>,
    sketches: &WriteSketches,
) -> Result<Vec<(i32, CompactThetaSketch)>, KrishivError> {
    let current = at.metadata.current_schema();
    let same_type = |schema: &Schema, field: i32, ty: &IcebergType| {
        schema
            .field_by_id(field)
            .and_then(|f| f.field_type.as_primitive_type())
            .is_some_and(|p| iceberg_type(p) == *ty)
    };
    let mut out = Vec::with_capacity(sketches.columns.len());
    let Some(parent) = at.parent.filter(|p| has_rows(p)) else {
        for (field, ty, own) in &sketches.columns {
            if same_type(current, *field, ty) {
                out.push((*field, own.clone()));
            }
        }
        return Ok(out);
    };
    let Some(file) = at
        .metadata
        .statistics_for_snapshot(parent.snapshot_id())
        .filter(|f| f.snapshot_id == parent.snapshot_id())
    else {
        return Ok(out);
    };
    let Some(parent_schema) = parent
        .schema_id()
        .and_then(|id| at.metadata.schema_by_id(id))
    else {
        return Ok(out);
    };
    for (field, ty, own) in &sketches.columns {
        if !same_type(current, *field, ty) || !same_type(parent_schema, *field, ty) {
            continue;
        }
        let Some(theirs) = read_sketch(io, file, *field).await? else {
            continue;
        };
        let mut u = union()?;
        u.update(own).map_err(sketch_error)?;
        u.update(&theirs).map_err(sketch_error)?;
        out.push((*field, u.to_sketch(true)));
    }
    Ok(out)
}

/// Whether `snapshot` holds live rows (`total-records` other than 0; a snapshot without the total counts as
/// holding rows).
fn has_rows(snapshot: &Snapshot) -> bool {
    snapshot
        .summary()
        .additional_properties
        .get(TOTAL_RECORDS)
        .and_then(|v| v.parse::<u64>().ok())
        .is_none_or(|n| n > 0)
}
