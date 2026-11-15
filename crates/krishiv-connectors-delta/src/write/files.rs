// SPDX-License-Identifier: Apache-2.0
//! The data files of one write (design note P3-T021 section 4.1, "Data files" and "Adds"):
//! rows grouped by partition, each group bound to the transaction's write state
//! ([`BoundWriteContext`]), turned into the physical schema by the kernel's
//! `logical_to_physical` expression (physical names and Parquet field ids under column mapping),
//! written as one Parquet file under the context's `write_dir`, and described by one add row
//! with Krishiv's own statistics ([`super::stats`]).
//!
//! Runs on the log pool (the Parquet handler's put and the evaluator are kernel calls).

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, StructArray};
use arrow_schema::{Field, Schema, SchemaRef};
use delta_kernel::expressions::{MapData, Scalar};
use delta_kernel::schema::{DataType, MapType, StructType};
use delta_kernel::transaction::{BoundWriteContext, WriteState};
use delta_kernel::{EngineData, EvaluationHandler as _, ParquetHandler as _};
use krishiv_common::error::{InternalError, KrishivError};

use super::stats::file_stats;
use crate::engine::eval::create_many;
use crate::engine::schema::{scalar_at, to_arrow_schema};
use crate::engine::{DeltaEngine, KrishivEngineData};
use crate::error::from_kernel;

/// What one write produced: its add rows (in the kernel's `add_files_schema`, `stats` holding
/// Krishiv's statistics) and its metrics.
#[derive(Debug, Default)]
pub(crate) struct Written {
    /// One row per file.
    pub(crate) adds: Vec<RecordBatch>,
    /// Files written.
    pub(crate) files: u64,
    /// Rows written.
    pub(crate) rows: u64,
    /// Bytes written.
    pub(crate) bytes: u64,
    /// The physical partition values of each partition written (the same map once per
    /// partition).
    pub(crate) partitions: Vec<HashMap<String, Option<String>>>,
    /// The logical partition values of each partition written, in the partition columns'
    /// order.
    pub(crate) partition_values: Vec<Vec<Scalar>>,
}

/// What the files of one write are written against.
pub(crate) struct Layout<'a> {
    /// The table's logical schema.
    pub(crate) schema: &'a StructType,
    /// The logical partition columns, in the table's order.
    pub(crate) partition_columns: &'a [String],
    /// The kernel's write state of the transaction.
    pub(crate) state: Arc<WriteState>,
    /// The kernel's stats schema of the transaction.
    pub(crate) stats_schema: &'a StructType,
    /// Whether the table supports deletion vectors (`tightBounds`).
    pub(crate) tight_bounds: bool,
    /// The table, for messages.
    pub(crate) table: &'a str,
}

/// Writes `batches` (rows of the table's logical columns, in its order) as data files.
///
/// # Errors
/// The kernel's partition-value and write-state errors; the evaluator's; storage errors.
pub(crate) fn write_files(
    e: &DeltaEngine,
    layout: &Layout<'_>,
    batches: &[RecordBatch],
) -> Result<Written, KrishivError> {
    let table = layout.table;
    let kernel = |err: &delta_kernel::KernelError| from_kernel(err, table);
    let logical: SchemaRef = Arc::new(to_arrow_schema(layout.schema).map_err(|e| kernel(&e))?);
    let mut groups: Vec<(Vec<Scalar>, Vec<RecordBatch>)> = Vec::new();
    let mut keys: HashMap<String, usize> = HashMap::new();
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let batch = conform(batch, &logical)?;
        if layout.partition_columns.is_empty() {
            match groups.first_mut() {
                Some((_, b)) => b.push(batch),
                None => groups.push((Vec::new(), vec![batch])),
            }
            continue;
        }
        let columns = layout
            .partition_columns
            .iter()
            .map(|p| {
                batch
                    .column_by_name(p)
                    .cloned()
                    .ok_or_else(|| internal(&format!("the partition column {p} is not written")))
            })
            .collect::<Result<Vec<ArrayRef>, _>>()?;
        let mut rows_of: Vec<(usize, Vec<u32>)> = Vec::new();
        for row in 0..batch.num_rows() {
            let values = columns
                .iter()
                .map(|c| scalar_at(c.as_ref(), row))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| kernel(&e))?;
            let key = format!("{values:?}");
            let group = match keys.get(&key) {
                Some(g) => *g,
                None => {
                    groups.push((values, Vec::new()));
                    keys.insert(key, groups.len() - 1);
                    groups.len() - 1
                }
            };
            let index = u32::try_from(row).map_err(|_| internal("a batch row past u32"))?;
            match rows_of.iter_mut().find(|(g, _)| *g == group) {
                Some((_, rows)) => rows.push(index),
                None => rows_of.push((group, vec![index])),
            }
        }
        for (group, rows) in rows_of {
            let indices = arrow_array::UInt32Array::from(rows);
            let taken = arrow_select::take::take_record_batch(&batch, &indices)
                .map_err(|e| internal(&format!("grouping rows by partition: {e}")))?;
            groups[group].1.push(taken);
        }
    }
    let mut written = Written::default();
    for (values, batches) in groups {
        let mut builder = layout.state.write_context_builder();
        if !layout.partition_columns.is_empty() {
            written.partition_values.push(values.clone());
            let map: HashMap<String, Scalar> = layout
                .partition_columns
                .iter()
                .cloned()
                .zip(values)
                .collect();
            builder = builder.with_partition_values(map);
        }
        let ctx = builder.build().map_err(|e| kernel(&e))?;
        let data = arrow_select::concat::concat_batches(&logical, &batches)
            .map_err(|e| internal(&format!("concatenating a partition's rows: {e}")))?;
        let add = write_one(e, layout, &ctx, &data)?;
        written.files += 1;
        written.rows += data.num_rows() as u64;
        written.bytes += add.1;
        written
            .partitions
            .push(ctx.physical_partition_values().clone());
        written.adds.push(add.0);
    }
    Ok(written)
}

/// Writes one partition's rows as one file and returns its add row and size.
fn write_one(
    e: &DeltaEngine,
    layout: &Layout<'_>,
    ctx: &BoundWriteContext,
    data: &RecordBatch,
) -> Result<(RecordBatch, u64), KrishivError> {
    let table = layout.table;
    let kernel = |err: &delta_kernel::KernelError| from_kernel(err, table);
    let logical_data = ctx.logical_data_schema();
    let input_schema: SchemaRef = Arc::new(to_arrow_schema(logical_data).map_err(|e| kernel(&e))?);
    let columns = logical_data
        .fields()
        .map(|f| {
            data.column_by_name(f.name())
                .cloned()
                .ok_or_else(|| internal(&format!("the column {} is not written", f.name())))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let input = RecordBatch::try_new(Arc::clone(&input_schema), columns)
        .map_err(|e| internal(&format!("a write's logical rows: {e}")))?;
    let evaluator = e
        .evaluation()
        .new_expression_evaluator(
            Arc::clone(logical_data),
            ctx.logical_to_physical(),
            DataType::Struct(Box::new(ctx.physical_data_schema().as_ref().clone())),
        )
        .map_err(|e| kernel(&e))?;
    let physical = evaluator
        .evaluate(&KrishivEngineData::new(input))
        .map_err(|e| kernel(&e))?;
    let physical = KrishivEngineData::from_engine_data(physical)
        .map_err(|e| kernel(&e))?
        .into_batch();
    let name = format!(
        "part-00000-{}.c000{}",
        file_uuid(),
        crate::engine::columnar::file_suffix()?
    );
    let url = ctx
        .write_dir()
        .join(&name)
        .map_err(|err| internal(&format!("the data file {name}: {err}")))?;
    let stats = file_stats(&physical, layout.stats_schema, layout.tight_bounds)?;
    let one: Box<dyn EngineData> = Box::new(KrishivEngineData::new(physical));
    let size = e
        .parquet()
        .write_parquet_file(url.clone(), Box::new(std::iter::once(Ok(one))))
        .map_err(|e| kernel(&e))?;
    let path = ctx.resolve_file_path(&url).map_err(|e| kernel(&e))?;
    let row = add_row(ctx, &path, size, stats, table)?;
    Ok((row, size))
}

/// The add row of one file: path, partition values, size, modification time, and the
/// statistics struct in place of the schema's empty one.
fn add_row(
    ctx: &BoundWriteContext,
    path: &str,
    size: u64,
    stats: StructArray,
    table: &str,
) -> Result<RecordBatch, KrishivError> {
    let kernel = |err: &delta_kernel::KernelError| from_kernel(err, table);
    let mandatory = StructType::new_unchecked([
        delta_kernel::schema::StructField::not_null("path", DataType::STRING),
        delta_kernel::schema::StructField::not_null(
            "partitionValues",
            MapType::new(DataType::STRING, DataType::STRING, true),
        ),
        delta_kernel::schema::StructField::not_null("size", DataType::LONG),
        delta_kernel::schema::StructField::not_null("modificationTime", DataType::LONG),
    ]);
    let mut values: Vec<(&String, &Option<String>)> =
        ctx.physical_partition_values().iter().collect();
    values.sort();
    let map = MapData::try_new(
        MapType::new(DataType::STRING, DataType::STRING, true),
        values.into_iter().map(|(k, v)| {
            (
                Scalar::String(k.clone()),
                v.as_ref().map_or(Scalar::Null(DataType::STRING), |v| {
                    Scalar::String(v.clone())
                }),
            )
        }),
    )
    .map_err(|e| kernel(&e))?;
    let size = i64::try_from(size).map_err(|_| internal("a file size past a long"))?;
    let base = create_many(
        &mandatory,
        &[vec![
            Scalar::String(path.to_owned()),
            Scalar::Map(map),
            Scalar::Long(size),
            Scalar::Long(now_ms()),
        ]],
    )
    .map_err(|e| kernel(&e))?;
    let mut fields: Vec<Field> = base
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    let mut columns = base.columns().to_vec();
    fields.push(Field::new("stats", stats.data_type().clone(), true));
    columns.push(Arc::new(stats));
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| internal(&format!("an add row: {e}")))
}

/// `batch` with each column in the type of the same column of `target` (the kernel's Arrow
/// form: the engine's string views, small decimals and zone labels to the kernel's).
pub(crate) fn conform(
    batch: &RecordBatch,
    target: &SchemaRef,
) -> Result<RecordBatch, KrishivError> {
    if batch.num_columns() != target.fields().len() {
        return Err(internal(&format!(
            "a write of {} columns into a table of {}",
            batch.num_columns(),
            target.fields().len()
        )));
    }
    let columns = batch
        .columns()
        .iter()
        .zip(target.fields())
        .map(|(c, f)| {
            if c.data_type() == f.data_type() {
                Ok(Arc::clone(c))
            } else {
                crate::engine::relabel::relabel(c, f.data_type())
                    .or_else(|_| arrow_cast::cast(c, f.data_type()))
                    .map_err(|e| {
                        internal(&format!(
                            "the column {} as {}: {e}",
                            f.name(),
                            f.data_type()
                        ))
                    })
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(Arc::clone(target), columns)
        .map_err(|e| internal(&format!("a write's rows in the table's types: {e}")))
}

/// A fresh random UUID (version 4 layout) for a file name.
pub(crate) fn file_uuid() -> String {
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let half = |salt: u64| {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(salt);
        h.write_u64(n);
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
        );
        h.finish()
    };
    let (a, b) = (half(1), half(2));
    let bits = (u128::from(a) << 64 | u128::from(b)) & !(0xF000_u128 << 64) | (0x4000_u128 << 64);
    let bits = bits & !(0xC000_u128 << 48) | (0x8000_u128 << 48);
    let hex = format!("{bits:032x}");
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// Milliseconds since the epoch.
pub(crate) fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_millis()).ok())
        .unwrap_or(0)
}

fn internal(message: &str) -> KrishivError {
    InternalError {
        message: message.to_owned(),
    }
    .into()
}
