// SPDX-License-Identifier: Apache-2.0
//! Krishiv's own per-file statistics (design note P3-T021 section 4.1, "Stats are ours"),
//! computed from the rows a file is written from, never read back from its footer (footer
//! bounds are cut at 4 KiB and lose exactness).
//!
//! The shape is the kernel's [`stats_schema`](delta_kernel::transaction::Transaction::stats_schema)
//! for the table (`delta.dataSkippingNumIndexedCols`, `delta.dataSkippingStatsColumns`, physical
//! names, no partition columns): `numRecords`, `nullCount` for every included leaf (arrays and
//! maps too), `minValues`/`maxValues` for the leaves delta-spark keeps bounds for, and
//! `tightBounds`. The values follow delta-spark 4.4.1's `StatisticsCollection.statsCollector`
//! (`spark/src/main/scala/org/apache/spark/sql/delta/stats/StatisticsCollection.scala:280-330`):
//! - `nullCount` counts a value as NULL when it or any struct above it is NULL (Spark's
//!   `c.isNull` on the nested field);
//! - min and max skip NULLs; a column with no non-NULL value has none (NULL here, omitted by the
//!   JSON);
//! - strings: the min is cut to its first 32 code points (`substring(min(c), 0, 32)`, `:296-298`),
//!   the max is cut and given a tie-breaker (`truncateMaxStringAgg`, `:889-923`), counted in
//!   UTF-16 units as on the JVM (`spark.databricks.io.skipping.stringPrefixLength`, default 32);
//! - `tightBounds = true` only on a table that supports deletion vectors (`:286-290`), NULL
//!   (omitted) otherwise.
//!
//! Floating-point bounds use Spark's ordering, where every NaN (whatever its sign bit: x86's
//! default NaN has it set) is one value above every number: Arrow's total order alone puts a
//! negative NaN below every number. Arrow orders `-0.0` below `0.0`, where Spark has them equal
//! (a recorded difference when a file holds both).

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, Int64Array, RecordBatch, StructArray};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType as ArrowType, Fields, SortOptions};
use delta_kernel::schema::{DataType, StructType};
use krishiv_common::error::{InternalError, KrishivError};

use crate::engine::schema::to_arrow_type;

/// delta-spark's `spark.databricks.io.skipping.stringPrefixLength` default.
pub const STRING_PREFIX_LENGTH: usize = 32;

/// `UTF8_MAX_CHARACTER` (U+10FFFF) as UTF-16 units.
const UTF8_MAX: [u16; 2] = [0xDBFF, 0xDFFF];
/// `ASCII_MAX_CHARACTER`.
const ASCII_MAX: u16 = 0x7F;

/// The string min bound delta-spark writes: `substring(min(c), 0, prefixLen)`
/// (`StatisticsCollection.scala:296-298`), Spark's `substring`, which counts code points
/// (`UTF8String.substringSQL`): the input when it has at most `prefix` characters, else its first
/// `prefix` characters (`truncateMinStringAgg`, the UTF-16 form, is not on this path).
#[must_use]
pub fn truncate_min(input: &str, prefix: usize) -> Option<String> {
    match input.char_indices().nth(prefix) {
        None => Some(input.to_owned()),
        Some((at, _)) => Some(input[..at].to_owned()),
    }
}

fn min_cut(units: &[u16], prefix: usize) -> usize {
    let high = |u: u16| (0xD800..0xDC00).contains(&u);
    let low = |u: u16| (0xDC00..0xE000).contains(&u);
    if high(units[prefix - 1]) && units.get(prefix).copied().is_some_and(low) {
        prefix + 1
    } else {
        prefix
    }
}

/// delta-spark's `truncateMaxStringAgg(prefixLen)`: the input when it has at most `prefix`
/// UTF-16 units; else the shortest prefix of at least `prefix` units (up to `2 * prefix`) that a
/// tie-breaker can follow: `U+007F` after an ASCII unit below it, `U+10FFFF` otherwise, skipping
/// `U+10FFFF` characters of the input; `None` when there is none within `2 * prefix`.
#[must_use]
pub fn truncate_max(input: &str, prefix: usize) -> Option<String> {
    let units: Vec<u16> = input.encode_utf16().collect();
    if units.len() <= prefix {
        return Some(input.to_owned());
    }
    if prefix == 0 {
        return None;
    }
    let max_len = 2 * prefix;
    let mut len = min_cut(&units, prefix);
    while len <= max_len {
        if len >= units.len() {
            return Some(input.to_owned());
        }
        if len + 1 < units.len() && units[len..len + 2] == UTF8_MAX {
            len += 2;
            continue;
        }
        let mut out = units[..len].to_vec();
        if units[len] < ASCII_MAX {
            out.push(ASCII_MAX);
        } else {
            out.extend_from_slice(&UTF8_MAX);
        }
        return String::from_utf16(&out).ok();
    }
    None
}

/// The statistics of the file written from `batch` (its physical columns), as one row of
/// `schema` (the kernel's stats schema for the table), in the Arrow form of that schema.
///
/// # Errors
/// `INTERNAL_ERROR` when a stats column is not in `batch`, or its type is not the one the
/// schema names.
pub fn file_stats(
    batch: &RecordBatch,
    schema: &StructType,
    tight_bounds: bool,
) -> Result<StructArray, KrishivError> {
    let mut acc = StatsAccumulator::default();
    acc.update(batch, schema)?;
    acc.finish(schema, tight_bounds)
}

/// Raw rows merged at once when a file's batches reach this many.
const MERGE_EVERY: usize = 64;

/// The statistics of one file accumulated over the batches it is written from: per batch a raw
/// row (row count, NULL counts, uncut bounds), merged every [`MERGE_EVERY`] batches and when the
/// file ends. A cut string bound of the merged rows equals the cut bound of the whole file (a
/// prefix keeps the order and the max's tie-breaker is applied once, to the uncut max).
#[derive(Debug, Default)]
pub(crate) struct StatsAccumulator {
    raw: Vec<StructArray>,
}

impl StatsAccumulator {
    /// Adds the rows of `batch` (physical columns).
    pub(crate) fn update(
        &mut self,
        batch: &RecordBatch,
        schema: &StructType,
    ) -> Result<(), KrishivError> {
        let root = Column {
            fields: batch.schema().fields().clone(),
            columns: batch.columns().to_vec(),
            nulls: None,
        };
        let mut fields = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();
        for f in schema.fields() {
            let array: ArrayRef = match f.name().as_str() {
                "numRecords" => Arc::new(Int64Array::from(vec![rows(batch.num_rows())?])),
                "nullCount" => Arc::new(nested(
                    &root,
                    struct_of(f.data_type())?,
                    Stat::NullCount,
                    false,
                )?),
                "minValues" => {
                    Arc::new(nested(&root, struct_of(f.data_type())?, Stat::Min, false)?)
                }
                "maxValues" => {
                    Arc::new(nested(&root, struct_of(f.data_type())?, Stat::Max, false)?)
                }
                "tightBounds" => continue,
                other => {
                    return Err(internal(&format!(
                        "an unknown field {other} in the kernel's stats schema"
                    )));
                }
            };
            fields.push(arrow_schema::Field::new(
                f.name(),
                array.data_type().clone(),
                true,
            ));
            arrays.push(array);
        }
        let row = StructArray::try_new(Fields::from(fields), arrays, None)
            .map_err(|e| internal(&format!("the stats row: {e}")))?;
        self.raw.push(row);
        if self.raw.len() >= MERGE_EVERY {
            let merged = self.merge(schema, None)?;
            self.raw = vec![merged];
        }
        Ok(())
    }

    /// The file's statistics: one row of `schema`, the bounds cut as delta-spark cuts them.
    pub(crate) fn finish(
        mut self,
        schema: &StructType,
        tight_bounds: bool,
    ) -> Result<StructArray, KrishivError> {
        if self.raw.is_empty() {
            return Err(internal("the stats of a file with no rows"));
        }
        self.merge(schema, Some(tight_bounds))
    }

    /// The raw rows as one: raw when `finish` is `None`, else the final row (with
    /// `tightBounds`).
    fn merge(
        &mut self,
        schema: &StructType,
        finish: Option<bool>,
    ) -> Result<StructArray, KrishivError> {
        let refs: Vec<&dyn Array> = self.raw.iter().map(|r| r as &dyn Array).collect();
        let all = arrow_select::concat::concat(&refs)
            .map_err(|e| internal(&format!("merging stats rows: {e}")))?;
        let all = all
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| internal("merged stats rows that are not a struct"))?;
        let root = Column {
            fields: all.fields().clone(),
            columns: all.columns().to_vec(),
            nulls: None,
        };
        let done = finish.is_some();
        let mut fields = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();
        for f in schema.fields() {
            let array: ArrayRef = match f.name().as_str() {
                "numRecords" => {
                    let n = root.column("numRecords")?;
                    Arc::new(Int64Array::from(vec![sum(n)?]))
                }
                "nullCount" => Arc::new(nested(
                    &root.level("nullCount")?,
                    struct_of(f.data_type())?,
                    Stat::Sum,
                    done,
                )?),
                "minValues" => Arc::new(nested(
                    &root.level("minValues")?,
                    struct_of(f.data_type())?,
                    Stat::Min,
                    done,
                )?),
                "maxValues" => Arc::new(nested(
                    &root.level("maxValues")?,
                    struct_of(f.data_type())?,
                    Stat::Max,
                    done,
                )?),
                "tightBounds" => match finish {
                    Some(tight) => Arc::new(BooleanArray::from(vec![tight.then_some(true)])),
                    None => continue,
                },
                other => {
                    return Err(internal(&format!(
                        "an unknown field {other} in the kernel's stats schema"
                    )));
                }
            };
            fields.push(arrow_schema::Field::new(
                f.name(),
                array.data_type().clone(),
                true,
            ));
            arrays.push(array);
        }
        StructArray::try_new(Fields::from(fields), arrays, None)
            .map_err(|e| internal(&format!("the stats row: {e}")))
    }
}

/// The sum of an Int64 column's values.
fn sum(column: &ArrayRef) -> Result<i64, KrishivError> {
    let c = column
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| internal("a stats count that is not a long"))?;
    c.iter()
        .flatten()
        .try_fold(0_i64, i64::checked_add)
        .ok_or_else(|| internal("a stats count that does not fit a long"))
}

/// Which statistic a struct of the stats schema holds.
#[derive(Clone, Copy)]
enum Stat {
    /// The NULLs of the data.
    NullCount,
    /// The sum of raw NULL counts.
    Sum,
    Min,
    Max,
}

/// A struct level of the data: its fields, columns and the validity of the rows that reach it
/// (every struct above it non-NULL).
struct Column {
    fields: Fields,
    columns: Vec<ArrayRef>,
    nulls: Option<NullBuffer>,
}

impl Column {
    fn column(&self, name: &str) -> Result<&ArrayRef, KrishivError> {
        self.fields
            .iter()
            .position(|d| d.name() == name)
            .map(|i| &self.columns[i])
            .ok_or_else(|| internal(&format!("the stats column {name} is missing")))
    }

    /// The struct column `name` as a level.
    fn level(&self, name: &str) -> Result<Self, KrishivError> {
        let s = self
            .column(name)?
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| internal(&format!("the stats column {name} is not a struct")))?;
        Ok(Self {
            fields: s.fields().clone(),
            columns: s.columns().to_vec(),
            nulls: None,
        })
    }
}

fn struct_of(t: &DataType) -> Result<&StructType, KrishivError> {
    match t {
        DataType::Struct(s) => Ok(s),
        other => Err(internal(&format!(
            "a stats field of type {other} where a struct was expected"
        ))),
    }
}

/// One row of `schema`, the statistic `stat` of each of its leaves over `data`; bounds are cut
/// and written as delta-spark writes them when `finish`, kept as they are otherwise.
fn nested(
    data: &Column,
    schema: &StructType,
    stat: Stat,
    finish: bool,
) -> Result<StructArray, KrishivError> {
    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for f in schema.fields() {
        let column = data.column(f.name()).map_err(|_| {
            internal(&format!(
                "the stats column {} is not among the written columns",
                f.name()
            ))
        })?;
        let nulls = NullBuffer::union(data.nulls.as_ref(), column.logical_nulls().as_ref());
        let out_type = to_arrow_type(f.data_type()).map_err(|e| internal(&e.to_string()))?;
        let counts = matches!(stat, Stat::NullCount | Stat::Sum);
        let array: ArrayRef = match (f.data_type(), column.data_type()) {
            (DataType::Struct(child), ArrowType::Struct(child_fields))
                if !counts || is_struct_count(child) =>
            {
                let s = column
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .ok_or_else(|| internal("a struct column that is not a struct array"))?;
                let level = Column {
                    fields: child_fields.clone(),
                    columns: s.columns().to_vec(),
                    nulls,
                };
                Arc::new(nested(&level, child, stat, finish)?)
            }
            _ => match stat {
                Stat::NullCount => {
                    let n = nulls.as_ref().map_or(0, NullBuffer::null_count);
                    Arc::new(Int64Array::from(vec![rows(n)?]))
                }
                Stat::Sum => Arc::new(Int64Array::from(vec![sum(column)?])),
                Stat::Min => extreme(column, nulls.as_ref(), false, &out_type, finish)?,
                Stat::Max => extreme(column, nulls.as_ref(), true, &out_type, finish)?,
            },
        };
        fields.push(arrow_schema::Field::new(
            f.name(),
            array.data_type().clone(),
            true,
        ));
        arrays.push(array);
    }
    StructArray::try_new(Fields::from(fields), arrays, None)
        .map_err(|e| internal(&format!("a stats struct: {e}")))
}

/// Whether a `nullCount` struct is a struct of counts (a struct column) rather than one count.
fn is_struct_count(s: &StructType) -> bool {
    s.fields().next().is_some()
}

/// Which values of a float column are NaN (any sign bit, any payload); `None` for other types.
fn nan_mask(column: &ArrayRef) -> Option<Vec<bool>> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float16Type, Float32Type, Float64Type};
    match column.data_type() {
        ArrowType::Float16 => Some(
            column
                .as_primitive::<Float16Type>()
                .values()
                .iter()
                .map(|v| v.is_nan())
                .collect(),
        ),
        ArrowType::Float32 => Some(
            column
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .map(|v| v.is_nan())
                .collect(),
        ),
        ArrowType::Float64 => Some(
            column
                .as_primitive::<Float64Type>()
                .values()
                .iter()
                .map(|v| v.is_nan())
                .collect(),
        ),
        _ => None,
    }
}

/// The smallest (or, with `max`, the largest) value of `column` among the rows `nulls` leaves
/// valid, as a one-row array of `out`; NULL when there is none. With `finish`, strings are cut
/// as delta-spark cuts them and floats written as it writes them.
fn extreme(
    column: &ArrayRef,
    nulls: Option<&NullBuffer>,
    max: bool,
    out: &ArrowType,
    finish: bool,
) -> Result<ArrayRef, KrishivError> {
    let cmp = arrow_ord::ord::make_comparator(column, column, SortOptions::default())
        .map_err(|e| internal(&format!("comparing a stats column: {e}")))?;
    let nan = nan_mask(column);
    let cmp = |i: usize, b: usize| match nan.as_deref().map(|m| (m[i], m[b])) {
        Some((true, true)) => std::cmp::Ordering::Equal,
        Some((true, false)) => std::cmp::Ordering::Greater,
        Some((false, true)) => std::cmp::Ordering::Less,
        _ => cmp(i, b),
    };
    let mut best: Option<usize> = None;
    for i in 0..column.len() {
        if nulls.is_some_and(|n| n.is_null(i)) {
            continue;
        }
        best = Some(match best {
            None => i,
            Some(b) => {
                let ord = cmp(i, b);
                let better = if max { ord.is_gt() } else { ord.is_lt() };
                if better { i } else { b }
            }
        });
    }
    let Some(at) = best else {
        return Ok(arrow_array::new_null_array(out, 1));
    };
    // `take` copies the value into fresh buffers: a `slice` would share (and keep alive) the
    // whole column's buffers for as long as the raw row lives (review 2, N1).
    let index = arrow_array::UInt32Array::from(vec![
        u32::try_from(at).map_err(|_| internal("a stats row index that does not fit a u32"))?,
    ]);
    let one = arrow_select::take::take(column.as_ref(), &index, None)
        .map_err(|e| internal(&format!("copying a stats value: {e}")))?;
    let one = arrow_cast::cast(&one, out)
        .map_err(|e| internal(&format!("a stats value as {out}: {e}")))?;
    if !finish {
        return Ok(one);
    }
    if let Some(text) = non_finite(&one) {
        // Jackson writes a non-finite double as a string (`"NaN"`, `"Infinity"`), as
        // delta-spark's stats hold it (`tests/fixtures/delta-4.4.1/write-stats`); the value
        // reaches the kernel's JSON as a string column.
        return Ok(Arc::new(arrow_array::StringArray::from(vec![text])));
    }
    if matches!(out, ArrowType::Float32 | ArrowType::Float64) {
        // -0.0 is written as 0.0, as delta-spark writes it.
        let v = arrow_cast::cast(&one, &ArrowType::Float64)
            .map_err(|e| internal(&format!("a float stats value: {e}")))?;
        let v = v
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .map(|a| a.value(0));
        if v == Some(0.0) {
            return arrow_cast::cast(&arrow_array::Float64Array::from(vec![0.0]), out)
                .map_err(|e| internal(&format!("a float stats value: {e}")));
        }
    }
    if !matches!(out, ArrowType::Utf8) {
        return Ok(one);
    }
    let text = one
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .ok_or_else(|| internal("a string stats value that is not a string array"))?
        .value(0);
    let cut = if max {
        truncate_max(text, STRING_PREFIX_LENGTH)
    } else {
        truncate_min(text, STRING_PREFIX_LENGTH)
    };
    Ok(Arc::new(arrow_array::StringArray::from(vec![cut])))
}

/// Jackson's text of a non-finite float value.
fn non_finite(one: &ArrayRef) -> Option<&'static str> {
    let v = match one.data_type() {
        ArrowType::Float32 => one
            .as_any()
            .downcast_ref::<arrow_array::Float32Array>()
            .map(|a| f64::from(a.value(0))),
        ArrowType::Float64 => one
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .map(|a| a.value(0)),
        _ => None,
    }?;
    if v.is_nan() {
        Some("NaN")
    } else if v == f64::INFINITY {
        Some("Infinity")
    } else if v == f64::NEG_INFINITY {
        Some("-Infinity")
    } else {
        None
    }
}

fn rows(n: usize) -> Result<i64, KrishivError> {
    i64::try_from(n).map_err(|_| internal("a row count that does not fit a long"))
}

fn internal(message: &str) -> KrishivError {
    InternalError {
        message: message.to_owned(),
    }
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The raw bounds own their values: a raw row does not keep the batch's buffers alive
    /// (review 2, N1). A 1 MiB string column leaves a raw row of a few hundred bytes, and the
    /// batch's value buffer is referenced by the batch alone once it is dropped.
    #[test]
    fn raw_bounds_do_not_pin_the_batch() {
        use delta_kernel::schema::StructField;
        let leaf = |t: DataType| StructType::new_unchecked([StructField::nullable("s", t)]);
        let schema = StructType::new_unchecked([
            StructField::nullable("numRecords", DataType::LONG),
            StructField::nullable(
                "nullCount",
                DataType::Struct(Box::new(leaf(DataType::LONG))),
            ),
            StructField::nullable(
                "minValues",
                DataType::Struct(Box::new(leaf(DataType::STRING))),
            ),
            StructField::nullable(
                "maxValues",
                DataType::Struct(Box::new(leaf(DataType::STRING))),
            ),
        ]);
        let values: Vec<String> = (0..1024)
            .map(|i| format!("{i:04}{}", "x".repeat(1024)))
            .collect();
        let column = arrow_array::StringArray::from(values);
        let buffer = column.values().clone();
        let batch = RecordBatch::try_from_iter([("s", Arc::new(column) as ArrayRef)]).unwrap();
        let mut acc = StatsAccumulator::default();
        acc.update(&batch, &schema).unwrap();
        drop(batch);
        let size: usize = acc.raw.iter().map(Array::get_array_memory_size).sum();
        assert!(size < 64 * 1024, "the raw row holds {size} bytes");
        // `into_mutable` succeeds only on the buffer's last reference.
        assert!(
            buffer.into_mutable().is_ok(),
            "a raw bound still references the batch's value buffer"
        );
        let done = acc.finish(&schema, false).unwrap();
        assert_eq!(done.len(), 1);
    }

    /// A NaN with the sign bit set (x86's default NaN, `0xFFF8_0000_0000_0000`) is above every
    /// number, as in Spark's ordering: the max, never the min, in one batch and across two.
    #[test]
    fn a_negative_nan_is_the_max_not_the_min() {
        use delta_kernel::schema::StructField;
        let leaf = |t: DataType| StructType::new_unchecked([StructField::nullable("x", t)]);
        let schema = StructType::new_unchecked([
            StructField::nullable("numRecords", DataType::LONG),
            StructField::nullable(
                "nullCount",
                DataType::Struct(Box::new(leaf(DataType::LONG))),
            ),
            StructField::nullable(
                "minValues",
                DataType::Struct(Box::new(leaf(DataType::DOUBLE))),
            ),
            StructField::nullable(
                "maxValues",
                DataType::Struct(Box::new(leaf(DataType::DOUBLE))),
            ),
        ]);
        let neg_nan = f64::from_bits(0xFFF8_0000_0000_0000);
        assert!(neg_nan.is_nan() && neg_nan.is_sign_negative());
        let batch = |v: Vec<f64>| {
            RecordBatch::try_from_iter([(
                "x",
                Arc::new(arrow_array::Float64Array::from(v)) as ArrayRef,
            )])
            .unwrap()
        };
        let bounds = |acc: StatsAccumulator| {
            let done = acc.finish(&schema, false).unwrap();
            let row = done
                .as_any()
                .downcast_ref::<arrow_array::StructArray>()
                .unwrap();
            let get = |name: &str| {
                let s = row.column_by_name(name).unwrap();
                let s = s
                    .as_any()
                    .downcast_ref::<arrow_array::StructArray>()
                    .unwrap();
                let x = s.column_by_name("x").unwrap();
                arrow_cast::display::array_value_to_string(x, 0).unwrap()
            };
            (get("minValues"), get("maxValues"))
        };
        let mut one = StatsAccumulator::default();
        one.update(&batch(vec![1.0, 5.0, neg_nan]), &schema)
            .unwrap();
        assert_eq!(bounds(one), ("1.0".to_owned(), "NaN".to_owned()));
        let mut two = StatsAccumulator::default();
        two.update(&batch(vec![neg_nan, 5.0]), &schema).unwrap();
        two.update(&batch(vec![1.0]), &schema).unwrap();
        assert_eq!(bounds(two), ("1.0".to_owned(), "NaN".to_owned()));
    }

    /// delta-spark's own cases (`StatsCollectionSuite` "truncateMaxStringAgg"), and the min's
    /// `substring`, prefix 32 unless named.
    #[test]
    fn string_bounds_are_cut_as_delta_spark_cuts_them() {
        let a32 = "a".repeat(32);
        assert_eq!(truncate_min(&a32, 32).as_deref(), Some(a32.as_str()));
        assert_eq!(truncate_max(&a32, 32).as_deref(), Some(a32.as_str()));
        let a33 = "a".repeat(33);
        assert_eq!(truncate_min(&a33, 32).as_deref(), Some(a32.as_str()));
        assert_eq!(truncate_max(&a33, 32), Some(format!("{a32}\u{7f}")));
        // A non-ASCII unit after the cut: the U+10FFFF tie-breaker.
        let accented = format!("{a32}\u{e9}\u{e9}");
        assert_eq!(
            truncate_max(&accented, 32),
            Some(format!("{a32}\u{10ffff}"))
        );
        // A surrogate pair across the cut stays whole.
        let emoji = format!("{}\u{1f600}bc", "a".repeat(31));
        assert_eq!(
            truncate_min(&emoji, 32),
            Some(format!("{}\u{1f600}", "a".repeat(31)))
        );
        assert_eq!(
            truncate_max(&emoji, 32),
            Some(format!("{}\u{1f600}\u{7f}", "a".repeat(31)))
        );
        // U+10FFFF characters after the cut are skipped; too many drop the max.
        let maxes = format!("{a32}{}b", "\u{10ffff}".repeat(2));
        assert_eq!(
            truncate_max(&maxes, 32),
            Some(format!("{a32}{}\u{7f}", "\u{10ffff}".repeat(2)))
        );
        let many = format!("{a32}{}b", "\u{10ffff}".repeat(20));
        assert_eq!(truncate_max(&many, 32), None);
        // Reaching the end while skipping keeps the input.
        let tail = format!("{a32}\u{10ffff}");
        assert_eq!(truncate_max(&tail, 32).as_deref(), Some(tail.as_str()));
        // A prefix of 0 keeps only the empty string.
        assert_eq!(truncate_min("", 0).as_deref(), Some(""));
        // The min counts code points (Spark's `substring`): 20 supplementary characters fit
        // a prefix of 32 whole, where 32 UTF-16 units would keep 16.
        let wide = "\u{1f600}".repeat(20);
        assert_eq!(truncate_min(&wide, 32).as_deref(), Some(wide.as_str()));
        let wider = "\u{1f600}".repeat(40);
        assert_eq!(truncate_min(&wider, 32), Some("\u{1f600}".repeat(32)));
        assert_eq!(truncate_min("abc", 0).as_deref(), Some(""));
        assert_eq!(truncate_max("abc", 0), None);
    }
}
