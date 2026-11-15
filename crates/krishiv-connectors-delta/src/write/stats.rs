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
//! - strings: the min is cut to a 32-character prefix (`truncateMinStringAgg`, `:866-883`), the
//!   max is cut and given a tie-breaker (`truncateMaxStringAgg`, `:889-923`), both counted in
//!   UTF-16 units as on the JVM (`spark.databricks.io.skipping.stringPrefixLength`, default 32);
//! - `tightBounds = true` only on a table that supports deletion vectors (`:286-290`), NULL
//!   (omitted) otherwise.
//!
//! Floating-point bounds use Arrow's total order, where NaN is above every number, as in
//! Spark's ordering; Arrow orders `-0.0` below `0.0`, where Spark has them equal (a recorded
//! difference when a file holds both).

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

/// delta-spark's `truncateMinStringAgg(prefixLen)`: the input when it has at most `prefix`
/// UTF-16 units, else its first `prefix` units (one more when they would split a surrogate
/// pair).
#[must_use]
pub fn truncate_min(input: &str, prefix: usize) -> Option<String> {
    let units: Vec<u16> = input.encode_utf16().collect();
    if units.len() <= prefix {
        return Some(input.to_owned());
    }
    if prefix == 0 {
        return None;
    }
    let cut = min_cut(&units, prefix);
    String::from_utf16(&units[..cut]).ok()
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
    let root = Column {
        fields: batch.schema().fields().clone(),
        columns: batch.columns().to_vec(),
        nulls: None,
    };
    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for f in schema.fields() {
        let out_type = to_arrow_type(f.data_type()).map_err(|e| internal(&e.to_string()))?;
        let array: ArrayRef = match f.name().as_str() {
            "numRecords" => Arc::new(Int64Array::from(vec![rows(batch.num_rows())?])),
            "nullCount" => Arc::new(nested(&root, struct_of(f.data_type())?, &Stat::NullCount)?),
            "minValues" => Arc::new(nested(&root, struct_of(f.data_type())?, &Stat::Min)?),
            "maxValues" => Arc::new(nested(&root, struct_of(f.data_type())?, &Stat::Max)?),
            "tightBounds" => Arc::new(BooleanArray::from(vec![tight_bounds.then_some(true)])),
            other => {
                return Err(internal(&format!(
                    "an unknown field {other} in the kernel's stats schema"
                )));
            }
        };
        let _ = out_type;
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

/// Which statistic a struct of the stats schema holds.
enum Stat {
    NullCount,
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

fn struct_of(t: &DataType) -> Result<&StructType, KrishivError> {
    match t {
        DataType::Struct(s) => Ok(s),
        other => Err(internal(&format!(
            "a stats field of type {other} where a struct was expected"
        ))),
    }
}

/// One row of `schema`, the statistic `stat` of each of its leaves over `data`.
fn nested(data: &Column, schema: &StructType, stat: &Stat) -> Result<StructArray, KrishivError> {
    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for f in schema.fields() {
        let i = data
            .fields
            .iter()
            .position(|d| d.name() == f.name())
            .ok_or_else(|| {
                internal(&format!(
                    "the stats column {} is not among the written columns",
                    f.name()
                ))
            })?;
        let column = &data.columns[i];
        let nulls = NullBuffer::union(data.nulls.as_ref(), column.logical_nulls().as_ref());
        let out_type = to_arrow_type(f.data_type()).map_err(|e| internal(&e.to_string()))?;
        let array: ArrayRef = match (f.data_type(), column.data_type()) {
            (DataType::Struct(child), ArrowType::Struct(child_fields))
                if !matches!(stat, Stat::NullCount) || is_struct_count(child) =>
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
                Arc::new(nested(&level, child, stat)?)
            }
            _ => match stat {
                Stat::NullCount => {
                    let n = nulls.as_ref().map_or(0, NullBuffer::null_count);
                    Arc::new(Int64Array::from(vec![rows(n)?]))
                }
                Stat::Min => extreme(column, nulls.as_ref(), false, &out_type)?,
                Stat::Max => extreme(column, nulls.as_ref(), true, &out_type)?,
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

/// The smallest (or, with `max`, the largest) value of `column` among the rows `nulls` leaves
/// valid, as a one-row array of `out`; NULL when there is none. Strings are cut as delta-spark
/// cuts them.
fn extreme(
    column: &ArrayRef,
    nulls: Option<&NullBuffer>,
    max: bool,
    out: &ArrowType,
) -> Result<ArrayRef, KrishivError> {
    let cmp = arrow_ord::ord::make_comparator(column, column, SortOptions::default())
        .map_err(|e| internal(&format!("comparing a stats column: {e}")))?;
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
    let one = column.slice(at, 1);
    let one = arrow_cast::cast(&one, out)
        .map_err(|e| internal(&format!("a stats value as {out}: {e}")))?;
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

    /// delta-spark's own cases (`StatsCollectionSuite` "truncateMaxStringAgg" and
    /// "truncateMinStringAgg"), prefix 32 unless named.
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
        assert_eq!(truncate_min("abc", 0), None);
        assert_eq!(truncate_max("abc", 0), None);
    }
}
