// SPDX-License-Identifier: Apache-2.0
//! The Iceberg column metrics of a data file Krishiv wrote (P3-T027, PR-36a;
//! design note §6.1), from the statistics the Parquet writer returns, as
//! Java's `ParquetMetrics` (apache-iceberg 1.12.0) derives them from the
//! footer, with float columns as Iceberg's Spark writer collects them
//! (`FloatFieldMetrics`: NaN counted, bounds over the other values):
//!
//! - mode `none`: nothing; `counts`: value and null counts (and NaN counts of
//!   float columns); `truncate(n)` and `full`: also lower and upper bounds.
//! - Only columns reachable through structs have metrics (Java drops lists
//!   and maps). A column without statistics has none at all.
//! - The value count is the row count (a column that is not repeated has one
//!   value, maybe null, per row); an unknown null count is left out.
//! - Bounds come from exact statistics only. A NaN bound leaves both out.
//!   Strings keep `n` code points: the lower bound is the prefix, the upper
//!   bound the prefix with its last code point incremented (skipping the
//!   surrogate range; none when every code point overflows), Java's
//!   `UnicodeUtil.truncateString{Min,Max}`; binaries the same by bytes
//!   (`BinaryUtil.truncateBinary{Min,Max}`). Other types are not truncated.
//!   A uuid column gets no bounds (the statistics hold it as a string).

use std::collections::BTreeMap;

use krishiv_io::{FileStatistics, ScalarValue};

use super::bounds::{encode, is_nan};
use super::metrics::MetricsMode;
use super::types::{IcebergType, SchemaField};

/// A data file's column metrics, by field id (Iceberg's `Metrics`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct FileMetrics {
    /// Values per column, nulls and NaNs included.
    pub(crate) value_counts: BTreeMap<i32, u64>,
    /// Nulls per column.
    pub(crate) null_counts: BTreeMap<i32, u64>,
    /// NaNs per float column.
    pub(crate) nan_counts: BTreeMap<i32, u64>,
    /// Lower bounds, serialised (Iceberg's single-value serialisation).
    pub(crate) lower_bounds: BTreeMap<i32, Vec<u8>>,
    /// Upper bounds, serialised.
    pub(crate) upper_bounds: BTreeMap<i32, Vec<u8>>,
}

/// The truncation length a mode gives bounds; 0 means no bounds (Java's
/// `ParquetMetrics.truncateLength`).
fn truncate_length(mode: MetricsMode) -> usize {
    match mode {
        MetricsMode::None | MetricsMode::Counts => 0,
        MetricsMode::Truncate(n) => n,
        // A write never runs under a mode that did not parse (the config
        // falls back, as Java's `parseMode`); no truncation is the safe
        // reading of `full`.
        MetricsMode::Full | MetricsMode::Unknown => usize::MAX,
    }
}

/// The metrics of a file whose writer returned `stats`, for the columns
/// `fields` (the table schema's, whose names the file uses), each under the
/// mode `mode(id)`.
pub(crate) fn file_metrics(
    stats: &FileStatistics,
    fields: &[SchemaField],
    mode: impl Fn(i32) -> MetricsMode,
) -> FileMetrics {
    let mut out = FileMetrics::default();
    let rows = stats.rows.exact().copied();
    for field in fields {
        let mode = mode(field.id);
        if mode == MetricsMode::None {
            continue;
        }
        let Some(col) = stats.columns.iter().find(|c| c.path == field.path) else {
            continue;
        };
        let Some(rows) = rows else {
            continue;
        };
        out.value_counts.insert(field.id, rows);
        if let Some(n) = col.null_count.exact() {
            out.null_counts.insert(field.id, *n);
        }
        if field.ty.is_float()
            && let Some(n) = col.nan_count.exact()
        {
            out.nan_counts.insert(field.id, *n);
        }
        let length = truncate_length(mode);
        if length == 0 {
            continue;
        }
        let (Some(min), Some(max)) = (col.min.exact(), col.max.exact()) else {
            continue;
        };
        if is_nan(min) || is_nan(max) || matches!(field.ty, IcebergType::Uuid) {
            continue;
        }
        if let Some(lower) =
            truncate_min(min, &field.ty, length).and_then(|v| encode(&v, &field.ty))
        {
            out.lower_bounds.insert(field.id, lower);
        }
        if let Some(upper) =
            truncate_max(max, &field.ty, length).and_then(|v| encode(&v, &field.ty))
        {
            out.upper_bounds.insert(field.id, upper);
        }
    }
    out
}

fn truncate_min(v: &ScalarValue, ty: &IcebergType, length: usize) -> Option<ScalarValue> {
    Some(match (ty, v) {
        (IcebergType::String, ScalarValue::Utf8(s)) => {
            ScalarValue::Utf8(truncate_string(s, length).to_owned())
        }
        (IcebergType::Binary, ScalarValue::Binary(b)) => {
            ScalarValue::Binary(b[..b.len().min(length)].to_vec())
        }
        _ => v.clone(),
    })
}

fn truncate_max(v: &ScalarValue, ty: &IcebergType, length: usize) -> Option<ScalarValue> {
    Some(match (ty, v) {
        (IcebergType::String, ScalarValue::Utf8(s)) => {
            ScalarValue::Utf8(truncate_string_max(s, length)?)
        }
        (IcebergType::Binary, ScalarValue::Binary(b)) => {
            ScalarValue::Binary(truncate_binary_max(b, length)?)
        }
        _ => v.clone(),
    })
}

/// The first `length` code points of `s` (Java's `UnicodeUtil.truncateString`).
pub(crate) fn truncate_string(s: &str, length: usize) -> &str {
    match s.char_indices().nth(length) {
        Some((at, _)) => &s[..at],
        None => s,
    }
}

/// The smallest string of at most `length` code points that is at least `s`
/// (Java's `UnicodeUtil.truncateStringMax`); `None` when none exists.
pub(crate) fn truncate_string_max(s: &str, length: usize) -> Option<String> {
    let prefix = truncate_string(s, length);
    if prefix.len() == s.len() {
        return Some(s.to_owned());
    }
    let mut chars: Vec<char> = prefix.chars().collect();
    while let Some(last) = chars.pop() {
        if let Some(next) = increment(last) {
            chars.push(next);
            return Some(chars.into_iter().collect());
        }
    }
    None
}

/// The next Unicode scalar value (Java's `incrementCodePoint`): U+D7FF goes
/// to U+E000; U+10FFFF overflows.
fn increment(c: char) -> Option<char> {
    match c {
        '\u{D7FF}' => Some('\u{E000}'),
        char::MAX => None,
        _ => char::from_u32(u32::from(c) + 1),
    }
}

/// The smallest byte string of at most `length` bytes that is at least `b`
/// (Java's `BinaryUtil.truncateBinaryMax`); `None` when every byte is 0xFF.
pub(crate) fn truncate_binary_max(b: &[u8], length: usize) -> Option<Vec<u8>> {
    if length >= b.len() {
        return Some(b.to_vec());
    }
    let mut out = b[..length].to_vec();
    while let Some(last) = out.pop() {
        if last != u8::MAX {
            out.push(last + 1);
            return Some(out);
        }
    }
    None
}
