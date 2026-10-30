// SPDX-License-Identifier: Apache-2.0
//! Spark's partitioned-write layout (P2-T082b unit 1): how a row's
//! partition columns become a directory, and what the files in it are
//! named.
//!
//! - A row with partition values `v1..vn` of columns `c1..cn` lands in
//!   `<root>/<c1>=<v1>/.../<cn>=<vn>/`, each value rendered as
//!   `CAST(v AS STRING)` reads (legacy mode, UTC) and then escaped with
//!   [`escape_partition_value`]. NULL and the empty string both become
//!   [`DEFAULT_PARTITION`], as in Spark's Hive-style partitioning.
//! - Files in a partition directory are named
//!   `part-<task:05>-<job uuid>.c<file:03><suffix>`, while unpartitioned
//!   output keeps [`spark_file_name`](crate::spark_file_name)
//!   (`part-<task:05>-<job uuid>-c<file:03><suffix>`).
//! - Escaping replaces one set of characters by `%` plus two upper-case
//!   hex digits (Spark 4.2's observed set): the C0 controls (U+0001
//!   through U+001F), DEL (U+007F), and `"`, `#`, `%`, `'`, `*`, `/`,
//!   `:`, `=`, `?`, `\`, `{`, `[`, `]`, `^`. Kept as is: `}` and space,
//!   `+`, `,`, `;`, `(`, `)`, `-`, `.`, `_` and every non-ASCII character.

use arrow_array::ArrayRef;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use krishiv_kernels::cast::{CastMode, CastOptions, FixedOffset, cast};

use crate::error::{Result, internal, unsupported};

/// What NULL and the empty string render as in a partition directory.
pub const DEFAULT_PARTITION: &str = "__HIVE_DEFAULT_PARTITION__";

#[cfg(test)]
mod tests;

/// Whether `c` is escaped in a partition path segment (Spark 4.2's
/// observed set): the C0 controls, DEL, and
/// `" # % ' * / : = ? \ { [ ] ^`.
fn escapes(c: char) -> bool {
    matches!(
        c,
        '\u{1}'
            ..='\u{1f}'
                | '\u{7f}'
                | '"'
                | '#'
                | '%'
                | '\''
                | '*'
                | '/'
                | ':'
                | '='
                | '?'
                | '\\'
                | '{'
                | '['
                | ']'
                | '^'
    )
}

/// Renders `value` as one partition path segment: each escapable
/// character becomes `%` plus two upper-case hex digits, everything
/// else (including `}`, space and non-ASCII text) stays as is.
#[must_use]
pub fn escape_partition_value(value: &str) -> String {
    if !value.chars().any(escapes) {
        return value.to_owned();
    }
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        if escapes(c) {
            out.push('%');
            let n = c as u32;
            out.push(
                char::from_digit(n >> 4, 16)
                    .unwrap_or('0')
                    .to_ascii_uppercase(),
            );
            out.push(
                char::from_digit(n & 0xF, 16)
                    .unwrap_or('0')
                    .to_ascii_uppercase(),
            );
        } else {
            out.push(c);
        }
    }
    out
}

/// Spark's data file name inside a partition directory:
/// `part-<task:05>-<job uuid>.c<file:03><suffix>`, e.g.
/// `part-00000-<uuid>.c000.snappy.parquet`. Unpartitioned output keeps
/// [`spark_file_name`](crate::spark_file_name) instead.
#[must_use]
pub fn spark_partitioned_file_name(task: u32, job_uuid: &str, file: u32, suffix: &str) -> String {
    format!("part-{task:05}-{job_uuid}.c{file:03}{suffix}")
}

/// The relative directory of a row's partition values:
/// `<name>=<segment>/...` for each column, in `partitionBy` order.
#[must_use]
pub fn partition_dir_name(names: &[String], segments: &[String]) -> String {
    names
        .iter()
        .zip(segments.iter())
        .map(|(n, s)| format!("{}={s}", escape_partition_value(n)))
        .collect::<Vec<_>>()
        .join("/")
}

/// The `CAST(v AS STRING)` text of every row of `column` (legacy mode, in
/// the session time zone `zone`, a fixed offset in seconds), or `None` for
/// NULL rows. String carriers read as is; anything
/// else goes through the cast kernel, so rendering matches the engine's
/// own casts.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a column type the cast kernel
/// cannot render as a string, and for a timestamp column when the session
/// time zone is not a fixed offset (`zone` is `None`).
pub fn partition_texts(column: &ArrayRef, zone: Option<i32>) -> Result<Vec<Option<String>>> {
    match column.data_type() {
        DataType::Utf8 => Ok(column
            .as_string::<i32>()
            .iter()
            .map(|v| v.map(str::to_owned))
            .collect()),
        DataType::LargeUtf8 => Ok(column
            .as_string::<i64>()
            .iter()
            .map(|v| v.map(str::to_owned))
            .collect()),
        DataType::Utf8View => Ok(column
            .as_string_view()
            .iter()
            .map(|v| v.map(str::to_owned))
            .collect()),
        _ => {
            let zone = match (zone, column.data_type()) {
                (Some(offset), _) => FixedOffset(offset),
                (None, DataType::Timestamp(_, Some(_))) => {
                    return Err(unsupported(
                        "partitioning by a timestamp column in a session time zone other than a fixed offset"
                            .to_owned(),
                    ));
                }
                (None, _) => FixedOffset(0),
            };
            let opts = CastOptions {
                mode: CastMode::Legacy,
                time_zone: &zone,
            };
            let strings = cast(column, &DataType::Utf8, &opts)?;
            let strings = strings.as_string::<i32>();
            Ok(strings.iter().map(|v| v.map(str::to_owned)).collect())
        }
    }
}

/// One partition path segment per row of `columns` (in column order):
/// the escaped [`partition_texts`], with NULL and the empty string as
/// [`DEFAULT_PARTITION`].
///
/// # Errors
/// As [`partition_texts`] with `zone`; `INTERNAL_ERROR` for columns of different
/// lengths.
pub fn partition_segments(columns: &[ArrayRef], zone: Option<i32>) -> Result<Vec<Vec<String>>> {
    let texts: Vec<Vec<Option<String>>> = columns
        .iter()
        .map(|c| partition_texts(c, zone))
        .collect::<Result<_>>()?;
    let rows = texts.first().map_or(0, Vec::len);
    if texts.iter().any(|t| t.len() != rows) {
        return Err(internal(
            "partition columns of different lengths".to_owned(),
        ));
    }
    let mut out = vec![Vec::with_capacity(columns.len()); rows];
    for (row, key) in out.iter_mut().enumerate() {
        for text in &texts {
            let segment = match &text[row] {
                Some(v) if !v.is_empty() => escape_partition_value(v),
                _ => DEFAULT_PARTITION.to_owned(),
            };
            key.push(segment);
        }
    }
    Ok(out)
}

/// Whether the cast kernel renders `data_type` as a string (what
/// [`partition_texts`] accepts beyond the string carriers).
#[must_use]
pub fn renders_as_string(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal128(_, _)
            | DataType::Decimal64(_, _)
            | DataType::Date32
            | DataType::Timestamp(_, _)
    )
}

/// Checks that the partition column of `data_type` can be rendered.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for any other type.
pub fn check_partition_column(name: &str, data_type: &DataType) -> Result<()> {
    if renders_as_string(data_type) {
        Ok(())
    } else {
        Err(unsupported(format!(
            "partitioning by a {data_type} column ({name})"
        )))
    }
}
