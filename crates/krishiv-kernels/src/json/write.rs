// SPDX-License-Identifier: Apache-2.0
//! The `to_json` writer: Arrow values to compact JSON text, as Spark's
//! `JacksonGenerator` writes them.
//!
//! Structs keep their fields in order (`sortKeys` sorts them by name),
//! with `NULL` fields dropped (`ignoreNullFields`, Spark's default);
//! maps keep insertion order (`sortKeys` sorts by rendered key, which
//! agrees with Spark's key order for string keys), arrays keep order. Strings are quoted and escaped, integral values print as
//! digits, doubles and floats print as Java's `toString` does, decimals
//! print plain, dates print as `yyyy-MM-dd` and timestamps in the session
//! zone as `yyyy-MM-dd'T'HH:mm:ss.SSSXXX` unless the options name other
//! patterns (non-finite floats and doubles are quoted, as Spark writes
//! them). Binary prints base64. A map key that is not a string prints
//! as its JSON value does, except a struct key, which prints as the JSON
//! array of its values (`{"[1,2]":...}`, as Spark does).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, Time64NanosecondType, TimestampMicrosecondType,
};
use arrow_array::{Array, ArrayRef, StringArray};
use arrow_schema::{DataType as ArrowType, TimeUnit};
use krishiv_common::error::{InternalError, KrishivError};

use super::JsonOptions;
use super::base64;
use crate::cast::TimeZoneRules;
use crate::datetime_pattern::{DateTimePattern, Purpose, Zone};
use crate::text::{java_decimal_string, java_double_string, java_float_string32};

/// Spark's default `dateFormat` for JSON output.
const DATE_DEFAULT: &str = "yyyy-MM-dd";
/// Spark's default `timestampFormat` for JSON output.
const TIMESTAMP_DEFAULT: &str = "yyyy-MM-dd'T'HH:mm:ss.SSSXXX";
/// Spark's default `timestampNTZFormat` for JSON output.
const TIMESTAMP_NTZ_DEFAULT: &str = "yyyy-MM-dd'T'HH:mm:ss.SSS";

/// Renders `values` as JSON text, one row per string.
pub fn write_json(
    values: &ArrayRef,
    opts: &JsonOptions,
    zone: &dyn TimeZoneRules,
) -> Result<ArrayRef, KrishivError> {
    let out = Output::new(opts)?;
    let ctx = Ctx { out: &out, zone };
    let mut strs = Vec::with_capacity(values.len());
    for row in 0..values.len() {
        if values.is_null(row) {
            strs.push(None);
            continue;
        }
        let mut text = String::new();
        write_value(values, row, &mut text, &ctx)?;
        strs.push(Some(text));
    }
    Ok(Arc::new(StringArray::from(strs)))
}

/// A row that fails while writing (a bad pattern for its type): the whole
/// batch fails, as Spark's generator fails the query.
fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// The compiled output patterns of one call.
struct Output {
    /// `dateFormat`, or Spark's default.
    date: DateTimePattern,
    /// `timestampFormat`, or Spark's default.
    timestamp: DateTimePattern,
    /// `timestampNTZFormat`, or Spark's default.
    timestamp_ntz: DateTimePattern,
    /// `timeFormat`; `None` writes `HH:mm:ss[.fraction]`.
    time: Option<DateTimePattern>,
    /// `sortKeys`.
    sort_keys: bool,
    /// `ignoreNullFields` (Spark's default): struct fields that are `NULL`
    /// are not written (maps and arrays still write `null`).
    ignore_null_fields: bool,
}

impl Output {
    /// The output patterns of `opts` (already compiled for formatting),
    /// with Spark's defaults where the options name none.
    fn new(opts: &JsonOptions) -> Result<Self, KrishivError> {
        let compile = |named: Option<&DateTimePattern>, default: &str| {
            named.cloned().map_or_else(
                || {
                    DateTimePattern::new(default, Purpose::Format)
                        .map_err(|e| internal(format!("the default JSON date pattern: {e}")))
                },
                Ok,
            )
        };
        Ok(Self {
            date: compile(opts.date_output.as_ref(), DATE_DEFAULT)?,
            timestamp: compile(opts.timestamp_output.as_ref(), TIMESTAMP_DEFAULT)?,
            timestamp_ntz: compile(opts.timestamp_ntz_output.as_ref(), TIMESTAMP_NTZ_DEFAULT)?,
            time: opts.time_output.clone(),
            sort_keys: opts.sort_keys,
            ignore_null_fields: opts.ignore_null_fields,
        })
    }
}

/// The writing context: patterns and the session zone.
struct Ctx<'a> {
    out: &'a Output,
    zone: &'a dyn TimeZoneRules,
}

/// One value (known valid) as JSON text.
fn write_value(
    col: &ArrayRef,
    row: usize,
    text: &mut String,
    ctx: &Ctx<'_>,
) -> Result<(), KrishivError> {
    if col.is_null(row) {
        text.push_str("null");
        return Ok(());
    }
    match col.data_type() {
        ArrowType::Null => text.push_str("null"),
        ArrowType::Boolean => {
            let v = col.as_boolean().value(row);
            text.push_str(if v { "true" } else { "false" });
        }
        ArrowType::Int8 => push_int(i64::from(col.as_primitive::<Int8Type>().value(row)), text),
        ArrowType::Int16 => push_int(i64::from(col.as_primitive::<Int16Type>().value(row)), text),
        ArrowType::Int32 => push_int(i64::from(col.as_primitive::<Int32Type>().value(row)), text),
        ArrowType::Int64 => push_int(col.as_primitive::<Int64Type>().value(row), text),
        ArrowType::UInt8 | ArrowType::UInt16 | ArrowType::UInt32 | ArrowType::UInt64 => {
            return Err(internal(format!(
                "a JSON value of Arrow type {}",
                col.data_type()
            )));
        }
        ArrowType::Float32 => {
            let v = col.as_primitive::<Float32Type>().value(row);
            // Bare `NaN` is not JSON: Spark quotes the non-finite values.
            if v.is_finite() {
                text.push_str(&java_float_string32(v));
            } else {
                push_quoted(&java_float_string32(v), text);
            }
        }
        ArrowType::Float64 => {
            let v = col.as_primitive::<Float64Type>().value(row);
            // Bare `NaN` is not JSON: Spark quotes the non-finite values.
            if v.is_finite() {
                text.push_str(&java_double_string(v));
            } else {
                push_quoted(&java_double_string(v), text);
            }
        }
        ArrowType::Decimal128(..) => {
            let v = col.as_primitive::<Decimal128Type>().value(row);
            let ArrowType::Decimal128(_, s) = col.data_type() else {
                return Err(internal("a decimal value".to_owned()));
            };
            text.push_str(&java_decimal_string(v, *s));
        }
        ArrowType::Utf8View | ArrowType::Utf8 | ArrowType::LargeUtf8 => {
            push_quoted(&col_str(col, row), text);
        }
        ArrowType::BinaryView | ArrowType::Binary | ArrowType::LargeBinary => {
            push_quoted(&base64::encode(&col_bytes(col, row)), text);
        }
        ArrowType::Date32 => {
            let days = col.as_primitive::<Date32Type>().value(row);
            let rendered = ctx
                .out
                .date
                .format_date(days)
                .map_err(|e| internal(format!("a to_json date pattern: {e}")))?;
            push_quoted(&rendered, text);
        }
        ArrowType::Time64(TimeUnit::Nanosecond) => {
            let nanos = col.as_primitive::<Time64NanosecondType>().value(row);
            let rendered = match &ctx.out.time {
                Some(pattern) => pattern
                    .format_local(nanos.div_euclid(1_000))
                    .map_err(|e| internal(format!("a to_json time pattern: {e}")))?,
                None => crate::cast::time::format_time(nanos),
            };
            push_quoted(&rendered, text);
        }
        ArrowType::Timestamp(TimeUnit::Microsecond, tz) => {
            let micros = col.as_primitive::<TimestampMicrosecondType>().value(row);
            let rendered = if tz.is_some() {
                let seconds = micros.div_euclid(1_000_000);
                let offset = ctx.zone.offset_at_utc(seconds);
                let zone = Zone {
                    id: &zone_id(offset),
                    rules: ctx.zone,
                };
                ctx.out
                    .timestamp
                    .format_timestamp(micros, zone)
                    .map_err(|e| internal(format!("a to_json timestamp pattern: {e}")))?
            } else {
                ctx.out
                    .timestamp_ntz
                    .format_local(micros)
                    .map_err(|e| internal(format!("a to_json timestamp pattern: {e}")))?
            };
            push_quoted(&rendered, text);
        }
        ArrowType::List(_) | ArrowType::LargeList(_) => write_list(col, row, text, ctx)?,
        ArrowType::Map(_, _) => write_map(col, row, text, ctx)?,
        ArrowType::Struct(_) => write_struct(col, row, text, ctx)?,
        other => {
            return Err(internal(format!("a JSON value of Arrow type {other}")));
        }
    }
    Ok(())
}

/// A string column's value at `row`.
fn col_str(col: &ArrayRef, row: usize) -> String {
    match col.data_type() {
        ArrowType::Utf8View => col.as_string_view().value(row).to_owned(),
        ArrowType::LargeUtf8 => col.as_string::<i64>().value(row).to_owned(),
        _ => col.as_string::<i32>().value(row).to_owned(),
    }
}

/// A binary column's value at `row`.
fn col_bytes(col: &ArrayRef, row: usize) -> Vec<u8> {
    match col.data_type() {
        ArrowType::BinaryView => col.as_binary_view().value(row).to_vec(),
        ArrowType::LargeBinary => col.as_binary::<i64>().value(row).to_vec(),
        _ => col.as_binary::<i32>().value(row).to_vec(),
    }
}

/// Pushes an integer.
fn push_int(v: i64, text: &mut String) {
    text.push_str(&v.to_string());
}

/// Pushes a quoted, escaped string.
pub(crate) fn push_quoted(s: &str, text: &mut String) {
    text.push('"');
    for c in s.chars() {
        match c {
            '"' => text.push_str("\\\""),
            '\\' => text.push_str("\\\\"),
            '\u{8}' => text.push_str("\\b"),
            '\u{c}' => text.push_str("\\f"),
            '\n' => text.push_str("\\n"),
            '\r' => text.push_str("\\r"),
            '\t' => text.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                text.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => text.push(c),
        }
    }
    text.push('"');
}

/// The fixed-offset zone id Spark normalises an offset to (`+HH:MM`,
/// `UTC` at zero).
fn zone_id(offset: i32) -> String {
    if offset == 0 {
        return "UTC".to_owned();
    }
    let sign = if offset < 0 { '-' } else { '+' };
    let abs = offset.unsigned_abs();
    format!(
        "{sign}{:02}:{:02}",
        abs.div_euclid(3600),
        abs.rem_euclid(3600).div_euclid(60)
    )
}

/// One list value.
fn write_list(
    col: &ArrayRef,
    row: usize,
    text: &mut String,
    ctx: &Ctx<'_>,
) -> Result<(), KrishivError> {
    let (offsets, values) = list_rows(col, row)?;
    text.push('[');
    for (i, index) in offsets.iter().enumerate() {
        if i > 0 {
            text.push(',');
        }
        write_value(&values, *index, text, ctx)?;
    }
    text.push(']');
    Ok(())
}

/// The element rows of list row `row`.
fn list_rows(col: &ArrayRef, row: usize) -> Result<(Vec<usize>, ArrayRef), KrishivError> {
    let bad = || internal(format!("a JSON list of Arrow type {}", col.data_type()));
    let (start, end, values) = match col.data_type() {
        ArrowType::List(_) => {
            let list = col.as_list::<i32>();
            let offsets = list.value_offsets();
            (
                usize::try_from(offsets.get(row).copied().unwrap_or(0)).map_err(|_| bad())?,
                usize::try_from(offsets.get(row.saturating_add(1)).copied().unwrap_or(0))
                    .map_err(|_| bad())?,
                Arc::clone(list.values()),
            )
        }
        ArrowType::LargeList(_) => {
            let list = col.as_list::<i64>();
            let offsets = list.value_offsets();
            (
                usize::try_from(offsets.get(row).copied().unwrap_or(0)).map_err(|_| bad())?,
                usize::try_from(offsets.get(row.saturating_add(1)).copied().unwrap_or(0))
                    .map_err(|_| bad())?,
                Arc::clone(list.values()),
            )
        }
        _ => return Err(bad()),
    };
    Ok(((start..end).collect(), values))
}

/// One struct value.
fn write_struct(
    col: &ArrayRef,
    row: usize,
    text: &mut String,
    ctx: &Ctx<'_>,
) -> Result<(), KrishivError> {
    let st = col.as_struct();
    text.push('{');
    let mut fields: Vec<(String, ArrayRef)> = st
        .columns()
        .iter()
        .zip(st.fields())
        .map(|(c, f)| (f.name().to_owned(), Arc::clone(c)))
        .collect();
    if ctx.out.sort_keys {
        fields.sort_by(|a, b| a.0.cmp(&b.0));
    }
    // `ignoreNullFields` (Spark's default) drops the null fields; the
    // comma tracks what was actually written.
    let mut first = true;
    for (name, child) in fields.iter() {
        if ctx.out.ignore_null_fields && child.is_null(row) {
            continue;
        }
        if !first {
            text.push(',');
        }
        first = false;
        push_quoted(name, text);
        text.push(':');
        write_value(child, row, text, ctx)?;
    }
    text.push('}');
    Ok(())
}

/// One map value.
fn write_map(
    col: &ArrayRef,
    row: usize,
    text: &mut String,
    ctx: &Ctx<'_>,
) -> Result<(), KrishivError> {
    let map = col.as_map();
    let offsets = map.value_offsets();
    let bad = || internal(format!("a JSON map of Arrow type {}", col.data_type()));
    let (start, end) = (
        usize::try_from(offsets.get(row).copied().unwrap_or(0)).map_err(|_| bad())?,
        usize::try_from(offsets.get(row.saturating_add(1)).copied().unwrap_or(0))
            .map_err(|_| bad())?,
    );
    let entries = map.entries();
    let keys = entries.column(0);
    let values = entries.column(1);
    // Keys render first (a struct key renders as the array of its values),
    // then sort by the rendered key with `sortKeys`.
    let mut rendered: Vec<(String, String)> = Vec::with_capacity(end.saturating_sub(start));
    for index in start..end {
        let mut key = String::new();
        write_key(keys, index, &mut key, ctx)?;
        let mut value = String::new();
        write_value(values, index, &mut value, ctx)?;
        rendered.push((key, value));
    }
    if ctx.out.sort_keys {
        rendered.sort_by(|a, b| a.0.cmp(&b.0));
    }
    text.push('{');
    for (i, (key, value)) in rendered.iter().enumerate() {
        if i > 0 {
            text.push(',');
        }
        push_quoted(key, text);
        text.push(':');
        text.push_str(value);
    }
    text.push('}');
    Ok(())
}

/// One map key: strings and binary stay text, anything else prints as its
/// JSON value does, except a struct, which prints as the array of its
/// values.
fn write_key(
    keys: &ArrayRef,
    index: usize,
    text: &mut String,
    ctx: &Ctx<'_>,
) -> Result<(), KrishivError> {
    match keys.data_type() {
        ArrowType::Utf8View | ArrowType::Utf8 | ArrowType::LargeUtf8 => {
            text.push_str(&col_str(keys, index));
            Ok(())
        }
        ArrowType::BinaryView | ArrowType::Binary | ArrowType::LargeBinary => {
            text.push_str(&base64::encode(&col_bytes(keys, index)));
            Ok(())
        }
        ArrowType::Struct(_) => {
            let st = keys.as_struct();
            text.push('[');
            for (i, child) in st.columns().iter().enumerate() {
                if i > 0 {
                    text.push(',');
                }
                write_value(child, index, text, ctx)?;
            }
            text.push(']');
            Ok(())
        }
        _ => write_value(keys, index, text, ctx),
    }
}
