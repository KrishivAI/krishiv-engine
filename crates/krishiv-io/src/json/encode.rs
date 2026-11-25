// SPDX-License-Identifier: Apache-2.0
//! The JSON encoder: batches into JSON-lines text, with Spark's value
//! renderings (numbers as JSON numbers, binary as base64, dates and
//! timestamps quoted).

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    TimestampMicrosecondType,
};
use arrow_array::{Array, RecordBatch};
use arrow_buffer::ArrowNativeType;
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use krishiv_kernels::datetime_pattern::Purpose;
use krishiv_kernels::text::{java_decimal_string, java_double_string, java_float_string32};

use super::convert::{encode_base64, kind_of};
use super::options::JsonOptions;
use crate::csv::convert::{Kind, interval_text};
use crate::error::{FormatError, FormatResult};
use crate::format::{Encoder, WrittenFile};
use crate::options::FormatOptions;
use crate::stats::{FileStatistics, Precision};
use crate::text::{Datetimes, LineSep};

pub(crate) struct JsonEncoder {
    o: JsonOptions,
    dt: Datetimes,
    /// Encoded bytes not yet taken.
    out: Vec<u8>,
    /// Bytes taken.
    taken: u64,
    names: Vec<String>,
    /// The leaf kind of each column; `None` for a nested column (array, struct, map).
    kinds: Vec<Option<Kind>>,
    fields: Vec<Field>,
    line: Vec<u8>,
    text: String,
}

/// Appends `s` as a JSON string (quotes and escapes) to `out`.
fn push_quoted(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Whether the column `f` is an array, struct or map. A calendar interval and a Variant are
/// Arrow structs too, but Spark's JSON writer treats them as leaves it refuses.
fn is_nested(f: &Field) -> bool {
    match f.data_type() {
        DataType::List(_) | DataType::LargeList(_) | DataType::Map(..) => true,
        DataType::Struct(_) => matches!(
            krishiv_types::DataType::from_arrow_field(f),
            Ok(krishiv_types::DataType::Struct(_))
        ),
        _ => false,
    }
}

/// Checks every type inside `f` is writable, so a refusal comes at planning, not mid-write.
fn check_field(f: &Field) -> FormatResult<()> {
    if !is_nested(f) {
        return kind_of(f)
            .map(drop)
            .map_err(|_| crate::error::unsupported_column("JSON", f));
    }
    match f.data_type() {
        DataType::List(c) | DataType::LargeList(c) => check_field(c),
        DataType::Struct(fields) => fields.iter().try_for_each(|c| check_field(c)),
        DataType::Map(entry, _) => match entry.data_type() {
            // A binary key has no text Spark writes (`toString` of a byte array).
            DataType::Struct(kv)
                if kv.len() == 2
                    && !matches!(
                        kv[0].data_type(),
                        DataType::Binary | DataType::LargeBinary | DataType::BinaryView
                    ) =>
            {
                kv.iter().try_for_each(|c| check_field(c))
            }
            _ => Err(crate::error::unsupported_column("JSON", f)),
        },
        _ => Err(crate::error::unsupported_column("JSON", f)),
    }
}

impl JsonEncoder {
    pub(crate) fn new(schema: &Schema, o: &FormatOptions) -> FormatResult<Self> {
        let opts = JsonOptions::new(o, true)?;
        let dt = Datetimes::new(o, Purpose::Format)?;
        let kinds = schema
            .fields()
            .iter()
            .map(|f| {
                if is_nested(f) {
                    check_field(f)?;
                    return Ok(None);
                }
                kind_of(f)
                    .map(Some)
                    .map_err(|_| crate::error::unsupported_column("JSON", f))
            })
            .collect::<FormatResult<Vec<_>>>()?;
        Ok(Self {
            o: opts,
            dt,
            out: Vec::new(),
            taken: 0,
            names: schema.fields().iter().map(|f| f.name().clone()).collect(),
            kinds,
            fields: schema.fields().iter().map(|f| f.as_ref().clone()).collect(),
            line: Vec::new(),
            text: String::new(),
        })
    }

    fn eol(&self) -> Vec<u8> {
        match &self.o.line_sep {
            LineSep::Default => b"\n".to_vec(),
            LineSep::Custom(s) => s.clone(),
        }
    }

    /// One value as JSON, appended to `buf`; `None` for NULL.
    fn push_value(
        dt: &Datetimes,
        buf: &mut String,
        a: &dyn Array,
        kind: &Kind,
        i: usize,
    ) -> FormatResult<Option<()>> {
        if a.is_null(i) {
            return Ok(None);
        }
        let write_err = |e: krishiv_common::error::KrishivError| FormatError::Other(e);
        match (kind, a.data_type()) {
            (Kind::Null, _) => return Ok(None),
            (Kind::Bool, _) => {
                buf.push_str(if a.as_boolean().value(i) {
                    "true"
                } else {
                    "false"
                });
            }
            (Kind::Int(8), _) => {
                buf.push_str(&a.as_primitive::<Int8Type>().value(i).to_string());
            }
            (Kind::Int(16), _) => {
                buf.push_str(&a.as_primitive::<Int16Type>().value(i).to_string());
            }
            (Kind::Int(32), _) => {
                buf.push_str(&a.as_primitive::<Int32Type>().value(i).to_string());
            }
            (Kind::Int(_), _) => {
                buf.push_str(&a.as_primitive::<Int64Type>().value(i).to_string());
            }
            // Non-finite values are written as quoted strings (Jackson's
            // `QUOTE_NON_NUMERIC_NUMBERS`), which stay valid JSON.
            (Kind::F32, _) => {
                let v = a.as_primitive::<Float32Type>().value(i);
                let text = java_float_string32(v);
                if v.is_finite() {
                    buf.push_str(&text);
                } else {
                    push_quoted(buf, &text);
                }
            }
            (Kind::F64, _) => {
                let v = a.as_primitive::<Float64Type>().value(i);
                let text = java_double_string(v);
                if v.is_finite() {
                    buf.push_str(&text);
                } else {
                    push_quoted(buf, &text);
                }
            }
            (Kind::Decimal(_, s), _) => {
                buf.push_str(&java_decimal_string(
                    crate::csv::convert::decimal_at(a, i),
                    *s,
                ));
            }
            (Kind::Str, DataType::Utf8View) => push_quoted(buf, a.as_string_view().value(i)),
            (Kind::Str, DataType::Utf8) => {
                push_quoted(buf, a.as_string::<i32>().value(i));
            }
            (Kind::Str, _) => push_quoted(buf, a.as_string::<i64>().value(i)),
            (Kind::Bin, DataType::BinaryView) => {
                push_quoted(buf, &encode_base64(a.as_binary_view().value(i)));
            }
            (Kind::Bin, DataType::Binary) => {
                push_quoted(buf, &encode_base64(a.as_binary::<i32>().value(i)));
            }
            (Kind::Bin, _) => {
                push_quoted(buf, &encode_base64(a.as_binary::<i64>().value(i)));
            }
            (Kind::Date, _) => {
                let s = dt
                    .format_date(a.as_primitive::<Date32Type>().value(i))
                    .map_err(write_err)?;
                push_quoted(buf, &s);
            }
            (Kind::Timestamp, _) => {
                let s = dt
                    .format_timestamp(a.as_primitive::<TimestampMicrosecondType>().value(i))
                    .map_err(write_err)?;
                push_quoted(buf, &s);
            }
            (Kind::YearMonth(..) | Kind::DayTime(..), _) => {
                push_quoted(buf, &interval_text(kind, a, i).unwrap_or_default());
            }
            (Kind::TimestampNtz, _) => {
                let s = dt
                    .format_ntz(a.as_primitive::<TimestampMicrosecondType>().value(i))
                    .map_err(write_err)?;
                push_quoted(buf, &s);
            }
        }
        Ok(Some(()))
    }

    /// One value of any type, nested or not, as JSON in `buf`; `None` for NULL. Spark's
    /// `JacksonGenerator`: an array writes its NULL elements as `null`, a struct leaves out
    /// its NULL fields under `ignoreNullFields`, a map writes the text of its key as the
    /// field name and a NULL value as `null`.
    fn push_nested(
        dt: &Datetimes,
        ignore_null: bool,
        buf: &mut String,
        a: &dyn Array,
        field: &Field,
        i: usize,
    ) -> FormatResult<Option<()>> {
        if a.is_null(i) {
            return Ok(None);
        }
        let item = |buf: &mut String, child: &dyn Array, f: &Field, j: usize| {
            if is_nested(f) {
                Self::push_nested(dt, ignore_null, buf, child, f, j)
            } else {
                let kind = kind_of(f).map_err(|_| crate::error::unsupported_column("JSON", f))?;
                Self::push_value(dt, buf, child, &kind, j)
            }
        };
        let list =
            |buf: &mut String, f: &Field, values: &dyn Array, range: std::ops::Range<usize>| {
                buf.push('[');
                for (n, j) in range.enumerate() {
                    if n > 0 {
                        buf.push(',');
                    }
                    if item(buf, values, f, j)?.is_none() {
                        buf.push_str("null");
                    }
                }
                buf.push(']');
                Ok::<(), FormatError>(())
            };
        match a.data_type() {
            DataType::List(f) => {
                let l = a.as_list::<i32>();
                let o = l.value_offsets();
                list(
                    buf,
                    f,
                    l.values().as_ref(),
                    o[i].as_usize()..o[i + 1].as_usize(),
                )?;
            }
            DataType::LargeList(f) => {
                let l = a.as_list::<i64>();
                let o = l.value_offsets();
                list(
                    buf,
                    f,
                    l.values().as_ref(),
                    o[i].as_usize()..o[i + 1].as_usize(),
                )?;
            }
            DataType::Struct(fields) => {
                let st = a.as_struct();
                buf.push('{');
                let mut first = true;
                for (f, c) in fields.iter().zip(st.columns()) {
                    let mut frag = String::new();
                    let kept = item(&mut frag, c.as_ref(), f, i)?;
                    if kept.is_none() && ignore_null {
                        continue;
                    }
                    if !first {
                        buf.push(',');
                    }
                    first = false;
                    push_quoted(buf, f.name());
                    buf.push(':');
                    buf.push_str(if kept.is_some() { &frag } else { "null" });
                }
                buf.push('}');
            }
            DataType::Map(entry, _) => {
                let m = a.as_map();
                let o = m.value_offsets();
                let DataType::Struct(kv) = entry.data_type() else {
                    return Err(crate::error::unsupported_column("JSON", field));
                };
                buf.push('{');
                for (n, j) in (o[i].as_usize()..o[i + 1].as_usize()).enumerate() {
                    if n > 0 {
                        buf.push(',');
                    }
                    // A map key is the text of Spark's internal value (`toString`): a date is its
                    // day count and a timestamp its microseconds.
                    let keys = m.keys();
                    match keys.data_type() {
                        DataType::Date32 => push_quoted(
                            buf,
                            &keys.as_primitive::<Date32Type>().value(j).to_string(),
                        ),
                        DataType::Timestamp(..) => push_quoted(
                            buf,
                            &keys
                                .as_primitive::<TimestampMicrosecondType>()
                                .value(j)
                                .to_string(),
                        ),
                        _ => {
                            let mut key = String::new();
                            item(&mut key, keys.as_ref(), &kv[0], j)?;
                            if key.starts_with('"') {
                                buf.push_str(&key);
                            } else {
                                push_quoted(buf, &key);
                            }
                        }
                    }
                    buf.push(':');
                    if item(buf, m.values().as_ref(), &kv[1], j)?.is_none() {
                        buf.push_str("null");
                    }
                }
                buf.push('}');
            }
            _ => return Err(crate::error::unsupported_column("JSON", field)),
        }
        Ok(Some(()))
    }

    /// One row as a JSON object line.
    fn row(&mut self, batch: &RecordBatch, i: usize) -> FormatResult<()> {
        self.text.clear();
        self.text.push('{');
        let mut first = true;
        for (((name, kind), field), a) in self
            .names
            .iter()
            .zip(&self.kinds)
            .zip(&self.fields)
            .zip(batch.columns().iter())
        {
            let mut frag = String::new();
            let kept = match kind {
                Some(kind) => Self::push_value(&self.dt, &mut frag, a.as_ref(), kind, i)?,
                None => Self::push_nested(
                    &self.dt,
                    self.o.ignore_null_fields,
                    &mut frag,
                    a.as_ref(),
                    field,
                    i,
                )?,
            };
            if kept.is_none() && self.o.ignore_null_fields {
                continue;
            }
            if !first {
                self.text.push(',');
            }
            first = false;
            // The field name never needs re-escaping per row: quote it
            // into the line directly.
            let mut quoted = String::with_capacity(name.len() + 2);
            push_quoted(&mut quoted, name);
            self.text.push_str(&quoted);
            self.text.push(':');
            match kept {
                Some(()) => self.text.push_str(&frag),
                None => self.text.push_str("null"),
            }
        }
        self.text.push('}');
        self.line.clear();
        self.o.charset.encode(&self.text, &mut self.line);
        self.line.extend_from_slice(&self.eol());
        self.out.extend_from_slice(&self.line);
        Ok(())
    }
}

impl Encoder for JsonEncoder {
    fn write(&mut self, batch: &RecordBatch) -> FormatResult<()> {
        for i in 0..batch.num_rows() {
            self.row(batch, i)?;
        }
        Ok(())
    }

    fn take_bytes(&mut self) -> Option<Bytes> {
        let out = std::mem::take(&mut self.out);
        self.taken += out.len() as u64;
        (!out.is_empty()).then(|| Bytes::from(out))
    }

    fn finish(self: Box<Self>) -> FormatResult<WrittenFile> {
        let size = self.taken + self.out.len() as u64;
        Ok(WrittenFile::new(
            Bytes::from(self.out),
            FileStatistics {
                bytes: Precision::Exact(size),
                ..FileStatistics::default()
            },
        ))
    }

    fn buffered_bytes(&self) -> usize {
        self.out.capacity() + self.line.capacity() + self.text.capacity()
    }

    fn written_bytes(&self) -> u64 {
        self.taken + self.out.len() as u64
    }
}
