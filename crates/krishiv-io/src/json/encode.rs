// SPDX-License-Identifier: Apache-2.0
//! The JSON encoder: batches into JSON-lines text, with Spark's value
//! renderings (numbers as JSON numbers, binary as base64, dates and
//! timestamps quoted).

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, Schema};
use bytes::Bytes;
use krishiv_kernels::datetime_pattern::Purpose;
use krishiv_kernels::text::{java_decimal_string, java_double_string, java_float_string32};

use super::convert::{encode_base64, kind_of};
use super::options::JsonOptions;
use crate::csv::convert::Kind;
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
    kinds: Vec<Kind>,
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

impl JsonEncoder {
    pub(crate) fn new(schema: &Schema, o: &FormatOptions) -> FormatResult<Self> {
        let opts = JsonOptions::new(o, true)?;
        let dt = Datetimes::new(o, Purpose::Format)?;
        let kinds = schema
            .fields()
            .iter()
            .map(|f| {
                kind_of(f.data_type()).map_err(|_| {
                    FormatError::Other(
                        krishiv_common::error::UnsupportedDataTypeForDatasource {
                            format: "JSON".to_owned(),
                            column_name: format!("`{}`", f.name()),
                            column_type: format!(
                                "\"{}\"",
                                crate::text::spark_type_name(f.data_type())
                            ),
                        }
                        .into(),
                    )
                })
            })
            .collect::<FormatResult<Vec<_>>>()?;
        Ok(Self {
            o: opts,
            dt,
            out: Vec::new(),
            taken: 0,
            names: schema.fields().iter().map(|f| f.name().clone()).collect(),
            kinds,
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
            (Kind::F32, _) => {
                buf.push_str(&java_float_string32(
                    a.as_primitive::<Float32Type>().value(i),
                ));
            }
            (Kind::F64, _) => {
                buf.push_str(&java_double_string(
                    a.as_primitive::<Float64Type>().value(i),
                ));
            }
            (Kind::Decimal(_, s), _) => {
                buf.push_str(&java_decimal_string(
                    a.as_primitive::<Decimal128Type>().value(i),
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
            (Kind::TimestampNtz, _) => {
                let s = dt
                    .format_ntz(a.as_primitive::<TimestampMicrosecondType>().value(i))
                    .map_err(write_err)?;
                push_quoted(buf, &s);
            }
        }
        Ok(Some(()))
    }

    /// One row as a JSON object line.
    fn row(&mut self, batch: &RecordBatch, i: usize) -> FormatResult<()> {
        self.text.clear();
        self.text.push('{');
        let mut first = true;
        for ((name, kind), a) in self
            .names
            .iter()
            .zip(&self.kinds)
            .zip(batch.columns().iter())
        {
            let mut frag = String::new();
            let kept = Self::push_value(&self.dt, &mut frag, a.as_ref(), kind, i)?;
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
