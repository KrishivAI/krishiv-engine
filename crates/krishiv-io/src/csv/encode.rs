// SPDX-License-Identifier: Apache-2.0
//! The CSV encoder: batches into text, with Spark's quoting rules.

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    TimestampMicrosecondType,
};
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, Schema};
use bytes::Bytes;
use krishiv_kernels::datetime_pattern::Purpose;
use krishiv_kernels::text::{java_decimal_string, java_double_string, java_float_string32};

use super::convert::Kind;
use super::options::{CsvOptions, Syntax};
use crate::error::{FormatError, FormatResult};
use crate::format::{Encoder, WrittenFile};
use crate::options::FormatOptions;
use crate::stats::{FileStatistics, Precision};
use crate::text::{Datetimes, LineSep};

pub(crate) struct CsvEncoder {
    o: CsvOptions,
    dt: Datetimes,
    /// Encoded bytes not yet taken.
    out: Vec<u8>,
    /// Bytes taken.
    taken: u64,
    kinds: Vec<Kind>,
    line: Vec<u8>,
    text: String,
}

/// Spark's text of a binary value: `[AB 01 FF]`.
fn binary_text(b: &[u8], out: &mut String) {
    out.push('[');
    for (i, byte) in b.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(&format!("{byte:02X}"));
    }
    out.push(']');
}

impl CsvEncoder {
    pub(crate) fn new(schema: &Schema, o: &FormatOptions) -> FormatResult<Self> {
        let opts = CsvOptions::new(o, true)?;
        let dt = Datetimes::new(o, Purpose::Format)?;
        let kinds = schema
            .fields()
            .iter()
            .map(|f| {
                Kind::of(f.data_type()).map_err(|_| {
                    FormatError::Other(
                        krishiv_common::error::UnsupportedDataTypeForDatasource {
                            format: "CSV".to_owned(),
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
        let mut enc = Self {
            out: Vec::new(),
            taken: 0,
            o: opts,
            dt,
            kinds,
            line: Vec::new(),
            text: String::new(),
        };
        if enc.o.header {
            let names: Vec<Option<String>> = schema
                .fields()
                .iter()
                .map(|f| Some(f.name().clone()))
                .collect();
            enc.row(names.iter().map(|n| n.as_deref()))?;
        }
        Ok(enc)
    }

    fn eol(&self) -> Vec<u8> {
        match &self.o.line_sep {
            LineSep::Default => b"\n".to_vec(),
            LineSep::Custom(s) => s.clone(),
        }
    }

    /// One value as the file shows it, appended to `self.text`.
    fn push_value(&mut self, value: Option<&str>) {
        let s: &Syntax = &self.o.syntax;
        // Without a quote character Univocity quotes with NUL.
        let quote = s.quote.unwrap_or('\0');
        let (text, is_null) = match value {
            None => (self.o.null_value.as_str(), true),
            Some(v) => (v, false),
        };
        let mut v = text;
        if s.ltrim {
            v = v.trim_start_matches(|c: char| c <= ' ');
        }
        if s.rtrim {
            v = v.trim_end_matches(|c: char| c <= ' ');
        }
        if v.is_empty() {
            if is_null {
                if self.o.quote_all {
                    self.text.push(quote);
                    self.text.push(quote);
                }
            } else {
                self.text.push_str(&s.empty_value);
            }
            return;
        }
        let needs = self.o.quote_all
            || v.starts_with(quote)
            || v.contains(s.sep.as_str())
            || v.contains(['\n', '\r'])
            || (self.o.escape_quotes && v.contains(quote))
            || matches!(&self.o.line_sep, LineSep::Custom(_))
                && self.eol_text().is_some_and(|e| v.contains(e.as_str()));
        if !needs {
            self.text.push_str(v);
            return;
        }
        self.text.push(quote);
        let eol = self.eol_text();
        for c in v.chars() {
            if c == '\n'
                && let Some(eol) = &eol
            {
                // A line feed in a value is written as the line separator.
                self.text.push_str(eol);
                continue;
            }
            if c == quote {
                if let Some(e) = s.escape {
                    self.text.push(e);
                }
            } else if Some(c) == s.escape
                && let Some(ee) = s.escape_escape
            {
                self.text.push(ee);
            }
            self.text.push(c);
        }
        self.text.push(quote);
    }

    fn eol_text(&self) -> Option<String> {
        match &self.o.line_sep {
            LineSep::Custom(b) => String::from_utf8(b.clone()).ok(),
            LineSep::Default => None,
        }
    }

    fn row<'a>(&mut self, values: impl Iterator<Item = Option<&'a str>>) -> FormatResult<()> {
        self.text.clear();
        for (i, v) in values.enumerate() {
            if i > 0 {
                self.text.push_str(&self.o.syntax.sep.clone());
            }
            self.push_value(v);
        }
        self.line.clear();
        self.o.charset.encode(&self.text, &mut self.line);
        self.line.extend_from_slice(&self.eol());
        self.out.extend_from_slice(&self.line);
        Ok(())
    }

    /// The text of row `i` of column `a`, `None` for NULL.
    fn cell(&self, a: &dyn Array, kind: &Kind, i: usize, buf: &mut String) -> FormatResult<bool> {
        if a.is_null(i) {
            return Ok(false);
        }
        let write_err = |e: krishiv_common::error::KrishivError| FormatError::Other(e);
        match (kind, a.data_type()) {
            (Kind::Null, _) => return Ok(false),
            (Kind::Bool, _) => buf.push_str(if a.as_boolean().value(i) {
                "true"
            } else {
                "false"
            }),
            (Kind::Int(8), _) => buf.push_str(&a.as_primitive::<Int8Type>().value(i).to_string()),
            (Kind::Int(16), _) => buf.push_str(&a.as_primitive::<Int16Type>().value(i).to_string()),
            (Kind::Int(32), _) => buf.push_str(&a.as_primitive::<Int32Type>().value(i).to_string()),
            (Kind::Int(_), _) => buf.push_str(&a.as_primitive::<Int64Type>().value(i).to_string()),
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
                buf.push_str(&java_decimal_string(super::convert::decimal_at(a, i), *s));
            }
            (Kind::Str, DataType::Utf8View) => buf.push_str(a.as_string_view().value(i)),
            (Kind::Str, DataType::Utf8) => buf.push_str(a.as_string::<i32>().value(i)),
            (Kind::Str, _) => buf.push_str(a.as_string::<i64>().value(i)),
            (Kind::Bin, DataType::BinaryView) => binary_text(a.as_binary_view().value(i), buf),
            (Kind::Bin, DataType::Binary) => binary_text(a.as_binary::<i32>().value(i), buf),
            (Kind::Bin, _) => binary_text(a.as_binary::<i64>().value(i), buf),
            (Kind::Date, _) => buf.push_str(
                &self
                    .dt
                    .format_date(a.as_primitive::<Date32Type>().value(i))
                    .map_err(write_err)?,
            ),
            (Kind::Timestamp, _) => buf.push_str(
                &self
                    .dt
                    .format_timestamp(a.as_primitive::<TimestampMicrosecondType>().value(i))
                    .map_err(write_err)?,
            ),
            (Kind::TimestampNtz, _) => buf.push_str(
                &self
                    .dt
                    .format_ntz(a.as_primitive::<TimestampMicrosecondType>().value(i))
                    .map_err(write_err)?,
            ),
        }
        Ok(true)
    }
}

impl Encoder for CsvEncoder {
    fn write(&mut self, batch: &RecordBatch) -> FormatResult<()> {
        let mut cell = String::new();
        let mut row: Vec<Option<String>> = Vec::with_capacity(batch.num_columns());
        for i in 0..batch.num_rows() {
            row.clear();
            for (a, kind) in batch.columns().iter().zip(&self.kinds) {
                cell.clear();
                row.push(
                    self.cell(a.as_ref(), kind, i, &mut cell)?
                        .then(|| cell.clone()),
                );
            }
            self.row(row.iter().map(Option::as_deref))?;
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
