// SPDX-License-Identifier: Apache-2.0
//! The `text` encoder: one string column into lines.

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema};
use bytes::Bytes;

use super::convert::str_value;
use super::options::TextOptions;
use crate::error::{FormatError, FormatResult, internal};
use crate::format::{Encoder, WrittenFile};
use crate::options::FormatOptions;
use crate::stats::{FileStatistics, Precision};
use crate::text::LineSep;

pub(crate) struct TextEncoder {
    o: TextOptions,
    /// Encoded bytes not yet taken.
    out: Vec<u8>,
    /// Bytes taken.
    taken: u64,
}

impl TextEncoder {
    pub(crate) fn new(schema: &Schema, o: &FormatOptions) -> FormatResult<Self> {
        let opts = TextOptions::new(o, true)?;
        if schema.fields().len() != 1 {
            return Err(FormatError::Other(
                krishiv_common::error::LegacyErrorTemp1290 {
                    schema_size: schema.fields().len().to_string(),
                }
                .into(),
            ));
        }
        let f = &schema.fields()[0];
        if !matches!(
            f.data_type(),
            DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8
        ) {
            return Err(FormatError::Other(crate::error::unsupported(format!(
                "the TEXT data source with a column of type {} (only a single string column can be written)",
                crate::text::spark_type_name(f.data_type())
            ))));
        }
        Ok(Self {
            o: opts,
            out: Vec::new(),
            taken: 0,
        })
    }
}

impl Encoder for TextEncoder {
    fn write(&mut self, batch: &RecordBatch) -> FormatResult<()> {
        let Some(a) = batch.columns().first() else {
            return Err(FormatError::Other(internal("a text batch without columns")));
        };
        for i in 0..batch.num_rows() {
            // A NULL is an empty line, as Spark's text writer writes it.
            if let Some(v) = str_value(a.as_ref(), i) {
                self.o.charset.encode(v, &mut self.out);
            }
            let eol = match &self.o.line_sep {
                LineSep::Default => b"\n".as_slice(),
                LineSep::Custom(s) => s.as_slice(),
            };
            self.out.extend_from_slice(eol);
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
        self.out.capacity()
    }

    fn written_bytes(&self) -> u64 {
        self.taken + self.out.len() as u64
    }
}
