// SPDX-License-Identifier: Apache-2.0
//! The `text` decoder: lines (or a whole file) into the single `value`
//! column, with Hadoop's split rule.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_schema::{DataType, SchemaRef};
use bytes::Bytes;

use super::options::TextOptions;
use crate::csv::convert::{Column, Val};
use crate::error::{FormatError, FormatResult, internal};
use crate::format::{Decoder, ScanUnitMetrics, Step, UnitPlan};
use crate::io::ByteRange;
use crate::request::{ROW_INDEX_COLUMN, ScanRequest};
use crate::text::reader::{LineReader, ReadChunk, WholeReader};
use crate::text::{BATCH_ROWS, TextMetadata};

/// What one output column is.
enum Out {
    /// The `value` column.
    Value,
    /// `_metadata.row_index`: the row's position among the file's rows.
    RowIndex,
}

enum Source {
    Lines(LineReader),
    Whole {
        reader: WholeReader,
        text: Option<String>,
        pos: usize,
    },
}

/// The next row of a source.
enum Next {
    /// Its text.
    Row(String),
    NeedData(Range<u64>),
    End,
}

pub(crate) struct TextDecoder {
    source: Source,
    o: TextOptions,
    outs: Vec<Out>,
    column: Column,
    schema: SchemaRef,
    /// Reads the lines before a split that starts after byte 0, to count
    /// the rows `_metadata.row_index` continues from (line mode only).
    prefix: Option<LineReader>,
    /// Rows accepted so far in the file, prefix included.
    row_no: i64,
    rows: usize,
    held_text: usize,
    done: bool,
    metrics: ScanUnitMetrics,
}

impl TextDecoder {
    pub(crate) fn new(
        m: &Arc<dyn crate::format::FileMetadata>,
        units: &UnitPlan,
        req: &ScanRequest,
        o: &crate::options::FormatOptions,
    ) -> FormatResult<Self> {
        let m: TextMetadata = crate::text::downcast(m)?;
        let opts = TextOptions::new(o, false)?;
        let schema = req.output_schema()?;
        let table = &req.table_schema;
        let names: Vec<&str> = table.fields().iter().map(|f| f.name().as_str()).collect();
        let mut outs = Vec::with_capacity(schema.fields().len());
        for f in schema.fields() {
            if f.name() == ROW_INDEX_COLUMN {
                outs.push(Out::RowIndex);
            } else if f.name() == "value" && names.contains(&"value") {
                outs.push(Out::Value);
            } else {
                return Err(crate::error::FormatError::Other(internal(format!(
                    "no column {} in a text file",
                    f.name()
                ))));
            }
        }
        let column = Column::new(&DataType::Utf8View)?;
        let split = units.units.first().map_or(ByteRange::new(0, 0), |u| {
            ByteRange::new(u.bytes.start, u.bytes.end)
        });
        let source = if opts.wholetext {
            Source::Whole {
                reader: WholeReader::new(m.size, split),
                text: None,
                pos: 0,
            }
        } else {
            Source::Lines(LineReader::new(m.size, split, true, opts.line_sep.clone()))
        };
        let metrics = ScanUnitMetrics {
            units: units.units.len() as u64,
            ..ScanUnitMetrics::default()
        };
        let prefix =
            (req.metadata_cols.row_index && split.start > 0 && !opts.wholetext).then(|| {
                LineReader::new(
                    m.size,
                    ByteRange::new(0, split.start),
                    true,
                    opts.line_sep.clone(),
                )
            });
        Ok(Self {
            source,
            o: opts,
            outs,
            column,
            schema,
            prefix,
            row_no: 0,
            rows: 0,
            held_text: 0,
            done: units.units.is_empty(),
            metrics,
        })
    }

    fn next(&mut self) -> FormatResult<Next> {
        loop {
            if self.prefix.is_some() {
                let prefix = self.prefix.as_mut().ok_or_else(|| {
                    FormatError::Other(internal("a text prefix reader is missing"))
                })?;
                match prefix.next_line()? {
                    ReadChunk::NeedData(range) => return Ok(Next::NeedData(range)),
                    ReadChunk::End => {
                        self.prefix = None;
                        continue;
                    }
                    ReadChunk::Line(_, _) => {
                        self.row_no += 1;
                        continue;
                    }
                }
            }
            match &mut self.source {
                Source::Lines(main) => match main.next_line()? {
                    ReadChunk::NeedData(range) => return Ok(Next::NeedData(range)),
                    ReadChunk::End => return Ok(Next::End),
                    // Empty lines are empty values, as in Hadoop.
                    ReadChunk::Line(bytes, _) => {
                        let line = self.o.charset.decode(&bytes).into_owned();
                        return Ok(Next::Row(line));
                    }
                },
                Source::Whole { reader, text, pos } => {
                    if text.is_none() {
                        if let Some(r) = reader.need() {
                            return Ok(Next::NeedData(r));
                        }
                        match reader.take() {
                            Some(bytes) => {
                                *text = Some(self.o.charset.decode(&bytes).into_owned());
                                *pos = 0;
                            }
                            None => return Ok(Next::End),
                        }
                    }
                    let Some(t) = text.as_deref() else {
                        return Ok(Next::End);
                    };
                    if *pos > 0 {
                        return Ok(Next::End);
                    }
                    *pos = 1;
                    return Ok(Next::Row(t.to_owned()));
                }
            }
        }
    }

    /// One row into the builder.
    fn add(&mut self, row: &str) {
        self.row_no += 1;
        if self.prefix.is_some() {
            // Rows before the split are only counted.
            return;
        }
        // The column holds every row; the row index is derived below.
        self.held_text += row.len();
        self.column.append(&Val::Text(row.to_owned()));
        self.rows += 1;
    }

    fn batch(&mut self) -> FormatResult<RecordBatch> {
        let value = self.column.finish();
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(self.outs.len());
        for out in &self.outs {
            match out {
                Out::Value => arrays.push(Arc::clone(&value)),
                Out::RowIndex => {
                    let first = self.row_no - self.rows as i64;
                    let col: Vec<i64> = (first..self.row_no).collect();
                    arrays.push(Arc::new(arrow_array::Int64Array::from(col)));
                }
            }
        }
        let rows = std::mem::take(&mut self.rows);
        self.held_text = 0;
        self.metrics.rows += rows as u64;
        RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            arrays,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .map_err(|e| FormatError::Corrupt(e.to_string()))
    }
}

impl Decoder for TextDecoder {
    fn step(&mut self) -> FormatResult<Step> {
        if self.done {
            return Ok(Step::Done);
        }
        loop {
            match self.next()? {
                Next::NeedData(r) => return Ok(Step::NeedData(vec![r])),
                Next::Row(row) => {
                    self.add(&row);
                    if self.rows >= BATCH_ROWS {
                        return Ok(Step::Batch(self.batch()?));
                    }
                }
                Next::End => {
                    self.done = true;
                    return if self.rows > 0 {
                        Ok(Step::Batch(self.batch()?))
                    } else {
                        Ok(Step::Done)
                    };
                }
            }
        }
    }

    fn push(&mut self, data: Vec<(Range<u64>, Bytes)>) -> FormatResult<()> {
        for (range, bytes) in data {
            self.metrics.bytes_pushed += bytes.len() as u64;
            match (&mut self.prefix, &mut self.source) {
                (Some(p), _) => p.push(&range, &bytes)?,
                (None, Source::Lines(r)) => r.push(&range, &bytes)?,
                (None, Source::Whole { reader, .. }) => reader.push(&range, &bytes)?,
            }
        }
        Ok(())
    }

    fn upcoming(&self, max_bytes: u64) -> Vec<Range<u64>> {
        match (&self.prefix, &self.source) {
            (Some(p), _) => p.upcoming(max_bytes),
            (None, Source::Lines(r)) => r.upcoming(max_bytes),
            (None, Source::Whole { reader, .. }) => reader.upcoming(max_bytes),
        }
    }

    fn buffered_bytes(&self) -> usize {
        let source = match &self.source {
            Source::Lines(r) => r.buffered_bytes(),
            Source::Whole { reader, text, .. } => {
                reader.buffered_bytes() + text.as_ref().map_or(0, String::capacity)
            }
        };
        source
            + self.prefix.as_ref().map_or(0, LineReader::buffered_bytes)
            + self.held_text
            + self.rows * 8 * self.outs.len().max(1)
    }

    fn metrics(&self) -> ScanUnitMetrics {
        self.metrics
    }
}
