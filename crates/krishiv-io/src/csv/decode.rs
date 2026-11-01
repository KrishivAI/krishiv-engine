// SPDX-License-Identifier: Apache-2.0
//! The CSV decoder: lines (or a whole file) into batches, with Spark's
//! header, comment, empty-line and parse-mode rules.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_schema::SchemaRef;
use bytes::Bytes;
use krishiv_common::error::{
    KrishivError, MalformedCsvRecord, MalformedRecordInParsingWithoutSuggestion,
};
use krishiv_kernels::datetime_pattern::Purpose;

use super::convert::{Column, Conv, Converter, Kind, Val};
use super::options::{CsvOptions, Mode};
use super::tokenizer::{Token, TokenizeError, Tokenizer};
use crate::error::{FormatError, FormatResult};
use crate::format::{Decoder, ScanUnitMetrics, Step, UnitPlan};
use crate::io::ByteRange;
use crate::request::{ROW_INDEX_COLUMN, ScanRequest};
use crate::text::reader::{LineReader, ReadChunk, WholeReader};
use crate::text::{BATCH_ROWS, Datetimes, TextMetadata};

/// What one output column is.
enum Out {
    /// The column of malformed records.
    Corrupt,
    /// A value from the record's token `index`.
    Data { index: usize, kind: Kind },
    /// `_metadata.row_index`: the record's position among the file's rows.
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

/// The next record of a source.
enum Next {
    /// Its raw text and its tokens (or why it has none).
    Record(String, Result<Vec<Token>, TokenizeError>),
    NeedData(Range<u64>),
    End,
}

pub(crate) struct CsvDecoder {
    source: Source,
    tok: Tokenizer,
    o: CsvOptions,
    dt: Datetimes,
    outs: Vec<Out>,
    columns: Vec<Column>,
    schema: SchemaRef,
    /// Names of the data columns, for the header check.
    data_names: Vec<String>,
    /// More than the requested columns can be dropped from tokens.
    pruned: bool,
    header_pending: bool,
    /// Reads the lines before a split that starts after byte 0, to count
    /// the rows `_metadata.row_index` continues from (only then).
    prefix: Option<LineReader>,
    /// Rows accepted so far in the file, prefix included.
    row_no: i64,
    rows: usize,
    held_text: usize,
    done: bool,
    metrics: ScanUnitMetrics,
}

/// Spark's `filterCommentAndEmpty`: lines that are white space only or start
/// with the comment character carry no record.
pub(super) fn skip_line(o: &CsvOptions, line: &str) -> bool {
    line.trim_matches(|c: char| c <= ' ').is_empty()
        || o.syntax.comment.is_some_and(|c| line.starts_with(c))
}

fn malformed_count(raw: &str) -> KrishivError {
    MalformedCsvRecord {
        bad_record: raw.to_owned(),
    }
    .into()
}

/// A malformed record in `FAILFAST` mode or a record that cannot be
/// tokenized: `FAILED_READ_FILE.NO_HINT` caused by the classified error.
fn failed(cause: KrishivError) -> FormatError {
    FormatError::Failed(cause)
}

fn fail_fast(raw: &str, cause: Option<KrishivError>) -> FormatError {
    let e: KrishivError = MalformedRecordInParsingWithoutSuggestion {
        bad_record: raw.to_owned(),
        fail_fast_mode: "FAILFAST".to_owned(),
    }
    .into();
    failed(match cause {
        Some(c) => e.with_cause(c),
        None => e,
    })
}

impl CsvDecoder {
    pub(crate) fn new(
        m: &Arc<dyn crate::format::FileMetadata>,
        units: &UnitPlan,
        req: &ScanRequest,
        o: &crate::options::FormatOptions,
    ) -> FormatResult<Self> {
        let m: TextMetadata = crate::text::downcast(m)?;
        let opts = CsvOptions::new(o, false)?;
        let dt = Datetimes::new(o, Purpose::Parse)?;
        let schema = req.output_schema()?;
        let table = &req.table_schema;
        // Token positions: the table's columns without the corrupt-record one.
        let data: Vec<&str> = table
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .filter(|n| *n != opts.corrupt_column)
            .collect();
        let mut outs = Vec::with_capacity(schema.fields().len());
        let mut columns = Vec::with_capacity(schema.fields().len());
        for f in schema.fields() {
            columns.push(Column::new(f.data_type())?);
            if f.name() == ROW_INDEX_COLUMN {
                outs.push(Out::RowIndex);
            } else if f.name() == &opts.corrupt_column {
                outs.push(Out::Corrupt);
            } else {
                let index = data
                    .iter()
                    .position(|n| *n == f.name())
                    .ok_or_else(|| crate::error::internal(format!("no column {}", f.name())))?;
                outs.push(Out::Data {
                    index,
                    kind: Kind::of(f.data_type())?,
                });
            }
        }
        let required = outs
            .iter()
            .filter(|c| matches!(c, Out::Data { .. }))
            .count();
        let pruned = opts.column_pruning && required < data.len();
        let split = units.units.first().map_or(ByteRange::new(0, 0), |u| {
            ByteRange::new(u.bytes.start, u.bytes.end)
        });
        let source = if opts.multi_line {
            Source::Whole {
                reader: WholeReader::new(m.size, split).compressed(m.codec),
                text: None,
                pos: 0,
            }
        } else {
            Source::Lines(
                LineReader::new(m.size, split, true, opts.line_sep.clone()).compressed(m.codec),
            )
        };
        let metrics = ScanUnitMetrics {
            units: units.units.len() as u64,
            ..ScanUnitMetrics::default()
        };
        let prefix =
            (req.metadata_cols.row_index && split.start > 0 && !opts.multi_line).then(|| {
                LineReader::new(
                    m.size,
                    ByteRange::new(0, split.start),
                    true,
                    opts.line_sep.clone(),
                )
            });
        Ok(Self {
            tok: Tokenizer::new(opts.syntax.clone()),
            header_pending: opts.header && (split.start == 0 || prefix.is_some()),
            prefix,
            row_no: 0,
            done: units.units.is_empty(),
            source,
            data_names: data.iter().map(|s| (*s).to_owned()).collect(),
            o: opts,
            dt,
            outs,
            columns,
            schema,
            pruned,
            rows: 0,
            held_text: 0,
            metrics,
        })
    }

    fn skip_line(&self, line: &str) -> bool {
        skip_line(&self.o, line)
    }

    fn next(&mut self) -> FormatResult<Next> {
        loop {
            let prefix = &mut self.prefix;
            match &mut self.source {
                Source::Lines(main) => match prefix.as_mut().unwrap_or(main).next_line()? {
                    ReadChunk::NeedData(range) => return Ok(Next::NeedData(range)),
                    ReadChunk::End if prefix.is_some() => {
                        // The rows before the split are counted.
                        *prefix = None;
                        self.header_pending = false;
                        continue;
                    }
                    ReadChunk::End => return Ok(Next::End),
                    ReadChunk::Line(bytes, offset) => {
                        let bytes = if offset == 0 {
                            bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&bytes)
                        } else {
                            &bytes[..]
                        };
                        let line = self.o.charset.decode(bytes).into_owned();
                        if self.skip_line(&line) {
                            continue;
                        }
                        let mut tokens = Vec::new();
                        let res = self.tok.record(&line, 0, &mut tokens).map(|_| tokens);
                        return Ok(Next::Record(line, res));
                    }
                },
                Source::Whole { reader, text, pos } => {
                    if text.is_none() {
                        if let Some(r) = reader.need() {
                            return Ok(Next::NeedData(r));
                        }
                        match reader.take() {
                            Some(bytes) => {
                                let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&bytes);
                                *text = Some(self.o.charset.decode(bytes).into_owned());
                                *pos = 0;
                            }
                            None => return Ok(Next::End),
                        }
                    }
                    let Some(t) = text.as_deref() else {
                        return Ok(Next::End);
                    };
                    let Some((raw, tokens, end)) =
                        self.tok.next_record(t, *pos, self.o.syntax.comment)
                    else {
                        return Ok(Next::End);
                    };
                    *pos = end;
                    return Ok(Next::Record(raw.to_owned(), tokens));
                }
            }
        }
    }

    /// Checks the header line against the schema (`enforceSchema=false`).
    fn check_header(&self, tokens: &[Token]) -> FormatResult<()> {
        if self.o.enforce_schema {
            return Ok(());
        }
        let names: Vec<&str> = tokens.iter().map(|t| t.as_deref().unwrap_or("")).collect();
        let same = names.len() == self.data_names.len()
            && names
                .iter()
                .zip(&self.data_names)
                .all(|(a, b)| a.eq_ignore_ascii_case(b));
        if same {
            Ok(())
        } else {
            Err(FormatError::Corrupt(format!(
                "the CSV header {names:?} does not match the schema {:?}",
                self.data_names
            )))
        }
    }

    /// One record into the builders.
    fn add(&mut self, raw: &str, tokens: &Result<Vec<Token>, TokenizeError>) -> FormatResult<()> {
        let tokens = match tokens {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = ?e, "a CSV record cannot be tokenized");
                return Err(failed(malformed_count(raw)));
            }
        };
        if self.header_pending {
            self.header_pending = false;
            return self.check_header(tokens);
        }
        let mut bad = !self.pruned && tokens.len() != self.data_names.len();
        let count_bad = bad;
        let conv = Converter {
            o: &self.o,
            dt: &self.dt,
        };
        let mut vals = Vec::with_capacity(self.outs.len());
        for out in &self.outs {
            vals.push(match out {
                Out::Corrupt => Val::Null,
                Out::RowIndex => Val::I64(self.row_no),
                Out::Data { index, kind } => {
                    let tok = tokens.get(*index).and_then(|t| t.as_deref());
                    match conv.convert(kind, tok) {
                        Ok(v) => v,
                        Err(Conv::Malformed) => {
                            bad = true;
                            Val::Null
                        }
                        Err(Conv::Fatal(e)) => return Err(FormatError::Other(e)),
                    }
                }
            });
        }
        if bad {
            match self.o.mode {
                Mode::DropMalformed => return Ok(()),
                Mode::FailFast => {
                    return Err(fail_fast(raw, count_bad.then(|| malformed_count(raw))));
                }
                Mode::Permissive => {
                    for (out, v) in self.outs.iter().zip(&mut vals) {
                        if matches!(out, Out::Corrupt) {
                            *v = Val::Text(raw.to_owned());
                        }
                    }
                }
            }
        }
        self.row_no += 1;
        if self.prefix.is_some() {
            // Rows before the split are only counted.
            return Ok(());
        }
        for (col, v) in self.columns.iter_mut().zip(&vals) {
            if let Val::Text(s) = v {
                self.held_text += s.len();
            }
            col.append(v);
        }
        self.rows += 1;
        Ok(())
    }

    fn batch(&mut self) -> FormatResult<RecordBatch> {
        let arrays: Vec<ArrayRef> = self.columns.iter_mut().map(Column::finish).collect();
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

impl Decoder for CsvDecoder {
    fn step(&mut self) -> FormatResult<Step> {
        if self.done {
            return Ok(Step::Done);
        }
        loop {
            match self.next()? {
                Next::NeedData(r) => return Ok(Step::NeedData(vec![r])),
                Next::Record(raw, tokens) => {
                    self.add(&raw, &tokens)?;
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
            + self.rows * 8 * self.columns.len()
    }

    fn metrics(&self) -> ScanUnitMetrics {
        self.metrics
    }
}
