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
use crate::request::{FieldMapping, ROW_INDEX_COLUMN, ScanRequest};
use crate::text::reader::{LineReader, ReadChunk, WholeReader};
use crate::text::{BATCH_ROWS, Charset, Datetimes, TextMetadata};

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
    /// A record of a read with no columns, not tokenized.
    Bare,
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
    /// `spark.sql.caseSensitive` (from the request's by-name mapping), for
    /// the header check.
    case_sensitive: bool,
    /// More than the requested columns can be dropped from tokens.
    pruned: bool,
    /// Pruned with no output column (`count()`): as Spark's
    /// `UnivocityParser.parse`, a record is an empty row and is not
    /// tokenized, so no record is malformed.
    bare: bool,
    header_pending: bool,
    /// Reads the lines before a split that starts after byte 0, to count
    /// the rows `_metadata.row_index` continues from (only then).
    prefix: Option<LineReader>,
    /// Rows accepted so far in the file, prefix included.
    row_no: i64,
    rows: usize,
    held_text: usize,
    done: bool,
    /// A reader error held back while the rows read before it are returned.
    failed: Option<FormatError>,
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
        let dt = Datetimes::csv(o, Purpose::Parse)?;
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
                    kind: Kind::of_field(f)?,
                });
            }
        }
        let required = outs
            .iter()
            .filter(|c| matches!(c, Out::Data { .. }))
            .count();
        let pruned = opts.column_pruning && required < data.len();
        // Lines are counted without tokenizing only when nothing is output:
        // a read of `_metadata.row_index` alone still tokenizes each line, so
        // it can fail under FAILFAST or `maxCharsPerColumn` where `count()`
        // does not (Spark has no CSV row index to compare with).
        let bare = pruned && outs.is_empty() && !opts.multi_line;
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
            failed: None,
            source,
            data_names: data.iter().map(|s| (*s).to_owned()).collect(),
            case_sensitive: matches!(
                req.mapping,
                FieldMapping::ByName {
                    case_sensitive: true
                }
            ),
            o: opts,
            dt,
            outs,
            columns,
            schema,
            pruned,
            bare,
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
                        if self.bare && !self.header_pending {
                            if skip_bare(&self.o, &bytes, offset == 0) {
                                continue;
                            }
                            return Ok(Next::Bare);
                        }
                        let line = line_text(self.o.charset, bytes, offset == 0);
                        if self.skip_line(&line) {
                            continue;
                        }
                        let mut tokens = Vec::with_capacity(self.data_names.len());
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

    /// Checks the header line against the schema (`enforceSchema=false`),
    /// as Spark 4.2's `CSVHeaderChecker`: names compare exactly under
    /// `spark.sql.caseSensitive`, else after lower-casing (Unicode, as
    /// Java's `toLowerCase`). With column pruning only the read columns
    /// are checked, each against the header token at its own position, and
    /// the header's length is not; without it the whole header is.
    fn check_header(&self, tokens: &[Token]) -> FormatResult<()> {
        if self.o.enforce_schema {
            return Ok(());
        }
        let header: Vec<&str> = tokens.iter().map(|t| t.as_deref().unwrap_or("")).collect();
        // (header names, schema names) in the order Spark compares them.
        let (found, wanted): (Vec<&str>, Vec<&str>) = if self.pruned {
            // The read columns in file order, as Spark's required schema.
            let mut read: Vec<usize> = self
                .outs
                .iter()
                .filter_map(|o| match o {
                    Out::Data { index, .. } => Some(*index),
                    _ => None,
                })
                .collect();
            read.sort_unstable();
            read.dedup();
            read.iter()
                .map(|&i| {
                    (
                        header.get(i).copied().unwrap_or(""),
                        self.data_names.get(i).map_or("", String::as_str),
                    )
                })
                .unzip()
        } else {
            if header.len() != self.data_names.len() {
                return Err(FormatError::Corrupt(format!(
                    "Number of column in CSV header is not equal to number of fields in the schema:\n \
                     Header length: {}, schema size: {}",
                    header.len(),
                    self.data_names.len()
                )));
            }
            (header, self.data_names.iter().map(String::as_str).collect())
        };
        let same = |a: &str, b: &str| {
            if self.case_sensitive {
                a == b
            } else {
                a == b || a.to_lowercase() == b.to_lowercase()
            }
        };
        match found.iter().zip(&wanted).find(|(h, s)| !same(h, s)) {
            None => Ok(()),
            Some((h, s)) => Err(FormatError::Corrupt(format!(
                "CSV header does not conform to the schema.\n Header: {}\n Schema: {}\nExpected: {s} but found: {h}",
                found.join(", "),
                wanted.join(", ")
            ))),
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

/// [`skip_line`] on a line's bytes when the comment character is ASCII (in
/// UTF-8, ISO-8859-1 and US-ASCII the characters up to a space and an ASCII
/// comment character are those bytes, and no other byte decodes to them);
/// otherwise on its text.
fn skip_bare(o: &CsvOptions, bytes: &[u8], first: bool) -> bool {
    let bytes = if first {
        bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes)
    } else {
        bytes
    };
    match o.syntax.comment {
        Some(c) if !c.is_ascii() => skip_line(o, &o.charset.decode(bytes)),
        comment => {
            bytes.iter().all(|b| *b <= b' ')
                || comment.is_some_and(|c| bytes.first() == Some(&(c as u8)))
        }
    }
}

/// A line's text: the bytes themselves when they are UTF-8 in a UTF-8 file
/// (no copy), else decoded by `charset`; the first line of a file drops a
/// UTF-8 byte order mark.
fn line_text(charset: Charset, bytes: Vec<u8>, first: bool) -> String {
    const BOM: &[u8] = b"\xEF\xBB\xBF";
    if first && bytes.starts_with(BOM) {
        return charset.decode(&bytes[BOM.len()..]).into_owned();
    }
    if charset == Charset::Utf8 {
        return match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
        };
    }
    charset.decode(&bytes).into_owned()
}

impl Decoder for CsvDecoder {
    fn step(&mut self) -> FormatResult<Step> {
        if let Some(e) = self.failed.take() {
            self.done = true;
            return Err(e);
        }
        if self.done {
            return Ok(Step::Done);
        }
        loop {
            let next = match self.next() {
                Ok(next) => next,
                // The rows read before a reader error (a compressed stream
                // that breaks) are returned first, as Spark's scan has handed
                // them on before the error reaches it.
                Err(e) if self.rows > 0 => {
                    self.failed = Some(e);
                    return Ok(Step::Batch(self.batch()?));
                }
                Err(e) => return Err(e),
            };
            match next {
                Next::NeedData(r) => return Ok(Step::NeedData(vec![r])),
                Next::Record(raw, tokens) => {
                    self.add(&raw, &tokens)?;
                    if self.rows >= BATCH_ROWS {
                        return Ok(Step::Batch(self.batch()?));
                    }
                }
                Next::Bare => {
                    self.row_no += 1;
                    if self.prefix.is_none() {
                        self.rows += 1;
                        if self.rows >= BATCH_ROWS {
                            return Ok(Step::Batch(self.batch()?));
                        }
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
