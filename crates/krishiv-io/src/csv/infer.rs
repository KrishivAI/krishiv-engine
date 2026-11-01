// SPDX-License-Identifier: Apache-2.0
//! Schema inference (Spark's `CSVInferSchema`), with `samplingRatio`
//! reproducing Spark's row sampling.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use futures::future::BoxFuture;
use krishiv_kernels::datetime_pattern::{DateTimePattern, Purpose};
use krishiv_kernels::text::{java_parse_double, java_parse_long};

use super::decode::skip_line;
use super::options::CsvOptions;
use super::sampling::Sampler;
use super::tokenizer::{Eol, Token, TokenizeError, Tokenizer};
use crate::error::{FormatError, FormatResult, internal};
use crate::io::{ByteRange, FileRef, IoContext};
use crate::options::FormatOptions;
use crate::split::{SplitConfig, SplitFile, plan};
use crate::text::Datetimes;
use crate::text::reader::{LineReader, ReadChunk, WholeReader};

/// The types inference moves through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum T {
    Null,
    Int,
    Long,
    Decimal(u8),
    Double,
    Date,
    Timestamp,
    Boolean,
    Str,
}

/// Spark's `TypeCoercion.findTightestCommonType` and
/// `CSVInferSchema.findCompatibleTypeForCSV`.
fn compatible(a: T, b: T) -> T {
    use T::{Date, Decimal, Double, Int, Long, Null, Str, Timestamp};
    if a == b {
        return a;
    }
    match (a, b) {
        (Null, t) | (t, Null) => t,
        (Str, _) | (_, Str) => Str,
        (Int, Long) | (Long, Int) => Long,
        (Int | Long | Decimal(_), Double) | (Double, Int | Long | Decimal(_)) => Double,
        (Decimal(p), Decimal(q)) => Decimal(p.max(q)),
        (Int, Decimal(p)) | (Decimal(p), Int) if p >= 10 => Decimal(p),
        (Long, Decimal(p)) | (Decimal(p), Long) if p >= 20 => Decimal(p),
        (Date, Timestamp) | (Timestamp, Date) => Timestamp,
        _ => T::Str,
    }
}

/// `new java.math.BigDecimal(s)`: precision and scale of a decimal literal.
fn big_decimal_shape(s: &str) -> Option<(i64, i64)> {
    let b = s.as_bytes();
    let mut i = usize::from(matches!(b.first(), Some(b'+' | b'-')));
    let (mut int, mut frac) = (0_i64, 0_i64);
    let mut digits = String::new();
    while i < b.len() && b[i].is_ascii_digit() {
        digits.push(char::from(b[i]));
        int += 1;
        i += 1;
    }
    if b.get(i) == Some(&b'.') {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            digits.push(char::from(b[i]));
            frac += 1;
            i += 1;
        }
    }
    if int + frac == 0 {
        return None;
    }
    let mut exp = 0_i64;
    if matches!(b.get(i), Some(b'e' | b'E')) {
        i += 1;
        let neg = b.get(i) == Some(&b'-');
        i += usize::from(matches!(b.get(i), Some(b'+' | b'-')));
        let from = i;
        let mut v = 0_i64;
        while i < b.len() && b[i].is_ascii_digit() {
            v = v.checked_mul(10)?.checked_add(i64::from(b[i] - b'0'))?;
            i += 1;
        }
        if i == from {
            return None;
        }
        exp = if neg { -v } else { v };
    }
    if i != b.len() {
        return None;
    }
    let stripped = digits.trim_start_matches('0');
    let precision = i64::try_from(stripped.len()).ok()?.max(1);
    Some((precision, frac.checked_sub(exp)?))
}

struct Infer<'a> {
    o: &'a CsvOptions,
    dt: &'a Datetimes,
    /// The strict date pattern (`dateFormat`, else `yyyy-MM-dd`).
    date: DateTimePattern,
}

impl Infer<'_> {
    fn field(&self, so_far: T, f: &str) -> T {
        if f.is_empty() || f == self.o.null_value {
            return so_far;
        }
        let t = match so_far {
            T::Null | T::Int => self.integer(f),
            T::Long => self.long(f),
            T::Decimal(_) => self.decimal(f),
            T::Double => self.double(f),
            T::Date => self.date_time(f),
            T::Timestamp => self.timestamp(f),
            T::Boolean => Self::boolean(f),
            T::Str => T::Str,
        };
        compatible(so_far, t)
    }

    fn integer(&self, f: &str) -> T {
        if krishiv_kernels::text::java_parse_integral(f, 32).is_some() {
            T::Int
        } else {
            self.long(f)
        }
    }

    fn long(&self, f: &str) -> T {
        if java_parse_long(f).is_some() {
            T::Long
        } else {
            self.decimal(f)
        }
    }

    fn decimal(&self, f: &str) -> T {
        let shaped = big_decimal_shape(&f.replace(',', ""));
        match shaped {
            Some((p, 0)) if p <= 38 => T::Decimal(u8::try_from(p).unwrap_or(38)),
            _ => self.double(f),
        }
    }

    fn double(&self, f: &str) -> T {
        if java_parse_double(f).is_some()
            || f == self.o.nan
            || f == self.o.neg_inf
            || f == self.o.pos_inf
        {
            T::Double
        } else {
            self.date_time(f)
        }
    }

    fn date_time(&self, f: &str) -> T {
        if matches!(self.date.parse_date(f), Ok(Some(_))) {
            T::Date
        } else {
            self.timestamp(f)
        }
    }

    fn timestamp(&self, f: &str) -> T {
        if matches!(self.dt.parse_timestamp(f), Ok(Some(_))) {
            T::Timestamp
        } else {
            Self::boolean(f)
        }
    }

    fn boolean(f: &str) -> T {
        if f.eq_ignore_ascii_case("true") || f.eq_ignore_ascii_case("false") {
            T::Boolean
        } else {
            T::Str
        }
    }

    fn row(&self, types: &mut [T], tokens: &[Token]) {
        for (t, tok) in types.iter_mut().zip(tokens) {
            *t = self.field(*t, tok.as_deref().unwrap_or(""));
        }
    }
}

/// Spark's `CSVUtils.makeSafeHeader`.
fn safe_header(tokens: &[Token], null_value: &str) -> Vec<String> {
    let lower = |s: &str| s.to_lowercase();
    let mut counts = std::collections::HashMap::<String, usize>::new();
    for t in tokens.iter().flatten() {
        *counts.entry(lower(t)).or_default() += 1;
    }
    tokens
        .iter()
        .enumerate()
        .map(|(i, t)| match t.as_deref() {
            None | Some("") => format!("_c{i}"),
            Some(v) if v == null_value => format!("_c{i}"),
            Some(v) if counts.get(&lower(v)).copied().unwrap_or(0) > 1 => format!("{v}{i}"),
            Some(v) => v.to_owned(),
        })
        .collect()
}

/// The column names: the header's (made safe), else `_c0`, `_c1`...
fn column_names(o: &CsvOptions, first: &[Token]) -> Vec<String> {
    if o.header {
        safe_header(first, &o.null_value)
    } else {
        (0..first.len()).map(|i| format!("_c{i}")).collect()
    }
}

fn arrow_type(t: T) -> DataType {
    match t {
        T::Int => DataType::Int32,
        T::Long => DataType::Int64,
        T::Decimal(p) => DataType::Decimal128(p, 0),
        T::Double => DataType::Float64,
        T::Date => DataType::Date32,
        T::Timestamp => DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        T::Boolean => DataType::Boolean,
        T::Null | T::Str => DataType::Utf8View,
    }
}

/// Spark's DDL name of an inferred column type, as `StructType.sql` spells
/// it (`INT`, `DECIMAL(10,0)`, `STRING`, ...).
pub(crate) fn ddl_type(t: &DataType) -> FormatResult<String> {
    Ok(match t {
        DataType::Int32 => "INT".to_owned(),
        DataType::Int64 => "BIGINT".to_owned(),
        DataType::Decimal128(p, s) => format!("DECIMAL({p},{s})"),
        DataType::Float64 => "DOUBLE".to_owned(),
        DataType::Date32 => "DATE".to_owned(),
        DataType::Timestamp(_, _) => "TIMESTAMP".to_owned(),
        DataType::Boolean => "BOOLEAN".to_owned(),
        DataType::Utf8View => "STRING".to_owned(),
        // `arrow_type` above only yields the types matched here.
        other => {
            return Err(FormatError::Other(internal(format!(
                "cannot render {other} in a CSV DDL"
            ))));
        }
    })
}

/// The DDL of a CSV literal (Spark's `schema_of_csv`): the literal parsed
/// as one record under `options` (the CSV reader's options, as Spark adds
/// `lineSep` `\uFFFF` so the literal is one record), its fields named
/// `_c0`, `_c1`, ... (Spark ignores the header option here), typed by the
/// same inference file reads use and rendered as Spark's `StructType.sql`
/// (`STRUCT<_c0: INT, _c1: STRING>`).
///
/// # Errors
/// What the CSV reader's options reject, and `INTERNAL_ERROR` for an empty
/// literal or a record the tokenizer cannot tokenize at all: Spark's
/// `SchemaOfCsvEvaluator` asserts the parsed record is not null (an empty
/// literal parses to no record, checked against Spark 4.2.0, which fails
/// such a query with `INTERNAL_ERROR` during constant folding), and its
/// parser throws past `maxColumns` (checked against Spark 4.2.0, whose
/// Univocity parser throws `TextParsingException` there).
pub fn ddl_of_string(csv: &str, options: &FormatOptions) -> FormatResult<String> {
    if csv.is_empty() {
        // Univocity parses an empty literal to no record, which Spark's
        // assert rejects; Krishiv's tokenizer would yield one empty field.
        return Err(FormatError::Other(internal(
            "the CSV literal is empty (Spark's parser yields no record for it)".to_owned(),
        )));
    }
    let opts = CsvOptions::new(options, false)?;
    let dt = Datetimes::new(options, Purpose::Parse)?;
    let date = DateTimePattern::new(
        options.str("dateFormat").unwrap_or("yyyy-MM-dd"),
        Purpose::Parse,
    )?;
    let inf = Infer {
        o: &opts,
        dt: &dt,
        date,
    };
    let mut syntax = opts.syntax.clone();
    // The lines are not split already (as for file lines): the whole
    // literal is one record, newlines inside values included.
    syntax.eol = Eol::EndOfText;
    let tok = Tokenizer::new(syntax);
    let mut tokens = Vec::new();
    tok.record(csv, 0, &mut tokens).map_err(|e| {
        // A stable message per failure (not the tokenizer error's debug
        // form, which names engine internals).
        let what = match e {
            TokenizeError::TooManyColumns => "has more columns than maxColumns",
            TokenizeError::ValueTooLong => "has a value longer than maxCharsPerColumn",
            TokenizeError::UnescapedQuote => "has an unescaped quote",
        };
        FormatError::Other(internal(format!(
            "the CSV literal cannot be tokenized ({what})"
        )))
    })?;
    let mut types = vec![T::Null; tokens.len()];
    inf.row(&mut types, &tokens);
    let mut fields = Vec::with_capacity(types.len());
    for (i, t) in types.iter().enumerate() {
        fields.push(format!("_c{i}: {}", ddl_type(&arrow_type(*t))?));
    }
    Ok(format!("STRUCT<{}>", fields.join(", ")))
}

/// Reads `range` of `file` line by line, calling `f` with each line until
/// it returns `false`.
async fn for_each_line(
    io: &IoContext,
    file: &FileRef,
    range: ByteRange,
    o: &CsvOptions,
    mut f: impl FnMut(&[u8], u64) -> FormatResult<bool>,
) -> FormatResult<()> {
    let codec = crate::text::codec::Codec::of_path(file.path.as_ref());
    let range = if codec.is_some() {
        ByteRange::whole(file.size)
    } else {
        range
    };
    let mut r = LineReader::new(file.size, range, true, o.line_sep.clone()).compressed(codec);
    loop {
        match r.next_line()? {
            ReadChunk::End => return Ok(()),
            ReadChunk::Line(line, at) => {
                if !f(&line, at)? {
                    return Ok(());
                }
            }
            ReadChunk::NeedData(need) => {
                let got = io.get_ranges(file, std::slice::from_ref(&need)).await?;
                let bytes = got
                    .into_iter()
                    .next()
                    .ok_or_else(|| FormatError::Other(internal("no bytes for a range")))?;
                r.push(&need, &bytes)?;
            }
        }
    }
}

/// The text of `file`, read whole (`multiLine`).
async fn whole_text(io: &IoContext, file: &FileRef, o: &CsvOptions) -> FormatResult<String> {
    let mut r = WholeReader::new(file.size, ByteRange::whole(file.size))
        .compressed(crate::text::codec::Codec::of_path(file.path.as_ref()));
    loop {
        if let Some(need) = r.need() {
            let got = io.get_ranges(file, std::slice::from_ref(&need)).await?;
            let bytes = got
                .into_iter()
                .next()
                .ok_or_else(|| FormatError::Other(internal("no bytes for a range")))?;
            r.push(&need, &bytes)?;
            continue;
        }
        let bytes = r.take().unwrap_or_default();
        let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&bytes);
        return Ok(o.charset.decode(bytes).into_owned());
    }
}

/// A byte size: digits and an optional `b`, `k`, `m`, `g` or `t` unit (with
/// an optional trailing `b`).
fn parse_size(v: &str) -> Option<u64> {
    let v = v.trim().to_ascii_lowercase();
    let digits = v.find(|c: char| !c.is_ascii_digit()).unwrap_or(v.len());
    let n: u64 = v[..digits].parse().ok()?;
    let unit = match &v[digits..] {
        "" | "b" => 1_u64,
        "k" | "kb" => 1 << 10,
        "m" | "mb" => 1 << 20,
        "g" | "gb" => 1 << 30,
        "t" | "tb" => 1 << 40,
        _ => return None,
    };
    n.checked_mul(unit)
}

fn size_option(o: &FormatOptions, name: &str, default: u64) -> FormatResult<u64> {
    let Some(v) = o.str(name) else {
        return Ok(default);
    };
    parse_size(v).ok_or_else(|| {
        FormatError::Other(crate::text::invalid_option(
            "csv",
            name,
            v,
            "expected a size",
        ))
    })
}

fn split_config(o: &FormatOptions) -> FormatResult<SplitConfig> {
    let int = |name: &str| -> FormatResult<Option<u64>> {
        match o.str(name) {
            None => Ok(None),
            Some(v) => v
                .trim()
                .parse::<u64>()
                .ok()
                .filter(|n| *n > 0)
                .map(Some)
                .ok_or_else(|| {
                    FormatError::Other(crate::text::invalid_option(
                        "csv",
                        name,
                        v,
                        "expected a positive integer",
                    ))
                }),
        }
    };
    let slots = std::thread::available_parallelism().map_or(1, |n| n.get() as u64);
    Ok(SplitConfig {
        max_partition_bytes: size_option(o, "inferenceMaxPartitionBytes", 128 << 20)?,
        open_cost_in_bytes: size_option(o, "inferenceOpenCostInBytes", 4 << 20)?,
        min_partition_num: int("inferenceMinPartitionNum")?
            .or(int("inferenceParallelism")?)
            .unwrap_or(slots)
            .max(1),
        max_partition_num: int("inferenceMaxPartitionNum")?,
    })
}

fn sampling_ratio(o: &FormatOptions) -> FormatResult<f64> {
    let Some(v) = o.str("samplingRatio") else {
        return Ok(1.0);
    };
    v.trim()
        .parse::<f64>()
        .ok()
        .filter(|r| *r > 0.0 && r.is_finite())
        .ok_or_else(|| {
            FormatError::Other(crate::text::invalid_option(
                "csv",
                "samplingRatio",
                v,
                "expected a number greater than 0",
            ))
        })
}

/// The schema of CSV `files`.
pub(crate) fn infer<'a>(
    io: &'a IoContext,
    files: &'a [FileRef],
    o: &'a FormatOptions,
) -> BoxFuture<'a, FormatResult<SchemaRef>> {
    Box::pin(async move {
        for f in files {
            crate::text::check_plain(f.path.as_ref())?;
        }
        let opts = CsvOptions::new(o, false)?;
        let dt = Datetimes::new(o, Purpose::Parse)?;
        let date =
            DateTimePattern::new(o.str("dateFormat").unwrap_or("yyyy-MM-dd"), Purpose::Parse)?;
        let inf = Infer {
            o: &opts,
            dt: &dt,
            date,
        };
        let infer_types = o.bool("inferSchema").unwrap_or(false);
        let ratio = sampling_ratio(o)?;
        let line_tok = {
            let mut s = opts.syntax.clone();
            s.eol = Eol::EndOfText;
            Tokenizer::new(s)
        };
        let tokenize = |line: &str| -> Option<Vec<Token>> {
            let mut t = Vec::new();
            line_tok.record(line, 0, &mut t).ok().map(|_| t)
        };
        let mut first: Option<String> = None;
        let mut names: Vec<String> = Vec::new();
        let mut types: Vec<T> = Vec::new();
        if opts.multi_line {
            let record_tok = Tokenizer::new(opts.syntax.clone());
            for file in files {
                let text = whole_text(io, file, &opts).await?;
                let (mut pos, mut first_of_file) = (0, true);
                while let Some((_, tokens, next)) =
                    record_tok.next_record(&text, pos, opts.syntax.comment)
                {
                    pos = next;
                    let Ok(tokens) = tokens else { break };
                    if first.is_none() {
                        first = Some(String::new());
                        names = column_names(&opts, &tokens);
                        types = vec![T::Null; names.len()];
                    }
                    // Every file's header is dropped.
                    if std::mem::take(&mut first_of_file) && opts.header {
                        continue;
                    }
                    if infer_types {
                        inf.row(&mut types, &tokens);
                    }
                }
            }
        } else {
            let listing: Vec<SplitFile> = files
                .iter()
                .map(|f| SplitFile {
                    size: f.size,
                    splittable: true,
                })
                .collect();
            let parts = plan(&split_config(o)?, &listing);
            // The first line: the first usable line of the first partition.
            'first: for p in &parts {
                for s in &p.splits {
                    let file = &files[s.file];
                    let mut found = None;
                    for_each_line(io, file, s.range, &opts, |bytes, at| {
                        let bytes = if at == 0 {
                            bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes)
                        } else {
                            bytes
                        };
                        let line = opts.charset.decode(bytes).into_owned();
                        if skip_line(&opts, &line) {
                            return Ok(true);
                        }
                        found = Some(line);
                        Ok(false)
                    })
                    .await?;
                    if found.is_some() {
                        first = found;
                        break 'first;
                    }
                }
            }
            let Some(first_line) = first.clone() else {
                return Ok(Arc::new(Schema::empty()));
            };
            let tokens = tokenize(&first_line).unwrap_or_default();
            names = column_names(&opts, &tokens);
            types = vec![T::Null; names.len()];
            if infer_types {
                let mut merged = vec![T::Null; names.len()];
                for p in &parts {
                    let mut sampler = (ratio <= 0.99).then(|| Sampler::new(ratio, p.index));
                    let mut part = vec![T::Null; names.len()];
                    for s in &p.splits {
                        let file = &files[s.file];
                        for_each_line(io, file, s.range, &opts, |bytes, at| {
                            if let Some(sm) = &mut sampler
                                && !sm.take()
                            {
                                return Ok(true);
                            }
                            let bytes = if at == 0 {
                                bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes)
                            } else {
                                bytes
                            };
                            let line = opts.charset.decode(bytes).into_owned();
                            if (opts.header && line == first_line) || skip_line(&opts, &line) {
                                return Ok(true);
                            }
                            if let Some(t) = tokenize(&line) {
                                inf.row(&mut part, &t);
                            }
                            Ok(true)
                        })
                        .await?;
                    }
                    for (m, t) in merged.iter_mut().zip(&part) {
                        *m = compatible(*m, *t);
                    }
                }
                types = merged;
            }
        }
        if first.is_none() {
            return Ok(Arc::new(Schema::empty()));
        }
        let fields: Vec<Field> = names
            .iter()
            .zip(&types)
            .map(|(n, t)| Field::new(n, arrow_type(*t), true))
            .collect();
        Ok(Arc::new(Schema::new(fields)))
    })
}
