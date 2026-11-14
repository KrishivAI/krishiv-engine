// SPDX-License-Identifier: Apache-2.0
//! Schema inference (Spark's `JsonInferSchema`), with `samplingRatio`
//! reproducing Spark's row sampling.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use futures::future::BoxFuture;
use krishiv_kernels::datetime_pattern::Purpose;
use krishiv_kernels::text::java_parse_long;

use super::decode::{JVal, Reason, Top, parse_line, parse_next};
use super::options::{JsonOptions, sampling_ratio};
use super::options::{Mode, Syntax};
use crate::csv::sampling::Sampler;
use crate::error::{FormatError, FormatResult, internal};
use crate::io::{ByteRange, FileRef, IoContext};
use crate::options::FormatOptions;
use crate::split::{SplitConfig, SplitFile, plan};
use crate::text::Datetimes;
use crate::text::reader::{LineReader, ReadChunk, WholeReader};
use krishiv_common::error::{
    ColumnAlreadyExists, InvalidJsonRecordType, MalformedRecordInParsingWithoutSuggestion,
};

/// The types inference moves through (Spark's inferred `DataType`s).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Ty {
    Null,
    Boolean,
    Long,
    Decimal(u8, i8),
    Double,
    Timestamp,
    Str,
    /// A struct, its fields sorted by name as Spark's `inferField` sorts
    /// them (`String.compareTo`, UTF-16 order).
    Struct(Vec<(String, Ty)>),
    /// An array of its element type.
    Array(Box<Ty>),
}

/// Java's `String.compareTo`: UTF-16 code unit order.
fn java_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

/// Spark's `JsonInferSchema.compatibleType` (default `StringType`): the
/// tightest common type of numbers, structs merged field by field, arrays
/// by element type, anything else a string.
fn compatible(a: Ty, b: Ty) -> Ty {
    use Ty::{Array, Decimal, Double, Long, Null, Str, Struct};
    if a == b {
        return a;
    }
    match (a, b) {
        (Null, t) | (t, Null) => t,
        (Struct(f1), Struct(f2)) => Struct(merge_fields(f1, f2)),
        (Array(e1), Array(e2)) => Array(Box::new(compatible(*e1, *e2))),
        (Double, Long | Decimal(_, _)) | (Long | Decimal(_, _), Double) => Double,
        (Decimal(p, s), Decimal(q, r)) => merge_decimals(p, s, q, r),
        // An integral type merges as the decimal that holds it (`Long` is
        // `DECIMAL(20, 0)`).
        (Long, Decimal(p, s)) | (Decimal(p, s), Long) => merge_decimals(20, 0, p, s),
        _ => Str,
    }
}

/// The decimal that holds both `(p, s)` and `(q, r)`, or `Double` when it
/// would need more than 38 digits (as Spark's `findTightestCommonType`).
fn merge_decimals(p: u8, s: i8, q: u8, r: i8) -> Ty {
    let scale = s.max(r);
    let range = (i16::from(p) - i16::from(s)).max(i16::from(q) - i16::from(r));
    match u8::try_from(range + i16::from(scale)) {
        Ok(precision) if precision <= 38 => Ty::Decimal(precision, scale),
        _ => Ty::Double,
    }
}

/// Two name-sorted field lists merged, as Spark merges sorted structs.
fn merge_fields(f1: Vec<(String, Ty)>, f2: Vec<(String, Ty)>) -> Vec<(String, Ty)> {
    let mut out = Vec::with_capacity(f1.len().max(f2.len()));
    let mut i1 = f1.into_iter().peekable();
    let mut i2 = f2.into_iter().peekable();
    loop {
        let order = match (i1.peek(), i2.peek()) {
            (Some((n1, _)), Some((n2, _))) => java_cmp(n1, n2),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => return out,
        };
        match order {
            std::cmp::Ordering::Less => out.extend(i1.next()),
            std::cmp::Ordering::Greater => out.extend(i2.next()),
            std::cmp::Ordering::Equal => {
                if let (Some((n, t1)), Some((_, t2))) = (i1.next(), i2.next()) {
                    out.push((n, compatible(t1, t2)));
                }
            }
        }
    }
}

/// What a record that is not an object does to the inferred root, by parse
/// mode (Spark's `withCorruptField`).
#[derive(Clone, Copy)]
struct Root<'a> {
    mode: Mode,
    corrupt: &'a str,
}

impl Root<'_> {
    /// The root struct with the corrupt-record column (PERMISSIVE), as it is
    /// (DROPMALFORMED), or the inference error (FAILFAST).
    fn with_corrupt(self, mut fields: Vec<(String, Ty)>, other: &Ty) -> FormatResult<Ty> {
        match self.mode {
            Mode::Permissive => {
                if !fields.iter().any(|(n, _)| n == self.corrupt) {
                    fields.push((self.corrupt.to_owned(), Ty::Str));
                    fields.sort_by(|(a, _), (b, _)| java_cmp(a, b));
                }
                Ok(Ty::Struct(fields))
            }
            Mode::DropMalformed => Ok(Ty::Struct(fields)),
            Mode::FailFast => Err(FormatError::Other(
                InvalidJsonRecordType {
                    fail_fast_mode: "FAILFAST".to_owned(),
                    invalid_type: format!("\"{}\"", sql_type(other)),
                }
                .into(),
            )),
        }
    }

    /// One record into the root type: an object, a top-level array (its
    /// elements are records), a scalar, or a record that does not parse
    /// (Spark's `handleJsonErrorsByParseMode`: the corrupt column alone in
    /// PERMISSIVE, nothing in DROPMALFORMED, an error in FAILFAST).
    fn absorb(self, inf: &Infer<'_>, root: Ty, rec: &Result<Top, Reason>) -> FormatResult<Ty> {
        let t = match rec {
            Ok(Top::Object(o)) => inf.object(&o.fields),
            Ok(Top::Array(items)) => Ty::Array(Box::new(
                items
                    .iter()
                    .fold(Ty::Null, |t, v| compatible(t, inf.field(v))),
            )),
            Ok(Top::Other(v)) => inf.field(v),
            Err(_) => match self.mode {
                Mode::Permissive => Ty::Struct(vec![(self.corrupt.to_owned(), Ty::Str)]),
                Mode::DropMalformed => return Ok(root),
                Mode::FailFast => {
                    return Err(FormatError::Other(
                        MalformedRecordInParsingWithoutSuggestion {
                            bad_record: self.corrupt.to_owned(),
                            fail_fast_mode: "FAILFAST".to_owned(),
                        }
                        .into(),
                    ));
                }
            },
        };
        self.merge(root, t)
    }

    /// Spark's `compatibleRootType`: top-level arrays are unwrapped, null
    /// records shape nothing, a struct meeting anything else takes the
    /// corrupt column.
    fn merge(self, a: Ty, b: Ty) -> FormatResult<Ty> {
        match (a, b) {
            (Ty::Array(t1), t2) => self.merge(*t1, t2),
            (t1, Ty::Array(t2)) => self.merge(t1, *t2),
            (s @ Ty::Struct(_), Ty::Null) | (Ty::Null, s @ Ty::Struct(_)) => Ok(s),
            (Ty::Struct(f), o) | (o, Ty::Struct(f)) if !matches!(o, Ty::Struct(_)) => {
                self.with_corrupt(f, &o)
            }
            (t1, t2) => Ok(compatible(t1, t2)),
        }
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
    prefers_decimal: bool,
    primitives_as_string: bool,
    infer_timestamp: bool,
    dt: &'a Datetimes,
}

impl Infer<'_> {
    /// Spark's `inferField`: the type of one value.
    fn field(&self, v: &JVal) -> Ty {
        match v {
            JVal::Null => Ty::Null,
            // Empty strings infer as nulls (Spark treats them alike until
            // the fields are merged).
            JVal::Str(s) if s.is_empty() => Ty::Null,
            JVal::Str(s) => {
                if self.prefers_decimal
                    && let Some(t) = self.decimal_of_string(s)
                {
                    t
                } else if self.infer_timestamp && matches!(self.dt.parse_timestamp(s), Ok(Some(_)))
                {
                    Ty::Timestamp
                } else {
                    Ty::Str
                }
            }
            JVal::Bool(_) | JVal::Num(_, _) if self.primitives_as_string => Ty::Str,
            JVal::Bool(_) => Ty::Boolean,
            JVal::Num(t, false) => {
                if java_parse_long(t).is_some() {
                    Ty::Long
                } else {
                    self.decimal_shape(t)
                }
            }
            JVal::Num(t, true) => {
                if self.prefers_decimal {
                    self.decimal_shape(t)
                } else {
                    Ty::Double
                }
            }
            JVal::Object(fields, _) => self.object(fields),
            JVal::Array(items, _) => Ty::Array(Box::new(
                items
                    .iter()
                    .fold(Ty::Null, |t, v| compatible(t, self.field(v))),
            )),
        }
    }

    /// The struct of an object's members, sorted by name.
    fn object(&self, fields: &[(String, JVal)]) -> Ty {
        let mut out: Vec<(String, Ty)> = fields
            .iter()
            .map(|(n, v)| (n.clone(), self.field(v)))
            .collect();
        out.sort_by(|(a, _), (b, _)| java_cmp(a, b));
        Ty::Struct(out)
    }

    /// A decimal literal as a `Ty`: `Double` when it cannot be a
    /// `Decimal128` (negative scale, precision above 38).
    fn decimal_shape(&self, t: &str) -> Ty {
        match big_decimal_shape(t) {
            Some((p, s)) if s >= 0 && p <= 38 && s <= 38 => {
                let (Ok(p), Ok(s)) = (u8::try_from(p.max(s).min(38)), i8::try_from(s)) else {
                    return Ty::Double;
                };
                Ty::Decimal(p.max(1), s)
            }
            _ => Ty::Double,
        }
    }

    /// A string that parses as a decimal, under `prefersDecimal`: the
    /// decimal of its precision and scale, as Spark's string branch infers
    /// it.
    fn decimal_of_string(&self, s: &str) -> Option<Ty> {
        let (p, sc) = big_decimal_shape(s)?;
        if sc < 0 || p > 38 || sc > p {
            return None;
        }
        Some(Ty::Decimal(u8::try_from(p).ok()?, i8::try_from(sc).ok()?))
    }
}

/// Spark's `canonicalizeType`: nulls become strings (or are dropped with
/// `dropFieldIfAllNull`), structs lose unnamed fields, and a struct with no
/// fields left (or an array of one) is dropped.
fn canonical(t: Ty, drop_all_null: bool) -> Option<Ty> {
    match t {
        Ty::Array(e) => canonical(*e, drop_all_null).map(|e| Ty::Array(Box::new(e))),
        Ty::Struct(fields) => {
            let kept: Vec<(String, Ty)> = fields
                .into_iter()
                .filter(|(n, _)| !n.is_empty())
                .filter_map(|(n, t)| canonical(t, drop_all_null).map(|t| (n, t)))
                .collect();
            (!kept.is_empty()).then_some(Ty::Struct(kept))
        }
        Ty::Null => (!drop_all_null).then_some(Ty::Str),
        other => Some(other),
    }
}

/// `COLUMN_ALREADY_EXISTS` when two fields of a struct anywhere in `t` have
/// the same name, ignoring case.
fn no_duplicates(t: &Ty) -> FormatResult<()> {
    match t {
        Ty::Array(e) => no_duplicates(e),
        Ty::Struct(fields) => {
            let mut seen = std::collections::HashSet::with_capacity(fields.len());
            for (n, t) in fields {
                let lower = n.to_lowercase();
                if !seen.insert(lower.clone()) {
                    return Err(FormatError::Other(
                        ColumnAlreadyExists {
                            column_name: format!("`{}`", lower.replace('`', "``")),
                        }
                        .into(),
                    ));
                }
                no_duplicates(t)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Spark's `toSQLType` of an inferred type (without the quotes).
fn sql_type(t: &Ty) -> String {
    match t {
        Ty::Null => "VOID".to_owned(),
        Ty::Boolean => "BOOLEAN".to_owned(),
        Ty::Long => "BIGINT".to_owned(),
        Ty::Decimal(p, s) => format!("DECIMAL({p},{s})"),
        Ty::Double => "DOUBLE".to_owned(),
        Ty::Timestamp => "TIMESTAMP".to_owned(),
        Ty::Str => "STRING".to_owned(),
        Ty::Struct(fields) => format!(
            "STRUCT<{}>",
            fields
                .iter()
                .map(|(n, t)| format!("{n}: {}", sql_type(t)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Ty::Array(e) => format!("ARRAY<{}>", sql_type(e)),
    }
}

fn arrow_type(t: &Ty) -> DataType {
    match t {
        Ty::Null | Ty::Str => DataType::Utf8View,
        Ty::Boolean => DataType::Boolean,
        Ty::Long => DataType::Int64,
        Ty::Decimal(p, s) => DataType::Decimal128(*p, *s),
        Ty::Double => DataType::Float64,
        Ty::Timestamp => DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        Ty::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|(n, t)| Field::new(n, arrow_type(t), true))
                .collect(),
        ),
        Ty::Array(e) => DataType::List(Arc::new(Field::new("element", arrow_type(e), true))),
    }
}

/// The JSON records of `file` from `range`, calling `f` with each parsed
/// object until it returns `false`.
struct Ctx<'a> {
    io: &'a IoContext,
    file: &'a FileRef,
    o: &'a JsonOptions,
}

/// Reads `range` of `ctx.file` record by record, calling `f` with each
/// object until it returns `false`.
async fn for_each_line(
    ctx: &Ctx<'_>,
    range: ByteRange,
    syntax: Syntax,
    mut f: impl FnMut(&Result<Top, Reason>) -> FormatResult<bool>,
) -> FormatResult<()> {
    let codec = crate::text::codec::Codec::of_path(ctx.file.path.as_ref());
    let range = if codec.is_some() {
        ByteRange::whole(ctx.file.size)
    } else {
        range
    };
    let mut r =
        LineReader::new(ctx.file.size, range, true, ctx.o.line_sep.clone()).compressed(codec);
    loop {
        match r.next_line()? {
            ReadChunk::End => return Ok(()),
            ReadChunk::Line(line, at) => {
                let line = if at == 0 {
                    line.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&line)
                } else {
                    &line[..]
                };
                if line
                    .iter()
                    .all(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
                {
                    continue;
                }
                let text = ctx.o.charset.decode(line).into_owned();
                if !f(&parse_line(&text, syntax))? {
                    return Ok(());
                }
            }
            ReadChunk::NeedData(need) => {
                let got = ctx
                    .io
                    .get_ranges(ctx.file, std::slice::from_ref(&need))
                    .await?;
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
async fn whole_text(ctx: &Ctx<'_>) -> FormatResult<String> {
    let mut r = WholeReader::new(ctx.file.size, ByteRange::whole(ctx.file.size))
        .compressed(crate::text::codec::Codec::of_path(ctx.file.path.as_ref()));
    loop {
        if let Some(need) = r.need() {
            let got = ctx
                .io
                .get_ranges(ctx.file, std::slice::from_ref(&need))
                .await?;
            let bytes = got
                .into_iter()
                .next()
                .ok_or_else(|| FormatError::Other(internal("no bytes for a range")))?;
            r.push(&need, &bytes)?;
            continue;
        }
        let bytes = r.take().unwrap_or_default();
        let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&bytes);
        return Ok(ctx.o.charset.decode(bytes).into_owned());
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
            "json",
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
                        "json",
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

/// The schema of JSON `files`.
pub(crate) fn infer<'a>(
    io: &'a IoContext,
    files: &'a [FileRef],
    o: &'a FormatOptions,
) -> BoxFuture<'a, FormatResult<SchemaRef>> {
    Box::pin(async move {
        for f in files {
            crate::text::check_plain(f.path.as_ref())?;
        }
        let opts = JsonOptions::new(o, false)?;
        let dt = Datetimes::new(o, Purpose::Parse)?;
        let syntax = Syntax::from(&opts);
        let inf = Infer {
            prefers_decimal: opts.prefers_decimal,
            primitives_as_string: opts.primitives_as_string,
            infer_timestamp: opts.infer_timestamp,
            dt: &dt,
        };
        let ratio = sampling_ratio(o)?;
        let ctx_of = |file: &'a FileRef| Ctx { io, file, o: &opts };
        let rm = Root {
            mode: opts.mode,
            corrupt: &opts.corrupt_column,
        };
        let mut root = Ty::Struct(Vec::new());
        if opts.multi_line {
            for file in files {
                let ctx = ctx_of(file);
                let text = whole_text(&ctx).await?;
                let mut pos = 0;
                loop {
                    match parse_next(&text, pos, syntax) {
                        Ok(None) => break,
                        Ok(Some((top, end))) => {
                            pos = end;
                            root = rm.absorb(&inf, root, &Ok(top))?;
                        }
                        // A tail that does not parse is one malformed
                        // record, as the reader sees it.
                        Err(reason) => {
                            root = rm.absorb(&inf, root, &Err(reason))?;
                            break;
                        }
                    }
                }
            }
        } else {
            let listing: Vec<SplitFile> = files
                .iter()
                .map(|f| SplitFile {
                    size: f.size,
                    // A compressed file is one split, sampled once.
                    splittable: crate::text::codec::Codec::of_path(f.path.as_ref()).is_none(),
                })
                .collect();
            let parts = plan(&split_config(o)?, &listing);
            for p in &parts {
                let mut sampler = (ratio <= 0.99).then(|| Sampler::new(ratio, p.index));
                for s in &p.splits {
                    let file = &files[s.file];
                    let ctx = ctx_of(file);
                    for_each_line(&ctx, s.range, syntax, |rec| {
                        if let Some(sm) = &mut sampler
                            && !sm.take()
                        {
                            return Ok(true);
                        }
                        root = rm.absorb(&inf, std::mem::replace(&mut root, Ty::Null), rec)?;
                        Ok(true)
                    })
                    .await?;
                }
            }
        }
        // Fields are sorted by name, as Spark's inference sorts them.
        let root = match canonical(root, opts.drop_field_if_all_null) {
            Some(t @ Ty::Struct(_)) => t,
            _ => Ty::Struct(Vec::new()),
        };
        // Spark refuses a schema whose names repeat at any level, ignoring
        // case (`checkSchemaColumnNameDuplication`, case-insensitive by
        // default), when the relation resolves.
        no_duplicates(&root)?;
        let Ty::Struct(root) = root else {
            return Err(FormatError::Other(internal(
                "the inferred root is a struct",
            )));
        };
        let fields: Vec<Field> = root
            .iter()
            .map(|(n, t)| Field::new(n, arrow_type(t), true))
            .collect();
        // Krishiv types nest at most `DataType::MAX_NESTING_DEPTH` deep:
        // a deeper inferred type is refused here, by the type system.
        for f in &fields {
            krishiv_types::DataType::from_arrow_field(f).map_err(FormatError::Other)?;
        }
        Ok(Arc::new(Schema::new(fields)))
    })
}
