// SPDX-License-Identifier: Apache-2.0
//! Schema inference (Spark's `JsonInferSchema`), with `samplingRatio`
//! reproducing Spark's row sampling.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use futures::future::BoxFuture;
use krishiv_kernels::datetime_pattern::Purpose;
use krishiv_kernels::json::decimal::{
    LONG_DECIMAL, NumberDecimal, is_not_decimal, merge_decimals, number_decimal, string_decimal,
};
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
    /// `TIMESTAMP_NTZ`.
    Ntz,
    /// `TIME(6)`.
    Time,
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
    use Ty::{Array, Decimal, Double, Long, Ntz, Null, Str, Struct, Timestamp};
    if a == b {
        return a;
    }
    match (a, b) {
        (Null, t) | (t, Null) => t,
        (Struct(f1), Struct(f2)) => Struct(merge_fields(f1, f2)),
        (Array(e1), Array(e2)) => Array(Box::new(compatible(*e1, *e2))),
        (Double, Long | Decimal(_, _)) | (Long | Decimal(_, _), Double) => Double,
        (Ntz, Timestamp) | (Timestamp, Ntz) => Timestamp,
        (Decimal(p, s), Decimal(q, r)) => decimal_or_double((p, s), (q, r)),
        // An integral type merges as the decimal that holds it (`Long` is
        // `DECIMAL(20, 0)`).
        (Long, Decimal(p, s)) | (Decimal(p, s), Long) => decimal_or_double(LONG_DECIMAL, (p, s)),
        _ => Str,
    }
}

/// The decimal that holds both, or `Double` when it would need more than 38
/// digits (Spark's `compatibleType`).
fn decimal_or_double(a: (u8, i8), b: (u8, i8)) -> Ty {
    merge_decimals(a, b).map_or(Ty::Double, |(p, s)| Ty::Decimal(p, s))
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

    /// One record into its partition's type: an object, a top-level array
    /// (its elements are records), a scalar, or a record that does not parse
    /// (Spark's `handleJsonErrorsByParseMode`: the corrupt column alone in
    /// PERMISSIVE, nothing in DROPMALFORMED, an error in FAILFAST).
    ///
    /// A partition has no type before its first record: Spark reduces each
    /// partition's record types without a seed (`reduceOption`) and only
    /// then folds the partitions into an empty struct, so the scalars before
    /// a partition's first object merge with each other first (`5` then
    /// `"str"` is `STRING`), and FAILFAST names that merged type.
    fn absorb(
        self,
        inf: &Infer<'_>,
        part: Option<Ty>,
        rec: &Result<Top, Reason>,
    ) -> FormatResult<Option<Ty>> {
        let typed = match rec {
            Ok(Top::Object(o)) => inf.object(&o.fields).map(Some),
            Ok(Top::Array(items)) => inf.array(items).map(Some),
            Ok(Top::Other(v)) => inf.field(v).map(Some),
            Err(_) => Ok(None),
        };
        // A number with no decimal form under `prefersDecimal` is a
        // `RuntimeException` in Spark's per-record `inferField`, so the
        // record goes to the parse mode as one that does not parse.
        let typed = match typed {
            Err(FormatError::Other(e)) if is_not_decimal(&e) => None,
            other => other?,
        };
        let t = match typed {
            Some(t) => t,
            None => match self.mode {
                Mode::Permissive => Ty::Struct(vec![(self.corrupt.to_owned(), Ty::Str)]),
                Mode::DropMalformed => return Ok(part),
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
        match part {
            None => Ok(Some(t)),
            Some(p) => self.merge(p, t).map(Some),
        }
    }

    /// A partition's type into the root (Spark's fold of the partition
    /// results into `StructType(Nil)`, in partition order here). Each
    /// partition first meets an empty struct, as Spark 4.2.0 does: a
    /// scalar-only partition adds the corrupt column as a string before the
    /// partitions merge.
    fn fold(self, root: Ty, part: Option<Ty>) -> FormatResult<Ty> {
        match part {
            None => Ok(root),
            Some(p) => {
                let p = self.merge(Ty::Struct(Vec::new()), p)?;
                self.merge(root, p)
            }
        }
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

struct Infer<'a> {
    prefers_decimal: bool,
    primitives_as_string: bool,
    infer_timestamp: bool,
    /// `spark.sql.timeType.enabled`: a `TIME` may be inferred.
    time_type: bool,
    /// `spark.sql.timestampType = TIMESTAMP_NTZ`: a zone-less timestamp
    /// text infers `TIMESTAMP_NTZ`.
    default_ntz: bool,
    /// `spark.sql.legacy.timeParserPolicy = LEGACY`.
    legacy: bool,
    dt: &'a Datetimes,
}

impl Infer<'_> {
    /// Spark's `inferField`: the type of one value.
    ///
    /// # Errors
    /// `NEGATIVE_SCALE_DISALLOWED` for a number with a negative scale under
    /// `prefersDecimal` (`1e3`), as Spark's `DecimalType` refuses it.
    fn field(&self, v: &JVal) -> FormatResult<Ty> {
        Ok(match v {
            JVal::Null => Ty::Null,
            // Empty strings infer as nulls (Spark treats them alike until
            // the fields are merged).
            JVal::Str(s) if s.is_empty() => Ty::Null,
            JVal::Str(s) => {
                if self.prefers_decimal
                    && let Some((p, sc)) = string_decimal(s)
                {
                    Ty::Decimal(p, sc)
                } else if self.infer_timestamp {
                    self.timestamp_of_string(s)
                } else {
                    Ty::Str
                }
            }
            JVal::Bool(_) | JVal::Num(_, _) if self.primitives_as_string => Ty::Str,
            JVal::Bool(_) => Ty::Boolean,
            JVal::Num(t, false) if java_parse_long(t).is_some() => Ty::Long,
            JVal::Num(_, true) if !self.prefers_decimal => Ty::Double,
            JVal::Num(t, _) => match number_decimal(t).map_err(FormatError::Other)? {
                NumberDecimal::Decimal(p, s) => Ty::Decimal(p, s),
                NumberDecimal::Double => Ty::Double,
            },
            JVal::Object(fields, _) => self.object(fields)?,
            JVal::Array(items, _) => self.array(items)?,
        })
    }

    /// An array of its merged element types.
    fn array(&self, items: &[JVal]) -> FormatResult<Ty> {
        let mut t = Ty::Null;
        for v in items {
            t = compatible(t, self.field(v)?);
        }
        Ok(Ty::Array(Box::new(t)))
    }

    /// The `inferTimestamp` arm of Spark's string inference (v4.3.0
    /// `JsonInferSchema` lines 181-205): a `TIME` (Spark 4.3, gate on), a
    /// `TIMESTAMP_NTZ` (default type NTZ), a `TIMESTAMP` under
    /// `timestampFormat` without the fallback, and under LEGACY a
    /// `TIMESTAMP` for any zone-less timestamp text (SPARK-46769). A parse
    /// failure of any kind is no match (`allCatch`, `parseOptional`).
    fn timestamp_of_string(&self, s: &str) -> Ty {
        if self.time_type && matches!(self.dt.parse_time(s), Ok(Some(_))) {
            Ty::Time
        } else if self.default_ntz && matches!(self.dt.parse_ntz_exact(s), Ok(Some(_))) {
            Ty::Ntz
        } else if matches!(self.dt.parse_timestamp_exact(s), Ok(Some(_)))
            || (self.legacy && krishiv_kernels::text::spark_string_to_timestamp_ntz(s).is_some())
        {
            Ty::Timestamp
        } else {
            Ty::Str
        }
    }

    /// The struct of an object's members, sorted by name.
    fn object(&self, fields: &[(String, JVal)]) -> FormatResult<Ty> {
        let mut out = Vec::with_capacity(fields.len());
        for (n, v) in fields {
            out.push((n.clone(), self.field(v)?));
        }
        out.sort_by(|(a, _), (b, _)| java_cmp(a, b));
        Ok(Ty::Struct(out))
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
        Ty::Ntz => "TIMESTAMP_NTZ".to_owned(),
        Ty::Time => "TIME(6)".to_owned(),
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
        Ty::Ntz => DataType::Timestamp(TimeUnit::Microsecond, None),
        Ty::Time => DataType::Time64(TimeUnit::Nanosecond),
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
    let codec = crate::text::codec::Codec::of_path(ctx.file.path.as_ref(), ctx.o.suffixes);
    let range = if codec.is_some() {
        ByteRange::whole(ctx.file.size)
    } else {
        range
    };
    let mut r = LineReader::new(ctx.file.size, range, true, ctx.o.line_sep.clone())
        .compressed(codec)
        .lenient(ctx.o.ignore_corrupt_files);
    loop {
        match r.next_line().map_err(|e| e.at(&ctx.file.display_path()))? {
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
    let codec = crate::text::codec::Codec::of_path(ctx.file.path.as_ref(), ctx.o.suffixes);
    let mut r = WholeReader::new(ctx.file.size, ByteRange::whole(ctx.file.size)).compressed(codec);
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
        let opts = JsonOptions::new(o, false)?;
        for f in files {
            crate::text::check_plain(f.path.as_ref(), opts.suffixes)?;
        }
        let dt = Datetimes::new(o, Purpose::Parse)?;
        let syntax = Syntax::from(&opts);
        let inf = Infer {
            prefers_decimal: opts.prefers_decimal,
            primitives_as_string: opts.primitives_as_string,
            infer_timestamp: opts.infer_timestamp,
            time_type: crate::text::infers_time(o),
            default_ntz: crate::text::default_ntz(o),
            legacy: crate::text::legacy_parser(o),
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
            // One partition, in listing order. Spark packs whole files into
            // combined splits of at least `openCostInBytes` (4 MiB), so small
            // files share one (observed); larger inputs split into several
            // [inferred], which only changes the type a FAILFAST error names.
            // Spark's file order within a split is Hadoop's and varied
            // between runs of the same files.
            let mut part = None;
            for file in files {
                let ctx = ctx_of(file);
                let text = whole_text(&ctx)
                    .await
                    .map_err(|e| e.at(&ctx.file.display_path()))?;
                // One record per file, its first value, as the reader
                // sees it (Spark's `JsonInferSchema` over one root).
                match parse_next(&text, 0, syntax) {
                    Ok(None) => {}
                    Ok(Some((top, _))) => part = rm.absorb(&inf, part, &Ok(top))?,
                    Err(reason) => part = rm.absorb(&inf, part, &Err(reason))?,
                }
            }
            root = rm.fold(root, part)?;
        } else {
            let listing: Vec<SplitFile> = files
                .iter()
                .map(|f| SplitFile {
                    size: f.size,
                    // A compressed file is one split, sampled once.
                    splittable: crate::text::codec::Codec::of_path(f.path.as_ref(), opts.suffixes)
                        .is_none(),
                })
                .collect();
            let parts = plan(&split_config(o)?, &listing);
            for p in &parts {
                let mut sampler = (ratio <= 0.99).then(|| Sampler::new(ratio, p.index));
                let mut part = None;
                for s in &p.splits {
                    let file = &files[s.file];
                    let ctx = ctx_of(file);
                    for_each_line(&ctx, s.range, syntax, |rec| {
                        if let Some(sm) = &mut sampler
                            && !sm.take()
                        {
                            return Ok(true);
                        }
                        part = rm.absorb(&inf, part.take(), rec)?;
                        Ok(true)
                    })
                    .await?;
                }
                root = rm.fold(std::mem::replace(&mut root, Ty::Null), part)?;
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
