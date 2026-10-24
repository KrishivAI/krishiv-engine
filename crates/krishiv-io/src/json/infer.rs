// SPDX-License-Identifier: Apache-2.0
//! Schema inference (Spark's `JsonInferSchema`), with `samplingRatio`
//! reproducing Spark's row sampling.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use futures::future::BoxFuture;
use krishiv_kernels::datetime_pattern::Purpose;
use krishiv_kernels::text::java_parse_long;

use super::decode::{JObject, JVal, Syntax, Top, parse_line, parse_next};
use super::options::{JsonOptions, sampling_ratio};
use crate::csv::sampling::Sampler;
use crate::error::{FormatError, FormatResult, internal, unsupported};
use crate::io::{ByteRange, FileRef, IoContext};
use crate::options::FormatOptions;
use crate::split::{SplitConfig, SplitFile, plan};
use crate::text::Datetimes;
use crate::text::reader::{LineReader, ReadChunk, WholeReader};
use krishiv_common::error::KrishivError;

/// The types inference moves through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum T {
    Null,
    Boolean,
    Long,
    Decimal(u8, i8),
    Double,
    Timestamp,
    Str,
}

/// Spark's `TypeCoercion.findTightestCommonType`, as `JsonInferSchema`
/// merges field types (flat records only).
fn compatible(a: T, b: T) -> T {
    use T::{Decimal, Double, Long, Null, Str};
    if a == b {
        return a;
    }
    match (a, b) {
        (Null, t) | (t, Null) => t,
        // String widening: any other type with a string is a string.
        #[expect(
            clippy::match_same_arms,
            reason = "the arm documents string widening; the wildcard below is the fallthrough"
        )]
        (Str, _) | (_, Str) => Str,
        (Double, Long | Decimal(_, _)) | (Long | Decimal(_, _), Double) => Double,
        (Decimal(p, s), Decimal(q, r)) => {
            let int = (i16::from(p) - i16::from(s)).max(i16::from(q) - i16::from(r));
            let scale = s.max(r);
            Decimal(
                u8::try_from(int + i16::from(scale))
                    .unwrap_or(u8::MAX)
                    .min(38),
                scale,
            )
        }
        (Long, Decimal(p, s)) | (Decimal(p, s), Long) => {
            let int = (i16::from(p) - i16::from(s)).max(19);
            Decimal(
                u8::try_from(int + i16::from(s)).unwrap_or(u8::MAX).min(38),
                s,
            )
        }
        _ => Str,
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
    dt: &'a Datetimes,
}

impl Infer<'_> {
    /// The type of one scalar, or an unsupported nested value.
    fn scalar(&self, so_far: T, field: &str, v: &JVal) -> Result<T, KrishivError> {
        if self.primitives_as_string {
            return match v {
                JVal::Nested => Err(nested(field)),
                _ => Ok(compatible(so_far, T::Str)),
            };
        }
        Ok(match v {
            JVal::Null => so_far,
            JVal::Nested => return Err(nested(field)),
            JVal::Bool(_) => compatible(so_far, T::Boolean),
            JVal::Num(t, false) => {
                if java_parse_long(t).is_some() {
                    compatible(so_far, T::Long)
                } else {
                    compatible(so_far, self.decimal_shape(t))
                }
            }
            JVal::Num(t, true) => {
                if self.prefers_decimal {
                    compatible(so_far, self.decimal_shape(t))
                } else {
                    compatible(so_far, T::Double)
                }
            }
            JVal::Str(s) => {
                if s.is_empty() {
                    compatible(so_far, T::Str)
                } else if matches!(self.dt.parse_timestamp(s), Ok(Some(_))) {
                    compatible(so_far, T::Timestamp)
                } else {
                    compatible(so_far, T::Str)
                }
            }
        })
    }

    /// A decimal literal as a `T`: `Double` when it cannot be a
    /// `Decimal128` (negative scale, precision above 38).
    fn decimal_shape(&self, t: &str) -> T {
        match big_decimal_shape(t) {
            Some((p, s)) if s >= 0 && p <= 38 && s <= 38 => {
                let (Ok(p), Ok(s)) = (u8::try_from(p.max(s).min(38)), i8::try_from(s)) else {
                    return T::Double;
                };
                T::Decimal(p.max(1), s)
            }
            _ => T::Double,
        }
    }
}

/// A nested value where only flat records are supported.
fn nested(field: &str) -> KrishivError {
    unsupported(format!(
        "a nested object or array in the JSON field `{field}` (only flat records are supported)"
    ))
}

fn arrow_type(t: T) -> DataType {
    match t {
        T::Null | T::Str => DataType::Utf8View,
        T::Boolean => DataType::Boolean,
        T::Long => DataType::Int64,
        T::Decimal(p, s) => DataType::Decimal128(p, s),
        T::Double => DataType::Float64,
        T::Timestamp => DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
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
    mut f: impl FnMut(&JObject) -> FormatResult<bool>,
) -> FormatResult<()> {
    let mut r = LineReader::new(ctx.file.size, range, true, ctx.o.line_sep.clone());
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
                match parse_line(&text, syntax) {
                    Ok(Top::Object(o)) => {
                        if !f(&o)? {
                            return Ok(());
                        }
                    }
                    Ok(Top::Other(_)) | Err(_) => {}
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
    let mut r = WholeReader::new(ctx.file.size, ByteRange::whole(ctx.file.size));
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
            dt: &dt,
        };
        let ratio = sampling_ratio(o)?;
        let ctx_of = |file: &'a FileRef| Ctx { io, file, o: &opts };
        let mut names: Vec<String> = Vec::new();
        let mut types: Vec<T> = Vec::new();
        if opts.multi_line {
            for file in files {
                let ctx = ctx_of(file);
                let text = whole_text(&ctx).await?;
                let mut pos = 0;
                // A truncated tail shapes nothing further.
                while let Ok(next) = parse_next(&text, pos, syntax) {
                    let Some((top, end)) = next else {
                        break;
                    };
                    pos = end;
                    let Top::Object(obj) = top else {
                        continue;
                    };
                    absorb(&inf, &mut names, &mut types, &obj)?;
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
            // Field order is by first appearance over the sampled records,
            // as in Spark.
            for p in &parts {
                let mut sampler = (ratio <= 0.99).then(|| Sampler::new(ratio, p.index));
                for s in &p.splits {
                    let file = &files[s.file];
                    let ctx = ctx_of(file);
                    for_each_line(&ctx, s.range, syntax, |o| {
                        if let Some(sm) = &mut sampler
                            && !sm.take()
                        {
                            return Ok(true);
                        }
                        absorb(&inf, &mut names, &mut types, o)?;
                        Ok(true)
                    })
                    .await?;
                }
            }
        }
        if opts.drop_field_if_all_null {
            let mut kept_names = Vec::new();
            let mut kept_types = Vec::new();
            for (n, t) in names.iter().zip(&types) {
                if *t != T::Null {
                    kept_names.push(n.clone());
                    kept_types.push(*t);
                }
            }
            names = kept_names;
            types = kept_types;
        }
        let fields: Vec<Field> = names
            .iter()
            .zip(&types)
            .map(|(n, t)| Field::new(n, arrow_type(*t), true))
            .collect();
        Ok(Arc::new(Schema::new(fields)))
    })
}

/// Adds a multi-line record's fields to the inference: new fields append
/// in first-appearance order.
fn absorb(
    inf: &Infer<'_>,
    names: &mut Vec<String>,
    types: &mut Vec<T>,
    obj: &JObject,
) -> FormatResult<()> {
    for (n, v) in &obj.fields {
        match names.iter().position(|x| x == n) {
            Some(i) => types[i] = inf.scalar(types[i], n, v)?,
            None => {
                names.push(n.clone());
                types.push(inf.scalar(T::Null, n, v)?);
            }
        }
    }
    Ok(())
}
