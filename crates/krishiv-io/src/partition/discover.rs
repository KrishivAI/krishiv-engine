// SPDX-License-Identifier: Apache-2.0
//! Partition discovery for reads by path (P2-T082b, design §2.5): the
//! `k=v` directories between a read's root and its files become columns.
//!
//! Spark 4.2's `PartitioningUtils`, as probed on the oracle:
//! - every directory between the root and a file must be `name=value`
//!   (names and values are `%XX`-unescaped); the files of one read must
//!   agree on the names and their order;
//! - `__HIVE_DEFAULT_PARTITION__` is NULL;
//! - with `spark.sql.sources.partitionColumnTypeInference.enabled` (default
//!   true) each value is read as the first of `int`, `bigint`,
//!   `decimal(p,0)` (integral values past a long, at most 38 digits),
//!   `double`, `date`, else `string` (`true`/`false` stay strings); a
//!   column's type is the widest of its values' types, and a mix with a
//!   non-numeric type is `string`;
//! - the partition columns follow the data columns.
//!
//! Timestamp-shaped values are refused (`KRISHIV_UNSUPPORTED_FEATURE`)
//! until the session time zone reaches the scan: reading them as strings
//! would silently differ from Spark.
//!
//! A scan needs only the columns: a file's values are the `k=v` segments
//! directly above it, as many as there are columns.

use arrow_array::{Array, ArrayRef, StringArray, StringViewArray};
use arrow_schema::{DataType, Field, FieldRef};
use krishiv_common::error::{ConflictingDirectoryStructures, ConflictingPartitionColumnNames};
use krishiv_kernels::cast::{CastMode, CastOptions, FixedOffset, cast};
use object_store::path::Path;
use std::sync::Arc;

use super::{DEFAULT_PARTITION, unsupported};
use crate::error::{Result, internal};

/// One `name=value` directory: the unescaped name and the raw value
/// (`None` for the default partition, which is NULL).
pub type Segment = (String, Option<String>);

/// Reverses [`escape_partition_value`](super::escape_partition_value):
/// `%XX` becomes the byte; anything else (including a malformed escape)
/// stays as written.
#[must_use]
pub fn unescape_partition_value(text: &str) -> String {
    if !text.contains('%') {
        return text.to_owned();
    }
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(h), Some(l)) = (hex(bytes[i + 1]), hex(bytes[i + 2]))
        {
            out.push(h << 4 | l);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex(b: u8) -> Option<u8> {
    char::from(b)
        .to_digit(16)
        .and_then(|d| u8::try_from(d).ok())
}

/// The `name=value` form of one directory name, or `None` when it has no
/// `=`, an empty name or an empty value (Spark does not read those as
/// partitions). `dir` is the name as stored: object stores return keys
/// as written, so only Hive's escaping is undone.
#[must_use]
pub fn parse_segment(dir: &str) -> Option<Segment> {
    let (name, value) = dir.split_once('=')?;
    if name.is_empty() || value.is_empty() {
        return None;
    }
    let value = unescape_partition_value(value);
    let value = (value != DEFAULT_PARTITION).then_some(value);
    Some((unescape_partition_value(name), value))
}

/// The partition columns of a read whose files have the segments `files`
/// (one list per file), typed as Spark infers them when `infer`, else all
/// strings. No files, or files without segments, have no partition
/// columns.
///
/// # Errors
/// `CONFLICTING_DIRECTORY_STRUCTURES` when files differ in depth,
/// `CONFLICTING_PARTITION_COLUMN_NAMES` when they differ in names, and
/// `KRISHIV_UNSUPPORTED_FEATURE` for a timestamp-shaped value.
pub fn partition_columns(files: &[(String, Vec<Segment>)], infer: bool) -> Result<Vec<FieldRef>> {
    let Some((first_path, first)) = files.first() else {
        return Ok(Vec::new());
    };
    for (path, segs) in files {
        if segs.len() != first.len() {
            return Err(ConflictingDirectoryStructures {
                discovered_base_paths: format!("{first_path}\n{path}"),
            }
            .into());
        }
        if segs
            .iter()
            .zip(first)
            .any(|(a, b)| !a.0.eq_ignore_ascii_case(&b.0))
        {
            let lists = |s: &[Segment]| {
                s.iter()
                    .map(|(n, _)| n.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            return Err(ConflictingPartitionColumnNames {
                distinct_part_col_lists: format!(
                    "\tPartition column name list #0: {}\n\tPartition column name list #1: {}",
                    lists(first),
                    lists(segs)
                ),
                suspicious_paths: format!("\t{first_path}\n\t{path}"),
            }
            .into());
        }
    }
    let mut fields = Vec::with_capacity(first.len());
    for (i, (name, _)) in first.iter().enumerate() {
        let mut ty: Option<Ty> = None;
        for (_, segs) in files {
            if let Some(v) = &segs[i].1 {
                let t = if infer { infer_one(v)? } else { Ty::Str };
                ty = Some(ty.map_or(t, |acc| merge(acc, t)));
            }
        }
        let data_type = ty.unwrap_or(Ty::Str).arrow();
        fields.push(Arc::new(Field::new(name, data_type, true)));
    }
    Ok(fields)
}

/// The inferred type of one partition value.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ty {
    Int,
    Long,
    Dec(u8),
    Double,
    Date,
    Str,
}

impl Ty {
    fn arrow(self) -> DataType {
        match self {
            Self::Int => DataType::Int32,
            Self::Long => DataType::Int64,
            Self::Dec(p) => DataType::Decimal128(p, 0),
            Self::Double => DataType::Float64,
            Self::Date => DataType::Date32,
            Self::Str => DataType::Utf8View,
        }
    }
}

/// The type both `a` and `b` values read as: Spark's
/// `findWiderTypeForPartitionColumn`, probed on the oracle. Integers widen
/// to `bigint`; an `int` meets a `double` as `double`; a decimal meets an
/// `int` as `decimal(max(p,10),0)` and a `bigint` as `decimal(max(p,20),0)`;
/// a `bigint` or decimal meeting a `double`, and anything meeting a date or
/// a string, is a string.
fn merge(a: Ty, b: Ty) -> Ty {
    match (a, b) {
        _ if a == b => a,
        (Ty::Dec(p), Ty::Dec(q)) => Ty::Dec(p.max(q)),
        (Ty::Dec(p), Ty::Int) | (Ty::Int, Ty::Dec(p)) => Ty::Dec(p.max(10)),
        (Ty::Dec(p), Ty::Long) | (Ty::Long, Ty::Dec(p)) => Ty::Dec(p.max(20)),
        (Ty::Int, Ty::Long) | (Ty::Long, Ty::Int) => Ty::Long,
        (Ty::Int, Ty::Double) | (Ty::Double, Ty::Int) => Ty::Double,
        _ => Ty::Str,
    }
}

fn infer_one(v: &str) -> Result<Ty> {
    if v.parse::<i32>().is_ok() {
        return Ok(Ty::Int);
    }
    if v.parse::<i64>().is_ok() {
        return Ok(Ty::Long);
    }
    if let Some(p) = decimal_precision(v) {
        return Ok(Ty::Dec(p));
    }
    if is_double(v) {
        return Ok(Ty::Double);
    }
    let b = v.as_bytes();
    if is_date(b) {
        return Ok(Ty::Date);
    }
    if is_timestamp(b) {
        return Err(unsupported(format!(
            "timestamp partition values ({v}): the session time zone does not reach partition discovery yet"
        )));
    }
    Ok(Ty::Str)
}

/// The precision of `v` read as a Java `BigDecimal` with a scale of at
/// most 0 (`1e3` is `decimal(4,0)`, `1.5e3` too, `5.` is `decimal(1,0)`,
/// `1.5` is no decimal), when it fits 38 digits.
fn decimal_precision(v: &str) -> Option<u8> {
    let b = v.as_bytes();
    let b = b
        .strip_prefix(b"-")
        .or_else(|| b.strip_prefix(b"+"))
        .unwrap_or(b);
    let (mantissa, exp) = match b.iter().position(|c| matches!(c, b'e' | b'E')) {
        Some(i) => (&b[..i], Some(&b[i + 1..])),
        None => (b, None),
    };
    let (int, frac) = match mantissa.iter().position(|&c| c == b'.') {
        Some(i) => (&mantissa[..i], &mantissa[i + 1..]),
        None => (mantissa, &[][..]),
    };
    let digits = |s: &[u8]| s.iter().all(u8::is_ascii_digit);
    if !digits(int) || !digits(frac) || (int.is_empty() && frac.is_empty()) {
        return None;
    }
    let exp: i64 = match exp {
        None => 0,
        Some(e) => {
            let (neg, e) = match e.split_first() {
                Some((b'-', rest)) => (true, rest),
                Some((b'+', rest)) => (false, rest),
                _ => (false, e),
            };
            if e.is_empty() || e.len() > 6 || !digits(e) {
                return None;
            }
            let n: i64 = std::str::from_utf8(e).ok()?.parse().ok()?;
            if neg { -n } else { n }
        }
    };
    let scale = i64::try_from(frac.len()).ok()? - exp;
    if scale > 0 {
        return None;
    }
    let unscaled = int
        .iter()
        .chain(frac)
        .skip_while(|c| **c == b'0')
        .count()
        .max(1);
    let precision = i64::try_from(unscaled).ok()? - scale;
    u8::try_from(precision).ok().filter(|p| *p <= 38)
}

/// Whether Java's `String.toDouble` reads `v`: a decimal or exponent form
/// with an optional `d`/`f` suffix, `NaN` or `Infinity`, signed.
fn is_double(v: &str) -> bool {
    let b = v.as_bytes();
    let signed = b
        .strip_prefix(b"-")
        .or_else(|| b.strip_prefix(b"+"))
        .unwrap_or(b);
    if signed == b"NaN" || signed == b"Infinity" {
        return true;
    }
    let body = signed
        .strip_suffix(b"d")
        .or_else(|| signed.strip_suffix(b"D"))
        .or_else(|| signed.strip_suffix(b"f"))
        .or_else(|| signed.strip_suffix(b"F"))
        .unwrap_or(signed);
    is_float(body)
}

/// `digits[.digits][e[+-]digits]` with at least one digit before or after
/// the point.
fn is_float(b: &[u8]) -> bool {
    let (mantissa, exp) = match b.iter().position(|c| matches!(c, b'e' | b'E')) {
        Some(i) => (&b[..i], Some(&b[i + 1..])),
        None => (b, None),
    };
    let (int, frac) = match mantissa.iter().position(|&c| c == b'.') {
        Some(i) => (&mantissa[..i], &mantissa[i + 1..]),
        None => (mantissa, &[][..]),
    };
    let digits = |s: &[u8]| s.iter().all(u8::is_ascii_digit);
    if !digits(int) || !digits(frac) || (int.is_empty() && frac.is_empty()) {
        return false;
    }
    exp.is_none_or(|e| {
        let e = e
            .strip_prefix(b"-")
            .or_else(|| e.strip_prefix(b"+"))
            .unwrap_or(e);
        !e.is_empty() && digits(e)
    })
}

fn is_date(b: &[u8]) -> bool {
    b.len() == 10
        && b.iter().enumerate().all(|(i, c)| {
            if matches!(i, 4 | 7) {
                *c == b'-'
            } else {
                c.is_ascii_digit()
            }
        })
        && {
            let date = StringArray::from(vec![String::from_utf8_lossy(b).into_owned()]);
            let utc = FixedOffset(0);
            let opts = CastOptions {
                mode: CastMode::Legacy,
                time_zone: &utc,
            };
            cast(&date, &DataType::Date32, &opts).is_ok_and(|a| a.null_count() == 0)
        }
}

/// A date followed by `T` or a space and `hh:mm`: what Spark reads as a
/// timestamp.
fn is_timestamp(b: &[u8]) -> bool {
    b.len() >= 16
        && is_date(&b[..10])
        && matches!(b[10], b' ' | b'T')
        && b[11..13].iter().all(u8::is_ascii_digit)
        && b[13] == b':'
        && b[14..16].iter().all(u8::is_ascii_digit)
}

/// The values of a file: the last `columns` directory names above `path`.
///
/// # Errors
/// `INTERNAL_ERROR` when the path has fewer directories, or one that is
/// not `name=value`: listing checked both.
pub fn file_values(path: &Path, columns: usize) -> Result<Vec<Option<String>>> {
    let parts: Vec<_> = path.parts().collect();
    let dirs = parts
        .len()
        .checked_sub(1)
        .and_then(|n| n.checked_sub(columns))
        .and_then(|from| parts.get(from..parts.len() - 1))
        .ok_or_else(|| internal(format!("{path} has fewer than {columns} directories")))?;
    dirs.iter()
        .map(|d| {
            parse_segment(d.as_ref()).map(|(_, v)| v).ok_or_else(|| {
                internal(format!(
                    "{path}: {} is not a partition directory",
                    d.as_ref()
                ))
            })
        })
        .collect()
}

/// A column of `rows` copies of `value` read as `field`'s type (NULL for
/// `None`).
///
/// # Errors
/// `INTERNAL_ERROR` when the value does not read as the type: discovery
/// inferred the type from the same values.
pub fn constant_column(field: &Field, value: Option<&str>, rows: usize) -> Result<ArrayRef> {
    let strings = StringArray::from(vec![value; rows]);
    if field.data_type() == &DataType::Utf8 {
        return Ok(Arc::new(strings));
    }
    if field.data_type() == &DataType::Utf8View {
        return Ok(Arc::new(StringViewArray::from(vec![value; rows])));
    }
    let utc = FixedOffset(0);
    let opts = CastOptions {
        mode: CastMode::Legacy,
        time_zone: &utc,
    };
    let out = cast(&strings, field.data_type(), &opts).map_err(|e| {
        internal(format!(
            "partition value {value:?} as {}: {e}",
            field.data_type()
        ))
    })?;
    if out.null_count() != strings.null_count() {
        return Err(internal(format!(
            "partition value {value:?} is not a {}",
            field.data_type()
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;

    fn seg(dir: &str) -> Segment {
        parse_segment(dir).unwrap()
    }

    fn cols(values: &[&[&str]], names: &[&str], infer: bool) -> Result<Vec<DataType>> {
        let files: Vec<(String, Vec<Segment>)> = values
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let segs = names
                    .iter()
                    .zip(*row)
                    .map(|(n, v)| seg(&format!("{n}={v}")))
                    .collect();
                (format!("f{i}"), segs)
            })
            .collect();
        partition_columns(&files, infer)
            .map(|fs| fs.iter().map(|f| f.data_type().clone()).collect())
    }

    fn one(values: &[&str]) -> DataType {
        let rows: Vec<Vec<&str>> = values.iter().map(|v| vec![*v]).collect();
        let rows: Vec<&[&str]> = rows.iter().map(Vec::as_slice).collect();
        cols(&rows, &["k"], true).unwrap().remove(0)
    }

    #[test]
    fn unescape_reverses_the_writer_and_keeps_malformed_escapes() {
        assert_eq!(unescape_partition_value("a%2Fb%3Dc"), "a/b=c");
        assert_eq!(unescape_partition_value("100%"), "100%");
        assert_eq!(unescape_partition_value("%zz%4"), "%zz%4");
        assert_eq!(unescape_partition_value("plain"), "plain");
        for c in ['/', '=', '%', ':', '\u{1}', '{'] {
            let text = c.to_string();
            assert_eq!(
                unescape_partition_value(&super::super::escape_partition_value(&text)),
                text
            );
        }
    }

    #[test]
    fn segments_need_a_name_and_the_default_partition_is_null() {
        assert_eq!(
            parse_segment("k=v"),
            Some(("k".to_owned(), Some("v".to_owned())))
        );
        assert_eq!(
            parse_segment("k=__HIVE_DEFAULT_PARTITION__"),
            Some(("k".to_owned(), None))
        );
        assert_eq!(parse_segment("k="), None);
        assert_eq!(parse_segment("=v"), None);
        assert_eq!(parse_segment("plain"), None);
        assert_eq!(parse_segment("a%3Db=1").unwrap().0, "a=b");
    }

    /// Each case was run on the Spark 4.2 oracle (`typeof(k)` of a read of
    /// directories `k=<value>`).
    #[test]
    fn types_are_inferred_as_spark_does() {
        let cases: &[(&[&str], DataType)] = &[
            (&["1", "1.5"], DataType::Float64),
            (&["12345678901", "1.5"], DataType::Utf8View),
            (&["1", "12345678901"], DataType::Int64),
            (&["1", "1e3"], DataType::Decimal128(10, 0)),
            (&["1e3", "1.5"], DataType::Utf8View),
            (&["1e3"], DataType::Decimal128(4, 0)),
            (&["1.5e3"], DataType::Decimal128(4, 0)),
            (&["1E3"], DataType::Decimal128(4, 0)),
            (&["5."], DataType::Decimal128(1, 0)),
            (&["1e-3"], DataType::Float64),
            (&[".5"], DataType::Float64),
            (
                &["123456789012345678901234567890", "1"],
                DataType::Decimal128(30, 0),
            ),
            (
                &["123456789012345678901234567890", "1.5"],
                DataType::Utf8View,
            ),
            (&["1e3", "12345678901"], DataType::Decimal128(20, 0)),
            (
                &["1e3", "123456789012345678901234567890"],
                DataType::Decimal128(30, 0),
            ),
            (
                &["123456789012345678901234567890", "12345678901"],
                DataType::Decimal128(30, 0),
            ),
            (&["NaN"], DataType::Float64),
            (&["NaN", "1"], DataType::Float64),
            (&["Infinity"], DataType::Float64),
            (&["NaN", "12345678901"], DataType::Utf8View),
            (&["1.5d"], DataType::Float64),
            (&["1", "2020-01-02"], DataType::Utf8View),
            (&["2020-01-02", "5"], DataType::Utf8View),
            (&["2020-01-02", "2021-12-31"], DataType::Date32),
            (&["007"], DataType::Int32),
            (&["+5", "-7"], DataType::Int32),
            (&["1234567890123456789"], DataType::Int64),
            (
                &["1234567890123456789012345678901234567890"],
                DataType::Float64,
            ),
            (&["2020-01-02 note"], DataType::Utf8View),
            (&["2020-13-45"], DataType::Utf8View),
            (&["true", "false"], DataType::Utf8View),
            (&["1", "a"], DataType::Utf8View),
            (&["1e"], DataType::Utf8View),
            (&["."], DataType::Utf8View),
        ];
        for (values, want) in cases {
            assert_eq!(&one(values), want, "{values:?}");
        }
    }

    #[test]
    fn no_inference_reads_strings_and_nulls_alone_are_strings() {
        assert_eq!(
            cols(&[&["1"]], &["k"], false).unwrap(),
            [DataType::Utf8View]
        );
        let files = vec![("f".to_owned(), vec![seg("k=__HIVE_DEFAULT_PARTITION__")])];
        assert_eq!(
            partition_columns(&files, true).unwrap()[0].data_type(),
            &DataType::Utf8View
        );
        assert!(partition_columns(&[], true).unwrap().is_empty());
    }

    #[test]
    fn a_timestamp_shaped_value_is_refused() {
        let e = cols(&[&["2020-01-02 03:04:05"]], &["k"], true).unwrap_err();
        assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
        assert!(cols(&[&["2020-01-02 03:04:05"]], &["k"], false).is_ok());
    }

    #[test]
    fn files_must_agree_on_depth_and_names() {
        let a = ("a".to_owned(), vec![seg("k=1")]);
        let b = ("b".to_owned(), vec![seg("k=1"), seg("j=2")]);
        let e = partition_columns(&[a.clone(), b], true).unwrap_err();
        assert_eq!(e.error_class().name(), "CONFLICTING_DIRECTORY_STRUCTURES");
        let c = ("c".to_owned(), vec![seg("j=1")]);
        let e = partition_columns(&[a, c], true).unwrap_err();
        assert_eq!(e.error_class().name(), "CONFLICTING_PARTITION_COLUMN_NAMES");
    }

    #[test]
    fn names_match_regardless_of_case() {
        let a = ("a".to_owned(), vec![seg("year=1")]);
        let b = ("b".to_owned(), vec![seg("Year=2")]);
        let cols = partition_columns(&[a, b], true).unwrap();
        assert_eq!(cols[0].name(), "year");
    }

    #[test]
    fn file_values_are_the_directories_above_the_file() {
        // Stores list keys as written: Hive's escape of `a/b` is `a%2Fb`.
        let p = Path::parse("root/year=2020/region=a%2Fb/f.parquet").unwrap();
        assert_eq!(
            file_values(&p, 2).unwrap(),
            [Some("2020".to_owned()), Some("a/b".to_owned())]
        );
        assert_eq!(file_values(&p, 0).unwrap(), Vec::<Option<String>>::new());
        assert!(file_values(&p, 4).is_err());
        assert!(file_values(&p, 3).is_err());
    }

    #[test]
    fn constants_read_as_the_column_type() {
        let int = Field::new("k", DataType::Int32, true);
        let a = constant_column(&int, Some("07"), 3).unwrap();
        assert_eq!(
            a.as_primitive::<arrow_array::types::Int32Type>().values(),
            &[7, 7, 7]
        );
        assert_eq!(constant_column(&int, None, 2).unwrap().null_count(), 2);
        let date = Field::new("k", DataType::Date32, true);
        let a = constant_column(&date, Some("1970-01-02"), 1).unwrap();
        assert_eq!(
            a.as_primitive::<arrow_array::types::Date32Type>().value(0),
            1
        );
        let dec = Field::new("k", DataType::Decimal128(30, 0), true);
        assert_eq!(
            constant_column(&dec, Some(&"9".repeat(30)), 1)
                .unwrap()
                .null_count(),
            0
        );
        let s = Field::new("k", DataType::Utf8, true);
        assert_eq!(
            constant_column(&s, Some("x"), 2)
                .unwrap()
                .as_string::<i32>()
                .value(1),
            "x"
        );
        assert!(constant_column(&int, Some("abc"), 1).is_err());
    }
}
