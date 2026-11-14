// SPDX-License-Identifier: Apache-2.0
//! A partition value read as the type a user schema gives its column
//! (`spark.read.schema(..)` naming a partition column), as Spark 4.2.0's
//! `PartitioningUtils` converts it: a Java parse per type, not a SQL cast.
//! Pinned by `tests/fixtures/spark-4.2.0/read-user-schema.json` (the
//! `typed ...` cases):
//!
//! | Type | Spark 4.2.0 |
//! |---|---|
//! | `tinyint`..`bigint` | `Integer.parseInt` / `Long.parseLong`: an optional sign and digits (`+1`, `007` read; `1.5`, ` 1` do not) |
//! | `float`, `double` | `Double.parseDouble` (`1.5`, `-2`, `1e3`) |
//! | `decimal(p,s)` | the value's `BigDecimal`; Parquet's vectorized reader then takes its *unscaled* digits at scale `s` (`1.5` as `decimal(5,2)` is `0.15`), a row-based reader (CSV) its value (`1.50`) |
//! | `boolean` | `true` / `false`, any case |
//! | `date` | `yyyy-MM-dd` (`1` does not read) |
//! | `binary` | the value's UTF-8 bytes |
//! | `timestamp` | `1` does not read |
//!
//! A value that does not read is `INVALID_PARTITION_VALUE`, or NULL under
//! `spark.sql.sources.validatePartitionColumns=false`. Values whose reading
//! the goldens do not pin (Java's `NaN`, `Infinity`, `1d`, padded numbers,
//! a `tinyint` past its range, decimals that do not fit, readable
//! timestamps) and other types are refused, never cast another way.

use arrow_schema::{DataType, Field};

/// Field-metadata key on a partition column whose values read as a user
/// type: [`USER_VALUE`] or [`USER_VALUE_UNSCALED`]. [`super::discover::constant_column`]
/// reads such a column's values through [`user_partition_value`].
pub const PARTITION_VALUE_KEY: &str = "krishiv.partition_value";
/// [`PARTITION_VALUE_KEY`]: Spark's conversion of a user type.
pub const USER_VALUE: &str = "user";
/// [`PARTITION_VALUE_KEY`]: as [`USER_VALUE`], decimals by their unscaled
/// digits (Spark's vectorized Parquet and ORC readers).
pub const USER_VALUE_UNSCALED: &str = "user_unscaled";

/// One partition value read as a user type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UserPartitionValue {
    /// NULL: the default partition.
    Null,
    /// The value, as text the type reads exactly (a legacy cast of it gives
    /// Spark's value).
    Text(String),
    /// Spark cannot read it as the type: `INVALID_PARTITION_VALUE`.
    Invalid,
    /// Not pinned to Spark: refused.
    Unsupported,
}

/// `field` marked to read its partition values as a user type.
#[must_use]
pub fn user_partition_field(field: Field, unscaled_decimals: bool) -> Field {
    let mut metadata = field.metadata().clone();
    metadata.insert(
        PARTITION_VALUE_KEY.to_owned(),
        if unscaled_decimals {
            USER_VALUE_UNSCALED
        } else {
            USER_VALUE
        }
        .to_owned(),
    );
    field.with_metadata(metadata)
}

/// Whether `field` reads its values as a user type, and whether decimals by
/// unscaled digits.
#[must_use]
pub fn user_value_mode(field: &Field) -> Option<bool> {
    match field
        .metadata()
        .get(PARTITION_VALUE_KEY)
        .map(String::as_str)
    {
        Some(USER_VALUE) => Some(false),
        Some(USER_VALUE_UNSCALED) => Some(true),
        _ => None,
    }
}

/// `value` (`None` for the default partition) read as `field`'s type, as
/// Spark reads a partition value as a user type; `unscaled_decimals` for a
/// vectorized reader.
#[must_use]
pub fn user_partition_value(
    field: &Field,
    value: Option<&str>,
    unscaled_decimals: bool,
) -> UserPartitionValue {
    use UserPartitionValue as V;
    let Some(v) = value else {
        return V::Null;
    };
    let text = || V::Text(v.to_owned());
    match field.data_type() {
        // Strings as written; binary as their UTF-8 bytes.
        DataType::Utf8
        | DataType::Utf8View
        | DataType::LargeUtf8
        | DataType::Binary
        | DataType::BinaryView
        | DataType::LargeBinary => text(),
        DataType::Int8 | DataType::Int16 | DataType::Int32 => match java_int(v, 32) {
            None => V::Invalid,
            Some(n) if fits(n, field.data_type()) => text(),
            // `Integer.parseInt(v).toByte` would wrap: not pinned.
            Some(_) => V::Unsupported,
        },
        DataType::Int64 => java_int(v, 64).map_or(V::Invalid, |_| text()),
        DataType::Float32 | DataType::Float64 => {
            if plain_number(v) {
                text()
            } else if java_double_only(v) {
                V::Unsupported
            } else {
                V::Invalid
            }
        }
        DataType::Decimal32(p, s)
        | DataType::Decimal64(p, s)
        | DataType::Decimal128(p, s)
        | DataType::Decimal256(p, s) => decimal(v, *p, *s, unscaled_decimals),
        DataType::Boolean => {
            if v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("false") {
                V::Text(v.to_ascii_lowercase())
            } else {
                V::Invalid
            }
        }
        DataType::Date32 => {
            if iso_date(v) {
                text()
            } else if v.bytes().all(|b| b.is_ascii_digit() || b == b'-') && !v.contains("--") {
                // `1`, `2024`, `2024-1-2`: `1` is pinned as invalid; the
                // other shapes a SQL cast would take are not pinned.
                if v.contains('-') {
                    V::Unsupported
                } else {
                    V::Invalid
                }
            } else {
                V::Invalid
            }
        }
        DataType::Timestamp(..) => {
            if v.len() >= 10 && iso_date(&v[..10]) {
                // Readable as a timestamp in the session zone, which the
                // scan does not have.
                V::Unsupported
            } else {
                V::Invalid
            }
        }
        _ => V::Unsupported,
    }
}

/// Java's `Integer.parseInt` / `Long.parseLong` (`bits` 32 or 64): an
/// optional sign and ASCII digits, in range.
fn java_int(v: &str, bits: u32) -> Option<i64> {
    let digits = v.strip_prefix(['+', '-']).unwrap_or(v);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n: i64 = v.strip_prefix('+').unwrap_or(v).parse().ok()?;
    (bits == 64 || i32::try_from(n).is_ok()).then_some(n)
}

fn fits(n: i64, t: &DataType) -> bool {
    match t {
        DataType::Int8 => i8::try_from(n).is_ok(),
        DataType::Int16 => i16::try_from(n).is_ok(),
        _ => i32::try_from(n).is_ok(),
    }
}

/// `[+-]digits[.digits][e[+-]digits]` (also `.5`, `5.`): what both Java's
/// `parseDouble` / `BigDecimal` and a SQL cast read alike.
fn plain_number(v: &str) -> bool {
    let b = v.strip_prefix(['+', '-']).unwrap_or(v).as_bytes();
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
    match exp {
        None => true,
        Some(e) => {
            let e = e
                .strip_prefix(b"+")
                .or_else(|| e.strip_prefix(b"-"))
                .unwrap_or(e);
            !e.is_empty() && digits(e)
        }
    }
}

/// A value Java's `parseDouble` may read but [`plain_number`] does not:
/// padding, `NaN`, `Infinity`, a `d`/`f` suffix, hexadecimal.
fn java_double_only(v: &str) -> bool {
    v.trim() != v
        || v.contains("NaN")
        || v.contains("Infinity")
        || v.ends_with(['d', 'D', 'f', 'F'])
        || {
            let b = v.trim_start_matches(['+', '-']);
            b.starts_with("0x") || b.starts_with("0X")
        }
}

/// `yyyy-MM-dd`, a valid date.
fn iso_date(v: &str) -> bool {
    let b = v.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return false;
    }
    let num = |s: &[u8]| -> Option<u32> {
        s.iter().try_fold(0_u32, |acc, c| {
            c.is_ascii_digit().then(|| acc * 10 + u32::from(c - b'0'))
        })
    };
    let (Some(y), Some(m), Some(d)) = (num(&b[..4]), num(&b[5..7]), num(&b[8..])) else {
        return false;
    };
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let days = match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    (1..=days).contains(&d)
}

/// A decimal partition value: Java's `BigDecimal` of `v`, at the column's
/// precision and scale, by its value (row readers) or by its unscaled digits
/// (`unscaled`, vectorized readers).
fn decimal(v: &str, precision: u8, scale: i8, unscaled: bool) -> UserPartitionValue {
    use UserPartitionValue as V;
    if !plain_number(v) {
        return if java_double_only(v) {
            V::Unsupported
        } else {
            V::Invalid
        };
    }
    if v.contains(['e', 'E']) {
        // An exponent changes the unscaled digits and the scale: not pinned.
        return V::Unsupported;
    }
    let Ok(scale) = usize::try_from(scale) else {
        return V::Unsupported;
    };
    let (negative, body) = match v.as_bytes().first() {
        Some(b'-') => (true, &v[1..]),
        Some(b'+') => (false, &v[1..]),
        _ => (false, v),
    };
    let (int, frac) = body.split_once('.').unwrap_or((body, ""));
    let sign = if negative { "-" } else { "" };
    let (int, frac) = if unscaled {
        // The unscaled digits, read at the column's scale.
        let digits = format!("{int}{frac}");
        let digits = digits.trim_start_matches('0');
        let padded = format!("{digits:0>width$}", width = scale + 1);
        let (i, f) = padded.split_at(padded.len() - scale);
        (i.to_owned(), f.to_owned())
    } else {
        if frac.len() > scale {
            // Rounding to the column's scale: not pinned.
            return V::Unsupported;
        }
        (int.to_owned(), format!("{frac:0<scale$}"))
    };
    let int = int.trim_start_matches('0');
    if int.len() + scale > usize::from(precision) {
        return V::Unsupported;
    }
    let int = if int.is_empty() { "0" } else { int };
    V::Text(if scale == 0 {
        format!("{sign}{int}")
    } else {
        format!("{sign}{int}.{frac}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(t: DataType, v: &str, unscaled: bool) -> UserPartitionValue {
        user_partition_value(&Field::new("k", t, true), Some(v), unscaled)
    }

    fn text(s: &str) -> UserPartitionValue {
        UserPartitionValue::Text(s.to_owned())
    }

    #[test]
    fn values_read_as_spark_4_2_reads_them() {
        use UserPartitionValue::{Invalid, Unsupported};
        // The goldens' `typed ...` cases.
        assert_eq!(read(DataType::Int32, "1.5", false), Invalid);
        assert_eq!(read(DataType::Int32, " 1", false), Invalid);
        assert_eq!(read(DataType::Int32, "+1", false), text("+1"));
        assert_eq!(read(DataType::Int32, "007", false), text("007"));
        assert_eq!(read(DataType::Int8, "2", false), text("2"));
        assert_eq!(read(DataType::Int8, "300", false), Unsupported);
        assert_eq!(read(DataType::Float64, "1e3", false), text("1e3"));
        assert_eq!(read(DataType::Float64, "NaN", false), Unsupported);
        assert_eq!(read(DataType::Float64, "abc", false), Invalid);
        assert_eq!(read(DataType::Boolean, "FALSE", false), text("false"));
        assert_eq!(read(DataType::Boolean, "1", false), Invalid);
        assert_eq!(
            read(DataType::Date32, "2024-01-02", false),
            text("2024-01-02")
        );
        assert_eq!(read(DataType::Date32, "1", false), Invalid);
        assert_eq!(read(DataType::Date32, "2024-1-2", false), Unsupported);
        let ts = DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None);
        assert_eq!(read(ts.clone(), "1", false), Invalid);
        assert_eq!(read(ts, "2024-01-02 03:04:05", false), Unsupported);
        assert_eq!(read(DataType::Binary, "1", false), text("1"));
        let list = DataType::List(std::sync::Arc::new(Field::new("e", DataType::Int32, true)));
        assert_eq!(read(list, "1", false), Unsupported);
        assert_eq!(
            user_partition_value(&Field::new("k", DataType::Int32, true), None, false),
            UserPartitionValue::Null
        );
    }

    #[test]
    fn decimals_read_by_value_or_by_unscaled_digits() {
        let d52 = DataType::Decimal128(5, 2);
        // Parquet (vectorized): the unscaled digits at the column's scale.
        assert_eq!(read(d52.clone(), "1", true), text("0.01"));
        assert_eq!(read(d52.clone(), "1.5", true), text("0.15"));
        assert_eq!(read(d52.clone(), "-12", true), text("-0.12"));
        assert_eq!(read(DataType::Decimal128(10, 0), "1.5", true), text("15"));
        // CSV (rows): the value.
        assert_eq!(read(d52.clone(), "1", false), text("1.00"));
        assert_eq!(read(d52.clone(), "1.5", false), text("1.50"));
        assert_eq!(read(d52.clone(), "-12", false), text("-12.00"));
        assert_eq!(
            read(d52.clone(), "1.555", false),
            UserPartitionValue::Unsupported
        );
        assert_eq!(
            read(d52.clone(), "12345", false),
            UserPartitionValue::Unsupported
        );
        assert_eq!(read(d52, "x", false), UserPartitionValue::Invalid);
    }
}
