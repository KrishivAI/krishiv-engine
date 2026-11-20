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
//! | `decimal(p,s)` | the value's `BigDecimal`; a row-based reader (CSV) reads its value (`1.5` as `decimal(5,2)` is `1.50`); Parquet's vectorized reader shows its *unscaled* digits at scale `s` (`0.15`) but filters on the value, so only a value written at the column's scale (`1.50`) reads alike both ways there |
//! | `boolean` | `true` / `false`, any case |
//! | `date` | `CAST(raw AS DATE)`: `2024`, `2024-1`, `2024-01-02 xyz` read; `1` and `abc` do not |
//! | `binary` | the value's UTF-8 bytes |
//! | `timestamp` | `CAST(value AS TIMESTAMP)` in the read's zone (the `timeZone` option, else the session zone); no fallback: the date fallback Spark tries has no zone and always fails |
//! | `timestamp_ntz` | `CAST(value AS TIMESTAMP_NTZ)`, else (ANSI) `CAST(CAST(raw AS DATE) AS TIMESTAMP_NTZ)` |
//!
//! Dates and timestamps are SQL casts (`castPartValueToDesiredType`), so the
//! session's ANSI mode decides: under ANSI a value that does not read raises,
//! otherwise the cast gives NULL and nothing is raised (even with
//! `validatePartitionColumns`). `raw` is the directory's escaped text,
//! `value` its unescaped form; a value whose two forms cast differently is
//! refused. The zone and the mode travel in the field's metadata
//! ([`PARTITION_VALUE_ZONE_KEY`], [`PARTITION_VALUE_ANSI_KEY`]), so the scan
//! reads values as listing checked them.
//!
//! A value that does not read is `INVALID_PARTITION_VALUE`, or NULL under
//! `spark.sql.sources.validatePartitionColumns=false`. Values whose reading
//! the goldens do not pin (Java's `NaN`, `Infinity`, `1d`, padded numbers,
//! a `tinyint` past its range, decimals that do not fit, a decimal on a
//! vectorized reader not written at the column's scale) and other types are
//! refused, never cast another way.

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, TimestampMicrosecondType};
use arrow_array::{Array, StringArray};
use arrow_schema::{DataType, Field, TimeUnit};
use krishiv_kernels::cast::{CastMode, CastOptions, FixedOffset, TimeZoneRules, cast};
use krishiv_kernels::datetime::{ZoneRef, resolve_zone};

/// Field-metadata key on a partition column whose values read as a user
/// type: [`USER_VALUE`] or [`USER_VALUE_EXACT_SCALE`]. [`super::discover::constant_column`]
/// reads such a column's values through [`user_partition_value`].
pub const PARTITION_VALUE_KEY: &str = "krishiv.partition_value";
/// [`PARTITION_VALUE_KEY`]: Spark's conversion of a user type.
pub const USER_VALUE: &str = "user";
/// [`PARTITION_VALUE_KEY`]: as [`USER_VALUE`], decimals only when written at
/// the column's scale (Spark's vectorized Parquet and ORC readers show their
/// unscaled digits but filter on their value; the two agree only then).
pub const USER_VALUE_EXACT_SCALE: &str = "user_exact_scale";
/// Field-metadata key on a user-typed date or timestamp partition column:
/// the zone id its values are cast in (the read's `timeZone` option, else
/// `spark.sql.session.timeZone`). Absent: UTC.
pub const PARTITION_VALUE_ZONE_KEY: &str = "krishiv.partition_value_zone";
/// Field-metadata key on a user-typed date or timestamp partition column:
/// `false` when `spark.sql.ansi.enabled` was off for the read (a value that
/// does not cast is NULL, not an error). Absent: ANSI.
pub const PARTITION_VALUE_ANSI_KEY: &str = "krishiv.partition_value_ansi";

/// Microseconds in a day.
const MICROS_PER_DAY: i64 = 86_400_000_000;

/// One partition value read as a user type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UserPartitionValue {
    /// NULL: the default partition.
    Null,
    /// The value, as text the type reads exactly (a legacy cast of it gives
    /// Spark's value).
    Text(String),
    /// A `DATE`: days since the epoch.
    Date(i32),
    /// A `TIMESTAMP` or `TIMESTAMP_NTZ`: microseconds since the epoch.
    Timestamp(i64),
    /// Spark cannot read it as the type: `INVALID_PARTITION_VALUE`.
    Invalid,
    /// Not pinned to Spark: refused.
    Unsupported,
}

/// `field` marked to read its partition values as a user type; a date or
/// timestamp column also records the zone id it casts in (`zone`) and
/// whether the read ran under ANSI (`ansi`).
#[must_use]
pub fn user_partition_field(
    field: Field,
    exact_scale_decimals: bool,
    zone: &str,
    ansi: bool,
) -> Field {
    let mut metadata = field.metadata().clone();
    if matches!(
        field.data_type(),
        DataType::Date32 | DataType::Timestamp(..)
    ) {
        metadata.insert(PARTITION_VALUE_ZONE_KEY.to_owned(), zone.to_owned());
        metadata.insert(PARTITION_VALUE_ANSI_KEY.to_owned(), ansi.to_string());
    }
    metadata.insert(
        PARTITION_VALUE_KEY.to_owned(),
        if exact_scale_decimals {
            USER_VALUE_EXACT_SCALE
        } else {
            USER_VALUE
        }
        .to_owned(),
    );
    field.with_metadata(metadata)
}

/// Whether `field` reads its values as a user type, and whether decimals only
/// at the column's scale.
#[must_use]
pub fn user_value_mode(field: &Field) -> Option<bool> {
    match field
        .metadata()
        .get(PARTITION_VALUE_KEY)
        .map(String::as_str)
    {
        Some(USER_VALUE) => Some(false),
        Some(USER_VALUE_EXACT_SCALE) => Some(true),
        _ => None,
    }
}

/// `value` (`None` for the default partition) read as `field`'s type, as
/// Spark reads a partition value as a user type; `exact_scale_decimals` for a
/// vectorized reader.
#[must_use]
pub fn user_partition_value(
    field: &Field,
    value: Option<&str>,
    exact_scale_decimals: bool,
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
        // Java also reads non-ASCII digits (`١`): not pinned.
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
            if v.chars().any(|c| !c.is_ascii() && c.is_numeric()) =>
        {
            V::Unsupported
        }
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
        | DataType::Decimal256(p, s) => decimal(v, *p, *s, exact_scale_decimals),
        DataType::Boolean => {
            if v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("false") {
                V::Text(v.to_ascii_lowercase())
            } else {
                V::Invalid
            }
        }
        DataType::Date32 | DataType::Timestamp(..) => datetime(field, v),
        _ => V::Unsupported,
    }
}

/// A date or timestamp partition value, as Spark 4.2.0's
/// `castPartValueToDesiredType` casts it.
fn datetime(field: &Field, v: &str) -> UserPartitionValue {
    use UserPartitionValue as V;
    let metadata = field.metadata();
    let ansi = metadata
        .get(PARTITION_VALUE_ANSI_KEY)
        .is_none_or(|a| a != "false");
    let zone = match metadata.get(PARTITION_VALUE_ZONE_KEY) {
        None => ZoneRef::Fixed(FixedOffset(0)),
        Some(z) => match resolve_zone(z) {
            Ok(zone) => zone,
            // Spark fails the read on a zone id it cannot resolve.
            Err(_) => return V::Unsupported,
        },
    };
    // Not under ANSI, the cast gives NULL and nothing is raised.
    let unread = if ansi { V::Invalid } else { V::Null };
    match field.data_type() {
        DataType::Date32 => match raw_date(v) {
            Raw::Date(d) => V::Date(d),
            Raw::None => unread,
            Raw::Differs => V::Unsupported,
        },
        DataType::Timestamp(TimeUnit::Microsecond, tz) => {
            match cast_one(v, field.data_type(), &zone) {
                Some(us) => V::Timestamp(us),
                None if !ansi => V::Null,
                // `TIMESTAMP`: Spark's fallback casts the date to a zoned
                // timestamp without a zone, which always raises.
                None if tz.is_some() => V::Invalid,
                None => match raw_date(v) {
                    Raw::Date(d) => V::Timestamp(i64::from(d) * MICROS_PER_DAY),
                    Raw::None => V::Invalid,
                    Raw::Differs => V::Unsupported,
                },
            }
        }
        _ => V::Unsupported,
    }
}

/// `CAST(raw AS DATE)` of a value, where `raw` is the directory's escaped
/// text.
enum Raw {
    Date(i32),
    None,
    /// The escaped and unescaped forms cast differently: the escaped text on
    /// disk is not known here.
    Differs,
}

fn raw_date(v: &str) -> Raw {
    let utc = FixedOffset(0);
    let plain = cast_one(v, &DataType::Date32, &utc);
    let escaped = super::escape_partition_value(v);
    if escaped != v && cast_one(&escaped, &DataType::Date32, &utc) != plain {
        return Raw::Differs;
    }
    plain.map_or(Raw::None, |d| {
        i32::try_from(d).map_or(Raw::Differs, Raw::Date)
    })
}

/// The legacy cast of the string `v` to `to` (`Date32` or `Timestamp(µs)`)
/// in `zone`: the days or microseconds, or `None` where Spark's cast gives
/// NULL (and ANSI raises).
fn cast_one(v: &str, to: &DataType, zone: &dyn TimeZoneRules) -> Option<i64> {
    let opts = CastOptions {
        mode: CastMode::Legacy,
        time_zone: zone,
    };
    let out = cast(&StringArray::from(vec![v]), to, &opts).ok()?;
    if out.is_null(0) {
        return None;
    }
    match to {
        DataType::Date32 => Some(i64::from(out.as_primitive::<Date32Type>().value(0))),
        _ => out
            .as_primitive_opt::<TimestampMicrosecondType>()
            .map(|a| a.value(0)),
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

/// A decimal partition value: Java's `BigDecimal` of `v`, at the column's
/// precision and scale, by its value; with `exact_scale` (vectorized readers)
/// only a value written at the column's scale.
fn decimal(v: &str, precision: u8, scale: i8, exact_scale: bool) -> UserPartitionValue {
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
    if exact_scale && frac.len() != scale {
        // A vectorized reader shows the unscaled digits at the column's scale
        // but filters on the value: they differ unless the written scale is
        // the column's, and neither can be reproduced as one value.
        return V::Unsupported;
    }
    if frac.len() > scale {
        // Rounding to the column's scale: not pinned.
        return V::Unsupported;
    }
    let (int, frac) = (int.to_owned(), format!("{frac:0<scale$}"));
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
            UserPartitionValue::Date(19_724)
        );
        assert_eq!(read(DataType::Date32, "1", false), Invalid);
        let ts = DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None);
        assert_eq!(read(ts, "1", false), Invalid);
        assert_eq!(read(DataType::Binary, "1", false), text("1"));
        let list = DataType::List(std::sync::Arc::new(Field::new("e", DataType::Int32, true)));
        assert_eq!(read(list, "1", false), Unsupported);
        assert_eq!(
            user_partition_value(&Field::new("k", DataType::Int32, true), None, false),
            UserPartitionValue::Null
        );
    }

    #[test]
    fn decimals_read_by_value_and_only_at_their_scale_on_vectorized_readers() {
        use UserPartitionValue::{Invalid, Unsupported};
        let d52 = DataType::Decimal128(5, 2);
        // Parquet (vectorized): only a value written at the column's scale,
        // where its shown and filtered values agree.
        assert_eq!(read(d52.clone(), "1.50", true), text("1.50"));
        assert_eq!(read(d52.clone(), "-3.25", true), text("-3.25"));
        assert_eq!(read(d52.clone(), "1.5", true), Unsupported);
        assert_eq!(read(d52.clone(), "1", true), Unsupported);
        assert_eq!(read(DataType::Decimal128(10, 0), "-12", true), text("-12"));
        assert_eq!(read(DataType::Decimal128(10, 0), "1.5", true), Unsupported);
        // CSV (rows): the value.
        assert_eq!(read(d52.clone(), "1", false), text("1.00"));
        assert_eq!(read(d52.clone(), "1.5", false), text("1.50"));
        assert_eq!(read(d52.clone(), "-12", false), text("-12.00"));
        assert_eq!(read(d52.clone(), "1.555", false), Unsupported);
        assert_eq!(read(d52.clone(), "12345", false), Unsupported);
        assert_eq!(read(d52, "x", false), Invalid);
    }

    fn typed(t: DataType, zone: &str, ansi: bool) -> Field {
        user_partition_field(Field::new("k", t, true), true, zone, ansi)
    }

    fn at(field: &Field, v: &str) -> UserPartitionValue {
        user_partition_value(field, Some(v), true)
    }

    const JAN_1_2024: i32 = 19_723;
    const DAY: i64 = 86_400_000_000;

    #[test]
    fn dates_read_as_spark_casts_the_raw_text() {
        use UserPartitionValue::{Date, Invalid, Null, Unsupported};
        let date = typed(DataType::Date32, "UTC", true);
        assert_eq!(at(&date, "2024"), Date(JAN_1_2024));
        assert_eq!(at(&date, "2024-1"), Date(JAN_1_2024));
        assert_eq!(at(&date, "2024-01-02 xyz"), Date(JAN_1_2024 + 1));
        assert_eq!(at(&date, "2024-01-02 03:04:05"), Date(JAN_1_2024 + 1));
        assert_eq!(at(&date, "abc"), Invalid);
        assert_eq!(at(&date, "1"), Invalid);
        // The zone does not move a date.
        let la = typed(DataType::Date32, "America/Los_Angeles", true);
        assert_eq!(at(&la, "2024"), Date(JAN_1_2024));
        // Not under ANSI the cast gives NULL, never an error.
        let legacy = typed(DataType::Date32, "UTC", false);
        assert_eq!(at(&legacy, "abc"), Null);
        assert_eq!(at(&legacy, "2024"), Date(JAN_1_2024));
        // A control character Spark escapes on disk: `2024%09` does not
        // read, `2024\t` would.
        assert_eq!(at(&date, "2024\t"), Unsupported);
        // An unknown zone id.
        assert_eq!(
            at(&typed(DataType::Date32, "Nowhere/x", true), "2024"),
            Unsupported
        );
        assert_eq!(at(&date, "2024-01-02T03"), Date(JAN_1_2024 + 1));
        assert_eq!(
            at(&Field::new("k", DataType::Date32, true), "2024"),
            Date(JAN_1_2024)
        );
    }

    #[test]
    fn timestamps_cast_in_the_recorded_zone() {
        use UserPartitionValue::{Invalid, Null, Timestamp};
        let ltz = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
        let ntz = DataType::Timestamp(TimeUnit::Microsecond, None);
        let jan1 = i64::from(JAN_1_2024) * DAY;
        let utc = typed(ltz.clone(), "UTC", true);
        assert_eq!(at(&utc, "2024"), Timestamp(jan1));
        assert_eq!(
            at(&utc, "2024-01-02 03:04:05"),
            Timestamp(jan1 + DAY + 11_045_000_000)
        );
        // Spark 4.2.0: a zoned timestamp has no working date fallback.
        assert_eq!(at(&utc, "2024-01-02 xyz"), Invalid);
        assert_eq!(at(&utc, "abc"), Invalid);
        let la = typed(ltz.clone(), "America/Los_Angeles", true);
        assert_eq!(at(&la, "2024"), Timestamp(jan1 + 8 * 3_600_000_000));
        assert_eq!(
            at(&la, "2024-01-02T03:04:05Z"),
            Timestamp(jan1 + DAY + 11_045_000_000)
        );
        let legacy = typed(ltz, "UTC", false);
        assert_eq!(at(&legacy, "2024-01-02 xyz"), Null);
        assert_eq!(at(&legacy, "1"), Null);
        // TIMESTAMP_NTZ: the zone does not apply; under ANSI a value that
        // does not cast falls back to its date.
        let n = typed(ntz.clone(), "America/Los_Angeles", true);
        assert_eq!(at(&n, "2024"), Timestamp(jan1));
        assert_eq!(at(&n, "2024-01-02 xyz"), Timestamp(jan1 + DAY));
        assert_eq!(at(&n, "abc"), Invalid);
        assert_eq!(at(&typed(ntz, "UTC", false), "2024-01-02 xyz"), Null);
        // Other timestamp units are not partition types.
        let ms = typed(
            DataType::Timestamp(TimeUnit::Millisecond, None),
            "UTC",
            true,
        );
        assert_eq!(at(&ms, "2024"), UserPartitionValue::Unsupported);
        assert_eq!(
            read(DataType::Int32, "١", false),
            UserPartitionValue::Unsupported
        );
    }
}
