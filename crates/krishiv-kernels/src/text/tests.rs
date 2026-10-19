// SPDX-License-Identifier: Apache-2.0
//! Expected values observed in Spark 4.2 (CSV reads and writes).

use super::*;
use crate::cast::FixedOffset;

#[test]
fn integers_follow_java_not_cast() {
    assert_eq!(java_parse_long("+1"), Some(1));
    assert_eq!(java_parse_long("-0"), Some(0));
    assert_eq!(java_parse_long("007"), Some(7));
    assert_eq!(java_parse_long("\u{661}\u{662}"), Some(12));
    assert_eq!(java_parse_long("-9223372036854775808"), Some(i64::MIN));
    for bad in [
        " 1",
        "1 ",
        "1.0",
        "0x10",
        "",
        "-",
        "+",
        "9223372036854775808",
        "1e3",
    ] {
        assert_eq!(java_parse_long(bad), None, "{bad:?}");
    }
    assert_eq!(java_parse_integral("127", 8), Some(127));
    assert_eq!(java_parse_integral("128", 8), None);
    assert_eq!(java_parse_integral("-129", 8), None);
    assert_eq!(java_parse_integral("-128", 8), Some(-128));
    assert_eq!(java_parse_integral("32768", 16), None);
    assert_eq!(java_parse_integral("2147483648", 32), None);
}

#[test]
fn floats_follow_java() {
    assert_eq!(java_parse_double(" 2.5 "), Some(2.5));
    assert_eq!(java_parse_double("1.5f"), Some(1.5));
    assert_eq!(java_parse_double("1d"), Some(1.0));
    assert_eq!(java_parse_double("0x1p3"), Some(8.0));
    assert_eq!(java_parse_double("1e999"), Some(f64::INFINITY));
    assert_eq!(java_parse_double("-Infinity"), Some(f64::NEG_INFINITY));
    assert!(java_parse_double("NaN").is_some_and(f64::is_nan));
    assert_eq!(java_parse_double(".5"), Some(0.5));
    assert_eq!(java_parse_double("5."), Some(5.0));
    for bad in ["inf", "Inf", "nan", "0x10", "", "1,0"] {
        assert_eq!(java_parse_double(bad), None, "{bad:?}");
    }
    assert_eq!(java_parse_float("3.4e38"), Some(3.4e38_f32));
    assert_eq!(java_double_string(1e7), "1.0E7");
    assert_eq!(java_double_string(-0.0), "-0.0");
    assert_eq!(java_double_string(f64::NEG_INFINITY), "-Infinity");
    assert_eq!(java_float_string32(1e10), "1.0E10");
    assert_eq!(java_float_string32(f32::NAN), "NaN");
}

#[test]
fn decimals_follow_big_decimal() {
    let d = |s: &str| java_parse_decimal(s, 5, 1);
    assert_eq!(d("1.25"), Some(13));
    assert_eq!(d("-1.25"), Some(-13));
    assert_eq!(d("1e1"), Some(100));
    assert_eq!(d(".5"), Some(5));
    assert_eq!(d("12345.6"), None);
    assert_eq!(d("1234.5"), Some(12345));
    for bad in [" 2.5", "2.5 ", "1.5f", "Inf", "0x10", ""] {
        assert_eq!(d(bad), None, "{bad:?}");
    }
    assert_eq!(java_decimal_string(1, 10), "1E-10");
    assert_eq!(java_decimal_string(-1, 7), "-1E-7");
    assert_eq!(java_decimal_string(1, 6), "0.000001");
    assert_eq!(java_decimal_string(1230, 2), "12.30");
    assert_eq!(java_decimal_string(-50, 2), "-0.50");
    assert_eq!(java_decimal_string(0, 10), "0E-10");
    assert_eq!(java_decimal_string(15, 8), "1.5E-7");
    assert_eq!(
        java_decimal_string(100_000_000_000_000_000_000, 0),
        "100000000000000000000"
    );
    assert_eq!(java_decimal_string(5, -2), "5E+2");
}

#[test]
fn default_datetime_grammars() {
    assert_eq!(spark_string_to_date(" 2020-01-01"), Some(18_262));
    assert_eq!(spark_string_to_date("2020-1-1"), Some(18_262));
    assert_eq!(spark_string_to_date("20200101"), None);
    let utc = FixedOffset(0);
    let ts = |s: &str| spark_string_to_timestamp(s, &utc, 10).unwrap();
    assert_eq!(ts("1970-01-01 01:00:00+01:00"), Some(0));
    assert_eq!(ts("1970-01-01T00:00:00Z"), Some(0));
    assert_eq!(ts("10:00:00"), Some(10 * 86_400_000_000 + 36_000_000_000));
    assert_eq!(ts("T10:00"), Some(10 * 86_400_000_000 + 36_000_000_000));
    assert_eq!(ts("1577836800"), None);
    assert!(spark_string_to_timestamp("2020-01-01 10:00:00 America/Los_Angeles", &utc, 0).is_err());
    assert_eq!(
        spark_string_to_timestamp_ntz("1970-01-01 00:00:01"),
        Some(1_000_000)
    );
    assert_eq!(spark_string_to_timestamp_ntz("1970-01-01 00:00:01Z"), None);
    assert_eq!(spark_string_to_timestamp_ntz("10:00:00"), None);
    assert_eq!(today_in(86_400_000_000 - 1, &FixedOffset(3600)), 1);
}
