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
    // Java reads UTF-16 units: a digit outside the BMP is two surrogates and no digit (Spark 4.2.0,
    // a CSV bigint/int column: `\u{663}` is 3, U+1D7D9 is NULL).
    assert_eq!(java_parse_long("\u{663}"), Some(3));
    assert_eq!(java_parse_long("\u{1d7d9}"), None);
    assert_eq!(java_parse_long("1\u{1d7da}"), None);
    assert_eq!(java_parse_integral("\u{1d7d9}", 32), None);
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
    // A region id in the text resolves against the bundled database:
    // 10:00 in Los Angeles (UTC-08:00 in January) is 18:00 UTC.
    assert_eq!(
        ts("2020-01-01 10:00:00 America/Los_Angeles"),
        Some(18_262 * 86_400_000_000 + 18 * 3_600_000_000)
    );
    // ... while an unknown one reads as no timestamp, as in Spark's
    // lenient readers.
    assert_eq!(
        spark_string_to_timestamp("2020-01-01 10:00:00 America/Nope", &utc, 0).unwrap(),
        None
    );
    assert_eq!(
        spark_string_to_timestamp_ntz("1970-01-01 00:00:01"),
        Some(1_000_000)
    );
    assert_eq!(spark_string_to_timestamp_ntz("1970-01-01 00:00:01Z"), None);
    assert_eq!(spark_string_to_timestamp_ntz("10:00:00"), None);
    assert_eq!(today_in(86_400_000_000 - 1, &FixedOffset(3600)), 1);
}

#[test]
fn a_decimal_must_stay_below_ten_to_the_precision() {
    assert_eq!(java_parse_decimal("999", 3, 0), Some(999));
    assert_eq!(java_parse_decimal("-999", 3, 0), Some(-999));
    // 10^precision itself does not fit.
    assert_eq!(java_parse_decimal("1000", 3, 0), None);
    assert_eq!(java_parse_decimal("-1000", 3, 0), None);
    // Rounding to the scale can reach the limit too.
    assert_eq!(java_parse_decimal("9.995", 3, 2), None);
    assert_eq!(java_parse_decimal("9.994", 3, 2), Some(999));
}

#[test]
fn infinities_print_with_their_sign_in_both_widths() {
    assert_eq!(java_double_string(f64::INFINITY), "Infinity");
    assert_eq!(java_double_string(f64::NEG_INFINITY), "-Infinity");
    assert_eq!(java_float_string32(f32::INFINITY), "Infinity");
    assert_eq!(java_float_string32(f32::NEG_INFINITY), "-Infinity");
    assert_eq!(java_float_string32(f32::NAN), "NaN");
    assert_eq!(java_float_string32(-1.5), "-1.5");
    assert_eq!(java_float_string32(0.0), "0.0");
}

#[test]
fn a_time_of_day_is_trimmed_of_whitespace_and_control_characters() {
    let utc = FixedOffset(0);
    let want = Some(10 * 86_400_000_000 + 36_000_000_000);
    let ts = |s: &str| spark_string_to_timestamp(s, &utc, 10).unwrap();
    assert_eq!(ts("10:00:00"), want);
    assert_eq!(ts(" 10:00:00 "), want);
    assert_eq!(ts("\t10:00:00\n"), want);
    assert_eq!(ts("\u{7f}10:00:00\u{7f}"), want);
}

#[test]
fn today_is_the_local_date_in_the_zone_named() {
    use crate::cast::timestamp::Zone;
    let day = 20_000_i64;
    let midnight = day * 86_400_000_000;
    // UTC midnight is still the day before at UTC-12, and the same day at UTC+1.
    assert_eq!(
        today_in_zone(&Zone::Session, &FixedOffset(0), midnight),
        Some(20_000)
    );
    assert_eq!(
        today_in_zone(&Zone::Session, &FixedOffset(-43_200), midnight),
        Some(19_999)
    );
    assert_eq!(
        today_in_zone(&Zone::Offset(-43_200), &FixedOffset(0), midnight),
        Some(19_999)
    );
    assert_eq!(
        today_in_zone(&Zone::Offset(3_600), &FixedOffset(0), midnight),
        Some(20_000)
    );
    assert_eq!(
        today_in_zone(
            &Zone::Offset(3_600),
            &FixedOffset(0),
            midnight - 3_601_000_000
        ),
        Some(19_999)
    );
    assert_eq!(
        today_in_zone(
            &Zone::Region("America/Los_Angeles".into()),
            &FixedOffset(0),
            midnight
        ),
        Some(19_999)
    );
    assert_eq!(
        today_in_zone(&Zone::Region("Not/AZone".into()), &FixedOffset(0), midnight),
        None
    );
    // The same on the instant that starts the day.
    assert_eq!(today_in(midnight, &FixedOffset(0)), 20_000);
    assert_eq!(today_in(midnight, &FixedOffset(-1)), 19_999);
}
