// SPDX-License-Identifier: Apache-2.0
//! `CAST` kernels against values and errors recorded from Spark 4.2, and
//! property tests (round trips, ANSI vs `try_cast` consistency, NULL rows
//! never raising).

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test helpers outside #[test] fns fail loudly"
)]

use std::sync::Arc;

use arrow_array::types::{Date32Type, Decimal128Type, Float64Type, Int32Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
    Int8Array, Int16Array, Int32Array, Int64Array, LargeStringArray, PrimitiveArray, StringArray,
    TimestampMicrosecondArray,
};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::error::ErrorClass;
use krishiv_kernels::KernelError;
use krishiv_kernels::cast::{CastMode, CastOptions, FixedOffset, cast};
use proptest::prelude::*;

const UTC: FixedOffset = FixedOffset(0);

fn opts(mode: CastMode) -> CastOptions<'static> {
    CastOptions {
        mode,
        time_zone: &UTC,
    }
}

fn run(a: &dyn Array, to: &DataType, mode: CastMode) -> Result<ArrayRef, KernelError> {
    cast(a, to, &opts(mode))
}

/// The single row of a one-row cast rendered as Spark's `CAST(... AS
/// STRING)` would show it (`NULL` for NULL), or the error.
#[expect(
    clippy::needless_pass_by_value,
    reason = "call sites build the operands inline"
)]
fn one(a: ArrayRef, to: DataType, mode: CastMode) -> String {
    match run(&a, &to, mode) {
        Ok(out) => {
            let s = cast(&out, &DataType::Utf8, &opts(CastMode::Ansi)).unwrap();
            let s = s.as_any().downcast_ref::<StringArray>().unwrap();
            if s.is_null(0) {
                "NULL".to_owned()
            } else {
                s.value(0).to_owned()
            }
        }
        Err(e) => e.to_string(),
    }
}

fn s(v: &str) -> ArrayRef {
    Arc::new(StringArray::from(vec![v]))
}

fn i32s(v: i32) -> ArrayRef {
    Arc::new(Int32Array::from(vec![v]))
}

fn f64s(v: f64) -> ArrayRef {
    Arc::new(Float64Array::from(vec![v]))
}

fn dec(v: i128, p: u8, sc: i8) -> ArrayRef {
    Arc::new(
        Decimal128Array::from(vec![v])
            .with_precision_and_scale(p, sc)
            .unwrap(),
    )
}

fn invalid(v: &str, to: &str) -> String {
    format!(
        "[CAST_INVALID_INPUT] The value {v} of the type \"STRING\" cannot be cast to \"{to}\" \
         because it is malformed. Correct the value as per the syntax, or change its target type. \
         Use `try_cast` to tolerate malformed input and return NULL instead. SQLSTATE: 22018"
    )
}

fn overflow(v: &str, from: &str, to: &str) -> String {
    format!(
        "[CAST_OVERFLOW] The value {v} of the type \"{from}\" cannot be cast to \"{to}\" due to an \
         overflow. Use `try_cast` to tolerate overflow and return NULL instead. SQLSTATE: 22003"
    )
}

fn out_of_range(v: &str, p: u8, sc: i8) -> String {
    format!(
        "[NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION]  {v} cannot be represented as Decimal({p}, \
         {sc}). If necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error, and \
         return NULL instead. SQLSTATE: 22003"
    )
}

use CastMode::{Ansi, Legacy, Try};

#[test]
fn string_to_numbers_match_spark() {
    let int = DataType::Int32;
    assert_eq!(one(s(" 12 "), int.clone(), Ansi), "12");
    // DEL is trimmed for integral targets (`trimAll`), not for decimal or
    // double ones (`String.trim`): Spark 4.0-4.2.
    for (to, mode) in [
        (int.clone(), Ansi),
        (DataType::Int8, Legacy),
        (DataType::Int64, Ansi),
    ] {
        assert_eq!(one(s("\u{7f}12\u{7f}"), to, mode), "12");
    }
    assert_eq!(
        one(s("1.5\u{7f}"), DataType::Decimal128(10, 2), Legacy),
        "NULL"
    );
    assert_eq!(one(s("1.5\u{7f}"), DataType::Float64, Legacy), "NULL");
    assert_eq!(one(s("12.0"), int.clone(), Ansi), invalid("'12.0'", "INT"));
    assert_eq!(one(s("12.7"), int.clone(), Legacy), "12");
    assert_eq!(
        one(s("2147483648"), int.clone(), Ansi),
        invalid("'2147483648'", "INT")
    );
    assert_eq!(one(s("2147483648"), int.clone(), Legacy), "NULL");
    assert_eq!(
        one(s("it's"), int.clone(), Ansi),
        invalid("'it\\'s'", "INT")
    );
    assert_eq!(one(s("abc"), int.clone(), Try), "NULL");
    // `try_cast` parses as ANSI does: a fraction is malformed, not dropped
    // (Spark 4.2 `try_cast.sql`: `TRY_CAST('1.23' AS int)` is NULL).
    for to in [DataType::Int8, int.clone(), DataType::Int64] {
        assert_eq!(one(s("1.23"), to.clone(), Try), "NULL");
        assert_eq!(one(s("-4.56"), to, Try), "NULL");
    }
    assert_eq!(one(s(" 7 "), int, Try), "7");
    assert_eq!(
        one(s("128"), DataType::Int8, Ansi),
        invalid("'128'", "TINYINT")
    );
    assert_eq!(one(s("32767.9"), DataType::Int16, Legacy), "32767");
    assert_eq!(
        one(s("-9223372036854775808"), DataType::Int64, Ansi),
        "-9223372036854775808"
    );
    let d = |p, sc| DataType::Decimal128(p, sc);
    assert_eq!(one(s("1e3"), d(10, 2), Ansi), "1000.00");
    assert_eq!(one(s(" 1.235 "), d(10, 2), Ansi), "1.24");
    assert_eq!(one(s("-1.235"), d(10, 2), Ansi), "-1.24");
    assert_eq!(one(s("-0.005"), d(3, 2), Ansi), "-0.01");
    assert_eq!(
        one(s("123.45"), d(3, 1), Ansi),
        out_of_range("123.45", 3, 1)
    );
    assert_eq!(one(s("123.45"), d(3, 1), Legacy), "NULL");
    assert_eq!(one(s("1d"), d(3, 1), Ansi), invalid("'1d'", "DECIMAL(3,1)"));
    assert_eq!(
        one(s(" 1e400 "), d(3, 1), Ansi),
        "[NUMERIC_OUT_OF_SUPPORTED_RANGE] The value  1e400  cannot be interpreted as a numeric \
         since it has more than 38 digits. SQLSTATE: 22003"
    );
    assert_eq!(one(s("1e400"), d(3, 1), Legacy), "NULL");
    assert_eq!(one(s("\u{661}"), d(5, 1), Legacy), "1.0");
    assert_eq!(one(s("12.345"), d(10, 0), Ansi), "12");
    assert_eq!(one(s("1.5"), DataType::Float64, Ansi), "1.5");
    assert_eq!(one(s("1.5d"), DataType::Float64, Ansi), "1.5");
    assert_eq!(one(s("inf"), DataType::Float64, Ansi), "Infinity");
    assert_eq!(
        one(s("abc"), DataType::Float64, Ansi),
        invalid("'abc'", "DOUBLE")
    );
    assert_eq!(
        one(s("3.4028236e38"), DataType::Float32, Legacy),
        "Infinity"
    );
    let large: ArrayRef = Arc::new(LargeStringArray::from(vec![" 7 ", "x"]));
    for (to, want) in [
        (DataType::Int8, "7"),
        (DataType::Int16, "7"),
        (DataType::Int32, "7"),
    ] {
        let out = run(&large, &to, Legacy).unwrap();
        let txt = run(&out, &DataType::Utf8, Ansi).unwrap();
        assert_eq!(
            txt.as_ref(),
            &StringArray::from(vec![Some(want), None]) as &dyn Array,
            "{to}"
        );
    }
    let out = run(&large, &DataType::Int64, Legacy).unwrap();
    assert_eq!(
        out.as_ref(),
        &Int64Array::from(vec![Some(7), None]) as &dyn Array
    );
    let out = run(&large, &d(3, 0), Try).unwrap();
    let txt = run(&out, &DataType::Utf8, Ansi).unwrap();
    assert_eq!(
        txt.as_ref(),
        &StringArray::from(vec![Some("7"), None]) as &dyn Array
    );
}

#[test]
fn strings_to_dates_match_spark() {
    let date = DataType::Date32;
    assert_eq!(one(s(" 2024-02-29 "), date.clone(), Ansi), "2024-02-29");
    assert_eq!(
        one(s("2024-02-29 garbage"), date.clone(), Ansi),
        "2024-02-29"
    );
    assert_eq!(one(s("2024"), date.clone(), Ansi), "2024-01-01");
    assert_eq!(one(s("+10000-01-01"), date.clone(), Ansi), "+10000-01-01");
    assert_eq!(
        one(s("2024-02-30"), date.clone(), Ansi),
        invalid("'2024-02-30'", "DATE")
    );
    assert_eq!(one(s("2024-02-30"), date.clone(), Legacy), "NULL");
    assert_eq!(one(s("1.5"), date.clone(), Try), "NULL");
    // `UTF8String.trimAll` also strips DEL (Spark 4.0-4.2, both modes).
    for mode in [Ansi, Legacy] {
        assert_eq!(
            one(s("\u{7f}2024-01-02 \u{7f}"), date.clone(), mode),
            "2024-01-02"
        );
    }
}

#[test]
fn strings_to_booleans_match_spark() {
    let b = DataType::Boolean;
    for t in ["t", "TRUE", " Yes\u{7f}", "y", "1"] {
        assert_eq!(one(s(t), b.clone(), Ansi), "true", "{t:?}");
    }
    for f in ["F", "false", "\tNo ", "n", "0"] {
        assert_eq!(one(s(f), b.clone(), Ansi), "false", "{f:?}");
    }
    assert_eq!(
        one(s("maybe"), b.clone(), Ansi),
        invalid("'maybe'", "BOOLEAN")
    );
    assert_eq!(one(s("2"), b.clone(), Ansi), invalid("'2'", "BOOLEAN"));
    assert_eq!(one(s(""), b.clone(), Ansi), invalid("''", "BOOLEAN"));
    for mode in [Legacy, Try] {
        assert_eq!(one(s("maybe"), b.clone(), mode), "NULL");
    }
    let out = run(
        &StringArray::from(vec![Some("t"), None, Some("x")]),
        &b,
        Legacy,
    )
    .unwrap();
    assert_eq!(
        out.as_ref(),
        &BooleanArray::from(vec![Some(true), None, None]) as &dyn Array
    );
}

#[test]
fn numbers_to_booleans_are_nonzero() {
    let b = DataType::Boolean;
    for mode in [Ansi, Legacy, Try] {
        assert_eq!(one(i32s(0), b.clone(), mode), "false");
        assert_eq!(one(i32s(-3), b.clone(), mode), "true");
        assert_eq!(one(dec(0, 5, 2), b.clone(), mode), "false");
        assert_eq!(one(dec(1, 5, 2), b.clone(), mode), "true");
        assert_eq!(one(f64s(0.0), b.clone(), mode), "false");
        assert_eq!(one(f64s(-0.0), b.clone(), mode), "false");
        assert_eq!(one(f64s(f64::NAN), b.clone(), mode), "true");
        assert_eq!(one(f64s(0.5), b.clone(), mode), "true");
    }
    let nulls = Int32Array::from(vec![Some(2), None, Some(0)]);
    let out = run(&nulls, &b, Ansi).unwrap();
    assert_eq!(
        out.as_ref(),
        &BooleanArray::from(vec![Some(true), None, Some(false)]) as &dyn Array
    );
}

/// Documented divergence: Spark 4.0-4.2 ANSI/try casts of a decimal with
/// precision > 18 to int/smallint/tinyint round through `double` (giving
/// 1 here); Krishiv truncates exactly like Spark's legacy cast.
#[test]
fn wide_decimal_to_int_truncates_exactly() {
    let nines = 99_999_999_999_999_999_999; // 0.99999999999999999999
    for mode in [Ansi, Legacy, Try] {
        assert_eq!(one(dec(nines, 20, 20), DataType::Int32, mode), "0");
    }
}

#[test]
fn numeric_casts_match_spark() {
    let i = |v: i32| -> ArrayRef { Arc::new(Int32Array::from(vec![v])) };
    let l = |v: i64| -> ArrayRef { Arc::new(Int64Array::from(vec![v])) };
    let f = |v: f64| -> ArrayRef { Arc::new(Float64Array::from(vec![v])) };
    let d = |p, sc| DataType::Decimal128(p, sc);
    assert_eq!(
        one(i(300), DataType::Int8, Ansi),
        overflow("300", "INT", "TINYINT")
    );
    assert_eq!(one(i(300), DataType::Int8, Legacy), "44");
    let sh: ArrayRef = Arc::new(Int16Array::from(vec![300_i16]));
    assert_eq!(
        one(sh, DataType::Int8, Ansi),
        overflow("300S", "SMALLINT", "TINYINT")
    );
    assert_eq!(
        one(l(3_000_000_000), DataType::Int32, Ansi),
        overflow("3000000000L", "BIGINT", "INT")
    );
    assert_eq!(
        one(l(3_000_000_000), DataType::Int32, Legacy),
        "-1294967296"
    );
    assert_eq!(one(l(3_000_000_000), DataType::Int32, Try), "NULL");
    assert_eq!(one(i(1234), d(3, 0), Ansi), out_of_range("1234", 3, 0));
    assert_eq!(one(i(5), d(3, 3), Ansi), out_of_range("5", 3, 3));
    assert_eq!(one(i(5), d(3, 2), Ansi), "5.00");
    assert_eq!(one(l(i64::MIN), d(19, 0), Ansi), "-9223372036854775808");
    assert_eq!(one(dec(123_456, 6, 3), d(4, 1), Ansi), "123.5");
    assert_eq!(one(dec(-12_345, 6, 2), d(4, 1), Ansi), "-123.5");
    assert_eq!(
        one(dec(123_456, 6, 3), d(3, 1), Ansi),
        out_of_range("123.456", 3, 1)
    );
    assert_eq!(one(dec(-995, 3, 2), d(2, 1), Legacy), "NULL");
    assert_eq!(
        one(dec(123_456_789_015, 12, 1), DataType::Int32, Ansi),
        overflow("12345678901.5BD", "DECIMAL(12,1)", "INT")
    );
    assert_eq!(
        one(dec(123_456_789_015, 12, 1), DataType::Int32, Legacy),
        "-539222987"
    );
    assert_eq!(one(dec(-19, 2, 1), DataType::Int32, Legacy), "-1");
    assert_eq!(one(dec(3007, 4, 1), DataType::Int8, Legacy), "44");
    assert_eq!(one(dec(-5, 2, 1), DataType::Int32, Ansi), "0");
    // Rescaling beyond 64 bits rounds half up too.
    let wide = dec(-12_345_678_901_234_567_890_123_456_789_015, 38, 2);
    assert_eq!(
        one(wide, d(38, 1), Ansi),
        "-123456789012345678901234567890.2"
    );
    assert_eq!(
        one(f(1e10), DataType::Int32, Ansi),
        overflow("1.0E10D", "DOUBLE", "INT")
    );
    assert_eq!(
        one(f(-2_147_483_649.0), DataType::Int32, Ansi),
        overflow("-2.147483649E9D", "DOUBLE", "INT")
    );
    assert_eq!(one(f(1e10), DataType::Int32, Legacy), "2147483647");
    assert_eq!(one(f(1e10), DataType::Int8, Legacy), "-1");
    assert_eq!(one(f(300.5), DataType::Int8, Legacy), "44");
    assert_eq!(one(f(-1.9), DataType::Int32, Legacy), "-1");
    assert_eq!(one(f(f64::NAN), DataType::Int16, Legacy), "0");
    assert_eq!(
        one(f(f64::NAN), DataType::Int32, Ansi),
        overflow("NaN", "DOUBLE", "INT")
    );
    assert_eq!(one(f(2_147_483_647.9), DataType::Int32, Ansi), "2147483647");
    assert_eq!(
        one(f(-2_147_483_648.9), DataType::Int32, Ansi),
        "-2147483648"
    );
    assert_eq!(
        one(f(9.223_372_036_854_776e18), DataType::Int64, Ansi),
        "9223372036854775807"
    );
    let ff: ArrayRef = Arc::new(Float32Array::from(vec![1.0e10_f32]));
    assert_eq!(
        one(Arc::clone(&ff), DataType::Int32, Ansi),
        overflow("1.0E10", "FLOAT", "INT")
    );
    assert_eq!(one(ff, DataType::Int64, Ansi), "10000000000");
    assert_eq!(one(f(1.005), d(3, 2), Ansi), "1.01");
    assert_eq!(one(f(0.1), d(20, 19), Ansi), "0.1000000000000000000");
    assert_eq!(one(f(1e20), d(38, 0), Ansi), "100000000000000000000");
    assert_eq!(
        one(f(123_456_789_012_345_678.0), d(38, 0), Ansi),
        "123456789012345680"
    );
    assert_eq!(one(f(f64::NAN), d(3, 1), Ansi), "NULL");
    assert_eq!(one(f(f64::INFINITY), d(5, 0), Ansi), "NULL");
    assert_eq!(one(f(1000.0), d(3, 1), Ansi), out_of_range("1000.0", 3, 1));
    assert_eq!(
        one(f(1e20), d(5, 0), Ansi),
        out_of_range("100000000000000000000", 5, 0)
    );
    assert_eq!(one(f(-1.25), d(3, 1), Ansi), "-1.3");
    assert_eq!(one(f(-0.0), d(3, 1), Ansi), "0.0");
    assert_eq!(one(f(1e-50), d(3, 1), Ansi), "0.0");
    let ff: ArrayRef = Arc::new(Float32Array::from(vec![1.15_f32]));
    assert_eq!(one(ff, d(3, 1), Ansi), "1.1");
    assert_eq!(
        one(dec(986_031_778_147_293_258, 18, 5), DataType::Float64, Ansi),
        "9.860317781472932E12"
    );
    assert_eq!(
        one(dec(9_007_199_254_740_993, 16, 0), DataType::Float64, Ansi),
        "9.007199254740992E15"
    );
    assert_eq!(one(dec(55, 2, 1), DataType::Float32, Ansi), "5.5");
    assert_eq!(
        one(l(i64::MAX), DataType::Float64, Ansi),
        "9.223372036854776E18"
    );
    assert_eq!(one(l(i64::MAX), DataType::Float32, Ansi), "9.223372E18");
    assert_eq!(one(i(16_777_217), DataType::Float32, Ansi), "1.6777216E7");
    assert_eq!(one(f(1.1), DataType::Float32, Ansi), "1.1");
}

#[test]
fn dates_and_timestamps_match_spark() {
    let day = 19_782; // 2024-02-29
    let d: ArrayRef = Arc::new(Date32Array::from(vec![day]));
    let ltz = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
    let ntz = DataType::Timestamp(TimeUnit::Microsecond, None);
    let la = FixedOffset(-8 * 3600);
    let la_opts = CastOptions {
        mode: Ansi,
        time_zone: &la,
    };
    let micros = |a: ArrayRef| {
        a.as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .value(0)
    };
    assert_eq!(micros(run(&d, &ltz, Ansi).unwrap()), 1_709_164_800_000_000);
    assert_eq!(
        micros(cast(&d, &ltz, &la_opts).unwrap()),
        1_709_193_600_000_000
    );
    assert_eq!(
        micros(cast(&d, &ntz, &la_opts).unwrap()),
        1_709_164_800_000_000
    );
    // 2024-02-29 03:00:00 UTC is 2024-02-28 in Los Angeles.
    let t: ArrayRef =
        Arc::new(TimestampMicrosecondArray::from(vec![1_709_175_600_000_000]).with_timezone("UTC"));
    let to_date = cast(&t, &DataType::Date32, &la_opts).unwrap();
    assert_eq!(
        to_date
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap()
            .value(0),
        day - 1
    );
    let as_ntz = cast(&t, &ntz, &la_opts).unwrap();
    assert_eq!(one_str(&as_ntz, &la_opts), "2024-02-28 19:00:00");
    let n: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![1_709_175_600_000_000]));
    assert_eq!(
        micros(cast(&n, &ltz, &la_opts).unwrap()),
        1_709_204_400_000_000
    );
    let before: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![-1]));
    let dd = run(&before, &DataType::Date32, Ansi).unwrap();
    assert_eq!(
        dd.as_any().downcast_ref::<Date32Array>().unwrap().value(0),
        -1
    );
    // Spark raises (unclassified "long overflow") in both modes.
    let far: ArrayRef = Arc::new(Date32Array::from(vec![i32::MAX]));
    for mode in [Ansi, Legacy] {
        assert_eq!(
            run(&far, &ltz, mode).unwrap_err().error_class(),
            ErrorClass::DatetimeOverflow
        );
    }
    assert_eq!(one_str(&t, &opts(Ansi)), "2024-02-29 03:00:00");
    let frac: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![1_709_210_096_120_000]));
    assert_eq!(one_str(&frac, &opts(Ansi)), "2024-02-29 12:34:56.12");
}

fn one_str(a: &ArrayRef, o: &CastOptions<'_>) -> String {
    let s = cast(a, &DataType::Utf8, o).unwrap();
    s.as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .value(0)
        .to_owned()
}

#[test]
fn renderings_match_spark() {
    let b: ArrayRef = Arc::new(BooleanArray::from(vec![Some(true), None]));
    let out = run(&b, &DataType::Utf8, Ansi).unwrap();
    assert_eq!(
        out.as_ref(),
        &StringArray::from(vec![Some("true"), None]) as &dyn Array
    );
    assert_eq!(one_str(&dec(-5, 2, 1), &opts(Ansi)), "-0.5");
    assert_eq!(one_str(&dec(0, 5, 3), &opts(Ansi)), "0.000");
    assert_eq!(one_str(&dec(100, 10, 9), &opts(Ansi)), "0.000000100");
    assert_eq!(one_str(&dec(150, 10, 2), &opts(Ansi)), "1.50");
    // Legacy mode renders like Java's BigDecimal.toString.
    for (v, p, s, want) in [
        (0, 38, 37, "0E-37"),
        (100, 10, 9, "1.00E-7"),
        (-120, 10, 9, "-1.20E-7"),
        (1000, 10, 9, "0.000001000"),
        (1_230_000_000, 10, 7, "123.0000000"),
    ] {
        assert_eq!(one_str(&dec(v, p, s), &opts(Legacy)), want);
    }
    assert_eq!(
        one_str(&dec(0, 38, 37), &opts(Try)),
        format!("0.{}", "0".repeat(37))
    );
    let i8a: ArrayRef = Arc::new(Int8Array::from(vec![-128_i8]));
    assert_eq!(one_str(&i8a, &opts(Ansi)), "-128");
    let neg_year: ArrayRef = Arc::new(Date32Array::from(vec![-719_893]));
    assert_eq!(one_str(&neg_year, &opts(Ansi)), "-0001-01-01");
}

#[test]
fn timestamps_and_numbers_match_spark() {
    let ltz = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
    let micros = |a: ArrayRef, mode| -> Option<i64> {
        let out = run(
            &a,
            &DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            mode,
        )
        .unwrap();
        let t = out
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        t.is_valid(0).then(|| t.value(0))
    };
    let f = |v: f64| -> ArrayRef { Arc::new(Float64Array::from(vec![v])) };
    let l = |v: i64| -> ArrayRef { Arc::new(Int64Array::from(vec![v])) };
    assert_eq!(
        micros(Arc::new(Int32Array::from(vec![1])), Ansi),
        Some(1_000_000)
    );
    assert_eq!(micros(f(-1.5), Ansi), Some(-1_500_000));
    assert_eq!(micros(f(1.000_000_5), Ansi), Some(1_000_000));
    assert_eq!(micros(f(-0.000_001_5), Ansi), Some(-1));
    assert_eq!(micros(dec(-10_000_005, 10, 7), Ansi), Some(-1_000_000));
    assert_eq!(micros(l(i64::MAX), Ansi), Some(i64::MAX));
    assert_eq!(micros(l(i64::MIN), Ansi), Some(i64::MIN));
    assert_eq!(
        micros(Arc::new(Float32Array::from(vec![1.1_f32])), Ansi),
        Some(1_100_000)
    );
    assert_eq!(
        micros(Arc::new(BooleanArray::from(vec![true])), Legacy),
        Some(1)
    );
    assert!(
        run(
            &(Arc::new(BooleanArray::from(vec![true])) as ArrayRef),
            &ltz,
            Ansi
        )
        .is_err()
    );
    assert_eq!(micros(f(f64::NAN), Legacy), None);
    assert_eq!(
        one(f(f64::NAN), ltz.clone(), Ansi),
        "[CAST_INVALID_INPUT] The value NaN of the type \"DOUBLE\" cannot be cast to \"TIMESTAMP\" \
         because it is malformed. Correct the value as per the syntax, or change its target type. \
         Use `try_cast` to tolerate malformed input and return NULL instead. SQLSTATE: 22018"
    );
    // Spark widens a float to double before the cast: the message names
    // DOUBLE (Spark 4.0-4.2).
    for v in [f32::NAN, f32::INFINITY] {
        let ff: ArrayRef = Arc::new(Float32Array::from(vec![v]));
        let e = run(&ff, &ltz, Ansi).unwrap_err().to_string();
        assert!(
            e.contains("of the type \"DOUBLE\" cannot be cast to \"TIMESTAMP\""),
            "{e}"
        );
    }
    assert_eq!(
        one(f(1e30), ltz, Ansi),
        overflow("1.0E36D", "DOUBLE", "BIGINT")
    );
    assert_eq!(micros(f(1e30), Legacy), Some(i64::MAX));
    let ts = |m: i64| -> ArrayRef {
        Arc::new(TimestampMicrosecondArray::from(vec![m]).with_timezone("UTC"))
    };
    let half = ts(-500_000);
    assert_eq!(one(Arc::clone(&half), DataType::Int64, Ansi), "-1");
    assert_eq!(one(Arc::clone(&half), DataType::Float64, Ansi), "-0.5");
    assert_eq!(
        one(Arc::clone(&half), DataType::Decimal128(10, 0), Ansi),
        "-1"
    );
    assert_eq!(
        one(Arc::clone(&half), DataType::Decimal128(10, 2), Ansi),
        "-0.50"
    );
    let leap = ts(1_709_210_096_123_456);
    assert_eq!(one(Arc::clone(&leap), DataType::Int32, Ansi), "1709210096");
    assert_eq!(
        one(Arc::clone(&leap), DataType::Int8, Ansi),
        overflow(
            "TIMESTAMP '2024-02-29 12:34:56.123456'",
            "TIMESTAMP",
            "TINYINT"
        )
    );
    assert_eq!(one(Arc::clone(&leap), DataType::Int8, Legacy), "NULL");
    assert_eq!(
        one(Arc::clone(&leap), DataType::Decimal128(5, 0), Ansi),
        out_of_range("1709210096.123456", 5, 0)
    );
    assert_eq!(
        one(Arc::clone(&leap), DataType::Float32, Ansi),
        "1.7092101E9"
    );
    assert!(run(&leap, &DataType::Boolean, Ansi).is_err());
}

#[test]
fn unsupported_and_identity() {
    let bin: ArrayRef = Arc::new(arrow_array::BinaryArray::from(vec![Some(&b"x"[..]), None]));
    let e = run(&bin, &DataType::Int32, Ansi).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    let text = run(&bin, &DataType::Utf8, Ansi).unwrap();
    assert_eq!(
        text.as_ref(),
        &StringArray::from(vec![Some("x"), None]) as &dyn Array
    );
    // STRING is valid UTF-8: ANSI raises Spark's validate_utf8 error,
    // legacy replaces (make_valid_utf8), try_cast gives NULL.
    let bad: ArrayRef = Arc::new(arrow_array::BinaryArray::from(vec![
        &[0xC3_u8, 0x28, 0xFF][..],
    ]));
    assert_eq!(
        run(&bad, &DataType::Utf8, Ansi).unwrap_err().to_string(),
        "[INVALID_UTF8_STRING] Invalid UTF8 byte sequence found in string: \\xC3\\x28\\xFF. SQLSTATE: 22029"
    );
    let fixed = run(&bad, &DataType::Utf8, Legacy).unwrap();
    assert_eq!(
        fixed.as_ref(),
        &StringArray::from(vec!["\u{FFFD}(\u{FFFD}"]) as &dyn Array
    );
    assert_eq!(run(&bad, &DataType::Utf8, Try).unwrap().null_count(), 1);
    // `STRING` bytes reinterpreted as `BINARY` never fail (Spark's
    // `CAST(string AS BINARY)` is the UTF-8 bytes): the union planner uses
    // this for binary-safe string branches, whose bytes are not necessarily
    // valid UTF-8.
    for mode in [Ansi, Legacy, Try] {
        let bytes = run(
            &StringArray::from(vec![Some("x"), None]),
            &DataType::Binary,
            mode,
        )
        .unwrap();
        assert_eq!(
            bytes.as_ref(),
            &arrow_array::BinaryArray::from(vec![Some(&b"x"[..]), None]) as &dyn Array
        );
    }
    assert!(run(&s("x"), &DataType::Boolean, Ansi).is_err());
    let d: ArrayRef = Arc::new(Date32Array::from(vec![1]));
    assert!(run(&d, &DataType::Int32, Ansi).is_err());
    let same = run(&d, &DataType::Date32, Ansi).unwrap();
    assert_eq!(same.as_ref(), d.as_ref());
    let neg: ArrayRef = Arc::new(
        Decimal128Array::from(vec![1_i128])
            .with_precision_and_scale(5, -2)
            .unwrap(),
    );
    assert!(run(&neg, &DataType::Int32, Ansi).is_err());
    assert!(run(&d, &DataType::Decimal128(5, -2), Ansi).is_err());
    let small: ArrayRef = Arc::new(StringArray::from(vec![Some("x"), None]));
    let large = run(&small, &DataType::LargeUtf8, Ansi).unwrap();
    assert_eq!(
        large.as_ref(),
        &LargeStringArray::from(vec![Some("x"), None]) as &dyn Array
    );
    let back = run(&large, &DataType::Utf8, Ansi).unwrap();
    assert_eq!(back.as_ref(), small.as_ref());
    let dur: ArrayRef = Arc::new(arrow_array::DurationMicrosecondArray::from(vec![1]));
    let e = run(&dur, &DataType::Utf8, Ansi).unwrap_err();
    assert!(e.to_string().contains("Duration"), "{e}");
    assert!(run(&dur, &DataType::Int64, Ansi).is_err());
    // TIMESTAMP in one zone name to another: the same instant.
    let t: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![7]).with_timezone("UTC"));
    let other = DataType::Timestamp(TimeUnit::Microsecond, Some("+08:00".into()));
    let moved = run(&t, &other, Ansi).unwrap();
    assert_eq!(
        moved
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .value(0),
        7
    );
    assert_eq!(moved.data_type(), &other);
}

#[test]
fn null_rows_never_raise() {
    // Malformed text and overflowing values hide under NULL slots.
    let base = StringArray::from(vec!["x", "1"]);
    let s = StringArray::new(
        base.offsets().clone(),
        base.values().clone(),
        Some(vec![false, true].into()),
    );
    let out = run(&s, &DataType::Int32, Ansi).unwrap();
    assert_eq!(
        out.as_ref(),
        &Int32Array::from(vec![None, Some(1)]) as &dyn Array
    );
    let l = Int64Array::new(vec![i64::MAX, 1].into(), Some(vec![false, true].into()));
    let out = run(&l, &DataType::Int8, Ansi).unwrap();
    assert_eq!(
        out.as_ref(),
        &Int8Array::from(vec![None, Some(1)]) as &dyn Array
    );
}

fn try_matches_ansi(a: &dyn Array, to: &DataType) {
    let tried = run(a, to, Try).unwrap();
    match run(a, to, Ansi) {
        Ok(ansi) => assert_eq!(ansi.as_ref(), tried.as_ref()),
        Err(e) => {
            let row = e.row().unwrap();
            assert!(tried.is_null(row) && a.is_valid(row), "{e}");
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: if cfg!(miri) { 2 } else { 256 }, failure_persistence: None, ..ProptestConfig::default() })]

    #[test]
    fn integral_round_trips_through_strings(v in any::<i64>(), w in -200_i32..200) {
        let a = Int64Array::from(vec![v]);
        let txt = run(&a, &DataType::Utf8, Ansi).unwrap();
        let back = run(&txt, &DataType::Int64, Ansi).unwrap();
        prop_assert_eq!(back.as_primitive_ref::<Int64Type>().value(0), v);
        let narrow = Int32Array::from(vec![w]);
        try_matches_ansi(&a, &DataType::Int32);
        try_matches_ansi(&narrow, &DataType::Int8);
        try_matches_ansi(&a, &DataType::Decimal128(12, 2));
    }

    #[test]
    fn doubles_round_trip_through_java_strings(bits in any::<u64>()) {
        let v = f64::from_bits(bits);
        let a = Float64Array::from(vec![v]);
        let txt = run(&a, &DataType::Utf8, Ansi).unwrap();
        let back = run(&txt, &DataType::Float64, Ansi).unwrap();
        let back = back.as_primitive_ref::<Float64Type>().value(0);
        prop_assert!(back.to_bits() == bits || (v.is_nan() && back.is_nan()));
        try_matches_ansi(&a, &DataType::Int32);
        try_matches_ansi(&a, &DataType::Decimal128(10, 3));
    }

    #[test]
    fn decimals_round_trip_and_rescale(v in -999_999_999_999_i128..999_999_999_999, s in 0_i8..12, t in 0_i8..14) {
        let a = Decimal128Array::from(vec![v]).with_precision_and_scale(12, s).unwrap();
        let txt = run(&a, &DataType::Utf8, Ansi).unwrap();
        let back = run(&txt, &DataType::Decimal128(12, s), Ansi).unwrap();
        prop_assert_eq!(back.as_primitive_ref::<Decimal128Type>().value(0), v);
        // Widening the scale is exact; narrowing rounds half up.
        let to = DataType::Decimal128(38, t);
        let r = run(&a, &to, Ansi).unwrap();
        let r = r.as_primitive_ref::<Decimal128Type>().value(0);
        let expect = if t >= s {
            v * 10_i128.pow(u32::try_from(t - s).unwrap())
        } else {
            let d = 10_i128.pow(u32::try_from(s - t).unwrap());
            let q = v / d;
            if (v % d).abs() * 2 >= d { q + v.signum() } else { q }
        };
        prop_assert_eq!(r, expect);
        try_matches_ansi(&a, &DataType::Decimal128(5, 1));
        try_matches_ansi(&a, &DataType::Int16);
    }

    #[test]
    fn dates_round_trip_through_strings(d in any::<i32>()) {
        let a = Date32Array::from(vec![d]);
        let txt = run(&a, &DataType::Utf8, Ansi).unwrap();
        let back = run(&txt, &DataType::Date32, Ansi).unwrap();
        prop_assert_eq!(back.as_primitive_ref::<Date32Type>().value(0), d);
        let ts = run(&a, &DataType::Timestamp(TimeUnit::Microsecond, None), Legacy);
        if let Ok(ts) = ts {
            let back = run(&ts, &DataType::Date32, Ansi).unwrap();
            prop_assert_eq!(back.as_primitive_ref::<Date32Type>().value(0), d);
        }
    }

    #[test]
    fn strings_parse_like_rust_where_both_apply(v in any::<i32>(), pad in "[ \t\n]{0,2}") {
        let text = format!("{pad}{v}{pad}");
        let a = StringArray::from(vec![text.as_str()]);
        let out = run(&a, &DataType::Int32, Ansi).unwrap();
        prop_assert_eq!(out.as_primitive_ref::<Int32Type>().value(0), v);
        try_matches_ansi(&a, &DataType::Int8);
        try_matches_ansi(&a, &DataType::Decimal128(6, 2));
        try_matches_ansi(&a, &DataType::Date32);
    }
}

trait AsPrimitiveRef {
    fn as_primitive_ref<T: arrow_array::ArrowPrimitiveType>(&self) -> &PrimitiveArray<T>;
}

impl AsPrimitiveRef for ArrayRef {
    fn as_primitive_ref<T: arrow_array::ArrowPrimitiveType>(&self) -> &PrimitiveArray<T> {
        self.as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .unwrap_or_else(|| panic!("{:?}", self.data_type()))
    }
}

/// A one-row `Decimal64` column.
fn dec64(v: i64, p: u8, sc: i8) -> ArrayRef {
    Arc::new(
        arrow_array::Decimal64Array::from(vec![v])
            .with_precision_and_scale(p, sc)
            .unwrap(),
    )
}

proptest! {
    // No failure file (Miri's isolation forbids `getcwd`); two cases under Miri.
    #![proptest_config(ProptestConfig {
        cases: if cfg!(miri) { 2 } else { 256 },
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    /// `Decimal64` is a carrier of `decimal(p <= 18, s)` (P2-T160): casts
    /// to and from it give the `Decimal128` cast's values and errors, and a
    /// result on the requested carrier.
    #[test]
    fn decimal64_casts_equal_decimal128(
        v in prop_oneof![Just(999_999_999_999_999_999_i64), Just(-1_i64), Just(0_i64), any::<i64>()],
        p in 1..=18_u8,
        s in 0..=18_i8,
        p2 in 1..=38_u8,
        s2 in 0..=20_i8,
        text in prop_oneof![Just("12.345".to_owned()), Just("-1e3".to_owned()), Just("abc".to_owned()), "-?[0-9]{1,20}(\\.[0-9]{0,6})?"],
        f in prop_oneof![Just(1.005_f64), Just(-0.5), Just(1e300), Just(f64::NAN), any::<f64>()],
    ) {
        let s = s.min(i8::try_from(p).unwrap());
        let s2 = s2.min(i8::try_from(p2).unwrap());
        let bound = 10_i64.pow(u32::from(p));
        let v = v % bound;
        for mode in [CastMode::Ansi, CastMode::Legacy] {
            // To decimal(p, s) on either carrier.
            let sources: [ArrayRef; 4] = [
                Arc::new(StringArray::from(vec![text.as_str()])),
                Arc::new(Float64Array::from(vec![f])),
                Arc::new(Int64Array::from(vec![v])),
                dec(i128::from(v), p, s),
            ];
            for a in &sources {
                let wide = run(a.as_ref(), &DataType::Decimal128(p, s), mode);
                let narrow = run(a.as_ref(), &DataType::Decimal64(p, s), mode);
                if let Ok(n) = &narrow {
                    prop_assert_eq!(n.data_type(), &DataType::Decimal64(p, s));
                }
                prop_assert_eq!(
                    one(Arc::clone(a), DataType::Decimal64(p, s), mode),
                    one(Arc::clone(a), DataType::Decimal128(p, s), mode)
                );
                prop_assert_eq!(wide.is_ok(), narrow.is_ok());
            }
            // From decimal(p, s) on either carrier.
            let mut targets = vec![
                DataType::Utf8,
                DataType::Float64,
                DataType::Float32,
                DataType::Int32,
                DataType::Int64,
                DataType::Int8,
                DataType::Boolean,
                DataType::Decimal128(p2, s2),
            ];
            if p2 <= 18 {
                targets.push(DataType::Decimal64(p2, s2));
            }
            for t in targets {
                prop_assert_eq!(
                    one(dec64(v, p, s), t.clone(), mode),
                    one(dec(i128::from(v), p, s), t, mode)
                );
            }
        }
    }
}
