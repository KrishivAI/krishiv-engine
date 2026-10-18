// SPDX-License-Identifier: Apache-2.0
//! Tests of the literal translation. The decimal and interval cases are what
//! Spark 4.2.0's `LiteralExpressionProtoConverter` returns for them (run in
//! the pinned Spark container).

use krishiv_common::error::ErrorClass;
use krishiv_types::{DataType, DecimalType};

use super::super::testing::proto_type;
use super::*;
use crate::proto::v4_2::DataType as DataTypeProto;
use crate::proto::v4_2::data_type as dt;
use crate::proto::v4_2::expression::literal as pl;

fn literal_of(value: LiteralType) -> expression::Literal {
    expression::Literal {
        literal_type: Some(value),
        data_type: None,
    }
}

fn typed(value: LiteralType, kind: Kind) -> expression::Literal {
    expression::Literal {
        literal_type: Some(value),
        data_type: Some(proto_type(kind)),
    }
}

fn ok(value: LiteralType) -> Literal {
    literal(&literal_of(value)).expect("translates")
}

fn dec(value: &str, precision: Option<i32>, scale: Option<i32>) -> LiteralType {
    LiteralType::Decimal(pl::Decimal {
        value: value.to_owned(),
        precision,
        scale,
    })
}

fn decimal_of(unscaled: i128, precision: u32, scale: u32) -> Literal {
    Literal::Decimal {
        unscaled,
        data_type: DecimalType::try_new(precision, scale).expect("valid"),
    }
}

#[test]
fn scalar_literals() {
    assert_eq!(ok(LiteralType::Boolean(true)), Literal::Boolean(true));
    assert_eq!(ok(LiteralType::Byte(-128)), Literal::Byte(-128));
    assert_eq!(ok(LiteralType::Short(300)), Literal::Short(300));
    assert_eq!(
        ok(LiteralType::Integer(i32::MIN)),
        Literal::Integer(i32::MIN)
    );
    assert_eq!(ok(LiteralType::Long(i64::MAX)), Literal::Long(i64::MAX));
    assert_eq!(ok(LiteralType::Float(1.25)), Literal::Float(1.25));
    assert_eq!(ok(LiteralType::Double(-0.5)), Literal::Double(-0.5));
    assert_eq!(
        ok(LiteralType::String("h\u{e9}llo".into())),
        Literal::String("h\u{e9}llo".into())
    );
    assert_eq!(
        ok(LiteralType::String(String::new())),
        Literal::String(String::new())
    );
    assert_eq!(
        ok(LiteralType::Binary(vec![0, 255])),
        Literal::Binary(vec![0, 255])
    );
    assert_eq!(ok(LiteralType::Date(18321)), Literal::Date(18321));
    assert_eq!(
        ok(LiteralType::Timestamp(1_582_938_123_456_789)),
        Literal::Timestamp(1_582_938_123_456_789)
    );
    assert_eq!(ok(LiteralType::TimestampNtz(-1)), Literal::TimestampNtz(-1));
}

#[test]
fn floats_keep_their_bits() {
    let nan = ok(LiteralType::Double(f64::NAN));
    assert_eq!(
        nan,
        Literal::Double(f64::NAN),
        "NaN equals itself in the IR"
    );
    assert_eq!(
        ok(LiteralType::Float(f32::INFINITY)),
        Literal::Float(f32::INFINITY)
    );
    assert_ne!(
        ok(LiteralType::Double(-0.0)),
        Literal::Double(0.0),
        "-0.0 is not 0.0"
    );
}

#[test]
fn byte_and_short_out_of_range_are_rejected() {
    // Spark truncates 300 to 44; a client never sends it, so it is an error.
    for value in [
        LiteralType::Byte(300),
        LiteralType::Byte(-129),
        LiteralType::Short(70_000),
    ] {
        let e = literal(&literal_of(value)).expect_err("out of range");
        assert_eq!(
            e.error_class(),
            ErrorClass::ConnectInvalidPlanAssertionFailure
        );
    }
}

#[test]
fn nulls_carry_their_type() {
    let null = |kind| ok(LiteralType::Null(proto_type(kind)));
    assert_eq!(
        null(Kind::Integer(dt::Integer::default())),
        Literal::Null(Some(DataType::Integer))
    );
    assert_eq!(
        null(Kind::Null(dt::Null::default())),
        Literal::Null(None),
        "NULL is void"
    );
    let decimal = null(Kind::Decimal(dt::Decimal {
        precision: Some(12),
        scale: Some(3),
        ..dt::Decimal::default()
    }));
    assert_eq!(
        decimal,
        Literal::Null(Some(DataType::Decimal(
            DecimalType::try_new(12, 3).unwrap()
        )))
    );
    // The root `data_type` field wins over the legacy one in `null`.
    let both = expression::Literal {
        literal_type: Some(LiteralType::Null(proto_type(Kind::Integer(
            dt::Integer::default(),
        )))),
        data_type: Some(proto_type(Kind::Long(dt::Long::default()))),
    };
    assert_eq!(literal(&both).unwrap(), Literal::Null(Some(DataType::Long)));
    // A null of a type Krishiv has not got fails by that type.
    let e = literal(&literal_of(LiteralType::Null(proto_type(Kind::Char(
        dt::Char::default(),
    )))))
    .expect_err("char");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn decimal_type_comes_from_the_digits_and_the_given_precision() {
    // (value, precision, scale) -> decimal(p,s) and unscaled: Spark's answers.
    let cases = [
        ("12.345", None, None, decimal_of(12_345, 5, 3)),
        ("12.345", Some(10), Some(0), decimal_of(12_345, 10, 3)),
        ("12.345", Some(4), None, decimal_of(12_345, 5, 3)),
        ("0.001", None, None, decimal_of(1, 3, 3)),
        ("0", None, None, decimal_of(0, 1, 0)),
        ("0.00", None, None, decimal_of(0, 2, 2)),
        ("-5", None, None, decimal_of(-5, 1, 0)),
        ("1E+3", None, None, decimal_of(1000, 4, 0)),
        ("1E+3", Some(10), Some(0), decimal_of(1000, 10, 0)),
        ("1.5E-2", None, None, decimal_of(15, 3, 3)),
        ("+7.50", None, None, decimal_of(750, 3, 2)),
        (".5", None, None, decimal_of(5, 1, 1)),
        ("5.", None, None, decimal_of(5, 1, 0)),
        ("1.50", Some(10), Some(4), decimal_of(15_000, 10, 4)),
        ("1.5", Some(5), Some(-1), decimal_of(15, 5, 1)),
        (
            "12345678901234567890123456789012345678",
            None,
            None,
            decimal_of(12_345_678_901_234_567_890_123_456_789_012_345_678, 38, 0),
        ),
        ("-0.0001", None, None, decimal_of(-1, 4, 4)),
    ];
    for (value, precision, scale, expected) in cases {
        let got = literal(&literal_of(dec(value, precision, scale)));
        assert_eq!(
            got.unwrap_or_else(|e| panic!("{value}: {e}")),
            expected,
            "{value} {precision:?} {scale:?}"
        );
    }
}

#[test]
fn declared_decimal_type_rounds_half_up() {
    let declared = |value: &str, precision, scale| {
        literal(&typed(
            dec(value, None, None),
            Kind::Decimal(dt::Decimal {
                precision,
                scale,
                ..dt::Decimal::default()
            }),
        ))
    };
    assert_eq!(
        declared("1.234", Some(5), Some(2)).unwrap(),
        decimal_of(123, 5, 2)
    );
    assert_eq!(
        declared("1.235", Some(5), Some(2)).unwrap(),
        decimal_of(124, 5, 2)
    );
    assert_eq!(
        declared("-1.235", Some(5), Some(2)).unwrap(),
        decimal_of(-124, 5, 2)
    );
    assert_eq!(
        declared("12.5", Some(5), None).unwrap(),
        decimal_of(13, 5, 0),
        "scale defaults to 0"
    );
    assert_eq!(
        declared("12.5", None, Some(3)).unwrap(),
        decimal_of(13, 10, 0),
        "decimal(10,0)"
    );
    assert_eq!(
        declared("1E+3", Some(10), Some(0)).unwrap(),
        decimal_of(1000, 10, 0)
    );
    assert_eq!(
        declared("1.5", Some(8), Some(4)).unwrap(),
        decimal_of(15_000, 8, 4)
    );
    assert_eq!(
        declared("0.4", Some(3), Some(0)).unwrap(),
        decimal_of(0, 3, 0)
    );
    assert_eq!(
        declared("0.5", Some(3), Some(0)).unwrap(),
        decimal_of(1, 3, 0)
    );
}

#[test]
fn decimal_errors() {
    let e = literal(&typed(
        dec("123.45", None, None),
        Kind::Decimal(dt::Decimal {
            precision: Some(3),
            scale: Some(2),
            ..dt::Decimal::default()
        }),
    ))
    .expect_err("does not fit");
    assert_eq!(
        e.error_class(),
        ErrorClass::NumericValueOutOfRangeWithSuggestion
    );
    assert!(e.message().contains("Decimal(3, 2)"), "{}", e.message());
    // More than 38 digits: the type itself is invalid.
    let e = literal(&literal_of(dec(&"9".repeat(39), None, None))).expect_err("39 digits");
    assert_eq!(
        e.error_class(),
        ErrorClass::DecimalPrecisionExceedsMaxPrecision
    );
    // 38 digits at scale 1 need decimal(39,1).
    let e = literal(&literal_of(dec(
        &format!("{}.5", "9".repeat(38)),
        None,
        None,
    )))
    .expect_err("wide");
    assert_eq!(
        e.error_class(),
        ErrorClass::DecimalPrecisionExceedsMaxPrecision
    );
    // Not numbers: Spark's NumberFormatException.
    for bad in [
        "",
        "abc",
        " 1",
        "1 ",
        "1e",
        "1e+",
        "--1",
        "1.2.3",
        ".",
        "+",
        "1e99999999999",
        "0x10",
    ] {
        let e = literal(&literal_of(dec(bad, None, None))).expect_err(bad);
        assert_eq!(
            e.error_class(),
            ErrorClass::ConnectInvalidPlanAssertionFailure,
            "{bad:?}"
        );
    }
    // An exponent past the digits that fit, and a huge text, stay bounded.
    let e = literal(&literal_of(dec("1E+40", None, None))).expect_err("1E+40");
    assert_eq!(
        e.error_class(),
        ErrorClass::DecimalPrecisionExceedsMaxPrecision
    );
    let e = literal(&typed(
        dec("1E+2147483647", None, None),
        Kind::Decimal(dt::Decimal {
            precision: Some(38),
            scale: Some(0),
            ..dt::Decimal::default()
        }),
    ))
    .expect_err("huge exponent");
    assert_eq!(
        e.error_class(),
        ErrorClass::NumericValueOutOfRangeWithSuggestion
    );
    let long = format!("0.{}", "0".repeat(100_000));
    assert!(
        literal(&literal_of(dec(&long, None, None))).is_err(),
        "scale over 38"
    );
    let shown = literal(&typed(
        dec(&format!("1{}", "0".repeat(60)), None, None),
        Kind::Decimal(dt::Decimal {
            precision: Some(5),
            scale: Some(0),
            ..dt::Decimal::default()
        }),
    ))
    .expect_err("too long");
    assert!(
        shown.message().len() < 200,
        "the value is cut: {}",
        shown.message()
    );
}

#[test]
fn interval_literals_default_and_declared_fields() {
    let ym = |value, kind: Option<dt::YearMonthInterval>| match kind {
        Some(k) => literal(&typed(
            LiteralType::YearMonthInterval(value),
            Kind::YearMonthInterval(k),
        )),
        None => literal(&literal_of(LiteralType::YearMonthInterval(value))),
    };
    let year_month = |months, start, end| Literal::YearMonthInterval { months, start, end };
    assert_eq!(
        ym(14, None).unwrap(),
        year_month(14, YearMonthField::Year, YearMonthField::Month)
    );
    let field = |start, end| dt::YearMonthInterval {
        start_field: start,
        end_field: end,
        ..dt::YearMonthInterval::default()
    };
    assert_eq!(
        ym(14, Some(field(Some(0), Some(0)))).unwrap(),
        year_month(14, YearMonthField::Year, YearMonthField::Year)
    );
    assert_eq!(
        ym(14, Some(field(Some(1), None))).unwrap(),
        year_month(14, YearMonthField::Month, YearMonthField::Month)
    );
    assert_eq!(
        ym(14, Some(field(None, Some(0)))).unwrap(),
        year_month(14, YearMonthField::Year, YearMonthField::Month),
        "an end alone is ignored, as in Spark"
    );
    assert!(
        ym(1, Some(field(Some(1), Some(0)))).is_err(),
        "MONTH TO YEAR"
    );
    assert!(ym(1, Some(field(Some(2), None))).is_err(), "no field 2");

    let dtm = |micros, start, end| {
        let kind = Kind::DayTimeInterval(dt::DayTimeInterval {
            start_field: start,
            end_field: end,
            ..dt::DayTimeInterval::default()
        });
        literal(&typed(LiteralType::DayTimeInterval(micros), kind))
    };
    let day_time = |micros, start, end| Literal::DayTimeInterval { micros, start, end };
    assert_eq!(
        ok(LiteralType::DayTimeInterval(90_061_000_000)),
        day_time(90_061_000_000, DayTimeField::Day, DayTimeField::Second)
    );
    assert_eq!(
        dtm(5, Some(1), Some(2)).unwrap(),
        day_time(5, DayTimeField::Hour, DayTimeField::Minute)
    );
    assert_eq!(
        dtm(5, Some(2), None).unwrap(),
        day_time(5, DayTimeField::Minute, DayTimeField::Minute)
    );
    assert!(dtm(5, Some(3), Some(1)).is_err(), "SECOND TO HOUR");
    assert!(dtm(5, Some(9), None).is_err(), "no field 9");
    assert!(dtm(5, Some(-1), None).is_err());
}

#[test]
fn a_declared_type_must_fit_the_value() {
    let e = literal(&typed(
        LiteralType::Integer(1),
        Kind::Long(dt::Long::default()),
    ))
    .expect_err("int as long");
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanIncompatibleLiteralDataType
    );
    assert!(
        e.message().contains("LONG") && e.message().contains("INTEGER"),
        "{}",
        e.message()
    );
    // The matching type is fine, and so is a null of any.
    assert_eq!(
        literal(&typed(
            LiteralType::Integer(1),
            Kind::Integer(dt::Integer::default())
        ))
        .unwrap(),
        Literal::Integer(1)
    );
    let unset = expression::Literal {
        literal_type: Some(LiteralType::Integer(1)),
        data_type: Some(DataTypeProto::default()),
    };
    assert_eq!(
        literal(&unset).unwrap_err().error_class(),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet
    );
}

#[test]
fn string_collation_in_the_declared_type() {
    let collated = |collation: &str| {
        literal(&typed(
            LiteralType::String("x".into()),
            Kind::String(dt::String {
                collation: collation.into(),
                ..dt::String::default()
            }),
        ))
    };
    assert_eq!(
        collated("UTF8_BINARY").unwrap(),
        Literal::String("x".into())
    );
    assert_eq!(collated("").unwrap(), Literal::String("x".into()));
    assert_eq!(
        collated("UTF8_LCASE").unwrap_err().error_class(),
        ErrorClass::KrishivUnsupportedFeature
    );
}

#[test]
fn a_literal_with_no_value_is_unsupported_literal_type() {
    let e = literal(&expression::Literal::default()).expect_err("unset");
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanUnsupportedLiteralType
    );
    assert!(
        e.message().contains("LITERALTYPE_NOT_SET"),
        "{}",
        e.message()
    );
}

#[test]
fn literals_the_ir_has_no_value_for_are_unsupported() {
    for (value, name) in [
        (
            LiteralType::CalendarInterval(pl::CalendarInterval::default()),
            "calendar_interval",
        ),
        (LiteralType::Array(pl::Array::default()), "array"),
        (LiteralType::Map(pl::Map::default()), "map"),
        (LiteralType::Struct(pl::Struct::default()), "struct"),
        (
            LiteralType::SpecializedArray(pl::SpecializedArray::default()),
            "specialized_array",
        ),
        (LiteralType::Time(pl::Time::default()), "time"),
    ] {
        let e = literal(&literal_of(value)).expect_err(name);
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{name}"
        );
        assert!(
            e.message()
                .contains(&format!("spark.connect.Expression.Literal.{name}")),
            "{name}: {}",
            e.message()
        );
    }
}

#[test]
fn a_decimal_literal_of_any_length_is_read_in_place() {
    // Trailing zeros are digits of the number (BigDecimal keeps them): 1.5
    // followed by a million zeros has a million significant digits.
    let long = format!("1.5{}", "0".repeat(1_000_000));
    let e = literal(&literal_of(dec(&long, None, None))).expect_err("precision past 38");
    assert_eq!(
        e.error_class(),
        ErrorClass::DecimalPrecisionExceedsMaxPrecision
    );
    let declared = Kind::Decimal(dt::Decimal {
        precision: Some(10),
        scale: Some(0),
        ..dt::Decimal::default()
    });
    let e = literal(&typed(dec(&long, None, None), declared)).expect_err("does not fit");
    assert_eq!(
        e.error_class(),
        ErrorClass::NumericValueOutOfRangeWithSuggestion
    );
    // Leading zeros are not digits: they cost nothing and change nothing.
    let zeros = "0".repeat(1_000_000);
    assert_eq!(
        literal(&literal_of(dec(&format!("{zeros}7"), None, None))).unwrap(),
        decimal_of(7, 1, 0)
    );
    assert_eq!(
        literal(&literal_of(dec(&format!("{zeros}.{zeros}"), None, None)))
            .map_err(|e| e.error_class()),
        Err(ErrorClass::DecimalPrecisionExceedsMaxPrecision),
        "0.000... has a scale of a million"
    );
    let e = literal(&literal_of(dec(&"9".repeat(1_000_000), None, None))).expect_err("wide");
    assert_eq!(
        e.error_class(),
        ErrorClass::DecimalPrecisionExceedsMaxPrecision
    );
}
