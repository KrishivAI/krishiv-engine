// SPDX-License-Identifier: Apache-2.0
//! Tests of the data type mapping.

use krishiv_common::error::ErrorClass;
use krishiv_types::{DataType, DecimalType};

use super::super::testing::proto_type;
use super::*;
use crate::proto::v4_2::data_type as dt;

#[test]
fn every_supported_kind_maps() {
    let cases = [
        (Kind::Boolean(dt::Boolean::default()), DataType::Boolean),
        (Kind::Byte(dt::Byte::default()), DataType::Byte),
        (Kind::Short(dt::Short::default()), DataType::Short),
        (Kind::Integer(dt::Integer::default()), DataType::Integer),
        (Kind::Long(dt::Long::default()), DataType::Long),
        (Kind::Float(dt::Float::default()), DataType::Float),
        (Kind::Double(dt::Double::default()), DataType::Double),
        (Kind::Binary(dt::Binary::default()), DataType::Binary),
        (Kind::Date(dt::Date::default()), DataType::Date),
        (
            Kind::Timestamp(dt::Timestamp::default()),
            DataType::Timestamp,
        ),
        (
            Kind::TimestampNtz(dt::TimestampNtz::default()),
            DataType::TimestampNtz,
        ),
        (Kind::String(dt::String::default()), DataType::String),
    ];
    for (kind, expected) in cases {
        assert_eq!(data_type(&proto_type(kind)).expect("maps"), expected);
    }
}

#[test]
fn string_collation() {
    let string = |collation: &str| {
        proto_type(Kind::String(dt::String {
            collation: collation.to_owned(),
            ..dt::String::default()
        }))
    };
    // Spark 4 reports the default string as UTF8_BINARY.
    assert_eq!(
        data_type(&string("UTF8_BINARY")).expect("default"),
        DataType::String
    );
    assert_eq!(
        data_type(&string("utf8_binary")).expect("default"),
        DataType::String
    );
    let e = data_type(&string("UTF8_LCASE")).expect_err("collated");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(
        e.message().contains("DataType.String.collation"),
        "{}",
        e.message()
    );
}

#[test]
fn decimal_precision_and_scale_follow_spark() {
    let decimal = |precision, scale| {
        data_type(&proto_type(Kind::Decimal(dt::Decimal {
            precision,
            scale,
            ..dt::Decimal::default()
        })))
    };
    let of = |p, s| DataType::Decimal(DecimalType::try_new(p, s).expect("valid"));
    assert_eq!(decimal(Some(12), Some(3)).expect("both"), of(12, 3));
    assert_eq!(decimal(Some(12), None).expect("precision only"), of(12, 0));
    assert_eq!(decimal(None, None).expect("neither"), of(10, 0));
    assert_eq!(
        decimal(None, Some(3)).expect("scale alone is ignored"),
        of(10, 0)
    );
    let e = decimal(Some(39), Some(0)).expect_err("too wide");
    assert_eq!(
        e.error_class(),
        ErrorClass::DecimalPrecisionExceedsMaxPrecision
    );
    let e = decimal(Some(5), Some(6)).expect_err("scale over precision");
    assert!(e.message().contains('6'), "{}", e.message());
    let e = decimal(Some(-1), Some(0)).expect_err("negative");
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanAssertionFailure
    );
}

#[test]
fn unsupported_and_unset_kinds() {
    let e = data_type(&DataTypeProto::default()).expect_err("unset");
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet
    );
    for kind in [
        Kind::Null(dt::Null::default()),
        Kind::Char(dt::Char::default()),
        Kind::VarChar(dt::VarChar::default()),
        Kind::Array(Box::default()),
        Kind::Struct(dt::Struct::default()),
        Kind::Map(Box::default()),
        Kind::Variant(dt::Variant::default()),
        Kind::CalendarInterval(dt::CalendarInterval::default()),
        Kind::YearMonthInterval(dt::YearMonthInterval::default()),
        Kind::DayTimeInterval(dt::DayTimeInterval::default()),
    ] {
        let e = data_type(&proto_type(kind)).expect_err("unsupported");
        assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
        assert!(
            e.message().contains("spark.connect.DataType."),
            "{}",
            e.message()
        );
    }
}

use crate::proto::v4_2::DataType as DataTypeProto;
