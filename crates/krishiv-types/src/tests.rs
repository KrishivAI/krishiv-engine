// SPDX-License-Identifier: Apache-2.0
use std::sync::Arc;

use arrow_schema::{DataType as ArrowType, Field, TimeUnit};
use krishiv_common::error::ErrorClass;

use crate::arrow::{SPARK_TYPE_METADATA_KEY, normalize_ingest_type};
use crate::{DataType, DecimalStorage, DecimalType, parse_data_type};

fn dec(p: u32, s: u32) -> DataType {
    match DecimalType::try_new(p, s) {
        Ok(d) => DataType::Decimal(d),
        Err(e) => panic!("decimal({p},{s}): {e}"),
    }
}

/// One instance of every in-scope type, decimals at the storage boundary.
fn all_types() -> Vec<DataType> {
    "boolean tinyint smallint int bigint float double string binary decimal(0,0) decimal(10,2) \
     decimal(18,18) decimal(19,0) decimal(38,38) date timestamp timestamp_ntz"
        .split(' ')
        .map(|s| parse_data_type(s).unwrap_or_else(|e| panic!("{s}: {e}")))
        .collect()
}

#[test]
fn spark_to_arrow_to_spark_round_trips_every_type() {
    for t in all_types() {
        let field = t.to_arrow_field("c", true);
        assert_eq!(DataType::from_arrow_field(&field).as_ref(), Ok(&t), "{t}");
        assert!(t.is_valid_storage(&t.to_arrow()), "{t}");
        if t != dec(0, 0) {
            assert_eq!(DataType::from_arrow(&t.to_arrow()), Ok(t.clone()), "{t}");
            assert!(field.metadata().is_empty(), "{t}");
        }
    }
}

#[test]
fn canonical_arrow_types() {
    let ts = |tz: Option<&str>| ArrowType::Timestamp(TimeUnit::Microsecond, tz.map(Arc::from));
    let cases = [
        (DataType::Boolean, ArrowType::Boolean),
        (DataType::Byte, ArrowType::Int8),
        (DataType::Short, ArrowType::Int16),
        (DataType::Integer, ArrowType::Int32),
        (DataType::Long, ArrowType::Int64),
        (DataType::Float, ArrowType::Float32),
        (DataType::Double, ArrowType::Float64),
        (DataType::String, ArrowType::Utf8View),
        (DataType::Binary, ArrowType::BinaryView),
        (dec(10, 2), ArrowType::Decimal128(10, 2)),
        (dec(38, 38), ArrowType::Decimal128(38, 38)),
        (DataType::Date, ArrowType::Date32),
        (DataType::Timestamp, ts(Some("UTC"))),
        (DataType::TimestampNtz, ts(None)),
    ];
    for (t, arrow) in cases {
        assert_eq!(t.to_arrow(), arrow, "{t}");
    }
}

#[test]
fn decimal_i64_fast_path_up_to_precision_18() {
    for p in 0..=38_u32 {
        let DataType::Decimal(d) = dec(p, 0) else {
            panic!("not a decimal");
        };
        assert_eq!(d.storage() == DecimalStorage::I64, p <= 18, "p={p}");
        // Only Decimal128 crosses an edge, whatever the internal storage.
        let p8 = d.precision().max(1);
        assert_eq!(
            DataType::Decimal(d).to_arrow(),
            ArrowType::Decimal128(p8, 0)
        );
        assert!(!DataType::Decimal(d).is_valid_storage(&ArrowType::Decimal64(p8, 0)));
    }
    assert!(!dec(10, 2).is_valid_storage(&ArrowType::Decimal128(10, 3)));
    let e = DataType::from_arrow(&ArrowType::Decimal64(9, 2)).map_err(|e| e.error_class());
    assert_eq!(e, Err(ErrorClass::KrishivUnsupportedFeature));
}

#[test]
fn decimal_0_0_is_stored_as_precision_1_with_a_tag() {
    use arrow_array::{Array, Decimal128Array};
    let t = dec(0, 0);
    let field = t.to_arrow_field("z", true);
    assert_eq!(field.data_type(), &ArrowType::Decimal128(1, 0));
    assert_eq!(
        field
            .metadata()
            .get(SPARK_TYPE_METADATA_KEY)
            .map(String::as_str),
        Some("decimal(0,0)")
    );
    assert_eq!(DataType::from_arrow_field(&field), Ok(t.clone()));
    // A real array of the field's type (Arrow rejects precision 0).
    let array = Decimal128Array::from(vec![Some(0), None]).with_precision_and_scale(1, 0);
    let array = array.unwrap_or_else(|e| panic!("{e}"));
    assert!(array.validate_decimal_precision(1).is_ok());
    assert_eq!(array.data_type(), field.data_type());
    assert!(
        Decimal128Array::from(vec![Some(0)])
            .with_precision_and_scale(0, 0)
            .is_err()
    );
    // Untagged Decimal128(0, 0) input still reads as decimal(0,0).
    assert_eq!(DataType::from_arrow(&ArrowType::Decimal128(0, 0)), Ok(t));
}

#[test]
fn timestamps_distinguish_ltz_and_ntz() {
    let any_zone = ArrowType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("+05:30")));
    assert_eq!(DataType::from_arrow(&any_zone), Ok(DataType::Timestamp));
    let none = ArrowType::Timestamp(TimeUnit::Microsecond, None);
    assert_eq!(DataType::from_arrow(&none), Ok(DataType::TimestampNtz));
    assert!(!DataType::Timestamp.is_valid_storage(&none));
    assert!(!DataType::TimestampNtz.is_valid_storage(&any_zone));
    // Any zone label (even "") is a timestamp; ingest re-labels it to UTC.
    let empty = ArrowType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("")));
    assert_eq!(DataType::from_arrow(&empty), Ok(DataType::Timestamp));
    assert!(!DataType::Timestamp.is_valid_storage(&any_zone));
    for a in [&any_zone, &empty] {
        assert_eq!(normalize_ingest_type(a), DataType::Timestamp.to_arrow());
    }
    assert_eq!(normalize_ingest_type(&none), none);
    assert_eq!(normalize_ingest_type(&ArrowType::Utf8), ArrowType::Utf8);
    for unit in [
        TimeUnit::Second,
        TimeUnit::Millisecond,
        TimeUnit::Nanosecond,
    ] {
        let e = DataType::from_arrow(&ArrowType::Timestamp(unit, None));
        assert_eq!(
            e.map_err(|e| e.error_class()),
            Err(ErrorClass::KrishivUnsupportedFeature)
        );
    }
}

#[test]
fn arrow_storage_alternatives() {
    for a in [ArrowType::Utf8, ArrowType::LargeUtf8, ArrowType::Utf8View] {
        assert_eq!(DataType::from_arrow(&a), Ok(DataType::String));
    }
    for a in [
        ArrowType::Binary,
        ArrowType::LargeBinary,
        ArrowType::BinaryView,
    ] {
        assert_eq!(DataType::from_arrow(&a), Ok(DataType::Binary));
        assert!(DataType::String.is_valid_storage(&a));
        assert!(!DataType::Binary.is_valid_storage(&ArrowType::Utf8));
    }
    for a in [
        ArrowType::Date64,
        ArrowType::UInt32,
        ArrowType::Float16,
        ArrowType::Null,
        ArrowType::Decimal32(5, 2),
        ArrowType::Decimal64(5, 2),
        ArrowType::Decimal256(10, 2),
        ArrowType::List(Arc::new(Field::new("item", ArrowType::Int32, true))),
    ] {
        let e = DataType::from_arrow(&a).map_err(|e| e.error_class());
        assert_eq!(e, Err(ErrorClass::KrishivUnsupportedFeature), "{a}");
    }
}

#[test]
fn arrow_decimal_bounds_follow_spark() {
    let class = |a: ArrowType| DataType::from_arrow(&a).map_err(|e| e.error_class());
    assert_eq!(class(ArrowType::Decimal128(38, 38)), Ok(dec(38, 38)));
    assert_eq!(
        class(ArrowType::Decimal128(39, 0)),
        Err(ErrorClass::DecimalPrecisionExceedsMaxPrecision)
    );
    assert_eq!(
        class(ArrowType::Decimal128(10, -2)),
        Err(ErrorClass::NegativeScaleDisallowed)
    );
    assert_eq!(
        class(ArrowType::Decimal128(5, 6)),
        Err(ErrorClass::LegacyErrorTemp1228)
    );
}

#[test]
fn binary_safe_string_field_carries_spark_type() {
    let f = DataType::String.to_arrow_field_with_storage("s", ArrowType::BinaryView, true);
    let f = f.unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(f.data_type(), &ArrowType::BinaryView);
    assert_eq!(
        f.metadata()
            .get(SPARK_TYPE_METADATA_KEY)
            .map(String::as_str),
        Some("string")
    );
    assert_eq!(DataType::from_arrow_field(&f), Ok(DataType::String));

    // Storage that already reads back as the type carries no tag.
    let f = DataType::String.to_arrow_field_with_storage("s", ArrowType::Utf8, false);
    assert!(f.is_ok_and(|f| f.metadata().is_empty()));

    let bad = DataType::Integer.to_arrow_field_with_storage("i", ArrowType::Int64, true);
    assert_eq!(
        bad.map_err(|e| e.error_class()),
        Err(ErrorClass::KrishivUnsupportedFeature)
    );
    let tagged = |tag: &str, arrow: ArrowType| {
        let f = Field::new("x", arrow, true).with_metadata(
            [(SPARK_TYPE_METADATA_KEY.to_owned(), tag.to_owned())]
                .into_iter()
                .collect::<std::collections::HashMap<_, _>>(),
        );
        DataType::from_arrow_field(&f).map_err(|e| e.error_class())
    };
    assert_eq!(
        tagged("int", ArrowType::Utf8),
        Err(ErrorClass::KrishivUnsupportedFeature)
    );
    assert_eq!(
        tagged("int(", ArrowType::Int32),
        Err(ErrorClass::ParseSyntaxError)
    );
}

#[test]
fn decimal_bounds_follow_spark() {
    // Spark's messages and parameters are checked against the fixture in
    // tests/spark_type_strings.rs.
    assert!(DecimalType::try_new(38, 38).is_ok() && DecimalType::try_new(0, 0).is_ok());
    let class = |p, s| DecimalType::try_new(p, s).map_err(|e| e.error_class());
    assert_eq!(
        class(39, 0),
        Err(ErrorClass::DecimalPrecisionExceedsMaxPrecision)
    );
    assert_eq!(class(39, 40), Err(ErrorClass::LegacyErrorTemp1228));
    assert_eq!(class(10, 0), Ok(DecimalType::USER_DEFAULT));
    assert_eq!(
        DataType::from(DecimalType::SYSTEM_DEFAULT).sql(),
        "DECIMAL(38,18)"
    );
}

#[test]
fn every_decimal_prints_and_reparses() {
    for p in 0..=38 {
        for s in 0..=p {
            let t = dec(p, s);
            for text in [
                t.simple_string(),
                t.catalog_string(),
                t.sql(),
                t.type_name(),
            ] {
                assert_eq!(parse_data_type(&text), Ok(t.clone()), "{text}");
            }
            assert_eq!(t.to_string().parse::<DataType>(), Ok(t));
        }
    }
}

#[test]
fn every_type_prints_and_reparses() {
    for t in all_types() {
        for text in [t.simple_string(), t.catalog_string(), t.sql()] {
            assert_eq!(parse_data_type(&text), Ok(t.clone()), "{text}");
            assert_eq!(
                parse_data_type(&format!(" \t{text}\n")),
                Ok(t.clone()),
                "{text}"
            );
            assert_eq!(
                parse_data_type(&text.to_ascii_uppercase()),
                Ok(t.clone()),
                "{text}"
            );
        }
    }
}

#[test]
fn parser_never_panics_on_token_soup() {
    let pieces: Vec<&str> = "int|decimal|timestamp|with|local|time|zone|(|)|,|38|39|-|1.5|`|<|>|\
         array|string|collate| |é|\u{0}|999999999999|\t|dec|numeric|without|:|struct"
        .split('|')
        .collect();
    // Deterministic xorshift generator: the same inputs on every run.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let n = u64::try_from(pieces.len()).unwrap_or(1);
    for _ in 0..20_000 {
        let mut s = String::new();
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        for _ in 0..(state % 8) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let i = usize::try_from(state % n).unwrap_or(0);
            s.push_str(pieces.get(i).copied().unwrap_or(""));
        }
        if let Ok(t) = parse_data_type(&s) {
            assert_eq!(parse_data_type(&t.simple_string()), Ok(t), "{s:?}");
        }
    }
}

#[test]
fn parser_error_classes() {
    let class = |s: &str| parse_data_type(s).map_err(|e| e.error_class());
    assert_eq!(class(" \t"), Err(ErrorClass::ParseEmptyStatement));
    assert_eq!(class("foo"), Err(ErrorClass::UnsupportedDatatype));
    assert_eq!(class("`int`"), Err(ErrorClass::UnsupportedDatatype));
    assert_eq!(class("int int"), Err(ErrorClass::ParseSyntaxError));
    assert_eq!(class("int /* x"), Err(ErrorClass::UnclosedBracketedComment));
    assert_eq!(class("inté"), Err(ErrorClass::InvalidIdentifier));
    assert_eq!(class("int /* a /* b */ */ -- c"), Ok(DataType::Integer));
    let e = parse_data_type("decimal(5,0099999999999)").err();
    let params = e
        .as_ref()
        .map(krishiv_common::error::KrishivError::message_parameters);
    assert_eq!(
        params,
        Some(
            &[
                ("scale", "99999999999".to_owned()),
                ("precision", "5".to_owned())
            ][..]
        )
    );
    assert_eq!(class("decimal(5,6)"), Err(ErrorClass::LegacyErrorTemp1228));
    assert_eq!(
        class("array<int>"),
        Err(ErrorClass::KrishivUnsupportedFeature)
    );
    let e = parse_data_type("timestamp(3)").err();
    assert_eq!(
        e.map(|e| e.to_string()).as_deref(),
        Some("[UNSUPPORTED_DATATYPE] Unsupported data type \"TIMESTAMP(3)\". SQLSTATE: 0A000")
    );
}

#[test]
fn trailing_unrecognized_character_gets_the_extra_input_hint() {
    // Spark's lexer has no token for `#`, `@`, `$`, `{` or `}`, so a trailing
    // one is UNRECOGNIZED and reported as extra input; a recognized
    // punctuation character is not.
    for c in ['#', '@', '$', '{', '}'] {
        let e = parse_data_type(&format!("int{c}")).unwrap_err();
        assert!(
            e.to_string().contains(&format!("extra input '{c}'")),
            "{c}: {e}"
        );
    }
    let e = parse_data_type("int;").unwrap_err();
    assert!(!e.to_string().contains("extra input"), "{e}");
}
