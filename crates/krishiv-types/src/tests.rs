// SPDX-License-Identifier: Apache-2.0
use std::sync::Arc;

use arrow_schema::{DataType as ArrowType, Field, TimeUnit};
use krishiv_common::error::ErrorClass;

use crate::arrow::{
    SPARK_METADATA_KEY, SPARK_TYPE_METADATA_KEY, canonical_arrow_type, client_arrow_type,
    normalize_ingest_type,
};
use crate::{
    ArrayType, DataType, DecimalStorage, DecimalType, MapType, Metadata, Srid, StructField,
    StructType, parse_data_type,
};

fn dec(p: u32, s: u32) -> DataType {
    match DecimalType::try_new(p, s) {
        Ok(d) => DataType::Decimal(d),
        Err(e) => panic!("decimal({p},{s}): {e}"),
    }
}

/// One instance of every in-scope type, decimals at the storage boundary.
fn all_types() -> Vec<DataType> {
    "boolean tinyint smallint int bigint float double string binary decimal(0,0) decimal(10,2) \
     decimal(18,18) decimal(19,0) decimal(38,38) date timestamp timestamp_ntz time(0) time(3) \
     time(6) geometry(0) geometry(3857) geometry(4326) geometry(any) geography(4326) geography(any)"
        .split(' ')
        .map(|s| parse_data_type(s).unwrap_or_else(|e| panic!("{s}: {e}")))
        .collect()
}

/// Types whose canonical Arrow type alone reads back as another type.
fn needs_tag(t: &DataType) -> bool {
    *t == dec(0, 0)
        || matches!(
            t,
            DataType::Time { precision: 0..=5 }
                | DataType::Geometry { .. }
                | DataType::Geography { .. }
        )
}

#[test]
fn spark_to_arrow_to_spark_round_trips_every_type() {
    for t in all_types() {
        let field = t.to_arrow_field("c", true);
        assert_eq!(DataType::from_arrow_field(&field).as_ref(), Ok(&t), "{t}");
        assert!(t.is_valid_storage(&t.to_arrow()), "{t}");
        assert_eq!(field.metadata().is_empty(), !needs_tag(&t), "{t}");
        if !needs_tag(&t) {
            assert_eq!(DataType::from_arrow(&t.to_arrow()), Ok(t.clone()), "{t}");
            assert!(field.metadata().is_empty(), "{t}");
        }
    }
}

#[test]
fn canonical_arrow_types() {
    let ts = |tz: Option<&str>| ArrowType::Timestamp(TimeUnit::Microsecond, tz.map(Arc::from));
    let cases = [
        (DataType::Null, ArrowType::Null),
        (DataType::Boolean, ArrowType::Boolean),
        (DataType::Byte, ArrowType::Int8),
        (DataType::Short, ArrowType::Int16),
        (DataType::Integer, ArrowType::Int32),
        (DataType::Long, ArrowType::Int64),
        (DataType::Float, ArrowType::Float32),
        (DataType::Double, ArrowType::Float64),
        (DataType::String, ArrowType::Utf8View),
        (DataType::Binary, ArrowType::BinaryView),
        (dec(10, 2), ArrowType::Decimal64(10, 2)),
        (dec(38, 38), ArrowType::Decimal128(38, 38)),
        (DataType::Date, ArrowType::Date32),
        (DataType::Timestamp, ts(Some("UTC"))),
        (DataType::TimestampNtz, ts(None)),
        (
            DataType::Time { precision: 2 },
            ArrowType::Time64(TimeUnit::Nanosecond),
        ),
    ];
    for (t, arrow) in cases {
        assert_eq!(t.to_arrow(), arrow, "{t}");
    }
    let spatial = ArrowType::Struct(
        vec![
            Field::new("srid", ArrowType::Int32, false),
            Field::new("wkb", ArrowType::BinaryView, false),
        ]
        .into(),
    );
    for srid in [Srid::Fixed(0), Srid::Any] {
        assert_eq!(DataType::Geometry { srid }.to_arrow(), spatial);
        assert_eq!(DataType::Geography { srid }.to_arrow(), spatial);
    }
}

#[test]
fn time_arrow_mapping() {
    let ns = ArrowType::Time64(TimeUnit::Nanosecond);
    assert_eq!(
        DataType::from_arrow(&ns),
        Ok(DataType::Time { precision: 6 })
    );
    // Any precision is stored as nanoseconds; other units are not a time.
    assert!(DataType::Time { precision: 0 }.is_valid_storage(&ns));
    for a in [
        ArrowType::Time64(TimeUnit::Microsecond),
        ArrowType::Time32(TimeUnit::Millisecond),
        ArrowType::Time32(TimeUnit::Second),
    ] {
        assert!(!DataType::Time { precision: 6 }.is_valid_storage(&a), "{a}");
        let e = DataType::from_arrow(&a).map_err(|e| e.error_class());
        assert_eq!(e, Err(ErrorClass::KrishivUnsupportedFeature), "{a}");
    }
    let field = DataType::Time { precision: 3 }.to_arrow_field("t", false);
    assert_eq!(
        field
            .metadata()
            .get(SPARK_TYPE_METADATA_KEY)
            .map(String::as_str),
        Some("time(3)")
    );
}

#[test]
fn spatial_fields_need_their_tag() {
    let g = DataType::Geometry { srid: Srid::Any };
    let field = g.to_arrow_field("g", true);
    assert_eq!(
        field
            .metadata()
            .get(SPARK_TYPE_METADATA_KEY)
            .map(String::as_str),
        Some("geometry(any)")
    );
    // An untagged struct is not a spatial type: it reads as a plain struct.
    let plain = DataType::from_arrow(field.data_type()).unwrap_or_else(|e| panic!("{e}"));
    assert_ne!(plain, g);
    assert_eq!(plain.simple_string(), "struct<srid:int,wkb:binary>");
    let binary_child = ArrowType::Struct(
        vec![
            Field::new("srid", ArrowType::Int32, false),
            Field::new("wkb", ArrowType::Binary, false),
        ]
        .into(),
    );
    assert!(!g.is_valid_storage(&binary_child));
    assert!(!g.is_valid_storage(&ArrowType::BinaryView));
    let retagged =
        Field::new("g", ArrowType::BinaryView, true).with_metadata(field.metadata().clone());
    assert_eq!(
        DataType::from_arrow_field(&retagged).map_err(|e| e.error_class()),
        Err(ErrorClass::KrishivUnsupportedFeature)
    );
}

#[test]
fn time_precision_bounds_follow_spark() {
    for p in 0..=6 {
        let t = DataType::time(p);
        assert_eq!(
            t.as_ref().map(DataType::simple_string),
            Ok(format!("time({p})"))
        );
    }
    let e = DataType::time(7).err();
    assert_eq!(
        e.map(|e| e.to_string()).as_deref(),
        Some(
            "[UNSUPPORTED_TIME_PRECISION] The seconds precision 7 of the TIME data type is out \
             of the supported range [0, 6]. SQLSTATE: 0A001"
        )
    );
    assert!(DataType::time(u32::MAX).is_err());
    assert_eq!(parse_data_type("time"), Ok(DataType::Time { precision: 6 }));
    assert_eq!(
        parse_data_type("time(2) without time zone"),
        Ok(DataType::Time { precision: 2 })
    );
}

#[test]
fn srids_follow_spark() {
    assert_eq!(
        DataType::geometry(Srid::Fixed(3857)),
        Ok(DataType::Geometry {
            srid: Srid::Fixed(3857)
        })
    );
    assert!(DataType::geography(Srid::Any).is_ok());
    for (srid, geographic) in [
        (1, false),
        (-1, false),
        (0, true),
        (3857, true),
        (9999, true),
    ] {
        let t = if geographic {
            DataType::geography(Srid::Fixed(srid))
        } else {
            DataType::geometry(Srid::Fixed(srid))
        };
        let e = t.err();
        assert_eq!(
            e.as_ref()
                .map(krishiv_common::error::KrishivError::error_class),
            Some(ErrorClass::StInvalidSridValue),
            "{srid}"
        );
        assert_eq!(
            e.map(|e| e.message_parameters().to_vec()),
            Some(vec![("srid", srid.to_string())])
        );
    }
    assert_eq!(Srid::crs(0, false), Some("SRID:0"));
    assert_eq!(Srid::crs(3857, false), Some("EPSG:3857"));
    assert_eq!(Srid::crs(4326, false), Some("OGC:CRS84"));
    assert_eq!(Srid::crs(4326, true), Some("OGC:CRS84"));
    assert_eq!(Srid::crs(3857, true), None);
    for s in [Srid::Any, Srid::Fixed(0), Srid::Fixed(4326)] {
        assert_eq!(Srid::from_i32(s.to_i32()), s);
    }
    assert_eq!(Srid::Any.to_i32(), -1);
    // `ANY` must close the argument list and end the type.
    for s in ["geometry(any", "geometry(any,1)", "geography(any) int"] {
        let e = parse_data_type(s).map_err(|e| e.error_class());
        assert_eq!(e, Err(ErrorClass::ParseSyntaxError), "{s}");
    }
}

#[test]
fn time_and_spatial_types_round_trip_through_json() {
    // Spark's JSON names from the type-string fixture.
    let cases = [
        (DataType::Time { precision: 0 }, "\"time(0)\""),
        (DataType::Time { precision: 6 }, "\"time(6)\""),
        (
            DataType::Geometry {
                srid: Srid::Fixed(0),
            },
            "\"geometry(SRID:0)\"",
        ),
        (
            DataType::Geometry {
                srid: Srid::Fixed(3857),
            },
            "\"geometry(EPSG:3857)\"",
        ),
        (
            DataType::Geometry {
                srid: Srid::Fixed(4326),
            },
            "\"geometry(OGC:CRS84)\"",
        ),
        (
            DataType::Geometry { srid: Srid::Any },
            "\"geometry(SRID:ANY)\"",
        ),
        (
            DataType::Geography {
                srid: Srid::Fixed(4326),
            },
            "\"geography(OGC:CRS84, SPHERICAL)\"",
        ),
        (
            DataType::Geography { srid: Srid::Any },
            "\"geography(SRID:ANY, SPHERICAL)\"",
        ),
    ];
    for (t, json) in cases {
        assert_eq!(t.json(), json, "{t}");
        assert_eq!(DataType::from_json(&t.json()).as_ref(), Ok(&t), "{t}");
    }
    // Nested under arrays, maps and structs too.
    let nested: DataType = ArrayType::new(
        MapType::new(
            DataType::Time { precision: 3 },
            DataType::Geometry { srid: Srid::Any },
            true,
        )
        .into(),
        true,
    )
    .into();
    assert_eq!(DataType::from_json(&nested.json()).as_ref(), Ok(&nested));
    let class = |s: &str| DataType::from_json(s).map_err(|e| e.error_class());
    assert_eq!(
        class("\"time(7)\""),
        Err(ErrorClass::UnsupportedTimePrecision)
    );
    assert_eq!(
        class("\"geography(SRID:0, SPHERICAL)\""),
        Err(ErrorClass::StInvalidSridValue)
    );
    assert_eq!(
        class("\"geometry(SRID:1)\""),
        Err(ErrorClass::InvalidJsonDataType)
    );
}

#[test]
fn decimal_i64_fast_path_up_to_precision_18() {
    for p in 0..=38_u32 {
        let DataType::Decimal(d) = dec(p, 0) else {
            panic!("not a decimal");
        };
        assert_eq!(d.storage() == DecimalStorage::I64, p <= 18, "p={p}");
        // The carrier follows the precision; every carrier is valid storage.
        let p8 = d.precision().max(1);
        let carrier = if p <= 18 {
            ArrowType::Decimal64(p8, 0)
        } else {
            ArrowType::Decimal128(p8, 0)
        };
        assert_eq!(DataType::Decimal(d).to_arrow(), carrier);
        assert!(DataType::Decimal(d).is_valid_storage(&ArrowType::Decimal64(p8, 0)));
        assert!(DataType::Decimal(d).is_valid_storage(&ArrowType::Decimal128(p8, 0)));
    }
    assert!(!dec(10, 2).is_valid_storage(&ArrowType::Decimal128(10, 3)));
    assert_eq!(
        DataType::from_arrow(&ArrowType::Decimal64(9, 2)),
        Ok(dec(9, 2))
    );
    assert_eq!(
        DataType::from_arrow(&ArrowType::Decimal32(9, 2)),
        Ok(dec(9, 2))
    );
}

#[test]
fn decimal_0_0_is_stored_as_precision_1_with_a_tag() {
    use arrow_array::{Array, Decimal64Array, Decimal128Array};
    let t = dec(0, 0);
    let field = t.to_arrow_field("z", true);
    assert_eq!(field.data_type(), &ArrowType::Decimal64(1, 0));
    assert_eq!(
        field
            .metadata()
            .get(SPARK_TYPE_METADATA_KEY)
            .map(String::as_str),
        Some("decimal(0,0)")
    );
    assert_eq!(DataType::from_arrow_field(&field), Ok(t.clone()));
    // A real array of the field's type (Arrow rejects precision 0).
    let array = Decimal64Array::from(vec![Some(0), None]).with_precision_and_scale(1, 0);
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
        ArrowType::Decimal256(10, 2),
        ArrowType::FixedSizeList(Arc::new(Field::new("item", ArrowType::Int32, true)), 2),
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
         array|string|collate| |é|\u{0}|999999999999|\t|dec|numeric|without|:|struct|geometry|\
         geography|any|4326|3857|0|6|7"
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
        class("array<char(3)>"),
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

#[test]
fn void_is_spark_null_type() {
    let t = DataType::Null;
    assert_eq!(t.simple_string(), "void");
    assert_eq!(t.sql(), "VOID");
    assert_eq!(t.type_name(), "void");
    assert_eq!(crate::parse_data_type("VOID"), Ok(DataType::Null));
    assert_eq!(DataType::from_arrow(&ArrowType::Null), Ok(DataType::Null));
}

fn ty(s: &str) -> DataType {
    parse_data_type(s).unwrap_or_else(|e| panic!("{s}: {e}"))
}

/// Nested types up to depth 4, with every nullability flag, metadata and
/// children that need an Arrow type tag.
fn nested_types() -> Vec<DataType> {
    let mut out: Vec<DataType> = [
        "array<int>",
        "array<array<array<array<int>>>>",
        "map<string,array<double>>",
        "map<map<int,int>,int>",
        "struct<>",
        "struct<a:int not null,b:string comment 'c',`x y`:binary>",
        "array<struct<a:int,b:map<string,array<decimal(0,0)>>>>",
        "map<string,struct<a:array<map<int,struct<b:timestamp_ntz not null>>>>>",
        "struct<a:struct<b:struct<c:struct<d:decimal(38,38)>>>>",
    ]
    .into_iter()
    .map(ty)
    .collect();
    out.push(ArrayType::new(DataType::Long, false).into());
    out.push(MapType::new(DataType::String, DataType::Date, false).into());
    out
}

#[test]
fn nested_types_round_trip_through_arrow_json_and_ddl() {
    for t in nested_types() {
        assert_eq!(DataType::from_arrow(&t.to_arrow()).as_ref(), Ok(&t), "{t}");
        let field = t.to_arrow_field("c", false);
        assert!(field.metadata().is_empty(), "{t}");
        assert_eq!(DataType::from_arrow_field(&field).as_ref(), Ok(&t), "{t}");
        assert_eq!(DataType::from_json(&t.json()).as_ref(), Ok(&t), "{t}");
        // `sql` keeps nullability, quoting and comments; DDL defaults only
        // the array and map flags.
        let defaults = !matches!(&t, DataType::Array(a) if !a.contains_null())
            && !matches!(&t, DataType::Map(m) if !m.value_contains_null());
        if defaults {
            assert_eq!(parse_data_type(&t.sql()).as_ref(), Ok(&t), "{t}");
        }
    }
    assert_eq!(ty("array<array<array<array<int>>>>").nesting_depth(), 4);
    assert_eq!(DataType::Integer.nesting_depth(), 0);
}

#[test]
fn nested_arrow_layout_follows_spark_arrow_utils() {
    let list = |f: Field| ArrowType::List(Arc::new(f));
    assert_eq!(
        ty("array<int>").to_arrow(),
        list(Field::new("element", ArrowType::Int32, true))
    );
    let ArrowType::Map(entries, sorted) = ty("map<string,int>").to_arrow() else {
        panic!("not a map");
    };
    assert!(!sorted && !entries.is_nullable());
    assert_eq!(entries.name(), "entries");
    let ArrowType::Struct(kv) = entries.data_type() else {
        panic!("map entries are not a struct");
    };
    assert_eq!((kv[0].name().as_str(), kv[0].is_nullable()), ("key", false));
    assert_eq!(
        (kv[1].name().as_str(), kv[1].is_nullable()),
        ("value", true)
    );
    // A child that needs a tag carries it, as a column would.
    let ArrowType::List(child) = ty("array<decimal(0,0)>").to_arrow() else {
        panic!("not a list");
    };
    assert_eq!(
        child
            .metadata()
            .get(SPARK_TYPE_METADATA_KEY)
            .map(String::as_str),
        Some("decimal(0,0)")
    );
    // Other layouts read too: `LargeList`, any child names.
    let large = ArrowType::LargeList(Arc::new(Field::new("item", ArrowType::Utf8, false)));
    assert_eq!(
        DataType::from_arrow(&large),
        Ok(ArrayType::new(DataType::String, false).into())
    );
    let commented = ty("struct<a:int comment 'c'>").to_arrow();
    let ArrowType::Struct(fields) = &commented else {
        panic!("not a struct");
    };
    assert_eq!(
        fields[0]
            .metadata()
            .get(SPARK_METADATA_KEY)
            .map(String::as_str),
        Some(r#"{"comment":"c"}"#)
    );
}

#[test]
fn nesting_deeper_than_the_limit_is_refused_everywhere() {
    let deep = |n: usize| format!("{}int{}", "array<".repeat(n), ">".repeat(n));
    let max = DataType::MAX_NESTING_DEPTH;
    let ok = ty(&deep(max));
    assert_eq!(ok.nesting_depth(), max);
    let class =
        |r: Result<DataType, krishiv_common::error::KrishivError>| r.map_err(|e| e.error_class());
    assert_eq!(
        class(parse_data_type(&deep(max + 1))),
        Err(ErrorClass::FailedToParseTooComplex)
    );
    assert_eq!(class(DataType::from_arrow(&ok.to_arrow())), Ok(ok.clone()));
    let too_deep: DataType = ArrayType::new(ok.clone(), true).into();
    assert_eq!(
        class(DataType::from_arrow(&too_deep.to_arrow())),
        Err(ErrorClass::FailedToParseTooComplex)
    );
    assert_eq!(class(DataType::from_json(&ok.json())), Ok(ok));
    assert_eq!(
        class(DataType::from_json(&too_deep.json())),
        Err(ErrorClass::FailedToParseTooComplex)
    );
    // Far deeper input fails cleanly, without exhausting the stack.
    assert!(parse_data_type(&deep(100_000)).is_err());
    assert!(DataType::from_json(&"[".repeat(100_000)).is_err());
}

#[test]
fn json_that_is_no_type_is_invalid_json_data_type() {
    for json in [
        "",
        "{}",
        r#""int""#,
        r#""bigint""#,
        r#"{"type":"array","elementType":"integer"}"#,
        r#"{"type":"array","elementType":"integer","containsNull":true,"x":1}"#,
        r#"{"type":"map","keyType":"integer","valueType":"integer","valueContainsNull":1}"#,
        r#"{"type":"struct","fields":[{"name":"a","type":"integer"}]}"#,
        r#"{"type":"struct","fields":[{"name":"a","type":"integer","nullable":true,"metadata":[]}]}"#,
        r#""decimal(1.5,0)""#,
    ] {
        let class = DataType::from_json(json).map_err(|e| e.error_class());
        assert_eq!(class, Err(ErrorClass::InvalidJsonDataType), "{json}");
    }
    let json = |s: &str| DataType::from_json(s).map_err(|e| e.error_class());
    assert_eq!(
        json(r#""decimal""#),
        Ok(DataType::Decimal(DecimalType::USER_DEFAULT))
    );
    assert_eq!(json(r#""decimal( 10 , 2 )""#), Ok(dec(10, 2)));
    assert_eq!(
        json(r#""decimal(10,-1)""#),
        Err(ErrorClass::NegativeScaleDisallowed)
    );
    assert_eq!(
        json(r#""char(3)""#),
        Err(ErrorClass::KrishivUnsupportedFeature)
    );
    // Metadata is optional, as in Spark.
    assert_eq!(
        json(r#"{"type":"struct","fields":[{"name":"a","type":"long","nullable":false}]}"#),
        Ok(StructType::new([StructField::new("a", DataType::Long, false)]).into())
    );
}

#[test]
fn spark_type_equalities() {
    let a = ty("struct<a:array<int>,b:map<string,int>>");
    let nullability: DataType = StructType::new([
        StructField::new("a", ArrayType::new(DataType::Integer, false).into(), false),
        StructField::new(
            "b",
            MapType::new(DataType::String, DataType::Integer, false).into(),
            true,
        )
        .with_metadata(Metadata::with_comment("c")),
    ])
    .into();
    assert_ne!(a, nullability);
    assert!(a.equals_ignore_nullability(&nullability));
    let upper = ty("struct<A:array<int>,b:map<string,int>>");
    assert!(!a.equals_ignore_nullability(&upper));
    assert!(a.equals_ignore_case_and_nullability(&upper));
    let renamed = ty("struct<x:array<int>,y:map<string,int>>");
    assert!(!a.equals_ignore_case_and_nullability(&renamed));
    assert!(a.equals_structurally(&renamed, false));
    assert!(!a.equals_structurally(&nullability, false));
    assert!(a.equals_structurally(&nullability, true));
    assert!(!a.equals_structurally(&ty("struct<a:array<bigint>,b:map<string,int>>"), true));
    // Union: the first side's names and metadata, either side's nulls.
    let merged = nullability.union_like_merge(&renamed);
    assert_eq!(
        merged.json(),
        r#"{"type":"struct","fields":[{"name":"a","type":{"type":"array","elementType":"integer","containsNull":true},"nullable":true,"metadata":{}},{"name":"b","type":{"type":"map","keyType":"string","valueType":"integer","valueContainsNull":true},"nullable":true,"metadata":{"comment":"c"}}]}"#
    );
}

#[test]
fn struct_strings_quote_names_and_escape_comments_as_spark() {
    let t = ty(r"struct<`a b`:int comment 'it\'s', `select`:string not null, _x1:int>");
    assert_eq!(t.simple_string(), "struct<a b:int,select:string,_x1:int>");
    assert_eq!(
        t.sql(),
        r"STRUCT<`a b`: INT COMMENT 'it\'s', select: STRING NOT NULL, _x1: INT>"
    );
    assert_eq!(parse_data_type(&t.sql()).as_ref(), Ok(&t));
    // The empty struct parses as `<>` and as `< >`.
    assert_eq!(ty("struct<>"), ty("struct< >"));
}

/// Metadata keys print sorted whatever order they arrive in: the JSON form,
/// and so the plan codec's bytes, must not depend on serde_json's map
/// (`preserve_order` enabled by any crate would change it).
#[test]
fn metadata_prints_with_sorted_keys() {
    let t = DataType::from_json(
        r#"{"type":"struct","fields":[{"name":"a","type":"integer","nullable":true,"metadata":{"z":1,"comment":"c","b":[true]}}]}"#,
    )
    .unwrap();
    assert!(
        t.json()
            .contains(r#""metadata":{"b":[true],"comment":"c","z":1}"#),
        "{}",
        t.json()
    );
}

#[test]
fn field_names_compare_as_java_ignores_case() {
    use crate::equals_ignore_case as eq;
    assert!(eq("abc", "ABC") && eq("é", "É") && eq("straße", "STRAßE"));
    assert!(!eq("straße", "STRASSE") && !eq("a", "b") && !eq("a", "aa"));
    let t = ty("struct<`é`:int>");
    assert!(t.equals_ignore_case_and_nullability(&ty("struct<`É`:int>")));
}

#[test]
fn a_type_tag_counts_towards_the_nesting_limit() {
    let inner = format!(
        "{}int{}",
        "array<".repeat(DataType::MAX_NESTING_DEPTH),
        ">".repeat(DataType::MAX_NESTING_DEPTH)
    );
    let tagged = Field::new("element", ArrowType::Int32, true).with_metadata(
        std::collections::HashMap::from([(SPARK_TYPE_METADATA_KEY.to_owned(), inner)]),
    );
    let outer = ArrowType::List(Arc::new(tagged));
    assert_eq!(
        DataType::from_arrow(&outer).map_err(|e| e.error_class()),
        Err(ErrorClass::FailedToParseTooComplex)
    );
}

/// A malformed WKB value is Spark's `WKB_PARSE_ERROR` with the problem and
/// the byte position.
#[test]
fn malformed_wkb_is_a_wkb_parse_error() {
    use crate::geo::{ByteOrder, SpatialKind, rewrite};
    let e = rewrite(&[], SpatialKind::Geometry, ByteOrder::Ndr).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::WkbParseError);
    assert_eq!(
        e.message_parameters().to_vec(),
        vec![
            ("parseError", "WKB data is empty or null".to_owned()),
            ("pos", "0".to_owned())
        ]
    );
}

/// `geometry(any)` and `geography(any)` nest like any other type: the token
/// after `(any)` is the enclosing type's, and only at the top level is more
/// input an error.
#[test]
fn any_srid_spatial_types_nest() {
    for (ddl, want) in [
        ("array<geometry(any)>", "array<geometry(any)>"),
        ("map<string,geography(any)>", "map<string,geography(any)>"),
        (
            "struct<a:geometry(any) not null,b:int>",
            "struct<a:geometry(any),b:int>",
        ),
    ] {
        let t = parse_data_type(ddl).unwrap_or_else(|e| panic!("{ddl}: {e}"));
        assert_eq!(t.simple_string(), want, "{ddl}");
    }
    let e = parse_data_type("geography(any) int").unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::ParseSyntaxError);
}

#[test]
fn decimal_carriers_map_through_nested_types() {
    let item = |t: ArrowType| Arc::new(Field::new("item", t, true));
    let list = |t| ArrowType::List(item(t));
    let row = |t| ArrowType::Struct(vec![Field::new("d", t, true)].into());
    assert_eq!(
        canonical_arrow_type(&row(list(ArrowType::Decimal128(18, 2)))),
        row(list(ArrowType::Decimal64(18, 2)))
    );
    assert_eq!(
        canonical_arrow_type(&ArrowType::Decimal32(5, 2)),
        ArrowType::Decimal64(5, 2)
    );
    assert_eq!(
        canonical_arrow_type(&ArrowType::Decimal128(19, 2)),
        ArrowType::Decimal128(19, 2)
    );
    assert_eq!(
        client_arrow_type(&row(list(ArrowType::Decimal64(18, 2)))),
        row(list(ArrowType::Decimal128(18, 2)))
    );
    let map = ArrowType::Map(item(row(ArrowType::Decimal64(3, 0))), false);
    assert_eq!(
        client_arrow_type(&map),
        ArrowType::Map(item(row(ArrowType::Decimal128(3, 0))), false)
    );
    let large = ArrowType::LargeList(item(ArrowType::Decimal64(3, 0)));
    assert_eq!(
        client_arrow_type(&large),
        ArrowType::LargeList(item(ArrowType::Decimal128(3, 0)))
    );
    let fixed = ArrowType::FixedSizeList(item(ArrowType::Decimal128(3, 0)), 2);
    assert_eq!(
        canonical_arrow_type(&fixed),
        ArrowType::FixedSizeList(item(ArrowType::Decimal64(3, 0)), 2)
    );
    assert_eq!(client_arrow_type(&ArrowType::Int32), ArrowType::Int32);
}
