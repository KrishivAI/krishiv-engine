// SPDX-License-Identifier: Apache-2.0
//! Variant kernels against Spark 4.2.0 (the `variant-field-extractions.sql`
//! goldens and the design note's oracle probes).

use std::sync::Arc;

use arrow_array::types::{Decimal128Type, Int32Type};
use arrow_array::{Array, ArrayRef, Int32Array, ListArray, StringArray, StructArray};
use arrow_schema::{DataType as ArrowType, Field};
use krishiv_common::error::ErrorClass;
use krishiv_types::variant::{VariantBuilder, VariantRef, VariantWriter};
use krishiv_types::{ArrayType, DataType, DecimalType, StructField, StructType};

use super::*;
use crate::cast::FixedOffset;

fn utc() -> FixedOffset {
    FixedOffset::parse("+00:00").unwrap()
}

fn column(json: &[Option<&str>]) -> ArrayRef {
    let mut b = VariantColumnBuilder::with_capacity(json.len());
    for j in json {
        match j {
            Some(j) => {
                let (m, v) = VariantBuilder::from_json(j).unwrap();
                b.append(&m, &v);
            }
            None => b.append_null(),
        }
    }
    b.finish().unwrap()
}

fn json_of(col: &ArrayRef) -> Vec<Option<String>> {
    let out = to_json(col.as_ref(), &utc()).unwrap();
    out.as_string_view()
        .iter()
        .map(|s| s.map(str::to_owned))
        .collect()
}

#[test]
fn prints_as_spark() {
    let col = column(&[
        Some(r#"{"b":1,"a":2}"#),
        Some("1.0"),
        Some("1.50"),
        Some("[0.10, 100, 1.00e1]"),
        Some(r#""a\"b\u0001""#),
        Some("null"),
        None,
        Some(r#"{ "price": 12345.678 }"#),
    ]);
    assert_eq!(
        json_of(&col),
        vec![
            Some(r#"{"a":2,"b":1}"#.to_owned()),
            Some("1".to_owned()),
            Some("1.5".to_owned()),
            Some("[0.1,100,10.0]".to_owned()),
            Some(r#""a\"b\u0001""#.to_owned()),
            Some("null".to_owned()),
            None,
            Some(r#"{"price":12345.678}"#.to_owned()),
        ]
    );
}

#[test]
fn parses_paths_as_spark() {
    use PathSegment::{Index, Key};
    assert_eq!(parse_path("$"), Some(vec![]));
    assert_eq!(
        parse_path("$.item[0].model"),
        Some(vec![Key("item".into()), Index(0), Key("model".into())])
    );
    assert_eq!(
        parse_path("$['field.name']"),
        Some(vec![Key("field.name".into())])
    );
    assert_eq!(parse_path(r#"$["a b"]"#), Some(vec![Key("a b".into())]));
    assert_eq!(
        parse_path("$.field-name"),
        Some(vec![Key("field-name".into())])
    );
    for bad in [
        "",
        "a",
        "$.",
        "$[",
        "$[x]",
        "$[-1]",
        "$[99999999999]",
        "$..a",
        "$['a",
    ] {
        assert_eq!(parse_path(bad), None, "{bad}");
    }
    let e = parse_path_or_raise("$[x]", "variant_get").unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InvalidVariantGetPath);
}

fn first<'a>(col: &'a VariantColumn) -> VariantRef<'a> {
    col.get(0).unwrap().unwrap()
}

#[test]
fn extracts_by_path() {
    let col = column(&[Some(
        r#"{ "item": [ { "model" : "basic", "price" : 6.12 }, { "model" : "medium" } ] }"#,
    )]);
    let col = VariantColumn::new(col.as_ref()).unwrap();
    let v = first(&col);
    let get = |p: &str| {
        extract(v, &parse_path(p).unwrap()).unwrap().map(|x| {
            let mut s = String::new();
            write_json(&x, &utc(), &mut s).unwrap();
            s
        })
    };
    assert_eq!(get("$.item[0].model").as_deref(), Some(r#""basic""#));
    assert_eq!(get("$.item[1].price"), None);
    assert_eq!(get("$.item[2]"), None);
    assert_eq!(get("$.item.model"), None);
    assert_eq!(get("$[0]"), None);
}

fn cast_rows(json: &[Option<&str>], to: &DataType, fail: bool) -> Result<ArrayRef, KrishivError> {
    let col = column(json);
    cast_from_variant(col.as_ref(), to, fail, &utc())
}

#[test]
fn casts_scalars_as_spark() {
    let dec = DataType::Decimal(DecimalType::try_new(5, 2).unwrap());
    let out = cast_rows(
        &[Some("30"), Some("\"12\""), Some("null"), None],
        &dec,
        true,
    )
    .unwrap();
    let out = out.as_primitive::<arrow_array::types::Decimal64Type>();
    assert_eq!(out.value(0), 3000);
    assert_eq!(out.value(1), 1200);
    assert!(out.is_null(2) && out.is_null(3));

    let e = cast_rows(
        &[Some("1"), Some("12345.678")],
        &DataType::Decimal(DecimalType::try_new(3, 2).unwrap()),
        true,
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InvalidVariantCast);
    assert!(e.to_string().contains("12345.678"), "{e}");
    assert!(e.to_string().contains("DECIMAL(3,2)"), "{e}");

    let out = cast_rows(
        &[Some("\"x\""), Some("3.7"), Some("[1]")],
        &DataType::Integer,
        false,
    )
    .unwrap();
    let out = out.as_primitive::<Int32Type>();
    assert!(out.is_null(0));
    assert_eq!(out.value(1), 3);
    assert!(out.is_null(2));

    let out = cast_rows(
        &[Some("30"), Some(r#"{"b":1,"a":[true]}"#), Some("\"s\"")],
        &DataType::String,
        true,
    )
    .unwrap();
    let out: Vec<_> = out.as_string_view().iter().collect();
    assert_eq!(
        out,
        vec![Some("30"), Some(r#"{"a":[true],"b":1}"#), Some("s")]
    );

    let out = cast_rows(&[Some("true")], &DataType::Boolean, true).unwrap();
    assert!(out.as_boolean().value(0));
    // A boolean is no date.
    assert!(cast_rows(&[Some("true")], &DataType::Date, true).is_err());
    let ts = cast_rows(&[Some("2")], &DataType::Timestamp, true).unwrap();
    assert_eq!(
        ts.as_primitive::<arrow_array::types::TimestampMicrosecondType>()
            .value(0),
        2_000_000
    );
}

#[test]
fn casts_containers() {
    let arr = DataType::Array(Arc::new(ArrayType::new(DataType::Integer, true)));
    let out = cast_rows(&[Some("[85, 92, null]"), Some("{}"), None], &arr, false).unwrap();
    let list = out.as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(list.value_length(0), 3);
    assert!(list.is_null(1) && list.is_null(2));
    let st = DataType::Struct(StructType::new(vec![
        StructField::new("a", DataType::Long, true),
        StructField::new("z", DataType::String, true),
    ]));
    let out = cast_rows(&[Some(r#"{"z":"q","a":7,"x":1}"#)], &st, true).unwrap();
    let s = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(
        s.column(0)
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(0),
        7
    );
    assert_eq!(s.column(1).as_string_view().value(0), "q");
}

#[test]
fn variant_target_rebuilds_the_dictionary() {
    let col = column(&[Some(r#"{"a":{"b":1},"c":2}"#)]);
    let c = VariantColumn::new(col.as_ref()).unwrap();
    let inner = extract(first(&c), &parse_path("$.a").unwrap())
        .unwrap()
        .unwrap();
    let out = cast_values(&[Some(inner)], &DataType::Variant, true, &utc()).unwrap();
    let out = VariantColumn::new(out.as_ref()).unwrap();
    let v = first(&out);
    // Only `b` is left in the dictionary.
    assert_eq!(v.metadata(), &[0x01, 0x01, 0x00, 0x01, b'b']);
    let mut s = String::new();
    write_json(&v, &utc(), &mut s).unwrap();
    assert_eq!(s, r#"{"b":1}"#);
}

#[test]
fn builds_from_values() {
    let ints: ArrayRef = Arc::new(Int32Array::from(vec![Some(30), None, Some(-300)]));
    let out = cast_to_variant(ints.as_ref(), &DataType::Integer).unwrap();
    assert_eq!(
        json_of(&out),
        vec![Some("30".into()), None, Some("-300".into())]
    );
    let strings: ArrayRef = Arc::new(StringArray::from(vec!["a"]));
    let out = cast_to_variant(strings.as_ref(), &DataType::String).unwrap();
    assert_eq!(json_of(&out), vec![Some(r#""a""#.into())]);
    let field = Arc::new(Field::new("element", ArrowType::Int32, true));
    let list: ArrayRef = Arc::new(
        ListArray::try_new(
            field,
            arrow_buffer::OffsetBuffer::new(vec![0, 2].into()),
            Arc::new(Int32Array::from(vec![Some(1), None])),
            None,
        )
        .unwrap(),
    );
    let t = DataType::Array(Arc::new(ArrayType::new(DataType::Integer, true)));
    let out = cast_to_variant(list.as_ref(), &t).unwrap();
    assert_eq!(json_of(&out), vec![Some("[1,null]".into())]);
    let dec: ArrayRef = Arc::new(
        arrow_array::PrimitiveArray::<Decimal128Type>::from(vec![1050_i128])
            .with_precision_and_scale(20, 2)
            .unwrap(),
    );
    let out = cast_to_variant(
        dec.as_ref(),
        &DataType::Decimal(DecimalType::try_new(20, 2).unwrap()),
    )
    .unwrap();
    assert_eq!(json_of(&out), vec![Some("10.5".into())]);
}

#[test]
fn duplicate_struct_keys_are_refused() {
    let st = DataType::Struct(StructType::new(vec![
        StructField::new("a", DataType::Integer, true),
        StructField::new("a", DataType::Integer, true),
    ]));
    let arrow = st.to_arrow();
    let ArrowType::Struct(fields) = arrow else {
        unreachable!()
    };
    let one: ArrayRef = Arc::new(Int32Array::from(vec![1]));
    let s: ArrayRef =
        Arc::new(StructArray::try_new(fields, vec![Arc::clone(&one), one], None).unwrap());
    let e = cast_to_variant(s.as_ref(), &st).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::VariantDuplicateKey);
}

#[test]
fn a_container_cast_raises_spark_s_first_error() {
    // Spark casts row by row: row 0's element fails before row 1's shape.
    let arr = DataType::Array(Arc::new(ArrayType::new(DataType::Integer, true)));
    let e = cast_rows(&[Some(r#"["x"]"#), Some("5")], &arr, true).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InvalidVariantCast);
    assert!(
        e.to_string().contains(r#""x""#) && e.to_string().contains("\"INT\""),
        "{e}"
    );
    // Within one object, fields go in stored (key) order, not struct order.
    let st = DataType::Struct(StructType::new(vec![
        StructField::new("z", DataType::Integer, true),
        StructField::new("a", DataType::Integer, true),
    ]));
    let e = cast_rows(&[Some(r#"{"z":"zz","a":"aa"}"#)], &st, true).unwrap_err();
    assert!(e.to_string().contains("aa"), "{e}");
    // Without failing, the bad element is NULL and the row stays.
    let out = cast_rows(&[Some(r#"["x", 2]"#)], &arr, false).unwrap();
    assert_eq!(out.as_list::<i32>().value(0).null_count(), 1);
}

#[test]
fn timestamps_print_the_offset_without_seconds() {
    let col = cast_to_variant(
        &(Arc::new(
            arrow_array::TimestampMicrosecondArray::from(vec![-2_208_988_800_000_000_i64])
                .with_timezone("UTC"),
        ) as ArrayRef),
        &DataType::Timestamp,
    )
    .unwrap();
    // Asia/Kolkata in 1900 is +05:21:10 (Madras time); Spark prints +05:21.
    let zone = crate::datetime::resolve_zone("Asia/Kolkata").unwrap();
    let out = to_json(col.as_ref(), &zone).unwrap();
    assert_eq!(
        out.as_string_view().value(0),
        r#""1900-01-01 05:21:10+05:21""#
    );
}

#[test]
fn a_string_casts_to_time() {
    let out = cast_rows(
        &[Some(r#""10:00:01.5""#)],
        &DataType::Time { precision: 6 },
        true,
    )
    .unwrap();
    assert_eq!(
        out.as_primitive::<arrow_array::types::Time64NanosecondType>()
            .value(0),
        36_001_500_000_000
    );
}

fn schema_of(json: &str) -> String {
    let (m, v) = VariantBuilder::from_json(json).unwrap();
    Schema::of(&VariantRef::new(&m, &v).unwrap()).unwrap().sql()
}

#[test]
fn schemas_print_as_spark() {
    // Spark 4.2.0 (design P2-T001c section 6, the fixtures' `schema` column
    // and the function's own examples).
    assert_eq!(schema_of("1.0"), "DECIMAL(1,0)");
    assert_eq!(schema_of("1.50"), "DECIMAL(2,1)");
    assert_eq!(schema_of("100.0"), "DECIMAL(3,0)");
    assert_eq!(schema_of("1e2"), "DOUBLE");
    assert_eq!(schema_of("1"), "BIGINT");
    assert_eq!(schema_of("null"), "VOID");
    assert_eq!(
        schema_of(r#"{"a":"x1","b":"s1","c":[1,2,{"d":1}]}"#),
        "OBJECT<a: STRING, b: STRING, c: ARRAY<VARIANT>>"
    );
    assert_eq!(schema_of(r#"[5, 1.50, "q", null]"#), "ARRAY<VARIANT>");
    assert_eq!(schema_of("[1, 2.5]"), "ARRAY<DECIMAL(21,1)>");
    assert_eq!(schema_of("[1.5, 1e2]"), "ARRAY<DOUBLE>");
    assert_eq!(schema_of("[[1], [null], []]"), "ARRAY<ARRAY<BIGINT>>");
    assert_eq!(
        schema_of(r#"[{"b":1}, {"a":true, "b":null}]"#),
        "ARRAY<OBJECT<a: BOOLEAN, b: BIGINT>>"
    );
    assert_eq!(schema_of(r#"{"a-b":1}"#), "OBJECT<`a-b`: BIGINT>");
    let merged = Schema::Long
        .merge(Schema::Boolean)
        .merge(Schema::Decimal(3, 2));
    assert_eq!(merged.sql(), "VARIANT");
}

#[test]
fn types_spark_does_not_read_are_unknown_primitives() {
    // A TIME value (primitive id 17): defined by the encoding, but Spark
    // 4.2.0 neither prints nor casts it.
    let metadata = [0x01, 0x00, 0x00];
    let mut value = vec![17 << 2];
    value.extend_from_slice(&3_600_000_000_i64.to_le_bytes());
    let v = VariantRef::new(&metadata, &value).unwrap();
    let mut s = String::new();
    let e = write_json(&v, &utc(), &mut s).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnknownPrimitiveTypeInVariant);
    let e = cast_values(&[Some(v)], &DataType::String, true, &utc()).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnknownPrimitiveTypeInVariant);
}

#[test]
fn review_cases_cast_as_spark() {
    // A date keeps its type into a string cast.
    let d: ArrayRef = Arc::new(arrow_array::Date32Array::from(vec![1]));
    let v = cast_to_variant(d.as_ref(), &DataType::Date).unwrap();
    let s = cast_from_variant(v.as_ref(), &DataType::String, true, &utc()).unwrap();
    assert_eq!(s.as_string_view().value(0), "1970-01-02");
    // A 38-digit decimal to a timestamp: no intermediate overflow.
    let ts = cast_rows(
        &[Some("12345678.901234567890123456789012345678")],
        &DataType::Timestamp,
        true,
    )
    .unwrap();
    assert_eq!(
        ts.as_primitive::<arrow_array::types::TimestampMicrosecondType>()
            .value(0),
        12_345_678_901_234
    );
    // An empty struct keeps the row count.
    let empty = DataType::Struct(StructType::new(vec![]));
    let out = cast_rows(&[Some("{}"), Some("{\"a\":1}")], &empty, true).unwrap();
    assert_eq!(out.len(), 2);
}

/// Every Spark type into a variant and back out, and the JSON of every kind.
#[test]
fn every_type_round_trips_through_a_variant() {
    use arrow_array::builder::{Int32Builder, MapBuilder, StringBuilder};
    use arrow_array::{
        BinaryArray, BooleanArray, Date32Array, Decimal64Array, Float32Array, Float64Array,
        Int8Array, Int16Array, Int64Array, LargeStringArray, StringViewArray,
        TimestampMicrosecondArray,
    };
    use krishiv_types::MapType;
    let dec = |p, s| DataType::Decimal(DecimalType::try_new(p, s).unwrap());
    let cases: Vec<(ArrayRef, DataType, &str)> = vec![
        (
            Arc::new(BooleanArray::from(vec![true])),
            DataType::Boolean,
            "true",
        ),
        (Arc::new(Int8Array::from(vec![-1])), DataType::Byte, "-1"),
        (
            Arc::new(Int16Array::from(vec![300])),
            DataType::Short,
            "300",
        ),
        (
            Arc::new(Int64Array::from(vec![1 << 40])),
            DataType::Long,
            "1099511627776",
        ),
        (
            Arc::new(Float32Array::from(vec![1.5])),
            DataType::Float,
            "1.5",
        ),
        (
            Arc::new(Float64Array::from(vec![f64::NAN])),
            DataType::Double,
            r#""NaN""#,
        ),
        (
            Arc::new(Float32Array::from(vec![f32::INFINITY])),
            DataType::Float,
            r#""Infinity""#,
        ),
        (
            Arc::new(
                Decimal64Array::from(vec![1250])
                    .with_precision_and_scale(6, 2)
                    .unwrap(),
            ),
            dec(6, 2),
            "12.5",
        ),
        (
            Arc::new(LargeStringArray::from(vec!["a\tb"])),
            DataType::String,
            r#""a\tb""#,
        ),
        (
            Arc::new(StringViewArray::from(vec!["v"])),
            DataType::String,
            r#""v""#,
        ),
        (
            Arc::new(BinaryArray::from(vec![&[1_u8, 2][..]])),
            DataType::Binary,
            r#""AQI=""#,
        ),
        (
            Arc::new(Date32Array::from(vec![19_000])),
            DataType::Date,
            r#""2022-01-08""#,
        ),
        (
            Arc::new(TimestampMicrosecondArray::from(vec![1_500_000]).with_timezone("UTC")),
            DataType::Timestamp,
            r#""1970-01-01 00:00:01.5+00:00""#,
        ),
        (
            Arc::new(TimestampMicrosecondArray::from(vec![60_000_000])),
            DataType::TimestampNtz,
            r#""1970-01-01 00:01:00""#,
        ),
    ];
    for (col, t, json) in cases {
        let v = cast_to_variant(col.as_ref(), &t).unwrap();
        assert_eq!(json_of(&v), vec![Some(json.to_owned())], "{t:?}");
        // And back to its own type and to a string.
        let back = cast_from_variant(v.as_ref(), &t, false, &utc()).unwrap();
        assert_eq!(back.len(), 1);
        cast_from_variant(v.as_ref(), &DataType::String, true, &utc()).unwrap();
    }
    // A map, a struct with a NULL field and a variant inside a variant.
    let mut m = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
    m.keys().append_value("k");
    m.values().append_null();
    m.append(true).unwrap();
    let map: ArrayRef = Arc::new(m.finish());
    let mt = DataType::Map(Arc::new(MapType::new(
        DataType::String,
        DataType::Integer,
        true,
    )));
    let v = cast_to_variant(map.as_ref(), &mt).unwrap();
    assert_eq!(json_of(&v), vec![Some(r#"{"k":null}"#.to_owned())]);
    let back = cast_from_variant(v.as_ref(), &mt, true, &utc()).unwrap();
    assert_eq!(back.len(), 1);
    let inner = column(&[Some(r#"{"x":[1]}"#), None]);
    let outer = cast_to_variant(inner.as_ref(), &DataType::Variant).unwrap();
    assert_eq!(json_of(&outer), vec![Some(r#"{"x":[1]}"#.to_owned()), None]);
    // A wrong carrier is an internal error, not a panic.
    let ints: ArrayRef = Arc::new(Int32Array::from(vec![1]));
    assert!(cast_to_variant(ints.as_ref(), &DataType::String).is_err());
    assert!(VariantColumn::new(ints.as_ref()).is_err());
}

/// Every variant scalar into the types it casts to, and the shapes it does not.
#[test]
fn every_lane_casts_as_spark() {
    let col = column(&[
        Some("true"),
        Some("7"),
        Some("1.5e0"),
        Some("2.25"),
        Some("\"3\""),
        Some("null"),
        None,
    ]);
    // Which of `true, 7, 1.5e0, 2.25, "3"` are NULL as each type
    // (`try_variant_get(parse_json(..), '$', type)` on Spark 4.2.0).
    let f = false;
    let t = true;
    for (to, want) in [
        (DataType::Long, [f, f, f, f, f]),
        (DataType::Double, [f, f, f, f, f]),
        (DataType::Float, [f, f, f, f, f]),
        (DataType::Boolean, [f, f, f, f, t]),
        (
            DataType::Decimal(DecimalType::try_new(10, 2).unwrap()),
            [f, f, f, f, f],
        ),
        (DataType::String, [f, f, f, f, f]),
        (DataType::Binary, [t, t, t, t, f]),
        (DataType::Timestamp, [t, f, f, f, t]),
        (DataType::TimestampNtz, [t, t, t, t, t]),
        (DataType::Date, [t, t, t, t, t]),
    ] {
        let out = cast_from_variant(col.as_ref(), &to, false, &utc()).unwrap();
        let got: Vec<bool> = (0..5).map(|i| out.is_null(i)).collect();
        assert_eq!(got, want, "{to:?}");
        // The variant null and the SQL NULL are NULL too.
        assert!(out.is_null(5) && out.is_null(6), "{to:?}");
    }
    // Containers of the wrong shape: NULL, or the error naming the value.
    let shapes = column(&[Some("1"), Some("[1]"), Some(r#"{"a":1}"#)]);
    let arr = DataType::Array(Arc::new(ArrayType::new(DataType::Long, true)));
    let map = DataType::Map(Arc::new(krishiv_types::MapType::new(
        DataType::String,
        DataType::Long,
        true,
    )));
    let st = DataType::Struct(StructType::new(vec![StructField::new(
        "a",
        DataType::Long,
        true,
    )]));
    for t in [&arr, &map, &st] {
        let out = cast_from_variant(shapes.as_ref(), t, false, &utc()).unwrap();
        assert_eq!(out.null_count(), 2, "{t:?}");
        let e = cast_from_variant(shapes.as_ref(), t, true, &utc()).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InvalidVariantCast, "{t:?}");
    }
    // A UUID only to a string; binary and dates print in JSON.
    let mut w = VariantWriter::new();
    w.uuid([0xab; 16]);
    let (m, v) = w.finish().unwrap();
    let r = VariantRef::new(&m, &v).unwrap();
    let s = cast_values(&[Some(r)], &DataType::String, true, &utc()).unwrap();
    assert_eq!(
        s.as_string_view().value(0),
        "abababab-abab-abab-abab-abababababab"
    );
    assert!(cast_values(&[Some(r)], &DataType::Long, true, &utc()).is_err());
    // Long seconds past the timestamp range are invalid.
    let big = column(&[Some("9223372036854775807")]);
    assert!(cast_from_variant(big.as_ref(), &DataType::Timestamp, true, &utc()).is_err());
    // A NULL column is all NULL.
    let nulls: ArrayRef = Arc::new(arrow_array::NullArray::new(2));
    let out = cast_from_variant(nulls.as_ref(), &DataType::Long, true, &utc()).unwrap();
    assert_eq!(out.null_count(), 2);
    // Paths raise on bad text.
    assert!(parse_path_or_raise("$.a", "variant_get").is_ok());
}

/// Spark 4.2.0 (`spark.sql.caseSensitive=false`): two objects of one size whose
/// sorted names match ignoring case, each pair with a tightest type, are one
/// object with the first's names (`findTypeForComplex`); otherwise fields merge.
#[test]
fn objects_matching_ignoring_case_merge_as_one() {
    for (json, want) in [
        (r#"[{"A":1},{"a":2}]"#, "ARRAY<OBJECT<A: BIGINT>>"),
        (
            r#"[{"A":1},{"a":"x"}]"#,
            "ARRAY<OBJECT<A: BIGINT, a: STRING>>",
        ),
        (
            r#"[{"A":1,"b":1},{"a":2}]"#,
            "ARRAY<OBJECT<A: BIGINT, a: BIGINT, b: BIGINT>>",
        ),
        (
            r#"[{"A":1,"b":1},{"a":2,"B":3}]"#,
            "ARRAY<OBJECT<A: BIGINT, B: BIGINT, a: BIGINT, b: BIGINT>>",
        ),
        (
            r#"[{"A":1,"b":1},{"b":2,"a":3}]"#,
            "ARRAY<OBJECT<A: BIGINT, b: BIGINT>>",
        ),
        (
            r#"[{"A":1},{"a":2.5}]"#,
            "ARRAY<OBJECT<A: BIGINT, a: DECIMAL(2,1)>>",
        ),
        (
            r#"[{"A":{"x":1}},{"a":{"X":2}}]"#,
            "ARRAY<OBJECT<A: OBJECT<x: BIGINT>>>",
        ),
        (
            r#"[{"b":1,"A":1},{"a":2,"B":2}]"#,
            "ARRAY<OBJECT<A: BIGINT, B: BIGINT, a: BIGINT, b: BIGINT>>",
        ),
    ] {
        assert_eq!(schema_of(json), want, "{json}");
    }
}
