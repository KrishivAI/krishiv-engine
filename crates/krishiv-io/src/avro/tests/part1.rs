// SPDX-License-Identifier: Apache-2.0
//! Tests, part of this module's test suite (split by size; the helpers are in the parent).

use super::*;

/// Types only `apache-avro`-written files exercise (an `enum`, a `fixed`,
/// a `bytes` decimal, a `uuid`, millis timestamps, a multi-union): our
/// reader maps them as Spark's `SchemaConverters` do.
#[test]
fn reads_apache_avro_types() {
    use apache_avro::types::Value as V;
    let env = Env::new();
    let o = env.options(&[]);
    let schema_json = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "e", "type": {"type": "enum", "name": "e", "symbols": ["A", "B"]}},
            {"name": "fx", "type": {"type": "fixed", "name": "fx", "size": 2}},
            {"name": "dec", "type": {"type": "bytes", "logicalType": "decimal", "precision": 9, "scale": 2}},
            {"name": "u", "type": {"type": "string", "logicalType": "uuid"}},
            {"name": "tm", "type": {"type": "long", "logicalType": "timestamp-millis"}},
            {"name": "lm", "type": {"type": "long", "logicalType": "local-timestamp-micros"}},
            {"name": "uni", "type": ["long", "null"]},
            {"name": "multi", "type": ["int", "string"]},
            {"name": "il", "type": ["int", "long"]},
            {"name": "fd", "type": ["null", "float", "double"]},
            {"name": "m3", "type": ["int", "string", "null"]}
        ]
    }"#;
    let uuid = "123e4567-e89b-12d3-a456-426614174000";
    let values = vec![
        V::Record(vec![
            ("e".to_owned(), V::Enum(1, "B".to_owned())),
            ("fx".to_owned(), V::Fixed(2, vec![0x01, 0x02])),
            (
                "dec".to_owned(),
                V::Decimal(apache_avro::Decimal::from(vec![0x30, 0x39])),
            ),
            ("u".to_owned(), V::Uuid(uuid.parse().unwrap())),
            ("tm".to_owned(), V::TimestampMillis(1_700_000_000_123)),
            (
                "lm".to_owned(),
                V::LocalTimestampMicros(1_700_000_000_123_456),
            ),
            ("uni".to_owned(), V::Union(0, Box::new(V::Long(7)))),
            (
                "multi".to_owned(),
                V::Union(1, Box::new(V::String("x".to_owned()))),
            ),
            ("il".to_owned(), V::Union(0, Box::new(V::Int(5)))),
            ("fd".to_owned(), V::Union(1, Box::new(V::Float(1.5)))),
            (
                "m3".to_owned(),
                V::Union(1, Box::new(V::String("s".to_owned()))),
            ),
        ]),
        V::Record(vec![
            ("e".to_owned(), V::Enum(0, "A".to_owned())),
            ("fx".to_owned(), V::Fixed(2, vec![0xFF, 0xFE])),
            (
                "dec".to_owned(),
                V::Decimal(apache_avro::Decimal::from(vec![0xFF])),
            ),
            ("u".to_owned(), V::Uuid(uuid.parse().unwrap())),
            ("tm".to_owned(), V::TimestampMillis(-1_000)),
            ("lm".to_owned(), V::LocalTimestampMicros(-1_000_000)),
            ("uni".to_owned(), V::Union(1, Box::new(V::Null))),
            ("multi".to_owned(), V::Union(0, Box::new(V::Int(-3)))),
            ("il".to_owned(), V::Union(1, Box::new(V::Long(7)))),
            ("fd".to_owned(), V::Union(2, Box::new(V::Double(2.5)))),
            ("m3".to_owned(), V::Union(2, Box::new(V::Null))),
        ]),
    ];
    let file = env.put("interop.avro", apache_write(schema_json, &values));
    let meta = env.metadata(&file, &o);
    let schema = meta.schema();
    let kinds: Vec<(String, DataType)> = schema
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone()))
        .collect();
    assert_eq!(
        kinds,
        vec![
            ("e".to_owned(), DataType::Utf8View),
            ("fx".to_owned(), DataType::BinaryView),
            ("dec".to_owned(), DataType::Decimal128(9, 2)),
            ("u".to_owned(), DataType::Utf8View),
            (
                "tm".to_owned(),
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
            ),
            (
                "lm".to_owned(),
                DataType::Timestamp(TimeUnit::Microsecond, None)
            ),
            ("uni".to_owned(), DataType::Int64),
            (
                "multi".to_owned(),
                DataType::Struct(
                    vec![
                        Field::new("member0", DataType::Int32, true),
                        Field::new("member1", DataType::Utf8View, true),
                    ]
                    .into(),
                )
            ),
            // Spark's union mapping: `int` and `long` are a `long`, `float`
            // and `double` a `double`, and a null branch is dropped.
            ("il".to_owned(), DataType::Int64),
            ("fd".to_owned(), DataType::Float64),
            (
                "m3".to_owned(),
                DataType::Struct(
                    vec![
                        Field::new("member0", DataType::Int32, true),
                        Field::new("member1", DataType::Utf8View, true),
                    ]
                    .into(),
                )
            ),
        ]
    );
    let table_req = ScanRequest::all(Arc::clone(&schema));
    let out_schema = mapped_schema(&schema, &table_req);
    let got = env.scan(&file, &table_req, &o).unwrap();
    let got = concat(&out_schema, &got);
    assert_eq!(rows(std::slice::from_ref(&got)), 2);
    let col = |name: &str| Arc::clone(got.column_by_name(name).unwrap());
    assert_eq!(
        col("e")
            .as_string_view()
            .iter()
            .flatten()
            .collect::<Vec<_>>(),
        vec!["B", "A"]
    );
    assert_eq!(
        col("fx")
            .as_binary_view()
            .iter()
            .flatten()
            .collect::<Vec<_>>(),
        vec![[0x01, 0x02].as_slice(), [0xFF, 0xFE].as_slice()]
    );
    assert_eq!(
        col("dec")
            .as_primitive::<Decimal128Type>()
            .iter()
            .flatten()
            .collect::<Vec<_>>(),
        vec![12345, -1]
    );
    assert_eq!(
        col("u")
            .as_string_view()
            .iter()
            .flatten()
            .collect::<Vec<_>>(),
        vec![uuid, uuid]
    );
    // `timestamp-millis` scales to microseconds, as Spark reads them.
    assert_eq!(
        col("tm")
            .as_primitive::<TimestampMicrosecondType>()
            .values()
            .to_vec(),
        vec![1_700_000_000_123_000, -1_000_000]
    );
    assert_eq!(
        col("lm")
            .as_primitive::<TimestampMicrosecondType>()
            .values()
            .to_vec(),
        vec![1_700_000_000_123_456, -1_000_000]
    );
    assert_eq!(
        col("uni")
            .as_primitive::<Int64Type>()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some(7), None]
    );
    let multi = col("multi").as_struct().clone();
    assert_eq!(
        multi
            .column(0)
            .as_primitive::<Int32Type>()
            .iter()
            .collect::<Vec<_>>(),
        vec![None, Some(-3)]
    );
    assert_eq!(
        multi.column(1).as_string_view().iter().collect::<Vec<_>>(),
        vec![Some("x"), None]
    );
    assert_eq!(
        col("il")
            .as_primitive::<Int64Type>()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some(5), Some(7)]
    );
    assert_eq!(
        col("fd")
            .as_primitive::<arrow_array::types::Float64Type>()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some(1.5), Some(2.5)]
    );
    let m3 = col("m3").as_struct().clone();
    assert_eq!((m3.is_valid(0), m3.is_null(1)), (true, true));
    assert_eq!(
        m3.column(1).as_string_view().iter().collect::<Vec<_>>(),
        vec![Some("s"), None]
    );
    // Files we write read back through `apache-avro` itself.
    let ours = env.encode(
        &meta.schema(),
        &env.scan(&file, &table_req, &o)
            .unwrap()
            .into_iter()
            .map(|b| {
                // Drop the mapped nullability: re-type to the file schema.
                retype_batch(&b, &meta.schema())
            })
            .collect::<Vec<_>>(),
        &o,
    );
    let back = apache_read(&ours);
    assert_eq!(back.len(), 2);
}

/// Every option parses, validates and takes effect.
#[test]
fn options_are_validated() {
    use crate::options::OptionWarning;
    let env = Env::new();
    // Unknown values fail with `KRISHIV_INVALID_FORMAT_OPTION_VALUE`.
    for (key, value) in [
        ("mode", "sometimes"),
        ("compression", "lz4"),
        ("datetimeRebaseMode", "sometimes"),
    ] {
        let user = [(key.to_owned(), option_value(key, value))]
            .into_iter()
            .collect();
        let e = AvroFormat
            .options()
            .parse(&user, &krishiv_common::config::Config::new());
        let e = e.unwrap_err();
        assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::KrishivInvalidFormatOptionValue,
            "{key}={value}"
        );
    }
    // An empty record name fails when the options are used.
    let o = env.options(&[("recordName", "")]);
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
    let e = AvroEncoder::new(&schema, &o).unwrap_err();
    assert_eq!(class_of(&e), ErrorClass::KrishivInvalidFormatOptionValue);
    // A bad `avroSchema` fails when it is used.
    for avro_schema in ["{", r#"{"type":"int"}"#] {
        let o = env.options(&[("avroSchema", avro_schema)]);
        let e = AvroEncoder::new(&schema, &o).unwrap_err();
        assert_eq!(
            class_of(&e),
            ErrorClass::KrishivInvalidFormatOptionValue,
            "{avro_schema}"
        );
    }
    // `recordName` and `recordNamespace` name the written record.
    let o = env.options(&[("recordName", "events"), ("recordNamespace", "example")]);
    let bytes = env.encode(&schema, &[], &o);
    // The header carries the schema JSON (the tail is the raw sync marker).
    for needle in [
        br#""name":"events""#.as_slice(),
        br#""namespace":"example""#.as_slice(),
    ] {
        assert!(
            bytes.windows(needle.len()).any(|w| w == needle),
            "no {} in the header",
            String::from_utf8_lossy(needle)
        );
    }
    // `ignoreExtension=false` is refused: extensions are always ignored.
    let o = env.options(&[("ignoreExtension", "false")]);
    let e = AvroEncoder::new(&schema, &o).unwrap_err();
    assert_eq!(class_of(&e), ErrorClass::KrishivUnsupportedFeature);
    // Every other option is accepted (here: parsed without a warning).
    for kv in [
        ("ignoreExtension", "true"),
        ("positionalFieldMatching", "true"),
        ("datetimeRebaseMode", "LEGACY"),
        ("mode", "DROPMALFORMED"),
        ("compression", "uncompressed"),
    ] {
        let user = [(kv.0.to_owned(), option_value(kv.0, kv.1))]
            .into_iter()
            .collect();
        let (o, warnings) = AvroFormat
            .options()
            .parse(&user, &krishiv_common::config::Config::new())
            .unwrap();
        assert!(warnings.is_empty());
        assert_eq!(o.ignored(), &[] as &[String]);
    }
    // An unknown option warns with a suggestion and is ignored (built from
    // parts: the typo gate rejects the literal misspelling).
    let typo = ["compress", "io"].concat();
    let user = [(typo.clone(), option_value(&typo, "snappy"))]
        .into_iter()
        .collect();
    let (o, warnings) = AvroFormat
        .options()
        .parse(&user, &krishiv_common::config::Config::new())
        .unwrap();
    assert_eq!(
        warnings,
        vec![OptionWarning::Unknown {
            key: typo.clone(),
            suggestion: Some("compression"),
        }]
    );
    assert_eq!(o.ignored(), &[typo]);
}

/// Each codec round-trips; the header names it the way Spark does.
/// Decimals of either carrier are written and read back on that carrier.
#[test]
fn decimals_round_trip_on_both_carriers() {
    let env = Env::new();
    let narrow: arrow_array::ArrayRef = Arc::new(
        arrow_array::Decimal64Array::from(vec![Some(150), None, Some(-275)])
            .with_precision_and_scale(10, 2)
            .unwrap(),
    );
    let wide: arrow_array::ArrayRef = Arc::new(
        arrow_array::Decimal128Array::from(vec![Some(150), None, Some(-275)])
            .with_precision_and_scale(20, 2)
            .unwrap(),
    );
    for (name, column) in [("narrow", narrow), ("wide", wide)] {
        let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
            "d",
            column.data_type().clone(),
            true,
        )]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![column]).unwrap();
        let o = env.options(&[]);
        let bytes = env.encode(&schema, std::slice::from_ref(&batch), &o);
        let file = env.put(&format!("decimal-{name}.avro"), bytes);
        let req = ScanRequest::all(Arc::clone(&schema));
        let got = env.scan(&file, &req, &o).unwrap();
        assert_eq!(concat(&schema, &got), batch, "{name}");
    }
}

#[test]
fn compression_codecs_round_trip() {
    let env = Env::new();
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("v", DataType::Utf8View, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringViewArray::from(vec![
            Some("x".to_owned());
            500
        ]))],
    )
    .unwrap();
    for (option, header) in [
        ("uncompressed", "null"),
        ("null", "null"),
        ("snappy", "snappy"),
        ("deflate", "deflate"),
        ("bzip2", "bzip2"),
    ] {
        let o = env.options(&[("compression", option)]);
        let bytes = env.encode(&schema, std::slice::from_ref(&batch), &o);
        assert!(
            bytes.windows(header.len()).any(|w| w == header.as_bytes()),
            "{option}: no {header} in the header"
        );
        let file = env.put(&format!("codec-{option}.avro"), bytes);
        let table_req = ScanRequest::all(Arc::clone(&schema));
        let out_schema = mapped_schema(&schema, &table_req);
        let got = env.scan(&file, &table_req, &o).unwrap();
        assert_eq!(concat(&out_schema, &got), retype_batch(&batch, &out_schema));
    }
    // `xz` and `zstandard` are accepted options but not enabled in
    // this build: `zstandard` is Spark's `CODEC_NOT_AVAILABLE` error.
    let refused = [
        ("xz", ErrorClass::KrishivUnsupportedFeature),
        (
            "zstandard",
            ErrorClass::CodecNotAvailableWithAvailableCodecsSuggestion,
        ),
    ];
    // `zstd` spells `zstandard`.
    let o = env.options(&[("compression", "zstd")]);
    let e = AvroEncoder::new(&schema, &o).unwrap_err();
    assert_eq!(
        class_of(&e),
        ErrorClass::CodecNotAvailableWithAvailableCodecsSuggestion
    );
    for (codec, class) in refused {
        let o = env.options(&[("compression", codec)]);
        let e = AvroEncoder::new(&schema, &o).unwrap_err();
        assert_eq!(class_of(&e), class, "{codec}");
        // A file headed with such a codec fails the metadata read the same way.
        let plain = env.options(&[("compression", "uncompressed")]);
        let mut bytes = env.encode(&schema, std::slice::from_ref(&batch), &plain);
        let needle = b"avro.codec";
        let at = bytes
            .windows(needle.len())
            .position(|w| w == needle)
            .unwrap();
        // The header holds the value length-delimited: `[len][content]`.
        let value_at = at + needle.len();
        assert_eq!(&bytes[value_at + 1..value_at + 5], b"null");
        let zigzag = |n: i64| {
            u8::try_from(
                n.unsigned_abs()
                    .wrapping_mul(2)
                    .wrapping_sub(u64::from(n.is_negative())),
            )
            .unwrap()
        };
        let replacement: Vec<u8> = [zigzag(codec.len() as i64)]
            .into_iter()
            .chain(codec.bytes())
            .collect();
        bytes.splice(value_at..value_at + 5, replacement);
        let file = env.put(&format!("codec-{codec}.avro"), bytes);
        let e = env
            .rt
            .block_on(
                env.io
                    .read_metadata(&AvroFormat, &file, MetadataNeeds::default(), &o),
            )
            .unwrap_err();
        assert_eq!(e.error_class(), class, "{codec}");
    }
}

/// The Spark-documented Avro → Arrow mapping, as `(writer fragment, Arrow
/// type or the expected error class)`.
#[test]
fn schema_mapping_matches_spark_docs() {
    let cases: &[(&str, Result<DataType, ErrorClass>)] = &[
        (r#""boolean""#, Ok(DataType::Boolean)),
        (r#""int""#, Ok(DataType::Int32)),
        (r#""long""#, Ok(DataType::Int64)),
        (r#""float""#, Ok(DataType::Float32)),
        (r#""double""#, Ok(DataType::Float64)),
        (r#""bytes""#, Ok(DataType::BinaryView)),
        (r#""string""#, Ok(DataType::Utf8View)),
        (r#""null""#, Ok(DataType::Null)),
        (
            r#"{"type":"record","name":"r","fields":[]}"#,
            Ok(DataType::Struct(Fields::empty())),
        ),
        (
            r#"{"type":"enum","name":"e","symbols":["A","B"]}"#,
            Ok(DataType::Utf8View),
        ),
        (
            r#"{"type":"fixed","name":"f","size":4}"#,
            Ok(DataType::BinaryView),
        ),
        (
            r#"{"type":"array","items":"long"}"#,
            Ok(DataType::List(Arc::new(Field::new(
                "element",
                DataType::Int64,
                false,
            )))),
        ),
        (
            r#"{"type":"map","values":"string"}"#,
            Ok(DataType::Map(
                Arc::new(Field::new(
                    "entries",
                    DataType::Struct(
                        vec![
                            Field::new("key", DataType::Utf8View, false),
                            Field::new("value", DataType::Utf8View, false),
                        ]
                        .into(),
                    ),
                    false,
                )),
                false,
            )),
        ),
        (
            r#"{"type":"fixed","name":"d","size":4,"logicalType":"decimal","precision":9,"scale":2}"#,
            Ok(DataType::Decimal128(9, 2)),
        ),
        (
            r#"{"type":"bytes","logicalType":"decimal","precision":10,"scale":2}"#,
            Ok(DataType::Decimal128(10, 2)),
        ),
        (
            r#"{"type":"int","logicalType":"date"}"#,
            Ok(DataType::Date32),
        ),
        (
            r#"{"type":"int","logicalType":"time-millis"}"#,
            Ok(DataType::Int32),
        ),
        (
            r#"{"type":"long","logicalType":"time-micros"}"#,
            Ok(DataType::Int64),
        ),
        (
            r#"{"type":"long","logicalType":"timestamp-millis"}"#,
            Ok(DataType::Timestamp(
                TimeUnit::Microsecond,
                Some("UTC".into()),
            )),
        ),
        (
            r#"{"type":"long","logicalType":"timestamp-micros"}"#,
            Ok(DataType::Timestamp(
                TimeUnit::Microsecond,
                Some("UTC".into()),
            )),
        ),
        (
            r#"{"type":"long","logicalType":"local-timestamp-millis"}"#,
            Ok(DataType::Timestamp(TimeUnit::Microsecond, None)),
        ),
        (
            r#"{"type":"long","logicalType":"local-timestamp-micros"}"#,
            Ok(DataType::Timestamp(TimeUnit::Microsecond, None)),
        ),
        (
            r#"{"type":"string","logicalType":"uuid"}"#,
            Ok(DataType::Utf8View),
        ),
        (
            r#"{"type":"fixed","name":"u","size":16,"logicalType":"uuid"}"#,
            Ok(DataType::BinaryView),
        ),
        (r#"["null","int"]"#, Ok(DataType::Int32)),
        (r#"["string","null"]"#, Ok(DataType::Utf8View)),
        // Unions follow Spark's `SchemaConverters`: nulls are dropped (the
        // union is nullable), then one branch is that branch, `int` and
        // `long` are a `long`, `float` and `double` a `double`, and anything
        // else a struct of `member0..N` over the non-null branches.
        (
            r#"["int","string","null"]"#,
            Ok(DataType::Struct(
                vec![
                    Field::new("member0", DataType::Int32, true),
                    Field::new("member1", DataType::Utf8View, true),
                ]
                .into(),
            )),
        ),
        (
            r#"["int","string"]"#,
            Ok(DataType::Struct(
                vec![
                    Field::new("member0", DataType::Int32, true),
                    Field::new("member1", DataType::Utf8View, true),
                ]
                .into(),
            )),
        ),
        (r#"["int","long"]"#, Ok(DataType::Int64)),
        (r#"["long","null","int"]"#, Ok(DataType::Int64)),
        (r#"["float","double"]"#, Ok(DataType::Float64)),
        (r#"["null","double","float"]"#, Ok(DataType::Float64)),
        (r#"["int"]"#, Ok(DataType::Int32)),
        (r#"["string"]"#, Ok(DataType::Utf8View)),
        (
            r#"{"type":"long","logicalType":"timestamp-nanos"}"#,
            Err(ErrorClass::KrishivUnsupportedFeature),
        ),
        (
            r#"{"type":"long","logicalType":"local-timestamp-nanos"}"#,
            Err(ErrorClass::KrishivUnsupportedFeature),
        ),
        (
            r#"{"type":"fixed","name":"d","size":12,"logicalType":"duration"}"#,
            Err(ErrorClass::KrishivUnsupportedFeature),
        ),
        (
            r#""not-a-type""#,
            Err(ErrorClass::FailedReadFileCannotReadFileFooter),
        ),
        (
            r#"{"type":"fixed","name":"d","size":40,"logicalType":"decimal","precision":80,"scale":0}"#,
            Err(ErrorClass::KrishivUnsupportedFeature),
        ),
    ];
    for (fragment, want) in cases {
        let json = format!(
            r#"{{"type":"record","name":"r","fields":[{{"name":"c","type":{fragment}}}]}}"#
        );
        let got: Result<(DataType, bool), ErrorClass> = (|| {
            let schema = AvroSchema::parse_str(&json)
                .map_err(|_| ErrorClass::FailedReadFileCannotReadFileFooter)?;
            let fields = fields_of_record(&schema).map_err(|e| class_of(&e))?;
            let field = fields.into_iter().next().ok_or(ErrorClass::InternalError)?;
            Ok((field.data_type().clone(), field.is_nullable()))
        })();
        let want: Result<(DataType, bool), ErrorClass> = want.clone().map(|dt| {
            let nullable = fragment.starts_with('[') && fragment.contains(r#""null""#);
            (dt, nullable)
        });
        assert_eq!(got, want, "{fragment}");
    }
}

/// Every mapped Arrow type, nullable with nulls: an encoded file scans back
/// to the rows written.
#[test]
fn round_trip_all_types() {
    let env = Env::new();
    let o = env.options(&[]);
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("b", DataType::Boolean, true),
        Field::new("i", DataType::Int32, true),
        Field::new("l", DataType::Int64, true),
        Field::new("f", DataType::Float32, true),
        Field::new("d", DataType::Float64, true),
        Field::new("s", DataType::Utf8View, true),
        Field::new("bin", DataType::BinaryView, true),
        Field::new("dec", DataType::Decimal128(10, 2), true),
        Field::new("date", DataType::Date32, true),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
        Field::new(
            "ntz",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "st",
            DataType::Struct(
                vec![
                    Field::new("x", DataType::Int32, true),
                    Field::new("y", DataType::Utf8View, true),
                ]
                .into(),
            ),
            true,
        ),
        Field::new(
            "arr",
            DataType::List(Arc::new(Field::new("element", DataType::Int64, false))),
            true,
        ),
        Field::new(
            "map",
            DataType::Map(
                Arc::new(Field::new(
                    "entries",
                    DataType::Struct(
                        vec![
                            Field::new("key", DataType::Utf8View, false),
                            Field::new("value", DataType::Int32, false),
                        ]
                        .into(),
                    ),
                    false,
                )),
                false,
            ),
            true,
        ),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)])),
            Arc::new(Int32Array::from(vec![Some(1), None, Some(-3)])),
            Arc::new(Int64Array::from(vec![None, Some(9_000_000_000), Some(-1)])),
            Arc::new(Float32Array::from(vec![Some(1.5), Some(f32::NAN), None])),
            Arc::new(Float64Array::from(vec![None, Some(-0.0), Some(2.25)])),
            Arc::new(StringViewArray::from(vec![Some("a"), None, Some("héllo")])),
            Arc::new(BinaryViewArray::from(vec![
                Some(b"\x00\xff".as_slice()),
                None,
                Some(b"".as_slice()),
            ])),
            Arc::new(
                Decimal128Array::from(vec![Some(12345), None, Some(-1)])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
            Arc::new(Date32Array::from(vec![Some(0), None, Some(20000)])),
            Arc::new(
                TimestampMicrosecondArray::from(vec![Some(1_700_000_000_000_000), None, Some(0)])
                    .with_timezone("UTC"),
            ),
            Arc::new(TimestampMicrosecondArray::from(vec![
                None,
                Some(1_700_000_000_000_000),
                Some(-1),
            ])),
            Arc::new(StructArray::new(
                Fields::from(vec![
                    Field::new("x", DataType::Int32, true),
                    Field::new("y", DataType::Utf8View, true),
                ]),
                vec![
                    Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])) as ArrayRef,
                    Arc::new(StringViewArray::from(vec![None, Some("q"), None])) as ArrayRef,
                ],
                Some(vec![true, false, true].into()),
            )),
            Arc::new(list_array(&[
                Some(vec![Some(1), Some(2)]),
                None,
                Some(vec![]),
            ])),
            Arc::new(map_array(&[
                Some(vec![("k1".to_owned(), Some(1)), ("k2".to_owned(), Some(2))]),
                None,
                Some(vec![]),
            ])),
        ],
    )
    .unwrap();
    let file = env.put(
        "rt.avro",
        env.encode(&schema, std::slice::from_ref(&batch), &o),
    );
    let meta = env.metadata(&file, &o);
    assert_eq!(meta.schema(), schema);
    let table_req = ScanRequest::all(Arc::clone(&schema));
    let out_schema = mapped_schema(&meta.schema(), &table_req);
    let got = env.scan(&file, &table_req, &o).unwrap();
    assert_eq!(concat(&out_schema, &got), retype_batch(&batch, &out_schema));
    assert_eq!(rows(&got), 3);
    // Empty batches: a header-only file scans to no rows.
    let empty = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(BooleanArray::from(Vec::<Option<bool>>::new())),
            Arc::new(Int32Array::from(Vec::<Option<i32>>::new())),
            Arc::new(Int64Array::from(Vec::<Option<i64>>::new())),
            Arc::new(Float32Array::from(Vec::<Option<f32>>::new())),
            Arc::new(Float64Array::from(Vec::<Option<f64>>::new())),
            Arc::new(StringViewArray::from(Vec::<Option<&str>>::new())),
            Arc::new(BinaryViewArray::from(Vec::<Option<&[u8]>>::new())),
            Arc::new(
                Decimal128Array::from(Vec::<Option<i128>>::new())
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
            Arc::new(Date32Array::from(Vec::<Option<i32>>::new())),
            Arc::new(
                TimestampMicrosecondArray::from(Vec::<Option<i64>>::new()).with_timezone("UTC"),
            ),
            Arc::new(TimestampMicrosecondArray::from(Vec::<Option<i64>>::new())),
            Arc::new(StructArray::new(
                Fields::from(vec![
                    Field::new("x", DataType::Int32, true),
                    Field::new("y", DataType::Utf8View, true),
                ]),
                vec![
                    Arc::new(Int32Array::from(Vec::<Option<i32>>::new())) as ArrayRef,
                    Arc::new(StringViewArray::from(Vec::<Option<&str>>::new())) as ArrayRef,
                ],
                None,
            )),
            Arc::new(list_array(&[])),
            Arc::new(map_array(&[])),
        ],
    )
    .unwrap();
    let file = env.put("empty.avro", env.encode(&schema, &[empty], &o));
    let got = env
        .scan(&file, &ScanRequest::all(Arc::clone(&schema)), &o)
        .unwrap();
    assert_eq!(rows(&got), 0);
}

/// One block per batch: splits at sync markers (even mid-block) read every
/// row exactly once, with file-relative row indexes.
#[test]
fn splits_at_sync_markers() {
    let env = Env::new();
    let o = env.options(&[]);
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, true)]));
    // Eight blocks of 500 rows: `flush` ends the block early.
    let mut enc = AvroEncoder::new(&schema, &o).unwrap();
    let mut out = Vec::new();
    for block in 0..8_u64 {
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(
                (0..500)
                    .map(|i| Some((block * 500 + i) as i64))
                    .collect::<Vec<_>>(),
            ))],
        )
        .unwrap();
        enc.write(&batch).unwrap();
        enc.flush().unwrap();
        while let Some(bytes) = enc.take_bytes() {
            out.extend_from_slice(&bytes);
        }
    }
    let done = Box::new(enc).finish().unwrap();
    out.extend_from_slice(&done.tail);
    let file = env.put("split.avro", out);
    let meta = env.metadata(&file, &o);
    assert_eq!(meta.statistics().rows.exact(), Some(&4000));
    // The whole file is eight blocks.
    let whole = ByteRange::whole(file.size);
    let plan = AvroFormat
        .plan_units(
            meta.as_ref(),
            &whole,
            &ScanRequest::all(Arc::clone(&schema)),
        )
        .unwrap();
    assert_eq!(plan.units.len(), 8, "one block per flush");
    // Halves and mid-block cuts: every row once, row indexes 0..4000.
    let mut cuts = vec![0, file.size / 7, file.size / 2, file.size - 20, file.size];
    cuts.sort_unstable();
    cuts.dedup();
    let mut seen = Vec::new();
    for w in cuts.windows(2) {
        let split = ByteRange::new(w[0], w[1]);
        let mut req = ScanRequest::all(Arc::clone(&schema));
        req.metadata_cols.row_index = true;
        let got = env.scan_split(&file, &req, &o, split).unwrap();
        for b in &got {
            let index = b.column_by_name("_tmp_metadata_row_index").unwrap();
            seen.extend(index.as_primitive::<Int64Type>().values().iter().copied());
            // The values travel with their row indexes.
            let values = b.column(0).as_primitive::<Int64Type>();
            for (k, &row) in index
                .as_primitive::<Int64Type>()
                .values()
                .iter()
                .enumerate()
            {
                assert_eq!(values.value(k), row, "row {row} holds another value");
            }
        }
    }
    seen.sort_unstable();
    assert_eq!(seen, (0..4000).collect::<Vec<_>>());
    // A split owning no block reads nothing.
    let plan = AvroFormat
        .plan_units(
            meta.as_ref(),
            &ByteRange::new(file.size, file.size),
            &ScanRequest::all(Arc::clone(&schema)),
        )
        .unwrap();
    assert!(plan.units.is_empty());
}

/// Corrupt files are typed errors, never panics: truncated, garbage and
/// flipped bits fail the footer read or the scan, per `mode`.
#[test]
fn corrupt_files_are_typed_errors() {
    let env = Env::new();
    let o = env.options(&[]);
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Utf8View, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(3)])),
            Arc::new(StringViewArray::from(vec![
                Some("x"),
                Some("yy"),
                Some("zzz"),
            ])),
        ],
    )
    .unwrap();
    let bytes = env.encode(&schema, &[batch.clone(), batch.clone(), batch.clone()], &o);
    assert!(bytes.len() > 64);
    // Truncations and garbage fail the footer read.
    let mut footer_cases = vec![
        ("empty".to_owned(), Vec::new()),
        ("garbage".to_owned(), b"not an avro file at all".to_vec()),
        (
            "bad magic".to_owned(),
            [b"Xbj\x01".as_slice(), &bytes[4..]].concat(),
        ),
    ];
    for cut in [1, 4, 17, bytes.len() / 2, bytes.len() - 20, bytes.len() - 1] {
        footer_cases.push((format!("truncated at {cut}"), bytes[..cut].to_vec()));
    }
    for (name, bad) in &footer_cases {
        let file = env.put(&format!("bad-{name}"), bad.clone());
        let e = env
            .rt
            .block_on(
                env.io
                    .read_metadata(&AvroFormat, &file, MetadataNeeds::default(), &o),
            )
            .unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::FailedReadFileCannotReadFileFooter,
            "{name}"
        );
    }
    // Record-level corruption: two blocks, the second block's data
    // overwritten, its sync marker intact. FAILFAST fails the scan,
    // PERMISSIVE returns the first block plus one null row, DROPMALFORMED
    // the first block. Uncompressed, so the damage hits record bytes and
    // not the codec.
    let plain = env.options(&[("compression", "uncompressed")]);
    let mut enc = AvroEncoder::new(&schema, &plain).unwrap();
    for _ in 0..2 {
        enc.write(&batch).unwrap();
        enc.flush().unwrap();
    }
    let mut two = Vec::new();
    while let Some(bytes) = enc.take_bytes() {
        two.extend_from_slice(&bytes);
    }
    let done = Box::new(enc).finish().unwrap();
    two.extend_from_slice(&done.tail);
    let meta = env
        .rt
        .block_on(env.io.read_metadata(
            &AvroFormat,
            &env.put("two.avro", two.clone()),
            MetadataNeeds::default(),
            &plain,
        ))
        .unwrap();
    let any: &dyn std::any::Any = meta.as_ref();
    let meta = any.downcast_ref::<super::super::AvroMetadata>().unwrap();
    assert_eq!(meta.blocks.len(), 2);
    let data = meta.blocks[1].data.clone();
    let (start, end) = (
        usize::try_from(data.start).unwrap(),
        usize::try_from(data.end).unwrap(),
    );
    let mut bad = two.clone();
    bad[start..end].fill(0xFF);
    let file = env.put("flip-data.avro", bad);
    let table_req = ScanRequest::all(Arc::clone(&schema));
    let out_schema = mapped_schema(&schema, &table_req);
    let failfast = env.options(&[("mode", "FAILFAST"), ("compression", "uncompressed")]);
    let e = env.scan(&file, &table_req, &failfast).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
    let permissive = env.options(&[("mode", "PERMISSIVE"), ("compression", "uncompressed")]);
    let got = env.scan(&file, &table_req, &permissive).unwrap();
    let got = concat(&out_schema, &got);
    assert_eq!(rows(std::slice::from_ref(&got)), 4);
    assert!(got.column(0).is_null(3));
    assert!(got.column(1).is_null(3));
    let drop = env.options(&[("mode", "DROPMALFORMED"), ("compression", "uncompressed")]);
    let got = env.scan(&file, &table_req, &drop).unwrap();
    let got = concat(&out_schema, &got);
    assert_eq!(rows(std::slice::from_ref(&got)), 3);
    // The null row takes the damaged record's file row (3), when selected;
    // a damaged record outside the selection adds nothing.
    let mut indexed = ScanRequest::all(Arc::clone(&schema));
    indexed.metadata_cols.row_index = true;
    let indexed_schema = mapped_schema(&schema, &indexed);
    let got = concat(
        &indexed_schema,
        &env.scan(&file, &indexed, &permissive).unwrap(),
    );
    assert_eq!(
        got.column_by_name("_tmp_metadata_row_index")
            .unwrap()
            .as_primitive::<Int64Type>()
            .values()
            .to_vec(),
        vec![0, 1, 2, 3]
    );
    indexed.selection = Some(crate::request::RowSelection::excluding(6, [3]));
    let got = env.scan(&file, &indexed, &permissive).unwrap();
    assert_eq!(rows(&got), 3);
    // A damaged sync marker fails the block scan in every mode: framing is
    // not a record, and the metadata read validates it.
    let sync_at = usize::try_from(meta.blocks[1].bytes.end).unwrap() - 1;
    for mode in ["FAILFAST", "PERMISSIVE", "DROPMALFORMED"] {
        let mut bad = two.clone();
        bad[sync_at] ^= 0xFF;
        let file = env.put(&format!("flip-sync-{mode}.avro"), bad);
        let o = env.options(&[("mode", mode), ("compression", "uncompressed")]);
        let e = env
            .rt
            .block_on(
                env.io
                    .read_metadata(&AvroFormat, &file, MetadataNeeds::default(), &o),
            )
            .unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::FailedReadFileCannotReadFileFooter,
            "{mode}"
        );
    }
}

/// Schema evolution through `avroSchema`: promoted, added (with a
/// default), dropped and reordered fields; by name, or by position with
/// `positionalFieldMatching`.
#[test]
fn schema_evolution_via_avro_schema() {
    use apache_avro::types::Value as V;
    let env = Env::new();
    let file_schema = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "a", "type": "int"},
            {"name": "b", "type": ["null", "string"]},
            {"name": "gone", "type": "long"}
        ]
    }"#;
    let values = vec![
        V::Record(vec![
            ("a".to_owned(), V::Int(1)),
            (
                "b".to_owned(),
                V::Union(1, Box::new(V::String("x".to_owned()))),
            ),
            ("gone".to_owned(), V::Long(10)),
        ]),
        V::Record(vec![
            ("a".to_owned(), V::Int(-5)),
            ("b".to_owned(), V::Union(0, Box::new(V::Null))),
            ("gone".to_owned(), V::Long(20)),
        ]),
    ];
    let file = env.put("evo.avro", apache_write(file_schema, &values));
    // Promote `a` to long, drop `gone`, add `c` with a default.
    let reader = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "a", "type": "long"},
            {"name": "b", "type": ["null", "string"]},
            {"name": "c", "type": ["null", "string"], "default": null}
        ]
    }"#;
    let o = env.options(&[("avroSchema", reader)]);
    let inferred = env.infer(&file, &o);
    assert_eq!(
        inferred
            .fields()
            .iter()
            .map(|f| (f.name().clone(), f.data_type().clone()))
            .collect::<Vec<_>>(),
        vec![
            ("a".to_owned(), DataType::Int64),
            ("b".to_owned(), DataType::Utf8View),
            ("c".to_owned(), DataType::Utf8View),
        ]
    );
    let table_req = ScanRequest::all(Arc::clone(&inferred));
    let out_schema = mapped_schema(&inferred, &table_req);
    let got = env.scan(&file, &table_req, &o).unwrap();
    let got = concat(&out_schema, &got);
    assert_eq!(
        got.column(0).as_primitive::<Int64Type>().values().to_vec(),
        vec![1, -5]
    );
    assert_eq!(
        got.column(1).as_string_view().iter().collect::<Vec<_>>(),
        vec![Some("x"), None]
    );
    assert_eq!(got.column(2).null_count(), 2);
    // An explicit non-null default fills missing fields.
    let reader = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "a", "type": "int"},
            {"name": "d", "type": "string", "default": "hi"}
        ]
    }"#;
    let o = env.options(&[("avroSchema", reader)]);
    let inferred = env.infer(&file, &o);
    let table_req = ScanRequest::all(Arc::clone(&inferred));
    let out_schema = mapped_schema(&inferred, &table_req);
    let got = env.scan(&file, &table_req, &o).unwrap();
    let got = concat(&out_schema, &got);
    assert_eq!(
        got.column(1).as_string_view().iter().collect::<Vec<_>>(),
        vec![Some("hi"), Some("hi")]
    );
    // A missing field without a default fails the scan.
    let reader = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "a", "type": "int"},
            {"name": "d", "type": "string"}
        ]
    }"#;
    let o = env.options(&[("avroSchema", reader)]);
    let table_req = ScanRequest::all(env.infer(&file, &o));
    let e = env.scan(&file, &table_req, &o).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
    // An unresolvable type fails the scan.
    let reader = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "a", "type": "string"}
        ]
    }"#;
    let o = env.options(&[("avroSchema", reader)]);
    let table_req = ScanRequest::all(env.infer(&file, &o));
    let e = env.scan(&file, &table_req, &o).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
    // Positional matching resolves by order, not by name.
    let positional_schema = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "x", "type": "int"},
            {"name": "y", "type": "int"}
        ]
    }"#;
    let swapped_values = vec![V::Record(vec![
        ("x".to_owned(), V::Int(1)),
        ("y".to_owned(), V::Int(2)),
    ])];
    let swapped_file = env.put(
        "positional.avro",
        apache_write(positional_schema, &swapped_values),
    );
    let reader = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "second", "type": ["null", "int"]},
            {"name": "first", "type": ["null", "int"]}
        ]
    }"#;
    let named = env.options(&[("avroSchema", reader)]);
    let inferred = env.infer(&swapped_file, &named);
    let table_req = ScanRequest::all(Arc::clone(&inferred));
    let out_schema = mapped_schema(&inferred, &table_req);
    let got = env.scan(&swapped_file, &table_req, &named).unwrap();
    let got = concat(&out_schema, &got);
    // By name there is no `second` or `first` field: both read their
    // defaults (null here).
    assert_eq!(got.column(0).null_count(), 1);
    assert_eq!(got.column(1).null_count(), 1);
    let positional = env.options(&[("avroSchema", reader), ("positionalFieldMatching", "true")]);
    let inferred = env.infer(&swapped_file, &positional);
    let table_req = ScanRequest::all(Arc::clone(&inferred));
    let out_schema = mapped_schema(&inferred, &table_req);
    let got = env.scan(&swapped_file, &table_req, &positional).unwrap();
    let got = concat(&out_schema, &got);
    // By position: `second` holds the first writer field's value.
    assert_eq!(
        got.column(0).as_primitive::<Int32Type>().values().to_vec(),
        vec![1]
    );
    assert_eq!(
        got.column(1).as_primitive::<Int32Type>().values().to_vec(),
        vec![2]
    );
}

/// `datetimeRebaseMode`: ancient dates fail in `EXCEPTION`, read as stored
/// in `CORRECTED` and are refused in `LEGACY`.
#[test]
fn datetime_rebase_modes() {
    use apache_avro::types::Value as V;
    let env = Env::new();
    let file_schema = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "d", "type": {"type": "int", "logicalType": "date"}},
            {"name": "t", "type": {"type": "long", "logicalType": "timestamp-micros"}}
        ]
    }"#;
    let values = vec![V::Record(vec![
        ("d".to_owned(), V::Date(-200_000)),
        ("t".to_owned(), V::TimestampMicros(-30_000_000_000_000)),
    ])];
    let file = env.put("rebase.avro", apache_write(file_schema, &values));
    // Modern values read in every mode.
    let modern = vec![V::Record(vec![
        ("d".to_owned(), V::Date(20000)),
        ("t".to_owned(), V::TimestampMicros(1_700_000_000_000_000)),
    ])];
    let modern_file = env.put("modern.avro", apache_write(file_schema, &modern));
    for mode in ["EXCEPTION", "CORRECTED", "LEGACY"] {
        let o = env.options(&[("datetimeRebaseMode", mode)]);
        let inferred = env.infer(&modern_file, &o);
        let got = env
            .scan(&modern_file, &ScanRequest::all(inferred), &o)
            .unwrap();
        assert_eq!(rows(&got), 1, "{mode}");
    }
    // Ancient values: EXCEPTION fails, CORRECTED reads them as stored,
    // LEGACY is refused.
    let inferred = env.infer(&file, &env.options(&[]));
    let e = env
        .scan(
            &file,
            &ScanRequest::all(Arc::clone(&inferred)),
            &env.options(&[]),
        )
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
    let o = env.options(&[("datetimeRebaseMode", "CORRECTED")]);
    let table_req = ScanRequest::all(Arc::clone(&inferred));
    let out_schema = mapped_schema(&inferred, &table_req);
    let got = env.scan(&file, &table_req, &o).unwrap();
    let got = concat(&out_schema, &got);
    assert_eq!(
        got.column(0).as_primitive::<Date32Type>().values().to_vec(),
        vec![-200_000]
    );
    assert_eq!(
        got.column(1)
            .as_primitive::<TimestampMicrosecondType>()
            .values()
            .to_vec(),
        vec![-30_000_000_000_000]
    );
    let o = env.options(&[("datetimeRebaseMode", "LEGACY")]);
    let e = env.scan(&file, &table_req, &o).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

/// File-relative row selections and `_metadata.row_index`: the selected
/// rows read back with their file positions.
#[test]
fn selection_and_row_index() {
    use crate::request::RowSelection;
    let env = Env::new();
    let o = env.options(&[]);
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(
            (0..1000).map(Some).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    let file = env.put("sel.avro", env.encode(&schema, &[batch], &o));
    let mut req = ScanRequest::all(Arc::clone(&schema));
    req.metadata_cols.row_index = true;
    req.selection = Some(RowSelection::excluding(1000, [7, 100, 999]));
    let out_schema = mapped_schema(&schema, &req);
    let got = env.scan(&file, &req, &o).unwrap();
    let got = concat(&out_schema, &got);
    assert_eq!(rows(std::slice::from_ref(&got)), 997);
    let index = got
        .column_by_name("_tmp_metadata_row_index")
        .unwrap()
        .as_primitive::<Int64Type>();
    let values = got.column(0).as_primitive::<Int64Type>();
    for (k, &row) in index.values().iter().enumerate() {
        assert_eq!(values.value(k), row);
        assert!(![7, 100, 999].contains(&row));
    }
    // Zero columns: row counts only.
    let mut req = ScanRequest::new(Arc::clone(&schema), Vec::new());
    req.metadata_cols.row_index = true;
    let out_schema = mapped_schema(&schema, &req);
    let got = env.scan(&file, &req, &o).unwrap();
    let got = concat(&out_schema, &got);
    assert_eq!(got.num_columns(), 1);
    assert_eq!(rows(std::slice::from_ref(&got)), 1000);
    let mut req = ScanRequest::new(Arc::clone(&schema), Vec::new());
    req.selection = Some(RowSelection::excluding(1000, [0]));
    let got = env.scan(&file, &req, &o).unwrap();
    assert_eq!(rows(&got), 999);
}
