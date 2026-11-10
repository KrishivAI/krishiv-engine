// SPDX-License-Identifier: Apache-2.0
//! Tests, part of this module's test suite (split by size; the helpers are in the parent).

use super::*;

/// Numeric promotions read as the Avro specification resolves them: `int`
/// widens to `long`, `float` and `double`, `long` to `float` and `double`,
/// and `float` to `double`.
#[test]
fn avro_numeric_promotions_read_as_spark_does() {
    use apache_avro::types::Value as V;
    let env = Env::new();
    let file_schema = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "a", "type": "int"},
            {"name": "b", "type": "long"},
            {"name": "c", "type": "float"}
        ]
    }"#;
    let values = vec![V::Record(vec![
        ("a".to_owned(), V::Int(3)),
        ("b".to_owned(), V::Long(1 << 40)),
        ("c".to_owned(), V::Float(1.5)),
    ])];
    let file = env.put("promote.avro", apache_write(file_schema, &values));
    let reader = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "a", "type": "long"},
            {"name": "b", "type": "double"},
            {"name": "c", "type": "double"}
        ]
    }"#;
    let o = env.options(&[("avroSchema", reader)]);
    let inferred = env.infer(&file, &o);
    let req = ScanRequest::all(Arc::clone(&inferred));
    let got = concat(
        &mapped_schema(&inferred, &req),
        &env.scan(&file, &req, &o).unwrap(),
    );
    let col = |name: &str| Arc::clone(got.column_by_name(name).unwrap());
    assert_eq!(
        col("a").as_primitive::<Int64Type>().values().to_vec(),
        vec![3]
    );
    assert_eq!(
        col("b")
            .as_primitive::<arrow_array::types::Float64Type>()
            .values()
            .to_vec(),
        vec![(1_i64 << 40) as f64]
    );
    assert_eq!(
        col("c")
            .as_primitive::<arrow_array::types::Float64Type>()
            .values()
            .to_vec(),
        vec![1.5]
    );
    // `long` to `float` and `int` to `float` and `double` widen too.
    let reader = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "a", "type": "double"},
            {"name": "b", "type": "float"},
            {"name": "c", "type": "double"}
        ]
    }"#;
    let o = env.options(&[("avroSchema", reader)]);
    let inferred = env.infer(&file, &o);
    let req = ScanRequest::all(Arc::clone(&inferred));
    let got = concat(
        &mapped_schema(&inferred, &req),
        &env.scan(&file, &req, &o).unwrap(),
    );
    let col = |name: &str| Arc::clone(got.column_by_name(name).unwrap());
    assert_eq!(
        col("a")
            .as_primitive::<arrow_array::types::Float64Type>()
            .values()
            .to_vec(),
        vec![3.0]
    );
    assert_eq!(
        col("b")
            .as_primitive::<arrow_array::types::Float32Type>()
            .values()
            .to_vec(),
        vec![(1_i64 << 40) as f32]
    );
    // And `int` to `float` directly.
    let reader = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "a", "type": "float"},
            {"name": "b", "type": "long"},
            {"name": "c", "type": "float"}
        ]
    }"#;
    let o = env.options(&[("avroSchema", reader)]);
    let inferred = env.infer(&file, &o);
    let req = ScanRequest::all(Arc::clone(&inferred));
    let got = concat(
        &mapped_schema(&inferred, &req),
        &env.scan(&file, &req, &o).unwrap(),
    );
    let col = |name: &str| Arc::clone(got.column_by_name(name).unwrap());
    assert_eq!(
        col("a")
            .as_primitive::<arrow_array::types::Float32Type>()
            .values()
            .to_vec(),
        vec![3.0]
    );
    // What the reader cannot hold fails in every mode.
    let reader = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "a", "type": "string"},
            {"name": "b", "type": "long"},
            {"name": "c", "type": "float"}
        ]
    }"#;
    let o = env.options(&[("avroSchema", reader)]);
    let inferred = env.infer(&file, &o);
    let req = ScanRequest::all(Arc::clone(&inferred));
    let e = scan_err(&env, &file, &req, &o);
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
}

/// Strings, bytes, enums, fixed and decimals convert as the specification
/// resolves them; unions take their branch, and a value no branch takes or
/// a null for a non-nullable field fails the file.
#[test]
fn avro_string_bytes_enum_and_union_resolution() {
    use apache_avro::types::Value as V;
    let env = Env::new();
    let uuid = "123e4567-e89b-12d3-a456-426614174000";
    let file_schema = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "g", "type": "bytes"},
            {"name": "h", "type": "string"},
            {"name": "e", "type": {"type": "enum", "name": "e", "symbols": ["A", "B"]}},
            {"name": "f", "type": {"type": "fixed", "name": "f", "size": 2}},
            {"name": "u", "type": {"type": "string", "logicalType": "uuid"}},
            {"name": "i", "type": ["int", "string"]},
            {"name": "j", "type": "int"},
            {"name": "n", "type": ["null", "int"]}
        ]
    }"#;
    let values = vec![V::Record(vec![
        ("g".to_owned(), V::Bytes(b"hi".to_vec())),
        ("h".to_owned(), V::String("yo".to_owned())),
        ("e".to_owned(), V::Enum(1, "B".to_owned())),
        ("f".to_owned(), V::Fixed(2, vec![0x01, 0x02])),
        ("u".to_owned(), V::Uuid(uuid.parse().unwrap())),
        (
            "i".to_owned(),
            V::Union(1, Box::new(V::String("x".to_owned()))),
        ),
        ("j".to_owned(), V::Int(9)),
        ("n".to_owned(), V::Union(0, Box::new(V::Null))),
    ])];
    let file = env.put("resolve.avro", apache_write(file_schema, &values));
    // `bytes` reads as `string` and back, `enum` as `string`, a plain
    // field as a single union, and a null union as its branch.
    let reader = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "g", "type": "string"},
            {"name": "h", "type": "bytes"},
            {"name": "e", "type": "string"},
            {"name": "f", "type": {"type": "fixed", "name": "f", "size": 2}},
            {"name": "u", "type": {"type": "string", "logicalType": "uuid"}},
            {"name": "i", "type": ["int", "string"]},
            {"name": "j", "type": ["null", "int"]},
            {"name": "n", "type": ["null", "int"]}
        ]
    }"#;
    let o = env.options(&[("avroSchema", reader)]);
    let inferred = env.infer(&file, &o);
    let req = ScanRequest::all(Arc::clone(&inferred));
    let got = concat(
        &mapped_schema(&inferred, &req),
        &env.scan(&file, &req, &o).unwrap(),
    );
    let col = |name: &str| Arc::clone(got.column_by_name(name).unwrap());
    assert_eq!(
        col("g").as_string_view().iter().collect::<Vec<_>>(),
        vec![Some("hi")]
    );
    assert_eq!(
        col("h").as_binary_view().iter().collect::<Vec<_>>(),
        vec![Some(b"yo".as_slice())]
    );
    assert_eq!(
        col("e").as_string_view().iter().collect::<Vec<_>>(),
        vec![Some("B")]
    );
    assert_eq!(
        col("j")
            .as_primitive::<Int32Type>()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some(9)]
    );
    assert_eq!(
        col("n")
            .as_primitive::<Int32Type>()
            .iter()
            .collect::<Vec<_>>(),
        vec![None]
    );
    // A `string` reads as the `enum` holding it.
    let reader = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "g", "type": "bytes"},
            {"name": "h", "type": {"type": "enum", "name": "q", "symbols": ["A", "yo"]}},
            {"name": "e", "type": {"type": "enum", "name": "e", "symbols": ["A", "B"]}},
            {"name": "f", "type": {"type": "fixed", "name": "f", "size": 2}},
            {"name": "u", "type": {"type": "string", "logicalType": "uuid"}},
            {"name": "i", "type": ["int", "string"]},
            {"name": "j", "type": "int"},
            {"name": "n", "type": ["null", "int"]}
        ]
    }"#;
    let o = env.options(&[("avroSchema", reader)]);
    let inferred = env.infer(&file, &o);
    let req = ScanRequest::all(Arc::clone(&inferred));
    let got = concat(
        &mapped_schema(&inferred, &req),
        &env.scan(&file, &req, &o).unwrap(),
    );
    let col = |name: &str| Arc::clone(got.column_by_name(name).unwrap());
    assert_eq!(
        col("h").as_string_view().iter().collect::<Vec<_>>(),
        vec![Some("yo")]
    );
    // The union members also read in `PERMISSIVE` mode, dry-checking the
    // member appends first.
    let permissive = env.options(&[("mode", "PERMISSIVE")]);
    let got = concat(
        &mapped_schema(&inferred, &req),
        &env.scan(&file, &req, &permissive).unwrap(),
    );
    assert_eq!(rows(std::slice::from_ref(&got)), 1);
    // Each mismatch names what it cannot hold: the same file read with
    // one field retyped.
    let base: &[(&str, &str)] = &[
        ("g", r#""bytes""#),
        ("h", r#""string""#),
        ("e", r#"{"type":"enum","name":"e","symbols":["A","B"]}"#),
        ("f", r#"{"type":"fixed","name":"f","size":2}"#),
        ("u", r#"{"type":"string","logicalType":"uuid"}"#),
        ("i", r#"["int","string"]"#),
        ("j", r#""int""#),
        ("n", r#"["null","int"]"#),
    ];
    let reader_with = |name: &str, fragment: &str| -> String {
        let fields = base
            .iter()
            .map(|(n, t)| {
                let t = if *n == name { fragment } else { t };
                format!(r#"{{"name":"{n}","type":{t}}}"#)
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(r#"{{"type":"record","name":"r","fields":[{fields}]}}"#)
    };
    // A plain field also reads as the union holding it.
    let o = env.options(&[("avroSchema", &reader_with("j", r#"["int","string"]"#))]);
    let inferred = env.infer(&file, &o);
    let req = ScanRequest::all(Arc::clone(&inferred));
    let got = concat(
        &mapped_schema(&inferred, &req),
        &env.scan(&file, &req, &o).unwrap(),
    );
    let member = got
        .column_by_name("j")
        .unwrap()
        .as_struct()
        .column(0)
        .as_primitive::<Int32Type>()
        .iter()
        .collect::<Vec<_>>();
    assert_eq!(member, vec![Some(9)]);
    // Each retyped field is a value the reader cannot hold: an enum as
    // an int, bytes as an int, a uuid as bytes, a fixed of another size
    // or as a string, a union no branch takes, a union branch the plain
    // reader rejects, and a null for a non-nullable field.
    for (name, fragment) in [
        ("h", r#""int""#),
        ("g", r#""int""#),
        ("u", r#""bytes""#),
        ("f", r#"{"type":"fixed","name":"f","size":4}"#),
        ("f", r#""string""#),
        ("i", r#"["int","long"]"#),
        ("i", r#""int""#),
        ("n", r#""int""#),
    ] {
        let o = env.options(&[("avroSchema", &reader_with(name, fragment))]);
        let inferred = env.infer(&file, &o);
        let req = ScanRequest::all(Arc::clone(&inferred));
        let e = scan_err(&env, &file, &req, &o);
        assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint, "{name}");
    }
    // A record takes the first union branch that holds it: the `int`
    // branch here, leaving the struct member null.
    let rec = env.put(
        "rec.avro",
        apache_write(
            r#"{"type":"record","name":"r","fields":[{"name":"a","type":"int"}]}"#,
            &[V::Record(vec![("a".to_owned(), V::Int(1))])],
        ),
    );
    let reader = r#"{"type":"record","name":"r","fields":[
        {"name":"a","type":[{"type":"record","name":"s","fields":[{"name":"b","type":"string"}]},"int"]}
    ]}"#;
    let o = env.options(&[("avroSchema", reader)]);
    let inferred = env.infer(&rec, &o);
    let req = ScanRequest::all(Arc::clone(&inferred));
    let got = concat(
        &mapped_schema(&inferred, &req),
        &env.scan(&rec, &req, &o).unwrap(),
    );
    let member = got
        .column_by_name("a")
        .unwrap()
        .as_struct()
        .column(1)
        .as_primitive::<Int32Type>()
        .iter()
        .collect::<Vec<_>>();
    assert_eq!(member, vec![Some(1)]);
    assert!(
        got.column_by_name("a")
            .unwrap()
            .as_struct()
            .column(0)
            .is_null(0)
    );
    // A record no reader union branch takes fails the file.
    let reader = r#"{"type":"record","name":"r","fields":[
        {"name":"a","type":[{"type":"record","name":"s","fields":[{"name":"b","type":"string"}]},"string"]}
    ]}"#;
    let o = env.options(&[("avroSchema", reader)]);
    let inferred = env.infer(&rec, &o);
    let req = ScanRequest::all(Arc::clone(&inferred));
    let e = scan_err(&env, &rec, &req, &o);
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
    // Bytes that are not UTF-8 cannot become a string.
    let raw = env.put(
        "raw.avro",
        apache_write(
            r#"{"type":"record","name":"r","fields":[{"name":"g","type":"bytes"}]}"#,
            &[V::Record(vec![("g".to_owned(), V::Bytes(vec![0xFF]))])],
        ),
    );
    let reader = r#"{"type":"record","name":"r","fields":[{"name":"g","type":"string"}]}"#;
    let o = env.options(&[("avroSchema", reader)]);
    let inferred = env.infer(&raw, &o);
    let req = ScanRequest::all(Arc::clone(&inferred));
    let e = scan_err(&env, &raw, &req, &o);
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidUtf8String,
        "bytes that are not UTF-8 cannot become a string"
    );
    // A missing field without a default fails; with one (or by position)
    // it reads the default.
    let thin = env.put(
        "thin.avro",
        apache_write(
            r#"{"type":"record","name":"r","fields":[{"name":"a","type":"int"}]}"#,
            &[V::Record(vec![("a".to_owned(), V::Int(1))])],
        ),
    );
    let reader = r#"{"type":"record","name":"r","fields":[
        {"name":"a","type":"int"},
        {"name":"b","type":"int"}
    ]}"#;
    let o = env.options(&[("avroSchema", reader)]);
    let inferred = env.infer(&thin, &o);
    let req = ScanRequest::all(Arc::clone(&inferred));
    let e = scan_err(&env, &thin, &req, &o);
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
    let reader = r#"{"type":"record","name":"r","fields":[
        {"name":"a","type":"int"},
        {"name":"b","type":"string","default":"d"}
    ]}"#;
    // By name the default fills the missing field; by position the
    // missing index does.
    for positional in [false, true] {
        let mut kv = vec![("avroSchema", reader)];
        if positional {
            kv.push(("positionalFieldMatching", "true"));
        }
        let o = env.options(&kv);
        let inferred = env.infer(&thin, &o);
        let req = ScanRequest::all(Arc::clone(&inferred));
        let got = concat(
            &mapped_schema(&inferred, &req),
            &env.scan(&thin, &req, &o).unwrap(),
        );
        assert_eq!(
            got.column_by_name("b")
                .unwrap()
                .as_string_view()
                .iter()
                .collect::<Vec<_>>(),
            vec![Some("d")],
            "positional: {positional}"
        );
    }
}

/// A nullable union reads as its plain branch: values pass through, and a
/// null takes the reader's null branch (or fails a non-nullable reader).
/// A decimal of another precision or scale fails the file.
#[test]
fn avro_nullable_unions_and_decimal_mismatches() {
    use apache_avro::types::Value as V;
    let env = Env::new();
    let file_schema = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "v", "type": ["null", "int"]},
            {"name": "d", "type": {"type": "bytes", "logicalType": "decimal", "precision": 9, "scale": 2}}
        ]
    }"#;
    let values = vec![
        V::Record(vec![
            ("v".to_owned(), V::Union(1, Box::new(V::Int(5)))),
            (
                "d".to_owned(),
                V::Decimal(apache_avro::Decimal::from(vec![0x30, 0x39])),
            ),
        ]),
        V::Record(vec![
            ("v".to_owned(), V::Union(0, Box::new(V::Null))),
            (
                "d".to_owned(),
                V::Decimal(apache_avro::Decimal::from(vec![0xFF])),
            ),
        ]),
    ];
    let file = env.put("opt.avro", apache_write(file_schema, &values));
    // The plain reader takes values through and fails nulls.
    let reader = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "v", "type": "int"},
            {"name": "d", "type": {"type": "bytes", "logicalType": "decimal", "precision": 9, "scale": 2}}
        ]
    }"#;
    let o = env.options(&[("avroSchema", reader)]);
    let inferred = env.infer(&file, &o);
    let req = ScanRequest::all(Arc::clone(&inferred));
    let e = scan_err(&env, &file, &req, &o);
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
    // A nullable reader takes both rows, nulling the union.
    let reader = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "v", "type": ["int", "string", "null"]},
            {"name": "d", "type": {"type": "bytes", "logicalType": "decimal", "precision": 9, "scale": 2}}
        ]
    }"#;
    let o = env.options(&[("avroSchema", reader)]);
    let inferred = env.infer(&file, &o);
    let req = ScanRequest::all(Arc::clone(&inferred));
    let got = concat(
        &mapped_schema(&inferred, &req),
        &env.scan(&file, &req, &o).unwrap(),
    );
    let member = got
        .column_by_name("v")
        .unwrap()
        .as_struct()
        .column(0)
        .as_primitive::<Int32Type>()
        .iter()
        .collect::<Vec<_>>();
    assert_eq!(member, vec![Some(5), None]);
    assert!(got.column_by_name("v").unwrap().as_struct().is_null(1));
    // A decimal of another precision or scale fails the file.
    let reader = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "v", "type": ["null", "int"]},
            {"name": "d", "type": {"type": "bytes", "logicalType": "decimal", "precision": 10, "scale": 2}}
        ]
    }"#;
    let o = env.options(&[("avroSchema", reader)]);
    let inferred = env.infer(&file, &o);
    let req = ScanRequest::all(Arc::clone(&inferred));
    let e = scan_err(&env, &file, &req, &o);
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
}

/// Ancient millis timestamps and local timestamps follow the rebase mode
/// like micros do: `EXCEPTION` fails, `CORRECTED` reads them as stored and
/// `LEGACY` is refused.
#[test]
fn avro_rebase_covers_millis_and_local_timestamps() {
    use apache_avro::types::Value as V;
    let env = Env::new();
    let file_schema = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "m", "type": {"type": "long", "logicalType": "timestamp-millis"}},
            {"name": "lm", "type": {"type": "long", "logicalType": "local-timestamp-millis"}}
        ]
    }"#;
    let values = vec![V::Record(vec![
        ("m".to_owned(), V::TimestampMillis(-30_000_000_000_000)),
        (
            "lm".to_owned(),
            V::LocalTimestampMillis(-30_000_000_000_000),
        ),
    ])];
    let file = env.put("ancient-millis.avro", apache_write(file_schema, &values));
    let inferred = env.infer(&file, &env.options(&[]));
    let req = ScanRequest::all(Arc::clone(&inferred));
    let e = scan_err(&env, &file, &req, &env.options(&[]));
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
    let o = env.options(&[("datetimeRebaseMode", "CORRECTED")]);
    let got = concat(
        &mapped_schema(&inferred, &req),
        &env.scan(&file, &req, &o).unwrap(),
    );
    assert_eq!(
        got.column_by_name("m")
            .unwrap()
            .as_primitive::<TimestampMicrosecondType>()
            .values()
            .to_vec(),
        vec![-30_000_000_000_000_000]
    );
    assert_eq!(
        got.column_by_name("lm")
            .unwrap()
            .as_primitive::<TimestampMicrosecondType>()
            .values()
            .to_vec(),
        vec![-30_000_000_000_000_000]
    );
    let o = env.options(&[("datetimeRebaseMode", "LEGACY")]);
    let e = scan_err(&env, &file, &req, &o);
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

/// A block cut short of its records and a union branch index no branch
/// has fail the record: the framing past a bad datum is unknowable, so
/// the rest of the block goes with it.
#[test]
fn avro_short_blocks_and_bad_union_branches_fail_records() {
    use apache_avro::types::Value as V;
    let env = Env::new();
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![Some(1), Some(2), Some(3)]))],
    )
    .unwrap();
    let plain = env.options(&[("compression", "uncompressed")]);
    let bytes = env.encode(&schema, std::slice::from_ref(&batch), &plain);
    let meta = env
        .rt
        .block_on(env.io.read_metadata(
            &AvroFormat,
            &env.put("short-src.avro", bytes.clone()),
            MetadataNeeds::default(),
            &plain,
        ))
        .unwrap();
    let any: &dyn std::any::Any = meta.as_ref();
    let start = usize::try_from(
        any.downcast_ref::<super::super::AvroMetadata>()
            .unwrap()
            .blocks[0]
            .bytes
            .start,
    )
    .unwrap();
    // Three rows take one count byte: claiming two decodes two and leaves
    // trailing bytes.
    assert_eq!(bytes[start], 0x06, "three rows take one count byte");
    let mut bad = bytes;
    bad[start] = 0x04;
    let file = env.put("short.avro", bad);
    let table_req = ScanRequest::all(Arc::clone(&schema));
    let e = scan_err(&env, &file, &table_req, &plain);
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
    // A union branch index past the last branch is undecodable too: the
    // block data starts with the branch byte.
    let uni_schema =
        r#"{"type":"record","name":"r","fields":[{"name":"u","type":["int","string"]}]}"#;
    let uni_values = vec![V::Record(vec![(
        "u".to_owned(),
        V::Union(0, Box::new(V::Int(7))),
    )])];
    let uni_bytes = apache_write(uni_schema, &uni_values);
    let uni = env.put("uni.avro", uni_bytes.clone());
    let meta = env
        .rt
        .block_on(
            env.io
                .read_metadata(&AvroFormat, &uni, MetadataNeeds::default(), &plain),
        )
        .unwrap();
    let any: &dyn std::any::Any = meta.as_ref();
    let range = any
        .downcast_ref::<super::super::AvroMetadata>()
        .unwrap()
        .blocks[0]
        .data
        .clone();
    let start = usize::try_from(range.start).unwrap();
    let mut bad = uni_bytes;
    assert_eq!(bad[start], 0x00, "branch zero takes one byte");
    bad[start] = 0x12;
    let file = env.put("uni-bad.avro", bad);
    let inferred = env.infer(&file, &env.options(&[]));
    let req = ScanRequest::all(Arc::clone(&inferred));
    let e = scan_err(&env, &file, &req, &env.options(&[]));
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
}

/// The writer covers the remaining Arrow layouts: narrow ints, owned and
/// large strings and bytes, times, millis timestamps, large and fixed-size
/// lists, maps with nullable values and wide decimals.
#[test]
fn avro_writer_covers_the_remaining_arrow_types() {
    use arrow_array::{
        LargeBinaryArray, LargeStringArray, StringArray, Time32MillisecondArray,
        Time64MicrosecondArray, TimestampMillisecondArray,
    };
    let env = Env::new();
    let o = env.options(&[]);
    let ll_items = arrow_array::LargeListArray::new(
        Arc::new(Field::new("element", DataType::Int32, true)),
        OffsetBuffer::new(vec![0_i64, 2, 2].into()),
        Arc::new(Int32Array::from(vec![Some(1), None])),
        Some(vec![true, false].into()),
    );
    let fsl_items = arrow_array::FixedSizeListArray::new(
        Arc::new(Field::new("element", DataType::Int32, true)),
        2,
        Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(3), Some(4)])),
        Some(vec![true, false].into()),
    );
    let map_value_field = Field::new("value", DataType::Int32, true);
    let map_entries_field = Arc::new(Field::new(
        "entries",
        DataType::Struct(
            vec![
                Field::new("key", DataType::Utf8View, false),
                map_value_field.clone(),
            ]
            .into(),
        ),
        false,
    ));
    let map_entries = StructArray::new(
        Fields::from(vec![
            Field::new("key", DataType::Utf8View, false),
            map_value_field,
        ]),
        vec![
            Arc::new(StringViewArray::from(vec![Some("k"), Some("j")])) as ArrayRef,
            Arc::new(Int32Array::from(vec![Some(1), None])) as ArrayRef,
        ],
        None,
    );
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("i8", DataType::Int8, true),
        Field::new("i16", DataType::Int16, true),
        Field::new("s", DataType::Utf8, true),
        Field::new("ls", DataType::LargeUtf8, true),
        Field::new("bin", DataType::Binary, true),
        Field::new("lbin", DataType::LargeBinary, true),
        Field::new("t32", DataType::Time32(TimeUnit::Millisecond), true),
        Field::new("t64", DataType::Time64(TimeUnit::Microsecond), true),
        Field::new(
            "tsms",
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            true,
        ),
        Field::new(
            "ll",
            DataType::LargeList(Arc::new(Field::new("element", DataType::Int32, true))),
            true,
        ),
        Field::new(
            "fsl",
            DataType::FixedSizeList(Arc::new(Field::new("element", DataType::Int32, true)), 2),
            true,
        ),
        Field::new(
            "m",
            DataType::Map(Arc::clone(&map_entries_field), false),
            true,
        ),
        Field::new("d38", DataType::Decimal128(38, 9), true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(arrow_array::Int8Array::from(vec![Some(-3), None])),
            Arc::new(arrow_array::Int16Array::from(vec![Some(300), None])),
            Arc::new(StringArray::from(vec![Some("a"), None])),
            Arc::new(LargeStringArray::from(vec![Some("b"), None])),
            Arc::new(arrow_array::BinaryArray::from(vec![
                Some(b"x".as_slice()),
                None,
            ])),
            Arc::new(LargeBinaryArray::from(vec![Some(b"y".as_slice()), None])),
            Arc::new(Time32MillisecondArray::from(vec![Some(1), None])),
            Arc::new(Time64MicrosecondArray::from(vec![Some(2), None])),
            Arc::new(
                TimestampMillisecondArray::from(vec![Some(1_700_000_000_123), None])
                    .with_timezone("UTC"),
            ),
            Arc::new(ll_items),
            Arc::new(fsl_items),
            Arc::new(arrow_array::MapArray::new(
                Arc::clone(&map_entries_field),
                OffsetBuffer::new(vec![0, 2, 2].into()),
                map_entries,
                Some(vec![true, false].into()),
                false,
            )),
            Arc::new(
                Decimal128Array::from(vec![Some(12345), None])
                    .with_precision_and_scale(38, 9)
                    .unwrap(),
            ),
        ],
    )
    .unwrap();
    let file = env.put(
        "types.avro",
        env.encode(&schema, std::slice::from_ref(&batch), &o),
    );
    let inferred = env.infer(&file, &o);
    // Narrow ints, times and millis timestamps read as their Avro types.
    assert_eq!(
        inferred.field_with_name("i8").unwrap().data_type(),
        &DataType::Int32
    );
    assert_eq!(
        inferred.field_with_name("t32").unwrap().data_type(),
        &DataType::Int32
    );
    assert_eq!(
        inferred.field_with_name("t64").unwrap().data_type(),
        &DataType::Int64
    );
    assert_eq!(
        inferred.field_with_name("tsms").unwrap().data_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
    );
    let req = ScanRequest::all(Arc::clone(&inferred));
    let got = concat(
        &mapped_schema(&inferred, &req),
        &env.scan(&file, &req, &o).unwrap(),
    );
    assert_eq!(rows(std::slice::from_ref(&got)), 2);
    let col = |name: &str| Arc::clone(got.column_by_name(name).unwrap());
    assert_eq!(
        col("i8")
            .as_primitive::<Int32Type>()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some(-3), None]
    );
    assert_eq!(
        col("tsms")
            .as_primitive::<TimestampMicrosecondType>()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some(1_700_000_000_123_000), None]
    );
}

/// A `null` column reads, and a damaged block nulled in `PERMISSIVE` (or
/// dropped in `DROPMALFORMED`) nulls every leaf: booleans, numbers,
/// strings, decimals, dates, timestamps, structs, lists and maps.
#[test]
fn avro_null_columns_and_permissive_null_rows() {
    use apache_avro::types::Value as V;
    let env = Env::new();
    let o = env.options(&[]);
    let file = env.put(
        "nulls.avro",
        apache_write(
            r#"{"type":"record","name":"r","fields":[{"name":"n","type":"null"}]}"#,
            &[V::Record(vec![("n".to_owned(), V::Null)])],
        ),
    );
    let inferred = env.infer(&file, &o);
    assert_eq!(
        inferred.field_with_name("n").unwrap().data_type(),
        &DataType::Null
    );
    let req = ScanRequest::all(Arc::clone(&inferred));
    let got = env.scan(&file, &req, &o).unwrap();
    assert_eq!(rows(&got), 1);
    // A null also passes the `DROPMALFORMED` dry check.
    let drop = env.options(&[("mode", "DROPMALFORMED")]);
    let got = env.scan(&file, &req, &drop).unwrap();
    assert_eq!(rows(&got), 1);
    // Every leaf nulled by one damaged block.
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("b", DataType::Boolean, true),
        Field::new("i", DataType::Int64, true),
        Field::new("s", DataType::Utf8View, true),
        Field::new("dec", DataType::Decimal128(10, 2), true),
        Field::new("date", DataType::Date32, true),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
        Field::new(
            "st",
            DataType::Struct(vec![Field::new("x", DataType::Int32, true)].into()),
            true,
        ),
        Field::new(
            "arr",
            DataType::List(Arc::new(Field::new("element", DataType::Int64, false))),
            true,
        ),
        Field::new(
            "m",
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
    let one = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(BooleanArray::from(vec![Some(true)])),
            Arc::new(Int64Array::from(vec![Some(1)])),
            Arc::new(StringViewArray::from(vec![Some("a")])),
            Arc::new(
                Decimal128Array::from(vec![Some(100)])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
            Arc::new(Date32Array::from(vec![Some(5)])),
            Arc::new(TimestampMicrosecondArray::from(vec![Some(7)]).with_timezone("UTC")),
            Arc::new(StructArray::new(
                Fields::from(vec![Field::new("x", DataType::Int32, true)]),
                vec![Arc::new(Int32Array::from(vec![Some(2)])) as ArrayRef],
                Some(vec![true].into()),
            )),
            Arc::new(list_array(&[Some(vec![Some(3)])])),
            Arc::new(map_array(&[Some(vec![("k".to_owned(), Some(4))])])),
        ],
    )
    .unwrap();
    let plain = env.options(&[("compression", "uncompressed")]);
    let mut enc = AvroEncoder::new(&schema, &plain).unwrap();
    for _ in 0..2 {
        enc.write(&one).unwrap();
        enc.flush().unwrap();
    }
    let mut two = Vec::new();
    while let Some(chunk) = enc.take_bytes() {
        two.extend_from_slice(&chunk);
    }
    let done = Box::new(enc).finish().unwrap();
    two.extend_from_slice(&done.tail);
    let meta = env
        .rt
        .block_on(env.io.read_metadata(
            &AvroFormat,
            &env.put("nullrows.avro", two.clone()),
            MetadataNeeds::default(),
            &plain,
        ))
        .unwrap();
    let any: &dyn std::any::Any = meta.as_ref();
    let data = any
        .downcast_ref::<super::super::AvroMetadata>()
        .unwrap()
        .blocks[1]
        .data
        .clone();
    let (start, end) = (
        usize::try_from(data.start).unwrap(),
        usize::try_from(data.end).unwrap(),
    );
    let mut bad = two;
    bad[start..end].fill(0xFF);
    let file = env.put("nullrows-bad.avro", bad);
    let table_req = ScanRequest::all(Arc::clone(&schema));
    let out_schema = mapped_schema(&schema, &table_req);
    let permissive = env.options(&[("mode", "PERMISSIVE"), ("compression", "uncompressed")]);
    let got = concat(
        &out_schema,
        &env.scan(&file, &table_req, &permissive).unwrap(),
    );
    assert_eq!(rows(std::slice::from_ref(&got)), 2);
    for name in ["b", "i", "s", "dec", "date", "ts", "st", "arr", "m"] {
        assert!(got.column_by_name(name).unwrap().is_null(1), "{name}");
    }
    let drop = env.options(&[("mode", "DROPMALFORMED"), ("compression", "uncompressed")]);
    let got = concat(&out_schema, &env.scan(&file, &table_req, &drop).unwrap());
    assert_eq!(rows(std::slice::from_ref(&got)), 1);
}

/// Large scans emit mid-file, and the prefetch gate accounts the
/// decoder: batches, upcoming ranges and decode estimates all run.
#[test]
fn avro_many_rows_emit_mid_scan_gate_and_prefetch() {
    let env = Env::new();
    let o = env.options(&[]);
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(
            (0..5000).map(Some).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    let file = env.put("many.avro", env.encode(&schema, &[batch], &o));
    // More than one batch of rows: the scan emits mid-file.
    let req = ScanRequest::all(Arc::clone(&schema));
    let got = env.scan(&file, &req, &o).unwrap();
    assert_eq!(rows(&got), 5000);
    // Through the prefetch gate the decoder reports its estimates.
    let pool: Arc<dyn krishiv_memory::MemoryPool> =
        krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
    let gate = Arc::new(crate::ScanGate::new(&pool, 1 << 20));
    let id = pool.register("scan", None);
    let req = ScanRequest::all(Arc::clone(&schema));
    let got = env.rt.block_on(async {
        let mut stream = ScanStream::open(
            &env.io,
            &AvroFormat,
            file.clone(),
            ByteRange::whole(file.size),
            &req,
            &o,
            krishiv_memory::Reservation::new(pool, id),
            ScanOptions::default(),
        )
        .await
        .unwrap()
        .with_gate(gate);
        let mut out = Vec::new();
        while let Some(batch) = stream.next_batch().await.unwrap() {
            out.push(batch);
        }
        out
    });
    assert_eq!(rows(&got), 5000);
    // A tiny prefetch budget settles the upcoming ranges early.
    let mut enc = AvroEncoder::new(&schema, &o).unwrap();
    for block in 0..3 {
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(vec![Some(block)]))],
        )
        .unwrap();
        enc.write(&batch).unwrap();
        enc.flush().unwrap();
    }
    let mut three = Vec::new();
    while let Some(chunk) = enc.take_bytes() {
        three.extend_from_slice(&chunk);
    }
    let done = Box::new(enc).finish().unwrap();
    three.extend_from_slice(&done.tail);
    let file = env.put("three.avro", three);
    let pool: Arc<dyn krishiv_memory::MemoryPool> =
        krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
    let id = pool.register("scan", None);
    let req = ScanRequest::all(Arc::clone(&schema));
    let got = env.rt.block_on(async {
        let mut stream = ScanStream::open(
            &env.io,
            &AvroFormat,
            file.clone(),
            ByteRange::whole(file.size),
            &req,
            &o,
            krishiv_memory::Reservation::new(pool, id),
            ScanOptions {
                prefetch_bytes: 10,
                ..ScanOptions::default()
            },
        )
        .await
        .unwrap();
        let mut out = Vec::new();
        while let Some(batch) = stream.next_batch().await.unwrap() {
            out.push(batch);
        }
        out
    });
    assert_eq!(rows(&got), 3);
}

/// A `target_file_bytes` write rolls files mid-batch, like the other
/// file writers: the encoder reports what it has buffered and written.
#[test]
fn avro_target_file_bytes_rolls_files() {
    let env = Env::new();
    let o = env.options(&[]);
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, true)]));
    let pool: Arc<dyn krishiv_memory::MemoryPool> =
        krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
    let id = pool.register("write", None);
    let files = env.rt.block_on(async {
        let mut writer = crate::FileWriter::new(
            &env.io,
            Arc::new(AvroFormat),
            &schema,
            o,
            crate::WriteOptions {
                store: StoreId::new(STORE),
                dir: Path::from("roll"),
                task: 0,
                job_uuid: "job-roll".to_owned(),
                max_records_per_file: 0,
                target_file_bytes: 1000,
                partition_by: Vec::new(),
                partition_names: Vec::new(),
                zone: Some(0),
                zone_name: None,
            },
            krishiv_memory::Reservation::new(pool, id),
        )
        .unwrap();
        for block in 0..10 {
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from(
                    (0..50).map(|i| Some(block * 50 + i)).collect::<Vec<_>>(),
                ))],
            )
            .unwrap();
            writer.write(&batch).await.unwrap();
        }
        writer.finish().await.unwrap()
    });
    assert!(files.len() >= 2, "one batch rolls into {files:?}");
    let rows: usize = files
        .iter()
        .map(|f| {
            let meta = env.rt.block_on(async {
                object_store::ObjectStoreExt::head(env.store.as_ref(), &f.path)
                    .await
                    .unwrap()
            });
            usize::try_from(meta.size).unwrap()
        })
        .sum();
    assert!(rows > 1000, "{rows}");
}

/// Files written by Spark 4.2's `spark-avro` (`tests/scripts/gen_avro_spark.py`,
/// one row of every supported type and a row of NULLs) in each block codec
/// read back as Spark showed them; `xz` and `zstandard` are the typed
/// refusals.
#[test]
fn spark_written_files_read_back_equal() {
    let env = Env::new();
    let fixture = |codec: &str| {
        let p = format!(
            "{}/../../tests/fixtures/spark-4.2.0/avro/people-{codec}.avro",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
    };
    let o = env.options(&[]);
    let read = |codec: &str| {
        let file = env.put(&format!("spark-{codec}.avro"), fixture(codec));
        let schema = env.infer(&file, &o);
        let req = ScanRequest::all(Arc::clone(&schema));
        let out = mapped_schema(&schema, &req);
        let got = env.scan(&file, &req, &o).unwrap();
        (schema, concat(&out, &got))
    };
    let (schema, plain) = read("uncompressed");
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        names,
        [
            "id", "s", "d", "f", "b", "bin", "dt", "ts", "ntz", "dec", "arr", "m", "st", "n", "ti",
            "si", "bg"
        ]
    );
    let types: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| format!("{}", f.data_type()))
        .collect();
    assert_eq!(plain.num_rows(), 2);
    let ty = |n: &str| schema.field_with_name(n).unwrap().data_type().clone();
    assert!(
        matches!(
            ty("ts"),
            DataType::Timestamp(TimeUnit::Microsecond, Some(_))
        ),
        "{types:?}"
    );
    assert!(
        matches!(ty("ntz"), DataType::Timestamp(TimeUnit::Microsecond, None)),
        "{types:?}"
    );
    assert!(
        matches!(ty("dec"), DataType::Decimal128(10, 2)),
        "{types:?}"
    );
    // Row 1: every value, as `df.show()` printed it.
    let c = |name: &str| Arc::clone(plain.column_by_name(name).unwrap());
    assert_eq!(c("id").as_primitive::<Int32Type>().value(0), 1);
    assert_eq!(c("s").as_string_view().value(0), "a");
    assert_eq!(
        c("d")
            .as_primitive::<arrow_array::types::Float64Type>()
            .value(0),
        1.5
    );
    assert_eq!(
        c("f")
            .as_primitive::<arrow_array::types::Float32Type>()
            .value(0),
        2.5
    );
    assert!(c("b").as_boolean().value(0));
    assert_eq!(c("bin").as_binary_view().value(0), b"x");
    assert_eq!(c("dt").as_primitive::<Date32Type>().value(0), 18_263);
    assert_eq!(
        c("ts").as_primitive::<TimestampMicrosecondType>().value(0),
        1_577_934_245_678_901
    );
    assert_eq!(
        c("ntz").as_primitive::<TimestampMicrosecondType>().value(0),
        1_577_934_245_678_901
    );
    assert_eq!(c("dec").as_primitive::<Decimal128Type>().value(0), 1234);
    assert_eq!(c("bg").as_primitive::<Int64Type>().value(0), 9_000_000_000);
    assert_eq!(c("ti").as_primitive::<Int32Type>().value(0), 5);
    assert_eq!(c("si").as_primitive::<Int32Type>().value(0), 6);
    assert!(c("n").is_null(0));
    // Row 2 is NULL in every column but the id.
    for n in names.iter().skip(1) {
        assert!(c(n).is_null(1), "{n} {types:?}");
    }
    for codec in ["snappy", "deflate", "bzip2"] {
        let (s, got) = read(codec);
        assert_eq!(s, schema, "{codec}");
        assert_eq!(got, plain, "{codec}");
    }
    for (codec, class) in [
        ("xz", ErrorClass::KrishivUnsupportedFeature),
        (
            "zstandard",
            ErrorClass::CodecNotAvailableWithAvailableCodecsSuggestion,
        ),
    ] {
        let file = env.put(&format!("spark-{codec}.avro"), fixture(codec));
        let e = env
            .rt
            .block_on(AvroFormat.infer_schema(&env.io, std::slice::from_ref(&file), &o))
            .unwrap_err();
        assert_eq!(class_of(&e), class, "{codec}");
    }
}

/// Writes one file per codec into `$KRISHIV_AVRO_DUMP` for
/// `tests/scripts/check_avro_written.py` (Spark 4.2 reads them).
#[test]
#[ignore = "writes files for the Spark check; run with KRISHIV_AVRO_DUMP=<dir>"]
fn dump_written_files_for_spark() {
    let Ok(dir) = std::env::var("KRISHIV_AVRO_DUMP") else {
        return;
    };
    let env = Env::new();
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("v", DataType::Utf8View, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from((0..500).collect::<Vec<_>>())),
            Arc::new(StringViewArray::from(
                (0..500).map(|i| Some(format!("v{i}"))).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    for codec in ["uncompressed", "snappy", "deflate", "bzip2"] {
        let o = env.options(&[("compression", codec)]);
        let bytes = env.encode(&schema, std::slice::from_ref(&batch), &o);
        std::fs::write(format!("{dir}/{codec}.avro"), bytes).unwrap();
    }
}
