// SPDX-License-Identifier: Apache-2.0
//! Tests, part of this module's test suite (split by size; the helpers are in the parent).

use super::*;

/// Nested nulls at every level read back as written.
#[test]
fn round_trip_nested_nulls() {
    let env = Env::new();
    let o = env.options(&[]);
    let inner = Fields::from(vec![
        Field::new("x", DataType::Int32, true),
        Field::new("y", DataType::Utf8View, true),
    ]);
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "st",
        DataType::Struct(inner.clone()),
        true,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StructArray::new(
            inner,
            vec![
                Arc::new(Int32Array::from(vec![Some(1), None, None])) as ArrayRef,
                Arc::new(StringViewArray::from(vec![None, Some("a"), None])) as ArrayRef,
            ],
            Some(vec![true, true, false].into()),
        ))],
    )
    .unwrap();
    let file = env.put(
        "nested.avro",
        env.encode(&schema, std::slice::from_ref(&batch), &o),
    );
    let table_req = ScanRequest::all(Arc::clone(&schema));
    let out_schema = mapped_schema(&schema, &table_req);
    let got = env.scan(&file, &table_req, &o).unwrap();
    assert_eq!(concat(&out_schema, &got), retype_batch(&batch, &out_schema));
}

/// The written schema JSON: names, namespaces, nullable unions, decimals
/// and nesting; unsupported Arrow types are refused.
#[test]
fn writer_schema_json() {
    let json = schema_json_of_arrow(
        &Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8View, true),
            Field::new("d", DataType::Decimal128(10, 2), true),
            Field::new(
                "st",
                DataType::Struct(vec![Field::new("x", DataType::Int64, true)].into()),
                true,
            ),
        ]),
        "topLevelRecord",
        "ns",
    )
    .unwrap();
    for fragment in [
        r#""name":"topLevelRecord""#,
        r#""namespace":"ns""#,
        r#"{"name":"a","type":"int"}"#,
        r#"{"name":"b","type":["null","string"]}"#,
        r#""logicalType":"decimal""#,
        r#""precision":10"#,
    ] {
        assert!(json.contains(fragment), "{json}");
    }
    // Unsupported Arrow types fail `check_write` before any task runs.
    let bad: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "dur",
        DataType::Duration(arrow_schema::TimeUnit::Nanosecond),
        true,
    )]));
    let e = AvroFormat
        .check_write(&bad, &env_options_none())
        .unwrap_err();
    assert_eq!(
        e.into_error("test").error_class(),
        ErrorClass::KrishivUnsupportedFeature
    );
    assert_eq!(AvroFormat.file_suffix(&env_options_none()), ".avro");
}

/// A file written by Spark 3.0 or later says how it was written: its rebase
/// mode comes from the header (`CORRECTED`, or `LEGACY` with the
/// `legacyDateTime` key; Spark 2.x files are `LEGACY`), whatever
/// `datetimeRebaseMode` says. Only files without a Spark version use the
/// option.
#[test]
fn spark_headers_choose_the_rebase_mode() {
    let env = Env::new();
    let version = ("org.apache.spark.version", "4.2.0");
    let legacy = ("org.apache.spark.legacyDateTime", "");
    for (name, meta, want_rows) in [
        ("v3.avro", vec![version], Some(1)),
        ("v3-legacy.avro", vec![version, legacy], None),
        ("v2.avro", vec![("org.apache.spark.version", "2.4.8")], None),
        ("unknown.avro", vec![], None),
    ] {
        let file = ancient_file(&env, name, &meta);
        // The default `EXCEPTION` mode applies only to the unknown writer.
        let o = env.options(&[]);
        let req = ScanRequest::all(env.infer(&file, &o));
        match (env.scan(&file, &req, &o), want_rows) {
            (Ok(got), Some(n)) => assert_eq!(rows(&got), n, "{name}"),
            (Err(e), None) => {
                let class = if name == "unknown.avro" {
                    ErrorClass::FailedReadFileNoHint
                } else {
                    ErrorClass::KrishivUnsupportedFeature
                };
                assert_eq!(e.error_class(), class, "{name}");
            }
            (got, want) => panic!("{name}: {got:?} against {want:?}"),
        }
        // The option does not override what the file says.
        let o = env.options(&[("datetimeRebaseMode", "CORRECTED")]);
        let got = env.scan(&file, &req, &o);
        match name {
            "v3.avro" | "unknown.avro" => assert_eq!(rows(&got.unwrap()), 1, "{name}"),
            _ => assert_eq!(
                got.unwrap_err().error_class(),
                ErrorClass::KrishivUnsupportedFeature,
                "{name}"
            ),
        }
    }
}

/// `compression` and `datetimeRebaseMode` fall back to Spark's session
/// confs; an option given explicitly wins.
#[test]
fn session_confs_back_the_options() {
    let env = Env::new();
    let mut conf = krishiv_common::config::Config::new();
    conf.set("spark.sql.avro.compression.codec", "deflate")
        .unwrap();
    conf.set("spark.sql.avro.datetimeRebaseModeInRead", "CORRECTED")
        .unwrap();
    let parse = |user: &[(&str, &str)]| {
        let user = user
            .iter()
            .map(|(k, v)| ((*k).to_owned(), option_value(k, v)))
            .collect();
        AvroFormat.options().parse(&user, &conf).unwrap().0
    };
    let o = parse(&[]);
    assert_eq!(o.str("compression"), Some("deflate"));
    assert_eq!(o.str("datetimeRebaseMode"), Some("CORRECTED"));
    let given = parse(&[
        ("compression", "uncompressed"),
        ("datetimeRebaseMode", "EXCEPTION"),
    ]);
    assert_eq!(given.str("compression"), Some("uncompressed"));
    assert_eq!(given.str("datetimeRebaseMode"), Some("EXCEPTION"));
    // Without a conf the defaults hold.
    let o = env.options(&[]);
    assert_eq!(o.str("compression"), Some("snappy"));
    assert_eq!(o.str("datetimeRebaseMode"), Some("EXCEPTION"));
    // The conf reaches the written header and the read.
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![Some(1)]))],
    )
    .unwrap();
    let bytes = env.encode(&schema, &[batch], &parse(&[]));
    assert!(bytes.windows(7).any(|w| w == b"deflate"));
    let file = ancient_file(&env, "conf.avro", &[]);
    let o = parse(&[]);
    let got = env
        .scan(&file, &ScanRequest::all(env.infer(&file, &o)), &o)
        .unwrap();
    assert_eq!(rows(&got), 1);
}

/// A corrupt file fails the read unless `mode` says otherwise (Spark's
/// default is `FAILFAST`).
#[test]
fn mode_defaults_to_failfast() {
    use super::super::options::{AvroOptions, Mode};
    let env = Env::new();
    let default = AvroOptions::new(&env.options(&[])).unwrap();
    assert_eq!(default.mode, Mode::FailFast);
    for (mode, want) in [
        ("PERMISSIVE", Mode::Permissive),
        ("DROPMALFORMED", Mode::DropMalformed),
        ("FAILFAST", Mode::FailFast),
    ] {
        let o = AvroOptions::new(&env.options(&[("mode", mode)])).unwrap();
        assert_eq!(o.mode, want, "{mode}");
    }
}

/// A record that decodes but that a later column rejects (an out-of-range
/// decimal, a timestamp that overflows microseconds) takes no value in the
/// earlier columns: `PERMISSIVE` keeps one null row at the record's own file
/// row, `DROPMALFORMED` drops the record and the rest of its block, and
/// `FAILFAST` fails. A row outside the selection is never read.
#[test]
fn a_record_a_column_rejects_leaves_no_partial_row() {
    use crate::request::RowSelection;
    use apache_avro::types::Value as V;
    let env = Env::new();
    let schema_json = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "a", "type": ["null", "int"]},
            {"name": "ts", "type": ["null", {"type": "long", "logicalType": "timestamp-millis"}]},
            {"name": "d", "type": ["null", {"type": "bytes", "logicalType": "decimal", "precision": 5, "scale": 2}]}
        ]
    }"#;
    let record = |a: i32, ts: i64, d: Vec<u8>| {
        V::Record(vec![
            ("a".to_owned(), V::Union(1, Box::new(V::Int(a)))),
            (
                "ts".to_owned(),
                V::Union(1, Box::new(V::TimestampMillis(ts))),
            ),
            (
                "d".to_owned(),
                V::Union(1, Box::new(V::Decimal(apache_avro::Decimal::from(d)))),
            ),
        ])
    };
    // 100 and 300 are `1.00` and `3.00`; 123456 has six digits.
    let cases = [
        ("timestamp", [(2, i64::MAX, vec![0x01, 0x2C])]),
        ("decimal", [(2, 2000, vec![0x01, 0xE2, 0x40])]),
    ];
    for (name, [(a, ts, d)]) in cases {
        let values = vec![
            record(1, 1000, vec![0x64]),
            record(a, ts, d),
            record(3, 3000, vec![0x01, 0x2C]),
        ];
        let file = env.put(&format!("{name}.avro"), apache_write(schema_json, &values));
        let o = env.options(&[]);
        let inferred = env.infer(&file, &o);
        let mut req = ScanRequest::all(Arc::clone(&inferred));
        req.metadata_cols.row_index = true;
        let out_schema = mapped_schema(&inferred, &req);
        let run = |o: &FormatOptions, req: &ScanRequest| {
            env.scan(&file, req, o).map(|got| concat(&out_schema, &got))
        };
        let a_of = |b: &RecordBatch| {
            b.column(0)
                .as_primitive::<Int32Type>()
                .iter()
                .collect::<Vec<_>>()
        };
        let ts_of = |b: &RecordBatch| {
            b.column(1)
                .as_primitive::<TimestampMicrosecondType>()
                .iter()
                .collect::<Vec<_>>()
        };
        let d_of = |b: &RecordBatch| {
            b.column(2)
                .as_primitive::<Decimal128Type>()
                .iter()
                .collect::<Vec<_>>()
        };
        let index_of = |b: &RecordBatch| {
            b.column_by_name("_tmp_metadata_row_index")
                .unwrap()
                .as_primitive::<Int64Type>()
                .values()
                .to_vec()
        };
        // The default fails the read.
        let e = run(&o, &req).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint, "{name}");
        // One null row, every column the same length, at the record's row.
        let permissive = env.options(&[("mode", "PERMISSIVE")]);
        let got = run(&permissive, &req).unwrap();
        assert_eq!(a_of(&got), vec![Some(1), None], "{name}");
        assert_eq!(ts_of(&got), vec![Some(1_000_000), None], "{name}");
        assert_eq!(d_of(&got), vec![Some(100), None], "{name}");
        assert_eq!(index_of(&got), vec![0, 1], "{name}");
        // The record and the rest of its block are dropped.
        let drop = env.options(&[("mode", "DROPMALFORMED")]);
        let got = run(&drop, &req).unwrap();
        assert_eq!(a_of(&got), vec![Some(1)], "{name}");
        assert_eq!(ts_of(&got), vec![Some(1_000_000)], "{name}");
        assert_eq!(d_of(&got), vec![Some(100)], "{name}");
        assert_eq!(index_of(&got), vec![0], "{name}");
        // Unselected, the bad record is never read, in any mode.
        let mut skipped = req.clone();
        skipped.selection = Some(RowSelection::excluding(3, [1]));
        for o in [&o, &permissive, &drop] {
            let got = run(o, &skipped).unwrap();
            assert_eq!(a_of(&got), vec![Some(1), Some(3)], "{name}");
            assert_eq!(index_of(&got), vec![0, 2], "{name}");
        }
    }
}

/// Map entries are written in key order (the last of equal keys wins) for
/// every UTF-8 key layout, so a file does not depend on the storage order.
#[test]
fn map_entries_are_written_in_key_order() {
    use arrow_array::{LargeStringArray, StringArray};
    let env = Env::new();
    let plain = env.options(&[("compression", "uncompressed")]);
    type Keys = fn(&[&str]) -> ArrayRef;
    let layouts: [(DataType, Keys); 3] = [
        (DataType::Utf8, |k| Arc::new(StringArray::from(k.to_vec()))),
        (DataType::LargeUtf8, |k| {
            Arc::new(LargeStringArray::from(k.to_vec()))
        }),
        (DataType::Utf8View, |k| {
            Arc::new(StringViewArray::from(k.to_vec()))
        }),
    ];
    // Row 0 in two storage orders (`a`=1, `b`=2, `c`=3), row 1 with a
    // repeated key.
    let orders: [&[(&str, i32)]; 2] = [
        &[("b", 2), ("c", 3), ("a", 1), ("a", 9), ("a", 1), ("x", 0)],
        &[("c", 3), ("a", 1), ("x", 0), ("b", 2), ("a", 9), ("a", 1)],
    ];
    for (key_type, keys) in layouts {
        let entry_fields = Fields::from(vec![
            Field::new("key", key_type.clone(), false),
            Field::new("value", DataType::Int32, false),
        ]);
        let map_type = DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(entry_fields.clone()),
                false,
            )),
            false,
        );
        let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("m", map_type, false)]));
        let mut files = Vec::new();
        for order in orders {
            // Row 0: `a`, `b`, `c` (three entries); row 1: `a` twice.
            let row0: Vec<(&str, i32)> = order
                .iter()
                .copied()
                .filter(|(k, v)| *k != "x" && !(*k == "a" && *v == 9))
                .collect();
            let row1 = [("a", 5), ("a", 9)];
            let all: Vec<(&str, i32)> = row0.iter().copied().chain(row1).collect();
            let key_names: Vec<&str> = all.iter().map(|(k, _)| *k).collect();
            let entries = StructArray::new(
                entry_fields.clone(),
                vec![
                    keys(&key_names),
                    Arc::new(Int32Array::from(
                        all.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
                    )) as ArrayRef,
                ],
                None,
            );
            let map = arrow_array::MapArray::new(
                Arc::new(Field::new(
                    "entries",
                    DataType::Struct(entry_fields.clone()),
                    false,
                )),
                OffsetBuffer::new(
                    vec![
                        0_i32,
                        i32::try_from(row0.len()).unwrap(),
                        i32::try_from(all.len()).unwrap(),
                    ]
                    .into(),
                ),
                entries,
                None,
                false,
            );
            let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(map)]).unwrap();
            let bytes = env.encode(&schema, &[batch], &plain);
            let file = env.put(&format!("map-{key_type}.avro"), bytes.clone());
            let meta = env.metadata(&file, &plain);
            let any: &dyn std::any::Any = meta.as_ref();
            let meta = any.downcast_ref::<super::super::AvroMetadata>().unwrap();
            let data = &meta.blocks[0].data;
            files.push(
                bytes[usize::try_from(data.start).unwrap()..usize::try_from(data.end).unwrap()]
                    .to_vec(),
            );
        }
        // A count (zigzag 3), the three entries as key and value (zigzag),
        // the closing zero block; then one entry (`a`: 9) and its zero block.
        let expected: Vec<u8> = vec![6, 2, b'a', 2, 2, b'b', 4, 2, b'c', 6, 0, 2, 2, b'a', 18, 0];
        assert_eq!(files[0], expected, "{key_type}");
        assert_eq!(files[1], expected, "{key_type}");
    }
}

/// An Arrow decimal is rescaled to the scale of the Avro schema (exactly, or
/// the write fails), checked against its precision and sign-extended to a
/// `fixed` size.
#[test]
fn decimals_are_rescaled_to_the_avro_schema() {
    let env = Env::new();
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "d",
        DataType::Decimal128(10, 4),
        true,
    )]));
    let batch = |values: Vec<Option<i128>>| {
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(
                Decimal128Array::from(values)
                    .with_precision_and_scale(10, 4)
                    .unwrap(),
            )],
        )
        .unwrap()
    };
    let wide = [("inexact", 12345), ("too wide", 9_999_999_900)];
    for (label, decimal_type, extra) in [
        (
            "bytes",
            r#"{"type":"bytes","logicalType":"decimal","precision":6,"scale":2}"#,
            vec![],
        ),
        (
            "fixed",
            r#"{"type":"fixed","name":"dec","size":3,"logicalType":"decimal","precision":6,"scale":2}"#,
            vec![],
        ),
        (
            // 1000.00 needs three bytes.
            "short fixed",
            r#"{"type":"fixed","name":"dec","size":2,"logicalType":"decimal","precision":6,"scale":2}"#,
            vec![("no room", 10_000_000)],
        ),
    ] {
        let avro_schema = format!(
            r#"{{"type":"record","name":"r","fields":[{{"name":"d","type":["null",{decimal_type}]}}]}}"#
        );
        let o = env.options(&[("avroSchema", &avro_schema)]);
        let good = batch(vec![Some(12300), Some(-4500), Some(0), None]);
        let file = env.put(
            &format!("dec-{label}.avro"),
            env.encode(&schema, &[good], &o),
        );
        let inferred = env.infer(&file, &o);
        let req = ScanRequest::all(Arc::clone(&inferred));
        let out_schema = mapped_schema(&inferred, &req);
        let got = concat(&out_schema, &env.scan(&file, &req, &o).unwrap());
        assert_eq!(
            got.column(0).data_type(),
            &DataType::Decimal128(6, 2),
            "{label}"
        );
        assert_eq!(
            got.column(0)
                .as_primitive::<Decimal128Type>()
                .iter()
                .collect::<Vec<_>>(),
            vec![Some(123), Some(-45), Some(0), None],
            "{label}"
        );
        // A value that is not a whole number of the Avro scale, has more
        // digits than the Avro precision or does not fit the `fixed` size is
        // a write error.
        for (what, value) in wide.into_iter().chain(extra) {
            let mut enc = AvroEncoder::new(&schema, &o).unwrap();
            let e = enc.write(&batch(vec![Some(value)])).unwrap_err();
            assert!(
                matches!(e, FormatError::WriteFailed(_)),
                "{label} {what}: {e:?}"
            );
        }
    }
}

/// The sync marker depends on the write job's seed, not only the schema, so
/// it cannot be computed from the schema; a job's file stays reproducible.
#[test]
fn sync_marker_comes_from_the_write_job_seed() {
    let env = Env::new();
    let o = env.options(&[]);
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![Some(1), Some(2)]))],
    )
    .unwrap();
    let write = |o: &FormatOptions| env.encode(&schema, std::slice::from_ref(&batch), o);
    let a = write(&o.clone().with_write_seed("job-a/0/0"));
    assert_eq!(a, write(&o.clone().with_write_seed("job-a/0/0")));
    let b = write(&o.clone().with_write_seed("job-b/0/0"));
    let c = write(&o.clone().with_write_seed("job-a/0/1"));
    let none = write(&o);
    assert_eq!(none, write(&o));
    for other in [&b, &c, &none] {
        assert_ne!(&a, other);
    }
    // Each file still reads back with the marker its header names.
    for bytes in [a, b, c, none] {
        let file = env.put("seeded.avro", bytes);
        let got = env
            .scan(&file, &ScanRequest::all(Arc::clone(&schema)), &o)
            .unwrap();
        assert_eq!(rows(&got), 2);
    }
}

/// An empty block (a count of zero) is a valid frame: the blocks after it
/// are still scanned and read.
#[test]
fn an_empty_block_does_not_end_the_block_scan() {
    let env = Env::new();
    let plain = env.options(&[("compression", "uncompressed")]);
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![Some(1), Some(2), Some(3)]))],
    )
    .unwrap();
    let mut enc = AvroEncoder::new(&schema, &plain).unwrap();
    let mut two = Vec::new();
    for _ in 0..2 {
        enc.write(&batch).unwrap();
        enc.flush().unwrap();
        while let Some(bytes) = enc.take_bytes() {
            two.extend_from_slice(&bytes);
        }
    }
    two.extend_from_slice(&Box::new(enc).finish().unwrap().tail);
    let meta = env.metadata(&env.put("two.avro", two.clone()), &plain);
    let any: &dyn std::any::Any = meta.as_ref();
    let meta = any.downcast_ref::<super::super::AvroMetadata>().unwrap();
    let at = usize::try_from(meta.blocks[1].bytes.start).unwrap();
    let mut spliced = two[..at].to_vec();
    spliced.extend_from_slice(&[0, 0]);
    spliced.extend_from_slice(&meta.sync);
    spliced.extend_from_slice(&two[at..]);
    let file = env.put("empty-block.avro", spliced);
    let meta = env.metadata(&file, &plain);
    let any: &dyn std::any::Any = meta.as_ref();
    let meta = any.downcast_ref::<super::super::AvroMetadata>().unwrap();
    assert_eq!(
        meta.blocks.iter().map(|b| b.count).collect::<Vec<_>>(),
        vec![3, 0, 3]
    );
    assert_eq!(meta.rows, 6);
    let mut req = ScanRequest::all(Arc::clone(&schema));
    req.metadata_cols.row_index = true;
    let got = env.scan(&file, &req, &plain).unwrap();
    assert_eq!(rows(&got), 6);
}

/// The metadata pass reads the header and, per block, its sync marker and
/// the next block's header; never the records.
#[test]
fn the_block_index_does_not_read_the_records() {
    let env = Env::new();
    let o = env.options(&[("compression", "uncompressed")]);
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8View, true)]));
    let text = "x".repeat(1000);
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringViewArray::from(vec![
            Some(text.as_str());
            4000
        ]))],
    )
    .unwrap();
    let file = env.put("big.avro", env.encode(&schema, &[batch], &o));
    assert!(file.size > 1 << 20, "a file of several megabytes");
    let before = crate::io::IoCounters::get(&env.io.counters().bytes_read);
    let meta = env.metadata(&file, &o);
    let read = crate::io::IoCounters::get(&env.io.counters().bytes_read) - before;
    let any: &dyn std::any::Any = meta.as_ref();
    let meta = any.downcast_ref::<super::super::AvroMetadata>().unwrap();
    assert!(meta.blocks.len() > 4, "{} blocks", meta.blocks.len());
    assert!(
        read < 200 << 10,
        "{read} bytes read for the metadata of {} bytes",
        file.size
    );
}

/// Defaults of fields the file lacks: a union default is the selected
/// branch (its own index), and `bytes`, `fixed` and decimal defaults are
/// ISO-8859-1 strings with one code point per byte.
#[test]
fn defaults_follow_the_avro_specification() {
    use super::super::schema::field_default;
    use apache_avro::types::Value as V;
    use serde_json::json;
    let parse = |json: &str| AvroSchema::parse_str(json).unwrap();
    // A union default belongs to the non-null branch, wherever it is.
    for (union, branch) in [(r#"["string","null"]"#, 0), (r#"["null","string"]"#, 1)] {
        let got = field_default(&parse(union), Some(&json!("hi"))).unwrap();
        assert_eq!(
            got,
            V::Union(branch, Box::new(V::String("hi".to_owned()))),
            "{union}"
        );
    }
    let null = field_default(&parse(r#"["string","null"]"#), Some(&json!(null))).unwrap();
    assert_eq!(null, V::Union(1, Box::new(V::Null)));
    // One code point per byte: U+00FF is the byte 0xFF.
    assert_eq!(
        field_default(&parse(r#""bytes""#), Some(&json!("\u{ff}\u{0}a"))).unwrap(),
        V::Bytes(vec![0xFF, 0x00, b'a'])
    );
    assert_eq!(
        field_default(
            &parse(r#"{"type":"fixed","name":"f","size":2}"#),
            Some(&json!("\u{ff}\u{1}"))
        )
        .unwrap(),
        V::Fixed(2, vec![0xFF, 0x01])
    );
    let decimal = parse(r#"{"type":"bytes","logicalType":"decimal","precision":9,"scale":2}"#);
    let V::Decimal(d) = field_default(&decimal, Some(&json!("\u{ff}\u{38}"))).unwrap() else {
        panic!("not a decimal");
    };
    assert_eq!(Vec::<u8>::try_from(&d).unwrap(), vec![0xFF, 0x38]);
    // Above U+00FF is not a byte; the fixed size counts bytes, not UTF-8.
    for (schema, default) in [
        (r#""bytes""#, "\u{100}"),
        (r#"{"type":"fixed","name":"f","size":2}"#, "\u{100}a"),
        (r#"{"type":"fixed","name":"f","size":2}"#, "\u{ff}"),
    ] {
        assert!(
            field_default(&parse(schema), Some(&json!(default))).is_err(),
            "{schema} {default:?}"
        );
    }
    // Read end to end: a file without the field gets the default.
    let env = Env::new();
    let file_schema = r#"{"type":"record","name":"r","fields":[{"name":"a","type":"int"}]}"#;
    let file = env.put(
        "defaults.avro",
        apache_write(file_schema, &[V::Record(vec![("a".to_owned(), V::Int(1))])]),
    );
    let reader = r#"{"type":"record","name":"r","fields":[
        {"name":"a","type":"int"},
        {"name":"b","type":"bytes","default":"\u00ff"},
        {"name":"u","type":["string","null"],"default":"hi"}
    ]}"#;
    let o = env.options(&[("avroSchema", reader)]);
    let inferred = env.infer(&file, &o);
    let req = ScanRequest::all(Arc::clone(&inferred));
    let out_schema = mapped_schema(&inferred, &req);
    let got = concat(&out_schema, &env.scan(&file, &req, &o).unwrap());
    assert_eq!(
        got.column(1).as_binary_view().iter().collect::<Vec<_>>(),
        vec![Some([0xFF].as_slice())]
    );
    assert_eq!(
        got.column(2).as_string_view().iter().collect::<Vec<_>>(),
        vec![Some("hi")]
    );
}

/// The file writer seeds each file's encoder with its job, task and file
/// number, so the same rows written by two jobs carry different markers.
#[test]
fn written_files_are_seeded_by_their_job() {
    let env = Env::new();
    let o = env.options(&[]);
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![Some(1), Some(2)]))],
    )
    .unwrap();
    let mut syncs = Vec::new();
    for job in ["job-a", "job-a", "job-b"] {
        let files = env.rt.block_on(async {
            let pool: Arc<dyn krishiv_memory::MemoryPool> =
                krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
            let id = pool.register("write", None);
            let mut writer = crate::FileWriter::new(
                &env.io,
                Arc::new(AvroFormat),
                &schema,
                o.clone(),
                crate::WriteOptions {
                    store: StoreId::new(STORE),
                    dir: Path::from(job),
                    task: 0,
                    job_uuid: job.to_owned(),
                    max_records_per_file: 0,
                    target_file_bytes: 0,
                    partition_by: Vec::new(),
                    partition_names: Vec::new(),
                    zone: Some(0),
                    zone_name: None,
                },
                krishiv_memory::Reservation::new(pool, id),
            )
            .unwrap();
            writer.write(&batch).await.unwrap();
            writer.finish().await.unwrap()
        });
        let meta = env.rt.block_on(async {
            object_store::ObjectStoreExt::head(env.store.as_ref(), &files[0].path)
                .await
                .unwrap()
        });
        let file = FileRef::from_meta(StoreId::new(STORE), &meta);
        let meta = env.metadata(&file, &o);
        let any: &dyn std::any::Any = meta.as_ref();
        syncs.push(
            any.downcast_ref::<super::super::AvroMetadata>()
                .unwrap()
                .sync,
        );
    }
    assert_eq!(syncs[0], syncs[1]);
    assert_ne!(syncs[0], syncs[2]);
}

/// Every `schema_kind` name, the `union_shape` edges and the writer
/// fragments of the remaining Arrow types: Spark's documented Avro mapping
/// (`SchemaConverters`, including its union rules).
#[test]
fn avro_schema_utilities_cover_every_kind_and_error() {
    use super::super::schema::{UnionShape, schema_kind, union_shape};
    let kinds: &[(&str, &str)] = &[
        (r#""null""#, "null"),
        (r#""boolean""#, "boolean"),
        (r#""int""#, "int"),
        (r#""long""#, "long"),
        (r#""float""#, "float"),
        (r#""double""#, "double"),
        (r#""bytes""#, "bytes"),
        (r#""string""#, "string"),
        (r#"{"type":"array","items":"int"}"#, "array"),
        (r#"{"type":"map","values":"int"}"#, "map"),
        (r#"["int","string"]"#, "union"),
        (r#"{"type":"record","name":"r","fields":[]}"#, "record"),
        (r#"{"type":"enum","name":"e","symbols":["A"]}"#, "enum"),
        (r#"{"type":"fixed","name":"f","size":2}"#, "fixed"),
        (
            r#"{"type":"bytes","logicalType":"decimal","precision":4,"scale":0}"#,
            "decimal",
        ),
        (r#""bigdecimal""#, "bigdecimal"),
        (r#"{"type":"string","logicalType":"uuid"}"#, "uuid"),
        (r#"{"type":"int","logicalType":"date"}"#, "date"),
        (
            r#"{"type":"int","logicalType":"time-millis"}"#,
            "time-millis",
        ),
        (
            r#"{"type":"long","logicalType":"time-micros"}"#,
            "time-micros",
        ),
        (
            r#"{"type":"long","logicalType":"timestamp-millis"}"#,
            "timestamp-millis",
        ),
        (
            r#"{"type":"long","logicalType":"timestamp-micros"}"#,
            "timestamp-micros",
        ),
        (
            r#"{"type":"long","logicalType":"timestamp-nanos"}"#,
            "timestamp-nanos",
        ),
        (
            r#"{"type":"long","logicalType":"local-timestamp-millis"}"#,
            "local-timestamp-millis",
        ),
        (
            r#"{"type":"long","logicalType":"local-timestamp-micros"}"#,
            "local-timestamp-micros",
        ),
        (
            r#"{"type":"long","logicalType":"local-timestamp-nanos"}"#,
            "local-timestamp-nanos",
        ),
        (
            r#"{"type":"fixed","name":"d","size":12,"logicalType":"duration"}"#,
            "duration",
        ),
    ];
    for (fragment, want) in kinds {
        // `"bigdecimal"` is not JSON the parser takes; the variant still
        // has its kind name.
        let schema = if *fragment == r#""bigdecimal""# {
            AvroSchema::BigDecimal
        } else {
            AvroSchema::parse_str(fragment).unwrap()
        };
        assert_eq!(schema_kind(&schema), *want, "{fragment}");
    }
    // A recursive reference names `ref`.
    let recursive =
        AvroSchema::parse_str(r#"{"type":"record","name":"n","fields":[{"name":"f","type":"n"}]}"#)
            .unwrap();
    let AvroSchema::Record(record) = &recursive else {
        panic!("not a record");
    };
    assert_eq!(schema_kind(&record.fields[0].schema), "ref");
    let e = fields_of_record(&recursive).unwrap_err();
    assert_eq!(
        class_of(&e),
        ErrorClass::KrishivUnsupportedFeature,
        "recursive schemas have no Spark type"
    );
    // Unions with no single branch, a pair or more than a pair: Spark maps
    // `int`+`long` to `long`, `float`+`double` to `double`, one branch to
    // itself and anything else to `member0..N` (or nothing, without
    // branches at all).
    let parse = |s: &str| AvroSchema::parse_str(s).unwrap();
    let (int, long, float, double, string) = (
        parse(r#""int""#),
        parse(r#""long""#),
        parse(r#""float""#),
        parse(r#""double""#),
        parse(r#""string""#),
    );
    assert!(matches!(union_shape(&[]), UnionShape::Members));
    assert!(matches!(
        union_shape(std::slice::from_ref(&int)),
        UnionShape::Single(_)
    ));
    assert!(matches!(
        union_shape(&[int.clone(), long]),
        UnionShape::Long
    ));
    assert!(matches!(union_shape(&[float, double]), UnionShape::Double));
    assert!(matches!(
        union_shape(&[int.clone(), string.clone()]),
        UnionShape::Members
    ));
    assert!(matches!(
        union_shape(&[int, string, parse(r#""null""#)]),
        UnionShape::Members
    ));
    // Only records are top-level tables.
    let e = fields_of_record(&parse(r#""int""#)).unwrap_err();
    assert_eq!(class_of(&e), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.to_string().contains("only records"), "{e}");
}

/// The writer fragments of the remaining Arrow types, and what the writer
/// refuses: record names are sanitized to Avro names, namespaces kept and
/// maps need UTF-8 keys.
#[test]
fn avro_writer_fragments_cover_the_remaining_arrow_types() {
    let entry = Arc::new(Field::new(
        "entries",
        DataType::Struct(
            vec![
                Field::new("key", DataType::Utf8View, false),
                Field::new("value", DataType::Int32, true),
            ]
            .into(),
        ),
        false,
    ));
    let schema = Schema::new(vec![
        // A bare `null` is already nullable: no union wraps it.
        Field::new("n", DataType::Null, false),
        Field::new("t32", DataType::Time32(TimeUnit::Millisecond), true),
        Field::new("t64", DataType::Time64(TimeUnit::Microsecond), true),
        Field::new(
            "tsms",
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            true,
        ),
        Field::new(
            "ltsms",
            DataType::Timestamp(TimeUnit::Millisecond, None),
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
        Field::new("m", DataType::Map(entry, false), true),
        Field::new("d38", DataType::Decimal128(38, 9), true),
    ]);
    // A record name outside Avro's name rules is sanitized: punctuation
    // becomes underscores and a leading digit is guarded.
    let json = schema_json_of_arrow(&schema, "9 lives!", "myns").unwrap();
    let parsed = AvroSchema::parse_str(&json).unwrap();
    let AvroSchema::Record(record) = &parsed else {
        panic!("not a record");
    };
    assert_eq!(record.fields.len(), 9);
    assert!(json.contains("\"_9_lives_\""), "{json}");
    // A map with a non-UTF-8 key and an unmapped Arrow type are refused.
    let bad_key = Schema::new(vec![Field::new(
        "m",
        DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(
                    vec![
                        Field::new("key", DataType::BinaryView, false),
                        Field::new("value", DataType::Int32, false),
                    ]
                    .into(),
                ),
                false,
            )),
            false,
        ),
        true,
    )]);
    let e = schema_json_of_arrow(&bad_key, "r", "").unwrap_err();
    assert!(e.to_string().contains("string keys"), "{e}");
    let bad_entries = Schema::new(vec![Field::new(
        "m",
        DataType::Map(
            Arc::new(Field::new("entries", DataType::Int32, false)),
            false,
        ),
        true,
    )]);
    assert!(schema_json_of_arrow(&bad_entries, "r", "").is_err());
    let duration = Schema::new(vec![Field::new(
        "d",
        DataType::Duration(TimeUnit::Millisecond),
        true,
    )]);
    let e = schema_json_of_arrow(&duration, "r", "").unwrap_err();
    assert!(e.to_string().contains("Duration"), "{e}");
}

/// Every field default the spec allows, and the corrupt ones: Spark reads a
/// missing field from its default, then null, and fails the file otherwise.
#[test]
fn avro_field_defaults_cover_every_scalar() {
    use super::super::schema::field_default;
    use apache_avro::types::Value as V;
    use serde_json::json;
    let parse = |s: &str| AvroSchema::parse_str(s).unwrap();
    // Null has no default but its error: null is only for nullable fields.
    // A non-null default for a null field is still null.
    assert!(field_default(&parse(r#""null""#), Some(&json!(null))).is_err());
    assert_eq!(
        field_default(&parse(r#""null""#), Some(&json!(5))).unwrap(),
        V::Null
    );
    let cases: &[(&str, serde_json::Value, V)] = &[
        (r#""boolean""#, json!(true), V::Boolean(true)),
        (r#""int""#, json!(5), V::Int(5)),
        (r#""long""#, json!(5), V::Long(5)),
        (r#""float""#, json!(1.5), V::Float(1.5)),
        (r#""double""#, json!(2), V::Double(2.0)),
        (r#""string""#, json!("hi"), V::String("hi".to_owned())),
        (
            r#"{"type":"enum","name":"e","symbols":["A","B"]}"#,
            json!("B"),
            V::Enum(1, "B".to_owned()),
        ),
        (
            r#"{"type":"int","logicalType":"date"}"#,
            json!(7),
            V::Date(7),
        ),
        (
            r#"{"type":"int","logicalType":"time-millis"}"#,
            json!(7),
            V::TimeMillis(7),
        ),
        (
            r#"{"type":"long","logicalType":"time-micros"}"#,
            json!(7),
            V::TimeMicros(7),
        ),
        (
            r#"{"type":"long","logicalType":"timestamp-millis"}"#,
            json!(7),
            V::TimestampMillis(7),
        ),
        (
            r#"{"type":"long","logicalType":"timestamp-micros"}"#,
            json!(7),
            V::TimestampMicros(7),
        ),
        (
            r#"{"type":"long","logicalType":"local-timestamp-millis"}"#,
            json!(7),
            V::LocalTimestampMillis(7),
        ),
        (
            r#"{"type":"long","logicalType":"local-timestamp-micros"}"#,
            json!(7),
            V::LocalTimestampMicros(7),
        ),
        (
            r#"{"type":"array","items":"int"}"#,
            json!([1, 2]),
            V::Array(vec![V::Int(1), V::Int(2)]),
        ),
        (
            r#"{"type":"map","values":"int"}"#,
            json!({"k": 3}),
            V::Map([(String::from("k"), V::Int(3))].into_iter().collect()),
        ),
    ];
    for (fragment, default, want) in cases {
        let got = field_default(&parse(fragment), Some(default)).unwrap();
        assert_eq!(&got, want, "{fragment}");
    }
    // A uuid reads from its text.
    let uuid = "123e4567-e89b-12d3-a456-426614174000";
    let got = field_default(
        &parse(r#"{"type":"string","logicalType":"uuid"}"#),
        Some(&json!(uuid)),
    )
    .unwrap();
    assert_eq!(got, V::Uuid(uuid.parse().unwrap()));
    assert!(
        field_default(
            &parse(r#"{"type":"string","logicalType":"uuid"}"#),
            Some(&json!("nope"))
        )
        .is_err()
    );
    // Out-of-range numbers and mismatched JSON fail.
    for (fragment, default) in [
        (r#""int""#, json!(9_999_999_999_i64)),
        (r#""long""#, json!(1.5)),
        (r#""int""#, json!("s")),
        (r#"{"type":"enum","name":"e","symbols":["A"]}"#, json!("Z")),
        (
            r#"{"type":"int","logicalType":"date"}"#,
            json!(9_999_999_999_i64),
        ),
        (
            r#"{"type":"long","logicalType":"timestamp-micros"}"#,
            json!("s"),
        ),
        (r#""boolean""#, json!(1)),
        (r#""double""#, json!("s")),
    ] {
        assert!(
            field_default(&parse(fragment), Some(&default)).is_err(),
            "{fragment} {default}"
        );
    }
    // Unions wider than a null and a branch take the first matching one;
    // one no branch takes is corrupt.
    let three = parse(r#"["int","string","null"]"#);
    assert_eq!(
        field_default(&three, Some(&json!("x"))).unwrap(),
        V::Union(1, Box::new(V::String("x".to_owned())))
    );
    let two = parse(r#"["int","string"]"#);
    assert_eq!(
        field_default(&two, Some(&json!(3))).unwrap(),
        V::Union(0, Box::new(V::Int(3)))
    );
    // No branch takes a float: with a null branch the null does, so the
    // corrupt default needs a union without one.
    assert!(field_default(&two, Some(&json!(1.5))).is_err());
    // Records default field by field; a missing field without a default
    // and without null is corrupt.
    let record = parse(
        r#"{"type":"record","name":"r","fields":[
            {"name":"a","type":"int","default":1},
            {"name":"u","type":["null","string"],"default":null}
        ]}"#,
    );
    let got = field_default(&record, Some(&json!({"a": 5}))).unwrap();
    assert_eq!(
        got,
        V::Record(vec![
            ("a".to_owned(), V::Int(5)),
            ("u".to_owned(), V::Union(0, Box::new(V::Null)))
        ])
    );
    let bare = parse(r#"{"type":"record","name":"r","fields":[{"name":"a","type":"int"}]}"#);
    assert!(field_default(&bare, Some(&json!({}))).is_err());
}

/// Reader schemas that do not parse, are not records or name types with
/// no Spark mapping; files with an unknown codec, a non-record writer
/// schema or a truncated block header; and inference without files.
#[test]
fn avro_reader_schema_and_header_errors() {
    let env = Env::new();
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![Some(1)]))],
    )
    .unwrap();
    let bytes = env.encode(&schema, std::slice::from_ref(&batch), &env.options(&[]));
    // An `avroSchema` that does not parse, is not a record, or names a
    // type with no Spark mapping fails inference.
    for (schema, needle) in [
        ("{nope", "does not parse"),
        (r#""int""#, "not a record"),
        (
            r#"{"type":"record","name":"r","fields":[{"name":"d","type":{"type":"fixed","name":"x","size":12,"logicalType":"duration"}}]}"#,
            "duration",
        ),
    ] {
        let o = env.options(&[("avroSchema", schema)]);
        let e = env
            .rt
            .block_on(AvroFormat.infer_schema(
                &env.io,
                std::slice::from_ref(&env.put("r.avro", bytes.clone())),
                &o,
            ))
            .unwrap_err();
        assert!(e.to_string().contains(needle), "{schema}: {e}");
    }
    // A file headed with an unknown codec names it.
    let mystery = splice_header_value(&bytes, b"avro.codec", b"wobble");
    let file = env.put("mystery.avro", mystery);
    let e = env
        .rt
        .block_on(env.io.read_metadata(
            &AvroFormat,
            &file,
            MetadataNeeds::default(),
            &env.options(&[]),
        ))
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::FailedReadFileCannotReadFileFooter
    );
    // A file whose writer schema is not a record is not a table.
    let plain = splice_header_value(&bytes, b"avro.schema", b"\"int\"");
    let file = env.put("plain.avro", plain);
    let e = env
        .rt
        .block_on(env.io.read_metadata(
            &AvroFormat,
            &file,
            MetadataNeeds::default(),
            &env.options(&[]),
        ))
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::FailedReadFileCannotReadFileFooter
    );
    // Inference without files is an internal error, never a data error.
    let e = env
        .rt
        .block_on(AvroFormat.infer_schema(&env.io, &[], &env.options(&[])))
        .unwrap_err();
    assert!(e.to_string().contains("without files"), "{e}");
    // A file cut inside the next block's header fails the footer read.
    let plain_options = env.options(&[("compression", "uncompressed")]);
    let wide: SchemaRef = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, true)]));
    let first = RecordBatch::try_new(
        Arc::clone(&wide),
        vec![Arc::new(Int64Array::from(vec![Some(1)]))],
    )
    .unwrap();
    let second = RecordBatch::try_new(
        Arc::clone(&wide),
        vec![Arc::new(Int64Array::from(
            (0..70).map(Some).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    let mut enc = AvroEncoder::new(&wide, &plain_options).unwrap();
    for batch in [&first, &second] {
        enc.write(batch).unwrap();
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
            &env.put("cut.avro", two.clone()),
            MetadataNeeds::default(),
            &plain_options,
        ))
        .unwrap();
    let any: &dyn std::any::Any = meta.as_ref();
    let end = any
        .downcast_ref::<super::super::AvroMetadata>()
        .unwrap()
        .blocks[0]
        .bytes
        .end;
    let end = usize::try_from(end).unwrap();
    // Seventy rows need two count bytes starting with a continuation byte,
    // so cutting one byte into the next header (or two, leaving the count
    // without its size) truncates the block header itself.
    assert_eq!(two[end], 0x8C, "the count must start with 0x8C");
    for cut in [end + 1, end + 2] {
        let file = env.put(&format!("cut-{cut}.avro"), two[..cut].to_vec());
        let e = env
            .rt
            .block_on(env.io.read_metadata(
                &AvroFormat,
                &file,
                MetadataNeeds::default(),
                &plain_options,
            ))
            .unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::FailedReadFileCannotReadFileFooter,
            "cut at {cut}"
        );
    }
    // A header larger than one fetch still parses: the loop keeps reading.
    let many: Vec<Field> = (0..3000)
        .map(|i| Field::new(format!("f{i:04}"), DataType::Int32, true))
        .collect();
    let wide_schema: SchemaRef = Arc::new(Schema::new(many));
    let one = RecordBatch::try_new(
        Arc::clone(&wide_schema),
        wide_schema
            .fields()
            .iter()
            .map(|_| Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef)
            .collect(),
    )
    .unwrap();
    let big = env.encode(&wide_schema, std::slice::from_ref(&one), &env.options(&[]));
    assert!(big.len() > 64 << 10, "the header must span fetches");
    let file = env.put("big.avro", big);
    let inferred = env.infer(&file, &env.options(&[]));
    assert_eq!(inferred.fields().len(), 3000);
    let req = ScanRequest::all(Arc::clone(&inferred));
    let got = env.scan(&file, &req, &env.options(&[])).unwrap();
    assert_eq!(rows(&got), 1);
}

/// The header walk: short buffers need more bytes, bad magic is rejected,
/// negative counts skip their size, huge counts and short entries fail.
#[test]
fn avro_parse_header_edge_cases() {
    use super::super::decode::{Parsed, parse_header};
    let magic = b"Obj\x01".to_vec();
    assert!(matches!(parse_header(&magic[..3]), Parsed::Need));
    assert!(matches!(parse_header(b"Xbj\x01"), Parsed::BadMagic));
    // A negative count without its byte size needs more bytes.
    let mut buf = magic.clone();
    buf.extend(zigzag(-2));
    assert!(matches!(parse_header(&buf), Parsed::Need));
    // With the size, the entries follow: one pair, then the end of the map
    // and the sync marker.
    let mut buf = magic.clone();
    buf.extend(zigzag(-1));
    buf.extend(zigzag(0));
    buf.extend(zigzag(1));
    buf.push(b'k');
    buf.extend(zigzag(1));
    buf.push(b'v');
    buf.extend(zigzag(0));
    buf.extend([9_u8; 16]);
    let Parsed::Done((meta, _)) = parse_header(&buf) else {
        panic!("want a parsed header");
    };
    assert_eq!(meta.get(b"k".as_slice()), Some(&b"v".to_vec()));
    // A map no reader would hold is not a header.
    let mut buf = magic.clone();
    buf.extend(zigzag(2_000_000));
    assert!(matches!(parse_header(&buf), Parsed::BadMagic));
    // A key cut short of its length needs more bytes.
    let mut buf = magic;
    buf.extend(zigzag(1));
    buf.extend(zigzag(4));
    buf.push(b'a');
    assert!(matches!(parse_header(&buf), Parsed::Need));
    // Entries without the sync marker need more bytes.
    let mut buf = b"Obj\x01".to_vec();
    buf.extend(zigzag(0));
    buf.extend([1, 2, 3]);
    assert!(matches!(parse_header(&buf), Parsed::Need));
}
