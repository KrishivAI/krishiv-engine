// SPDX-License-Identifier: Apache-2.0
use std::sync::Arc;

use arrow_array::builder::{ListBuilder, MapBuilder, StringViewBuilder};
use arrow_array::{
    Array, ArrayRef, BinaryViewArray, Int32Array, Int64Array, RecordBatch, StringViewArray,
    StructArray, TimestampMicrosecondArray,
};
use arrow_ipc::CompressionType;
use arrow_schema::{DataType as ArrowType, Field, Fields, Schema, TimeUnit};
use krishiv_connect::proto::v4_3::data_type::Kind;
use krishiv_types::{
    ArrayType, Collation, DataType, DayTimeField, DecimalType, MapType, Metadata, Srid,
    StructField, StructType, YearMonthField,
};

use super::arrow::{
    EncodeSettings, chunks, compression_codec, empty_batch, ipc_stream, slices, spark_batch,
};
use super::schema::{proto_schema, proto_type};

fn settings(max_rows: Option<usize>, max_bytes: usize) -> EncodeSettings {
    EncodeSettings {
        max_rows,
        max_bytes,
        compression: None,
        chunk_size: None,
        time_zone: None,
    }
}

fn kind(t: &DataType) -> Kind {
    proto_type(t).unwrap().kind.unwrap()
}

#[test]
fn types_map_as_sparks_converter_maps_them() {
    assert!(matches!(kind(&DataType::Null), Kind::Null(_)));
    assert!(matches!(kind(&DataType::Boolean), Kind::Boolean(_)));
    assert!(matches!(kind(&DataType::Byte), Kind::Byte(_)));
    assert!(matches!(kind(&DataType::Short), Kind::Short(_)));
    assert!(matches!(kind(&DataType::Integer), Kind::Integer(_)));
    assert!(matches!(kind(&DataType::Float), Kind::Float(_)));
    assert!(matches!(kind(&DataType::Double), Kind::Double(_)));
    assert!(matches!(kind(&DataType::Binary), Kind::Binary(_)));
    assert!(matches!(kind(&DataType::Date), Kind::Date(_)));
    assert!(matches!(kind(&DataType::Timestamp), Kind::Timestamp(_)));
    assert!(matches!(
        kind(&DataType::TimestampNtz),
        Kind::TimestampNtz(_)
    ));
    assert!(matches!(
        kind(&DataType::CalendarInterval),
        Kind::CalendarInterval(_)
    ));
    assert!(matches!(kind(&DataType::Variant), Kind::Variant(_)));
    let Kind::String(s) = kind(&DataType::String) else {
        panic!()
    };
    assert_eq!(s.collation, "", "the default string sends no collation");
    let Kind::String(s) = kind(&DataType::collated_string(
        Collation::parse("UNICODE_CI").unwrap(),
    )) else {
        panic!()
    };
    assert_eq!(s.collation, "UNICODE_CI");
    let Kind::Decimal(d) = kind(&DataType::Decimal(DecimalType::try_new(12, 3).unwrap())) else {
        panic!()
    };
    assert_eq!((d.precision, d.scale), (Some(12), Some(3)));
    let Kind::Time(t) = kind(&DataType::time(3).unwrap()) else {
        panic!()
    };
    assert_eq!(t.precision, Some(3));
    let Kind::Geometry(g) = kind(&DataType::geometry(Srid::Any).unwrap()) else {
        panic!()
    };
    assert_eq!(g.srid, -1);
    let Kind::Geography(g) = kind(&DataType::geography(Srid::Fixed(4326)).unwrap()) else {
        panic!()
    };
    assert_eq!(g.srid, 4326);
    let Kind::YearMonthInterval(i) =
        kind(&DataType::year_month_interval(YearMonthField::Year, YearMonthField::Month).unwrap())
    else {
        panic!()
    };
    assert_eq!((i.start_field, i.end_field), (Some(0), Some(1)));
    let Kind::DayTimeInterval(i) =
        kind(&DataType::day_time_interval(DayTimeField::Hour, DayTimeField::Second).unwrap())
    else {
        panic!()
    };
    assert_eq!((i.start_field, i.end_field), (Some(1), Some(3)));
    for (f, n) in [(DayTimeField::Day, 0), (DayTimeField::Minute, 2)] {
        let Kind::DayTimeInterval(i) = kind(&DataType::day_time_interval(f, f).unwrap()) else {
            panic!()
        };
        assert_eq!(i.start_field, Some(n));
    }
    let Kind::Array(a) = kind(&ArrayType::new(DataType::Long, false).into()) else {
        panic!()
    };
    assert!(!a.contains_null);
    assert!(matches!(a.element_type.unwrap().kind, Some(Kind::Long(_))));
    let Kind::Map(m) = kind(&MapType::new(DataType::String, DataType::Integer, true).into()) else {
        panic!()
    };
    assert!(m.value_contains_null);
    let mut map = serde_json::Map::new();
    map.insert("comment".to_owned(), "c".into());
    let st = StructType::new(vec![
        StructField::new("a", DataType::Integer, false),
        StructField::new("b", DataType::String, true).with_metadata(Metadata::from_map(map)),
    ]);
    let Kind::Struct(s) = kind(&st.into()) else {
        panic!()
    };
    assert_eq!(s.fields[0].metadata, None, "empty metadata is not sent");
    assert_eq!(s.fields[1].metadata.as_deref(), Some(r#"{"comment":"c"}"#));
    assert!(s.fields[1].nullable);
}

#[test]
fn a_result_schema_reads_the_spark_type_of_each_field() {
    let schema = Schema::new(vec![
        Field::new("id", ArrowType::Int64, false),
        Field::new("s", ArrowType::Utf8View, true),
    ]);
    let Some(Kind::Struct(s)) = proto_schema(&schema).unwrap().kind else {
        panic!()
    };
    assert_eq!(s.fields.len(), 2);
    assert!(matches!(
        s.fields[1].data_type.as_ref().unwrap().kind,
        Some(Kind::String(_))
    ));
}

fn column_batch(name: &str, column: ArrayRef) -> RecordBatch {
    let field = Field::new(name, column.data_type().clone(), true);
    RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![column]).unwrap()
}

#[test]
fn views_become_utf8_and_binary_also_when_nested() {
    let strings: ArrayRef = Arc::new(StringViewArray::from(vec![Some("a"), None]));
    let out = spark_batch(&column_batch("s", strings), None).unwrap();
    assert_eq!(out.schema().field(0).data_type(), &ArrowType::Utf8);
    assert_eq!(out.column(0).null_count(), 1);

    let bins: ArrayRef = Arc::new(BinaryViewArray::from(vec![Some(&b"x"[..]), None]));
    let out = spark_batch(&column_batch("b", bins), None).unwrap();
    assert_eq!(out.schema().field(0).data_type(), &ArrowType::Binary);

    let mut list = ListBuilder::new(StringViewBuilder::new());
    list.values().append_value("x");
    list.append(true);
    list.append(false);
    let list: ArrayRef = Arc::new(list.finish());
    let out = spark_batch(&column_batch("l", list), None).unwrap();
    let ArrowType::List(item) = out.schema().field(0).data_type().clone() else {
        panic!()
    };
    assert_eq!(item.data_type(), &ArrowType::Utf8);
    assert_eq!(out.column(0).null_count(), 1);

    let mut map = MapBuilder::new(None, StringViewBuilder::new(), StringViewBuilder::new());
    map.keys().append_value("k");
    map.values().append_value("v");
    map.append(true).unwrap();
    let map: ArrayRef = Arc::new(map.finish());
    let out = spark_batch(&column_batch("m", map), None).unwrap();
    let ArrowType::Map(entries, _) = out.schema().field(0).data_type().clone() else {
        panic!()
    };
    let ArrowType::Struct(kv) = entries.data_type() else {
        panic!()
    };
    assert_eq!(kv[0].data_type(), &ArrowType::Utf8);
    assert_eq!(kv[1].data_type(), &ArrowType::Utf8);

    let st: ArrayRef = Arc::new(StructArray::new(
        Fields::from(vec![Field::new("x", ArrowType::Utf8View, true)]),
        vec![Arc::new(StringViewArray::from(vec!["q"])) as ArrayRef],
        None,
    ));
    let out = spark_batch(&column_batch("st", st), None).unwrap();
    let ArrowType::Struct(f) = out.schema().field(0).data_type().clone() else {
        panic!()
    };
    assert_eq!(f[0].data_type(), &ArrowType::Utf8);
}

#[test]
fn timestamps_carry_the_session_time_zone() {
    let ts: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![0_i64]).with_timezone("UTC"));
    let tz: Arc<str> = Arc::from("Asia/Kolkata");
    let out = spark_batch(&column_batch("t", Arc::clone(&ts)), Some(&tz)).unwrap();
    assert_eq!(
        out.schema().field(0).data_type(),
        &ArrowType::Timestamp(TimeUnit::Microsecond, Some(Arc::clone(&tz)))
    );
    let kept = spark_batch(&column_batch("t", ts), None).unwrap();
    assert_eq!(
        kept.schema().field(0).data_type(),
        &ArrowType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
    );
    let ntz: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![0_i64]));
    let out = spark_batch(&column_batch("n", ntz), Some(&tz)).unwrap();
    assert_eq!(
        out.schema().field(0).data_type(),
        &ArrowType::Timestamp(TimeUnit::Microsecond, None)
    );
}

#[test]
fn variant_and_spatial_columns_take_sparks_shape() {
    use krishiv_types::arrow::{VARIANT_METADATA_FIELD, VARIANT_VALUE_FIELD};
    let variant = DataType::Variant.to_arrow_field("v", true);
    let ArrowType::Struct(children) = variant.data_type().clone() else {
        panic!("{variant:?}")
    };
    let columns: Vec<ArrayRef> = children
        .iter()
        .map(|_| Arc::new(BinaryViewArray::from(vec![&b"\x01"[..]])) as ArrayRef)
        .collect();
    let array: ArrayRef = Arc::new(StructArray::new(children, columns, None));
    let batch = RecordBatch::try_new(Arc::new(Schema::new(vec![variant])), vec![array]).unwrap();
    let out = spark_batch(&batch, None).unwrap();
    let field = out.schema().field(0).clone();
    assert_eq!(
        field.metadata().get("variant").map(String::as_str),
        Some("true")
    );
    let ArrowType::Struct(f) = field.data_type() else {
        panic!()
    };
    assert_eq!(f[0].name(), VARIANT_VALUE_FIELD);
    assert_eq!(f[1].name(), VARIANT_METADATA_FIELD);

    let geometry = DataType::geometry(Srid::Fixed(4326)).unwrap();
    let field = geometry.to_arrow_field("g", true);
    let ArrowType::Struct(children) = field.data_type().clone() else {
        panic!("{field:?}")
    };
    let array: ArrayRef = Arc::new(StructArray::new(
        children,
        vec![
            Arc::new(Int32Array::from(vec![4326])) as ArrayRef,
            Arc::new(BinaryViewArray::from(vec![&b"wkb"[..]])) as ArrayRef,
        ],
        None,
    ));
    let batch = RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![array]).unwrap();
    let out = spark_batch(&batch, None).unwrap();
    let ArrowType::Struct(f) = out.schema().field(0).data_type().clone() else {
        panic!()
    };
    assert_eq!(f[1].data_type(), &ArrowType::Binary);
}

#[test]
fn slices_cut_at_rows_and_bytes() {
    let ids: ArrayRef = Arc::new(Int64Array::from_iter_values(0..25_000));
    let batch = column_batch("id", ids);
    let rows: Vec<usize> = slices(&batch, &settings(Some(10_000), usize::MAX))
        .iter()
        .map(RecordBatch::num_rows)
        .collect();
    assert_eq!(rows, [10_000, 10_000, 5_000]);
    // About 8 bytes a row: 8,000 bytes is about 1,000 rows.
    let by_size = slices(&batch, &settings(None, 8_000));
    assert!(by_size.len() >= 25, "{}", by_size.len());
    assert!(by_size.iter().all(|b| b.num_rows() <= 1_000));
    // The first row always goes.
    let one = slices(&batch, &settings(None, 1));
    assert_eq!(one[0].num_rows(), 1);
    assert!(slices(&batch.slice(0, 0), &settings(None, 1)).is_empty());
}

#[test]
fn ipc_streams_chunks_and_codecs() {
    let ids: ArrayRef = Arc::new(Int64Array::from_iter_values(0..100));
    let batch = column_batch("id", ids);
    for codec in [
        None,
        Some(CompressionType::ZSTD),
        Some(CompressionType::LZ4_FRAME),
    ] {
        let bytes = ipc_stream(&batch, codec).unwrap();
        let read: Vec<RecordBatch> =
            arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(bytes), None)
                .unwrap()
                .collect::<Result<_, _>>()
                .unwrap();
        assert_eq!(read, vec![batch.clone()]);
    }
    assert_eq!(compression_codec(Some("ZSTD")), Some(CompressionType::ZSTD));
    assert_eq!(
        compression_codec(Some("lz4")),
        Some(CompressionType::LZ4_FRAME)
    );
    assert_eq!(compression_codec(Some("none")), None);
    assert_eq!(compression_codec(None), None);
    assert_eq!(
        chunks(&[1, 2, 3, 4, 5], 2),
        vec![&[1, 2][..], &[3, 4], &[5]]
    );
    let none: &[u8] = &[];
    assert_eq!(chunks(none, 2), vec![none]);
    let empty = empty_batch(&batch.schema(), None).unwrap();
    assert_eq!(empty.num_rows(), 0);
}

fn running_op() -> crate::operation::Operation {
    crate::operation::Operation::new(
        crate::operation::new_operation_id().unwrap(),
        [0; 32],
        Arc::from(Vec::new()),
        true,
    )
}

fn failure() -> krishiv_common::error::KrishivError {
    krishiv_common::error::InternalError {
        message: "boom".to_owned(),
    }
    .into()
}

/// How the stream of `op` ends once its responses are sent.
fn end_of(op: &crate::operation::Operation) -> Option<tonic::Status> {
    let g = op.replay().attach(None);
    match op.replay().next(g, 1, true) {
        crate::operation::replay::Next::Finished(end) => end,
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_failed_operation_is_finished_before_its_error_is_sent() {
    let op = running_op();
    super::settle(&op, &errors(), Err(super::Stop::Failed(failure())));
    let status = end_of(&op).unwrap();
    assert!(status.message().contains("INTERNAL_ERROR"), "{status:?}");
    // Its error is recorded: an interrupt now interrupts nothing.
    assert!(!op.cancel());
}

#[test]
fn a_closed_operation_finishes_its_stream() {
    let op = running_op();
    super::settle(&op, &errors(), Err(super::Stop::Gone));
    assert!(end_of(&op).is_none());
    assert!(!op.cancel());
}

#[test]
fn an_interrupt_before_the_end_wins_over_the_error() {
    // Spark: an error raised after the interrupt is replaced by
    // OPERATION_CANCELED (`ExecuteThreadRunner.scala:150-178`).
    let op = running_op();
    assert!(op.cancel());
    super::settle(&op, &errors(), Err(super::Stop::Failed(failure())));
    let status = end_of(&op).unwrap();
    assert!(
        status.message().contains("OPERATION_CANCELED"),
        "{status:?}"
    );
}

/// A session's error cache, for `settle`.
fn errors() -> crate::errors::SessionErrors {
    crate::errors::SessionErrors::new(krishiv_session::Session::new().unwrap().conf())
}

/// Review M1: a setup that fails is `FAILED`; one whose RPC was dropped
/// (the client cancelled during analysis) is `CANCELLED`.
#[test]
fn a_setup_failure_is_failed_and_a_dropped_setup_cancelled() {
    use crate::operation::{OperationState, Operations};
    let operations = Arc::new(Operations::default());
    let failed = Arc::new(running_op());
    operations.register(&failed, 10).unwrap();
    let id = failed.id().as_str().to_owned();
    super::Registered {
        operations: Arc::clone(&operations),
        op: Some(Arc::clone(&failed)),
    }
    .fail();
    assert!(operations.is_empty());
    assert_eq!(operations.state_of(&id), Some(OperationState::Failed));

    let operations = Arc::new(Operations::default());
    let dropped = Arc::new(running_op());
    operations.register(&dropped, 10).unwrap();
    drop(super::Registered {
        operations: Arc::clone(&operations),
        op: Some(Arc::clone(&dropped)),
    });
    assert!(operations.is_empty());
    assert_eq!(
        operations.state_of(dropped.id().as_str()),
        Some(OperationState::Cancelled)
    );
    assert!(dropped.replay().is_closed());
}
