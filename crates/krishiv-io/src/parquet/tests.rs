// SPDX-License-Identifier: Apache-2.0
//! Parquet reader tests over files written by arrow-rs (and by its
//! low-level writer, for what `ArrowWriter` refuses to write). The generic
//! conformance suite and the interoperability checks live in
//! `krishiv-testkit`.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Int16Type, Int32Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, Int32Array, RecordBatch, StringArray, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_common::error::{ErrorClass, KrishivError};
use object_store::ObjectStore;
use object_store::memory::InMemory;
use object_store::path::Path;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use super::ParquetFormat;
use crate::request::{CmpOp, ColumnPath, FieldMapping, PushdownExpr, RowSelection, ScalarValue};
use crate::{
    ByteRange, FIELD_ID_KEY, FileFormat, FileRef, FormatOptions, IoConfig, IoContext, IoScheduler,
    MetadataNeeds, ScanOptions, ScanRequest, ScanStream, StoreId,
};

pub(super) struct Env {
    pub(super) rt: tokio::runtime::Runtime,
    pub(super) io: IoContext,
    store: Arc<InMemory>,
}

const STORE: &str = "memory://pq";

impl Env {
    pub(super) fn new() -> Self {
        Self::with_config(IoConfig::default())
    }

    fn with_config(config: IoConfig) -> Self {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()
            .unwrap();
        let store = Arc::new(InMemory::new());
        let sched = IoScheduler::new(config);
        sched.register_store(
            StoreId::new(STORE),
            Arc::clone(&store) as Arc<dyn ObjectStore>,
        );
        Self {
            rt,
            io: sched.context(),
            store,
        }
    }

    pub(super) fn put(&self, name: &str, bytes: Vec<u8>) -> FileRef {
        self.rt.block_on(async {
            let path = Path::from(name);
            object_store::ObjectStoreExt::put(self.store.as_ref(), &path, bytes.into())
                .await
                .unwrap();
            let meta = object_store::ObjectStoreExt::head(self.store.as_ref(), &path)
                .await
                .unwrap();
            FileRef::from_meta(StoreId::new(STORE), &meta)
        })
    }

    pub(super) fn options(&self, kv: &[(&str, &str)]) -> FormatOptions {
        self.options_over(&[], kv)
    }

    /// The options of `kv` over a session that holds `session`.
    pub(super) fn options_over(
        &self,
        session: &[(&str, &str)],
        kv: &[(&str, &str)],
    ) -> FormatOptions {
        let (conf, _) =
            krishiv_common::config::Config::from_pairs(session.iter().copied()).unwrap();
        let user = kv
            .iter()
            .map(|(k, v)| {
                (
                    (*k).to_owned(),
                    krishiv_common::secret::OptionValue::from_user(k, (*v).to_owned()),
                )
            })
            .collect();
        ParquetFormat.options().parse(&user, &conf).unwrap().0
    }

    pub(super) fn scan(
        &self,
        file: &FileRef,
        req: &ScanRequest,
        o: &FormatOptions,
    ) -> Result<Vec<RecordBatch>, KrishivError> {
        Ok(self.scan_metered(file, req, o)?.0)
    }

    /// As [`Self::scan`], also returning the decoder's counters.
    pub(super) fn scan_metered(
        &self,
        file: &FileRef,
        req: &ScanRequest,
        o: &FormatOptions,
    ) -> Result<(Vec<RecordBatch>, crate::ScanUnitMetrics), KrishivError> {
        self.rt.block_on(async {
            let pool: Arc<dyn krishiv_memory::MemoryPool> =
                krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
            let id = pool.register("scan", None);
            let mut s = ScanStream::open(
                &self.io,
                &ParquetFormat,
                file.clone(),
                ByteRange::whole(file.size),
                req,
                o,
                krishiv_memory::Reservation::new(pool, id),
                ScanOptions::default(),
            )
            .await?;
            let mut out = Vec::new();
            while let Some(b) = s.next_batch().await? {
                out.push(b);
            }
            Ok((out, s.metrics()))
        })
    }
}

pub(super) fn write(batches: &[RecordBatch], props: WriterProperties) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut w = ArrowWriter::try_new(&mut buf, batches[0].schema(), Some(props)).unwrap();
    for b in batches {
        w.write(b).unwrap();
    }
    w.close().unwrap();
    buf
}

pub(super) fn rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

fn ints(name: &str, values: impl IntoIterator<Item = i32>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int32, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from_iter_values(values))]).unwrap()
}

/// Acceptance (P2-T080a): a range predicate on a sorted key skips at least
/// 90% of the row groups.
#[test]
fn sorted_key_ranges_skip_most_row_groups() {
    let env = Env::new();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(1000))
        .build();
    let bytes = write(&[ints("k", 0..100_000)], props);
    let file = env.put("sorted.parquet", bytes);
    let o = env.options(&[]);
    let meta = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), &o),
        )
        .unwrap();
    let schema = meta.schema();
    let mut req = ScanRequest::all(Arc::clone(&schema));
    req.predicate = Some(PushdownExpr::And(vec![
        PushdownExpr::cmp(
            ColumnPath::new("k"),
            CmpOp::GtEq,
            ScalarValue::Int32(41_500),
        ),
        PushdownExpr::cmp(ColumnPath::new("k"), CmpOp::Lt, ScalarValue::Int32(45_000)),
    ]));
    let plan = ParquetFormat
        .plan_units(meta.as_ref(), &ByteRange::whole(file.size), &req)
        .unwrap();
    assert_eq!(plan.owned, 100);
    assert_eq!(plan.units.len(), 4, "row groups 41..=44 hold the range");
    assert!(plan.pruned_by_statistics * 10 >= plan.owned * 9);
    let out = env.scan(&file, &req, &o).unwrap();
    assert_eq!(rows(&out), 4000);
    let first = out[0].column(0).as_primitive::<Int32Type>().value(0);
    assert_eq!(first, 41_000);
}

/// `TIMESTAMP(MILLIS)` columns (Spark's `outputTimestampType=TIMESTAMP_MILLIS`)
/// read as Spark reads them: microsecond timestamps, with and without a
/// zone; their statistics prune in microseconds too.
#[test]
fn millisecond_timestamps_read_as_microseconds() {
    use arrow_array::TimestampMillisecondArray;
    use arrow_array::types::TimestampMicrosecondType;
    use arrow_schema::TimeUnit;

    let env = Env::new();
    let millis: Vec<i64> = (0..4000).map(|i| i * 1000 - 1_999_999).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            false,
        ),
        Field::new(
            "ntz",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(TimestampMillisecondArray::from(millis.clone()).with_timezone("UTC")),
            Arc::new(TimestampMillisecondArray::from(millis.clone())),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(1000))
        .build();
    let file = env.put("ms.parquet", write(&[batch], props));
    let o = env.options(&[]);
    let meta = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), &o),
        )
        .unwrap();
    let schema = meta.schema();
    assert_eq!(
        schema.field(0).data_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
    );
    assert_eq!(
        schema.field(1).data_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, None)
    );
    let out = env
        .scan(&file, &ScanRequest::all(Arc::clone(&schema)), &o)
        .unwrap();
    let got: Vec<i64> = out
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_primitive::<TimestampMicrosecondType>()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(got, millis.iter().map(|v| v * 1000).collect::<Vec<_>>());
    // Row group 2 holds [1 ms, 999_001 ms]: only scaled to microseconds do
    // its bounds keep it (and only it) for this predicate.
    let mut req = ScanRequest::all(Arc::clone(&schema));
    req.predicate = Some(PushdownExpr::And(vec![
        PushdownExpr::cmp(
            ColumnPath::new("ts"),
            CmpOp::GtEq,
            ScalarValue::TimestampMicros(500_000_000),
        ),
        PushdownExpr::cmp(
            ColumnPath::new("ts"),
            CmpOp::Lt,
            ScalarValue::TimestampMicros(600_000_000),
        ),
    ]));
    let plan = ParquetFormat
        .plan_units(meta.as_ref(), &ByteRange::whole(file.size), &req)
        .unwrap();
    assert_eq!(plan.units.iter().map(|u| u.index).collect::<Vec<_>>(), [2]);
    // A millisecond value beyond the microsecond range fails the scan (Spark's
    // `multiplyExact` fails too), never wraps.
    let schema = Arc::new(Schema::new(vec![Field::new(
        "ts",
        DataType::Timestamp(TimeUnit::Millisecond, None),
        false,
    )]));
    let huge = RecordBatch::try_new(
        schema,
        vec![Arc::new(TimestampMillisecondArray::from(vec![
            i64::MAX / 10,
        ]))],
    )
    .unwrap();
    let file = env.put(
        "huge.parquet",
        write(&[huge], WriterProperties::builder().build()),
    );
    let schema = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), &o),
        )
        .unwrap()
        .schema();
    let e = env.scan(&file, &ScanRequest::all(schema), &o).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
}

/// Statistics of timestamp leaves scale by their stored unit, whether the
/// file says it with a logical type or only with a legacy converted type;
/// nanoseconds have no microsecond statistics.
#[test]
fn timestamp_statistics_scale_by_the_stored_unit() {
    use parquet::basic::{ConvertedType, Repetition, Type as PhysicalType};
    use parquet::schema::types::{ColumnDescriptor, ColumnPath as LeafPath, Type};

    let leaf = |logical: Option<parquet::basic::LogicalType>, converted: ConvertedType| {
        let t = Type::primitive_type_builder("t", PhysicalType::INT64)
            .with_repetition(Repetition::REQUIRED)
            .with_logical_type(logical)
            .with_converted_type(converted)
            .build()
            .unwrap();
        ColumnDescriptor::new(Arc::new(t), 0, 0, LeafPath::from("t"))
    };
    let ts = |unit| {
        Some(parquet::basic::LogicalType::Timestamp(
            parquet::basic::TimestampType {
                is_adjusted_to_u_t_c: true,
                unit,
            },
        ))
    };
    let per_unit = super::meta::micros_per_unit;
    assert_eq!(
        per_unit(&leaf(
            ts(parquet::basic::TimeUnit::MILLIS),
            ConvertedType::NONE
        )),
        Some(1000)
    );
    assert_eq!(
        per_unit(&leaf(
            ts(parquet::basic::TimeUnit::NANOS),
            ConvertedType::NONE
        )),
        None
    );
    assert_eq!(
        per_unit(&leaf(None, ConvertedType::TIMESTAMP_MILLIS)),
        Some(1000)
    );
    assert_eq!(
        per_unit(&leaf(None, ConvertedType::TIMESTAMP_MICROS)),
        Some(1)
    );
    assert_eq!(per_unit(&leaf(None, ConvertedType::NONE)), None);
}

/// Splits own row groups by their midpoint: any split points read every row
/// once.
#[test]
fn splits_read_each_row_group_once() {
    let env = Env::new();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(700))
        .build();
    let file = env.put("split.parquet", write(&[ints("k", 0..10_000)], props));
    let o = env.options(&[]);
    let meta = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), &o),
        )
        .unwrap();
    let req = ScanRequest::all(meta.schema());
    for parts in [2u64, 3, 7] {
        let step = file.size.div_ceil(parts);
        let owned: usize = (0..parts)
            .map(|i| {
                let split = ByteRange::new(i * step, ((i + 1) * step).min(file.size));
                ParquetFormat
                    .plan_units(meta.as_ref(), &split, &req)
                    .unwrap()
                    .units
                    .len()
            })
            .sum();
        assert_eq!(owned, 15);
    }
}

/// Unsigned integers read as Spark types them; strings as `Utf8View`.
#[test]
fn unsigned_integers_are_widened_and_strings_are_views() {
    let env = Env::new();
    let schema = Arc::new(Schema::new(vec![
        Field::new("u8", DataType::UInt8, true),
        Field::new("u16", DataType::UInt16, true),
        Field::new("u32", DataType::UInt32, true),
        Field::new("u64", DataType::UInt64, true),
        Field::new("s", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(UInt8Array::from(vec![Some(255), None])) as ArrayRef,
            Arc::new(UInt16Array::from(vec![Some(65_535), Some(1)])),
            Arc::new(UInt32Array::from(vec![Some(u32::MAX), Some(2)])),
            Arc::new(UInt64Array::from(vec![Some(u64::MAX), Some(3)])),
            Arc::new(StringArray::from(vec![Some("a"), None])),
        ],
    )
    .unwrap();
    let file = env.put("u.parquet", write(&[batch], WriterProperties::default()));
    let o = env.options(&[]);
    let meta = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), &o),
        )
        .unwrap();
    let types: Vec<DataType> = meta
        .schema()
        .fields()
        .iter()
        .map(|f| f.data_type().clone())
        .collect();
    assert_eq!(
        types,
        [
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::Decimal128(20, 0),
            DataType::Utf8View
        ]
    );
    // Unsigned statistics are never used.
    let st = meta.statistics();
    assert!(
        st.columns
            .iter()
            .all(|c| c.path.root() == "s" || c.min.exact().is_none())
    );
    let out = env
        .scan(&file, &ScanRequest::all(meta.schema()), &o)
        .unwrap();
    let b = &out[0];
    assert_eq!(b.column(0).as_primitive::<Int16Type>().value(0), 255);
    assert!(b.column(0).is_null(1));
    assert_eq!(
        b.column(2).as_primitive::<Int64Type>().value(0),
        i64::from(u32::MAX)
    );
    assert_eq!(
        b.column(3).as_primitive::<Decimal128Type>().value(0),
        i128::from(u64::MAX)
    );
    assert_eq!(b.column(4).as_string_view().value(0), "a");
}

/// A projection reads only its leaves, in the requested order, with nested
/// pruning; selections and row indexes are file-relative.
#[test]
fn projection_selection_and_row_index() {
    let env = Env::new();
    let inner = arrow_schema::Fields::from(vec![
        Field::new("x", DataType::Int32, true),
        Field::new("y", DataType::Int32, true),
    ]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("st", DataType::Struct(inner.clone()), true),
    ]));
    let n = 3000;
    let a = Int32Array::from_iter_values(0..n);
    let st = arrow_array::StructArray::new(
        inner,
        vec![
            Arc::new(Int32Array::from_iter_values(0..n)) as ArrayRef,
            Arc::new(Int32Array::from_iter_values((0..n).map(|v| -v))) as ArrayRef,
        ],
        None,
    );
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(a), Arc::new(st)]).unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(1000))
        .build();
    let file = env.put("p.parquet", write(&[batch], props));
    let o = env.options(&[]);
    let file_schema = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), &o),
        )
        .unwrap()
        .schema();
    let mut req = ScanRequest::new(
        Arc::clone(&file_schema),
        vec![ColumnPath::nested(["st", "y"]), ColumnPath::new("a")],
    );
    req.selection = Some(RowSelection::from_ranges([10..12, 2500..2502]));
    req.metadata_cols.row_index = true;
    let out = env.scan(&file, &req, &o).unwrap();
    let b = arrow_select::concat::concat_batches(&out[0].schema(), &out).unwrap();
    assert_eq!(b.num_columns(), 3);
    let st = b.column(0).as_struct();
    assert_eq!(st.num_columns(), 1);
    assert_eq!(
        st.column(0).as_primitive::<Int32Type>().values(),
        &[-10, -11, -2500, -2501]
    );
    assert_eq!(
        b.column(1).as_primitive::<Int32Type>().values(),
        &[10, 11, 2500, 2501]
    );
    assert_eq!(
        b.column(2).as_primitive::<Int64Type>().values(),
        &[10, 11, 2500, 2501]
    );
    // A count scan: rows only.
    let count = env
        .scan(&file, &ScanRequest::new(file_schema, Vec::new()), &o)
        .unwrap();
    assert_eq!(rows(&count), 3000);
    assert!(count.iter().all(|b| b.num_columns() == 0));
}

/// Writes one `BYTE_ARRAY (STRING)` column with the given values through
/// the low-level writer (arrow refuses invalid UTF-8 in a string array).
fn raw_strings(values: &[&[u8]]) -> Vec<u8> {
    use parquet::data_type::{ByteArray, ByteArrayType};
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;
    let schema = Arc::new(parse_message_type("message m { required binary s (STRING); }").unwrap());
    let mut buf = Vec::new();
    let mut w = SerializedFileWriter::new(&mut buf, schema, Arc::default()).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = rg.next_column().unwrap().unwrap();
    let data: Vec<ByteArray> = values.iter().map(|v| ByteArray::from(v.to_vec())).collect();
    col.typed::<ByteArrayType>()
        .write_batch(&data, None, None)
        .unwrap();
    col.close().unwrap();
    rg.close().unwrap();
    w.close().unwrap();
    buf
}

/// Owner decision: strings are valid UTF-8. Invalid bytes in a `STRING`
/// column fail the scan with `INVALID_UTF8_STRING`, also when corrupt files
/// are ignored.
#[test]
fn invalid_utf8_in_a_string_column_is_an_error() {
    let env = Env::new();
    let good = env.put("good.parquet", raw_strings(&[b"ok", b"fine"]));
    let bad = env.put("bad.parquet", raw_strings(&[b"ok", b"\xff\xfeA"]));
    let o = env.options(&[]);
    let schema = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &good, MetadataNeeds::default(), &o),
        )
        .unwrap()
        .schema();
    assert_eq!(schema.field(0).data_type(), &DataType::Utf8View);
    let req = ScanRequest::all(schema);
    assert_eq!(rows(&env.scan(&good, &req, &o).unwrap()), 2);
    let e = env.scan(&bad, &req, &o).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InvalidUtf8String);
    assert!(e.to_string().contains("bad.parquet"), "{e}");
}

/// A footer whose column chunk points outside the file is a footer error,
/// not a panic; so are truncated and garbage files and encrypted footers.
#[test]
fn corrupt_footers_are_errors() {
    let env = Env::new();
    let o = env.options(&[]);
    let good = write(&[ints("k", 0..100)], WriterProperties::default());
    let read = |name: &str, bytes: Vec<u8>| {
        let f = env.put(name, bytes);
        env.rt
            .block_on(
                env.io
                    .read_metadata(&ParquetFormat, &f, MetadataNeeds::default(), &o),
            )
            .map(|_| ())
    };
    assert!(read("ok", good.clone()).is_ok());
    for (name, bytes) in [
        ("tiny", b"PAR1".to_vec()),
        ("garbage", vec![7u8; 4096]),
        ("truncated", good[good.len() / 2..].to_vec()),
    ] {
        let e = read(name, bytes).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::FailedReadFileCannotReadFileFooter,
            "{name}: {e}"
        );
    }
    let mut encrypted = good.clone();
    let n = encrypted.len();
    encrypted[n - 4..].copy_from_slice(b"PARE");
    assert_eq!(
        read("enc", encrypted).unwrap_err().error_class(),
        ErrorClass::KrishivUnsupportedFeature
    );
    // Every single-byte change of the footer is an error or a valid read.
    let footer_len = u32::from_le_bytes(good[n - 8..n - 4].try_into().unwrap()) as usize;
    for i in (n - 8 - footer_len..n - 8).step_by(3) {
        let mut b = good.clone();
        b[i] ^= 0x5A;
        let f = env.put("flip", b);
        let r =
            env.rt.block_on(
                env.io
                    .read_metadata(&ParquetFormat, &f, MetadataNeeds::default(), &o),
            );
        if let Ok(m) = r {
            // Reading what such a footer describes is an error or rows.
            let _ = env.scan(&f, &ScanRequest::all(m.schema()), &o);
        }
    }
}

/// Files the `parquet_read` fuzz target flagged: footers whose Thrift lists
/// declare billions of elements in a few bytes, or whose bytes the decoder
/// misreads. The footer guard must reject each for the reason in the
/// fixture's third column, before arrow-rs decodes it (no loop over the
/// declared count, no allocation sized by it), and the read reports a footer
/// error.
#[test]
fn fuzz_artifacts_are_footer_errors_from_the_guard() {
    let env = Env::new();
    let o = env.options(&[]);
    let mut seen = 0;
    for line in include_str!("../../tests/fixtures/fuzz/parquet_read.hex").lines() {
        let mut columns = line.splitn(3, ' ');
        let (Some(name), Some(hex), Some(reason)) = (
            columns.next().filter(|_| !line.starts_with('#')),
            columns.next(),
            columns.next(),
        ) else {
            continue;
        };
        let bytes: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        let n = bytes.len();
        let footer_len = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
        assert_eq!(
            super::thrift_guard::check(&bytes[n - 8 - footer_len..n - 8]),
            Err(reason),
            "{name}"
        );
        let f = env.put(name, bytes);
        let r =
            env.rt.block_on(
                env.io
                    .read_metadata(&ParquetFormat, &f, MetadataNeeds::default(), &o),
            );
        let e = r.map(|_| ()).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::FailedReadFileCannotReadFileFooter,
            "{name}: {e}"
        );
        seen += 1;
    }
    assert_eq!(seen, 7);
}

/// A table field without an id matches its file column by name even when
/// another table field claims the same column by id (both read the value;
/// Spark 4.2's vectorized reader returns 0 for one of them).
#[test]
fn field_id_name_fallback_reads_the_value() {
    let env = Env::new();
    let id = |n: &str, i: &str| {
        Field::new(n, DataType::Int32, true).with_metadata(std::collections::HashMap::from([(
            FIELD_ID_KEY.to_owned(),
            i.to_owned(),
        )]))
    };
    let file_schema: SchemaRef = Arc::new(Schema::new(vec![id("a", "1")]));
    let batch = RecordBatch::try_new(
        Arc::clone(&file_schema),
        vec![Arc::new(Int32Array::from(vec![10, 20]))],
    )
    .unwrap();
    let file = env.put("ids.parquet", write(&[batch], WriterProperties::default()));
    let o = env.options(&[]);
    let table = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        id("x", "1"),
    ]));
    let mut req = ScanRequest::all(table);
    req.mapping = FieldMapping::FieldId;
    let out = env.scan(&file, &req, &o).unwrap();
    assert_eq!(
        out[0].column(0).as_primitive::<Int32Type>().values(),
        &[10, 20]
    );
    assert_eq!(
        out[0].column(1).as_primitive::<Int32Type>().values(),
        &[10, 20]
    );
}

/// `binaryAsString` reads unannotated binary as strings (and changes the
/// cached schema: it is part of the cache key).
#[test]
fn binary_as_string() {
    let env = Env::new();
    let schema = Arc::new(Schema::new(vec![Field::new("b", DataType::Binary, true)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(arrow_array::BinaryArray::from(vec![
            b"hi".as_ref(),
        ]))],
    )
    .unwrap();
    let file = env.put("b.parquet", write(&[batch], WriterProperties::default()));
    let read = |o: &FormatOptions| {
        env.rt
            .block_on(
                env.io
                    .read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), o),
            )
            .unwrap()
            .schema()
            .field(0)
            .data_type()
            .clone()
    };
    assert_eq!(read(&env.options(&[])), DataType::BinaryView);
    let o = env.options(&[("binaryAsString", "true")]);
    assert_eq!(read(&o), DataType::Utf8View);
    let out = env
        .scan(
            &file,
            &ScanRequest::all(Arc::new(Schema::new(vec![Field::new(
                "b",
                DataType::Utf8View,
                true,
            )]))),
            &o,
        )
        .unwrap();
    assert_eq!(out[0].column(0).as_string_view().value(0), "hi");
    // What was read as a string writes back as one.
    let written =
        futures::FutureExt::now_or_never(ParquetFormat.writer(&env.io, out[0].schema(), &o));
    assert!(matches!(written, Some(Ok(_))));
}

/// A fuzzer-found file whose dictionary index points past its dictionary
/// (arrow-rs slices out of bounds): the scan fails with a typed error.
#[test]
fn an_out_of_range_dictionary_index_is_an_error() {
    let env = Env::new();
    let bytes = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/parquet/corrupt/dict-index-out-of-range.parquet"),
    )
    .unwrap();
    let file = env.put("crash.parquet", bytes);
    let o = env.options(&[]);
    let meta = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), &o),
        )
        .unwrap();
    let e = env
        .scan(&file, &ScanRequest::all(meta.schema()), &o)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
    // The decoder is not used again after a contained failure.
    let units = ParquetFormat
        .plan_units(
            meta.as_ref(),
            &ByteRange::whole(file.size),
            &ScanRequest::all(meta.schema()),
        )
        .unwrap();
    let pm = super::downcast(Arc::clone(&meta)).unwrap();
    let mut d =
        super::decode::ParquetDecoder::new(&pm, &units, &ScanRequest::all(meta.schema()), &o)
            .unwrap();
    assert!(format!("{d:?}").contains("ParquetDecoder"));
    let mut failed = false;
    for _ in 0..64 {
        match crate::Decoder::step(&mut d) {
            Ok(crate::Step::NeedData(ranges)) => {
                let data = env.rt.block_on(env.io.get_ranges(&file, &ranges)).unwrap();
                if crate::Decoder::push(&mut d, ranges.into_iter().zip(data).collect()).is_err() {
                    failed = true;
                    break;
                }
            }
            Ok(crate::Step::Done) => break,
            Ok(_) => {}
            Err(_) => {
                failed = true;
                break;
            }
        }
    }
    assert!(failed);
    assert!(
        crate::Decoder::step(&mut d).is_err(),
        "a failed decoder is not reused"
    );
}

/// Nested unsigned integers widen; dictionary and large-list columns read
/// in engine types; a path into a list prunes after decoding.
#[test]
fn nested_types_read_in_engine_types() {
    let env = Env::new();
    let st_fields = arrow_schema::Fields::from(vec![Field::new("u", DataType::UInt32, true)]);
    let st = arrow_array::StructArray::new(
        st_fields.clone(),
        vec![Arc::new(UInt32Array::from(vec![Some(7), None])) as ArrayRef],
        None,
    );
    let item = Arc::new(Field::new("item", DataType::UInt8, true));
    let list = arrow_array::LargeListArray::new(
        Arc::clone(&item),
        arrow_buffer::OffsetBuffer::<i64>::from_lengths([2, 0]),
        Arc::new(UInt8Array::from(vec![1, 2])),
        None,
    );
    let inner = arrow_schema::Fields::from(vec![
        Field::new("x", DataType::Int32, true),
        Field::new("y", DataType::UInt16, true),
    ]);
    let structs = arrow_array::StructArray::new(
        inner.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
            Arc::new(UInt16Array::from(vec![4, 5, 6])) as ArrayRef,
        ],
        None,
    );
    let lst = arrow_array::ListArray::new(
        Arc::new(Field::new("item", DataType::Struct(inner), true)),
        arrow_buffer::OffsetBuffer::from_lengths([1, 2]),
        Arc::new(structs),
        None,
    );
    let dict: arrow_array::DictionaryArray<arrow_array::types::Int32Type> =
        vec!["a", "b"].into_iter().collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("st", DataType::Struct(st_fields), true),
        Field::new("ll", list.data_type().clone(), true),
        Field::new("l", lst.data_type().clone(), true),
        Field::new("d", dict.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(st), Arc::new(list), Arc::new(lst), Arc::new(dict)],
    )
    .unwrap();
    let file = env.put(
        "nested.parquet",
        write(&[batch], WriterProperties::default()),
    );
    let o = env.options(&[]);
    let meta = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), &o),
        )
        .unwrap();
    let fs = meta.schema();
    let DataType::Struct(st) = fs.field(0).data_type() else {
        panic!("st")
    };
    assert_eq!(st[0].data_type(), &DataType::Int64);
    let DataType::LargeList(e) = fs.field(1).data_type() else {
        panic!("ll")
    };
    assert_eq!(e.data_type(), &DataType::Int16);
    assert_eq!(fs.field(3).data_type(), &DataType::Utf8View);
    let out = env
        .scan(&file, &ScanRequest::all(Arc::clone(&fs)), &o)
        .unwrap();
    assert_eq!(
        out[0]
            .column(0)
            .as_struct()
            .column(0)
            .as_primitive::<Int64Type>()
            .value(0),
        7
    );
    assert_eq!(out[0].column(3).as_string_view().value(1), "b");
    // A path into the list of structs reads the whole column, then prunes.
    let req = ScanRequest::new(fs, vec![ColumnPath::nested(["l", "y"])]);
    let out = env.scan(&file, &req, &o).unwrap();
    let values = out[0]
        .column(0)
        .as_list::<i32>()
        .values()
        .as_struct()
        .clone();
    assert_eq!(values.num_columns(), 1);
    assert_eq!(
        values.column(0).as_primitive::<Int32Type>().values(),
        &[4, 5, 6]
    );
}

/// A footer larger than the speculative tail read costs one more read.
#[test]
fn a_large_footer_takes_a_second_read() {
    let env = Env::with_config(IoConfig {
        footer_read_bytes: 8,
        ..IoConfig::default()
    });
    let file = env.put(
        "f.parquet",
        write(&[ints("k", 0..10)], WriterProperties::default()),
    );
    let o = env.options(&[]);
    let m = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), &o),
        )
        .unwrap();
    assert_eq!(m.statistics().rows, crate::Precision::Exact(10));
    let mut bytes = write(&[ints("k", 0..10)], WriterProperties::default());
    let n = bytes.len();
    bytes[n - 8..n - 4].copy_from_slice(&u32::MAX.to_le_bytes());
    let bad = env.put("huge.parquet", bytes);
    let e = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &bad, MetadataNeeds::default(), &o),
        )
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::FailedReadFileCannotReadFileFooter
    );
}

/// Spark's calendar rules: the writer's Spark version decides, then the
/// mode. Values a rebase would change are refused; ambiguous values fail as
/// Spark's EXCEPTION mode does; other values read as stored.
#[test]
fn calendar_rules_follow_spark() {
    use parquet::file::metadata::KeyValue;
    let env = Env::new();
    let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Date32, true)]));
    let file = |name: &str, day: i32, kv: Vec<KeyValue>| {
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(arrow_array::Date32Array::from(vec![day, 10_957]))],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_key_value_metadata((!kv.is_empty()).then_some(kv))
            .build();
        env.put(name, write(&[batch], props))
    };
    let version = |v: &str| {
        vec![KeyValue::new(
            "org.apache.spark.version".into(),
            v.to_owned(),
        )]
    };
    let ancient = -200_000; // before 1582-10-15
    let modern = 0;
    let spark2_old = file("s2o.parquet", ancient, version("2.4.8"));
    let spark2_new = file("s2n.parquet", modern, version("2.4.8"));
    let spark4_old = file("s4o.parquet", ancient, version("4.2.0"));
    let other_old = file("oo.parquet", ancient, Vec::new());
    let req = ScanRequest::all(Arc::clone(&schema));
    let class = |f: &FileRef, mode: &str| {
        let o = env.options(&[("datetimeRebaseMode", mode)]);
        env.scan(f, &req, &o).err().map(|e| e.error_class())
    };
    let unsupported = Some(ErrorClass::KrishivUnsupportedFeature);
    let ancient_class = Some(ErrorClass::InconsistentBehaviorCrossVersionReadAncientDatetime);
    for mode in ["CORRECTED", "EXCEPTION", "LEGACY"] {
        // A Spark 2 file is rebased by Spark in every mode.
        assert_eq!(class(&spark2_old, mode), unsupported, "{mode}");
        assert_eq!(class(&spark2_new, mode), None, "{mode}");
        // A Spark 3+ file is read as stored in every mode.
        assert_eq!(class(&spark4_old, mode), None, "{mode}");
    }
    assert_eq!(class(&other_old, "CORRECTED"), None);
    assert_eq!(class(&other_old, "EXCEPTION"), ancient_class);
    assert_eq!(class(&other_old, "LEGACY"), unsupported);
    // A read without dates checks nothing.
    let o = env.options(&[("datetimeRebaseMode", "LEGACY")]);
    let count = ScanRequest::new(Arc::clone(&schema), Vec::new());
    assert!(env.scan(&spark2_old, &count, &o).is_ok());
}

/// A date conjunct does not prune a file whose dates Spark rebases (a Spark
/// 2 file): Spark rebases the pushed literal to the stored calendar before
/// comparing (`ParquetFilters`, LEGACY spec); Krishiv does not, so the read
/// goes on to the calendar guard instead of trusting stored bounds. A Spark 3+
/// file prunes as usual. (The decoder strips the predicate too, with the read
/// mode; here statistics decide the conjunct first, so this test does not
/// exercise that strip on its own.)
#[test]
fn rebased_dates_do_not_prune() {
    use parquet::file::metadata::KeyValue;
    let env = Env::new();
    let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Date32, true)]));
    let file = |name: &str, version: &str| {
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(arrow_array::Date32Array::from(vec![
                -200_000, 10_957,
            ]))],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_key_value_metadata(Some(vec![KeyValue::new(
                "org.apache.spark.version".into(),
                version.to_owned(),
            )]))
            .build();
        env.put(name, write(&[batch], props))
    };
    let o = env.options(&[]);
    // Outside the stored [min, max]: statistics alone would rule the row group out.
    let mut req = ScanRequest::all(Arc::clone(&schema));
    req.predicate = Some(PushdownExpr::cmp(
        ColumnPath::new("d"),
        CmpOp::Eq,
        ScalarValue::Date32(-300_000),
    ));
    let pruned = |f: &FileRef| {
        let meta = env
            .rt
            .block_on(
                env.io
                    .read_metadata(&ParquetFormat, f, MetadataNeeds::default(), &o),
            )
            .unwrap();
        ParquetFormat
            .plan_units(meta.as_ref(), &ByteRange::whole(f.size), &req)
            .unwrap()
            .pruned_by_statistics
    };
    assert_eq!(pruned(&file("s4.parquet", "4.2.0")), 1);
    let spark2 = file("s2.parquet", "2.4.8");
    assert_eq!(pruned(&spark2), 0);
    // So the read reaches the guard, which refuses the value a rebase would change.
    assert_eq!(
        env.scan(&spark2, &req, &o).err().map(|e| e.error_class()),
        Some(ErrorClass::KrishivUnsupportedFeature)
    );
    // The same predicate as a row-level prefix reaches the guard in both
    // forms. A smoke check only: the row filter is never used while a guarded
    // column is in the output, so the decoder's prefix strip is defensive.
    let mut prefixed = ScanRequest::all(Arc::clone(&schema));
    prefixed.row_level = crate::request::RowLevel::Prefix(req.predicate.clone());
    assert_eq!(
        env.scan(&spark2, &prefixed, &o)
            .err()
            .map(|e| e.error_class()),
        Some(ErrorClass::KrishivUnsupportedFeature)
    );
    // A literal after the switch point compares the same in both calendars:
    // it still prunes a Spark 2 file (2024-10-03, above the stored maximum).
    let mut modern = ScanRequest::all(Arc::clone(&schema));
    modern.predicate = Some(PushdownExpr::cmp(
        ColumnPath::new("d"),
        CmpOp::GtEq,
        ScalarValue::Date32(20_000),
    ));
    let pruned_with = |f: &FileRef, req: &ScanRequest, o: &crate::FormatOptions| {
        let meta = env
            .rt
            .block_on(
                env.io
                    .read_metadata(&ParquetFormat, f, MetadataNeeds::default(), o),
            )
            .unwrap();
        ParquetFormat
            .plan_units(meta.as_ref(), &ByteRange::whole(f.size), req)
            .unwrap()
            .pruned_by_statistics
    };
    assert_eq!(pruned_with(&spark2, &modern, &o), 1);
    // A file not written by Spark is rebased only in LEGACY mode, which statistics pruning
    // sees through the metadata (the mode is part of its cache key).
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(arrow_array::Date32Array::from(vec![
            -200_000, 10_957,
        ]))],
    )
    .unwrap();
    let other = env.put(
        "other.parquet",
        write(&[batch], WriterProperties::default()),
    );
    assert_eq!(pruned_with(&other, &req, &o), 1);
    let legacy = env.options(&[("datetimeRebaseMode", "LEGACY")]);
    assert_eq!(pruned_with(&other, &req, &legacy), 0);
    assert_eq!(pruned_with(&other, &modern, &legacy), 1);
}

/// A runtime filter on a rebased date column is dropped, whatever its keys:
/// its values are in the reader's calendar, the stored bounds in Spark's.
/// Without the drop, a modern key set prunes a Spark 2 file's row group
/// through bounds that mean different days (join rows would go missing).
#[test]
fn runtime_filters_on_rebased_dates_do_not_prune() {
    use crate::{FilterId, FilterSnapshot, KeyFilterBuilder, KeyFilterOptions, RuntimeFilter};
    use parquet::file::metadata::KeyValue;
    let env = Env::new();
    let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Date32, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(arrow_array::Date32Array::from(vec![
            -200_000, 10_957,
        ]))],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_key_value_metadata(Some(vec![KeyValue::new(
            "org.apache.spark.version".into(),
            "2.4.8".to_owned(),
        )]))
        .build();
    let file = env.put("s2-rf.parquet", write(&[batch], props));
    let o = env.options(&[]);
    for key in [-300_000, 20_000] {
        let mut b = KeyFilterBuilder::new(DataType::Date32, KeyFilterOptions::default()).unwrap();
        b.add(&arrow_array::Date32Array::from(vec![key]), None)
            .unwrap();
        let filter = Arc::new(RuntimeFilter::new(
            FilterId(1),
            ColumnPath::new("d"),
            DataType::Date32,
        ));
        filter.publish(FilterSnapshot::Final(Arc::new(b.finish())));
        let mut req = ScanRequest::all(Arc::clone(&schema));
        req.dynamic
            .push(crate::ScanFilter::new(Arc::clone(&filter), &schema).unwrap());
        let meta = env
            .rt
            .block_on(
                env.io
                    .read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), &o),
            )
            .unwrap();
        let plan = ParquetFormat
            .plan_units(meta.as_ref(), &ByteRange::whole(file.size), &req)
            .unwrap();
        assert_eq!(plan.pruned_by_statistics, 0, "key {key}");
        // The read reaches the calendar guard instead of silently pruning.
        assert_eq!(
            env.scan(&file, &req, &o).err().map(|e| e.error_class()),
            Some(ErrorClass::KrishivUnsupportedFeature),
            "key {key}"
        );
    }
}

/// The decoder's prefetch hint stops at the budget, whole row groups only.
#[test]
fn upcoming_ranges_respect_the_budget() {
    let env = Env::new();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(1000))
        .build();
    let file = env.put("u.parquet", write(&[ints("k", 0..5000)], props));
    let o = env.options(&[]);
    let meta = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), &o),
        )
        .unwrap();
    let req = ScanRequest::all(meta.schema());
    let units = ParquetFormat
        .plan_units(meta.as_ref(), &ByteRange::whole(file.size), &req)
        .unwrap();
    let pm = super::downcast(meta).unwrap();
    let d = super::decode::ParquetDecoder::new(&pm, &units, &req, &o).unwrap();
    let all = crate::Decoder::upcoming(&d, u64::MAX);
    assert_eq!(all.len(), 5);
    let one = all[0].end - all[0].start;
    assert_eq!(crate::Decoder::upcoming(&d, one).len(), 1);
    assert!(crate::Decoder::upcoming(&d, one - 1).is_empty());
    assert_eq!(crate::Decoder::buffered_bytes(&d), 0);
}

/// Runtime filters (P2-T163): a filter published before the split opens
/// prunes row groups at planning; one published while the split is read
/// prunes the row groups not started yet at the next boundary. The table
/// types the key `bigint` over an `int` file column (keys compare widened).
#[test]
#[cfg(not(loom))] // Runtime filters use loom's primitives under `--cfg loom`.
fn runtime_filters_prune_row_groups_at_open_and_at_boundaries() {
    use crate::{FilterId, FilterSnapshot, KeyFilterBuilder, KeyFilterOptions, RuntimeFilter};

    let env = Env::new();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(1000))
        .build();
    let bytes = write(&[ints("k", 0..100_000)], props);
    let file = env.put("sorted.parquet", bytes);
    let o = env.options(&[]);
    let table = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true)]));
    let mut b = KeyFilterBuilder::new(DataType::Int64, KeyFilterOptions::default()).unwrap();
    b.add(&arrow_array::Int64Array::from(vec![41_500, 77_000]), None)
        .unwrap();
    let keys = Arc::new(b.finish());
    let request = |filter: &Arc<RuntimeFilter>| {
        let mut req = ScanRequest::all(Arc::clone(&table));
        req.dynamic
            .push(crate::ScanFilter::new(Arc::clone(filter), &table).unwrap());
        req
    };

    // Published before the split opens: two row groups are read.
    let early = Arc::new(RuntimeFilter::new(
        FilterId(1),
        ColumnPath::new("k"),
        DataType::Int64,
    ));
    early.publish(FilterSnapshot::Final(Arc::clone(&keys)));
    let out = env.scan(&file, &request(&early), &o).unwrap();
    assert_eq!(rows(&out), 2000);
    let c = early.stats().counts();
    assert_eq!(c.units_pruned, 98);
    assert!(c.bytes_pruned > 0);

    // Published after the first row group was read: the rest are pruned
    // at the next boundary.
    let late = Arc::new(RuntimeFilter::new(
        FilterId(2),
        ColumnPath::new("k"),
        DataType::Int64,
    ));
    let req = request(&late);
    let read = env.rt.block_on(async {
        let pool: Arc<dyn krishiv_memory::MemoryPool> =
            krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
        let id = pool.register("scan", None);
        let mut s = ScanStream::open(
            &env.io,
            &ParquetFormat,
            file.clone(),
            ByteRange::whole(file.size),
            &req,
            &o,
            krishiv_memory::Reservation::new(pool, id),
            ScanOptions::default(),
        )
        .await
        .unwrap();
        let first = s.next_batch().await.unwrap().unwrap();
        assert_eq!(first.num_rows(), 1000, "row group 0, before the filter");
        late.publish(FilterSnapshot::Final(Arc::clone(&keys)));
        let mut total = first.num_rows();
        while let Some(b) = s.next_batch().await.unwrap() {
            total += b.num_rows();
        }
        (total, s.metrics().units_pruned)
    });
    assert_eq!(read, (3000, 97));
    assert_eq!(late.stats().counts().units_pruned, 97);
}

/// `lateMaterialization` is the session's `krishiv.io.lateMaterialization`
/// unless the read sets it; anything but `auto`, `on` and `off` is refused.
#[test]
fn late_materialization_defaults_to_the_session_setting() {
    let env = Env::new();
    let key = "krishiv.io.lateMaterialization";
    let policy = |session: &[(&str, &str)], read: &[(&str, &str)]| {
        env.options_over(session, read)
            .str("lateMaterialization")
            .map(str::to_owned)
    };
    assert_eq!(policy(&[], &[]).as_deref(), Some("auto"));
    assert_eq!(policy(&[(key, "off")], &[]).as_deref(), Some("off"));
    assert_eq!(policy(&[(key, "on")], &[]).as_deref(), Some("on"));
    assert_eq!(
        policy(&[(key, "off")], &[("lateMaterialization", "on")]).as_deref(),
        Some("on")
    );
    let bad = krishiv_common::config::Config::from_pairs([(key, "sometimes")]);
    assert!(bad.is_err(), "{bad:?}");
}

/// `BINARY` decimals, which arrow-rs reads only as `Decimal128`, reach the query
/// on the canonical carrier: at the top level and inside a list.
#[test]
fn binary_decimals_read_on_the_canonical_carrier() {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Decimal64Type;
    use parquet::data_type::{ByteArray, ByteArrayType};
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;
    let message = "message m {
        required binary f (DECIMAL(10,2));
        optional group l (LIST) {
            repeated group list { optional binary element (DECIMAL(10,2)); }
        }
    }";
    let schema = Arc::new(parse_message_type(message).unwrap());
    let be = |v: i32| ByteArray::from(v.to_be_bytes().to_vec());
    let mut buf = Vec::new();
    let mut w = SerializedFileWriter::new(&mut buf, schema, Arc::default()).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut flat = rg.next_column().unwrap().unwrap();
    flat.typed::<ByteArrayType>()
        .write_batch(&[be(150)], None, None)
        .unwrap();
    flat.close().unwrap();
    let mut nested = rg.next_column().unwrap().unwrap();
    nested
        .typed::<ByteArrayType>()
        .write_batch(&[be(150), be(-275)], Some(&[3, 3]), Some(&[0, 1]))
        .unwrap();
    nested.close().unwrap();
    rg.close().unwrap();
    w.close().unwrap();

    let env = Env::new();
    let file = env.put("binary-decimals.parquet", buf);
    let o = env.options(&[]);
    let d = DataType::Decimal64(10, 2);
    let table: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("f", d.clone(), false),
        Field::new(
            "l",
            DataType::List(Arc::new(Field::new("element", d.clone(), true))),
            true,
        ),
    ]));
    let got = env.scan(&file, &ScanRequest::all(table), &o).unwrap();
    let batch = &got[0];
    assert_eq!(batch.column(0).data_type(), &d);
    assert_eq!(
        batch.column(0).as_primitive::<Decimal64Type>().value(0),
        150
    );
    let list = batch.column(1).as_list::<i32>();
    assert_eq!(list.values().data_type(), &d);
    assert_eq!(
        list.values()
            .as_primitive::<Decimal64Type>()
            .values()
            .to_vec(),
        [150, -275]
    );
}

/// A scan's demand reads (no prefetch budget, so every data range is read
/// when the decoder asks for it) run on the scheduler's I/O runtime, not
/// inline in the caller's poll: a remote response body is never handled on
/// a compute driver (the FileScan remote-poll stalls, queue-opus-b1).
#[test]
fn demand_reads_run_on_the_io_runtime() {
    let io_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .thread_name("scan-test-io")
        .enable_all()
        .build()
        .unwrap();
    let caller = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let inner = Arc::new(InMemory::new());
    let fault = crate::io::FaultStore::new(Arc::clone(&inner) as Arc<dyn ObjectStore>);
    let sched = IoScheduler::with_runtime(IoConfig::default(), io_rt.handle().clone());
    sched.register_store(
        StoreId::new(STORE),
        Arc::clone(&fault) as Arc<dyn ObjectStore>,
    );
    let io = sched.context();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(1000))
        .build();
    let bytes = write(&[ints("k", 0..5000)], props);
    let o = Env::new().options(&[]);
    let batches = std::thread::Builder::new()
        .name("scan-test-caller".into())
        .spawn(move || {
            caller.block_on(async {
                let path = Path::from("demand.parquet");
                object_store::ObjectStoreExt::put(inner.as_ref(), &path, bytes.into())
                    .await
                    .unwrap();
                let meta = object_store::ObjectStoreExt::head(inner.as_ref(), &path)
                    .await
                    .unwrap();
                let file = FileRef::from_meta(StoreId::new(STORE), &meta);
                let meta = io
                    .read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), &o)
                    .await
                    .unwrap();
                let req = ScanRequest::all(meta.schema());
                // The footer is cached now: what follows is the scan's own reads.
                fault.take_get_threads();
                let pool: Arc<dyn krishiv_memory::MemoryPool> =
                    krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
                let id = pool.register("scan", None);
                let options = ScanOptions {
                    prefetch_bytes: 0,
                    ..ScanOptions::default()
                };
                let mut s = ScanStream::open(
                    &io,
                    &ParquetFormat,
                    file.clone(),
                    ByteRange::whole(file.size),
                    &req,
                    &o,
                    krishiv_memory::Reservation::new(pool, id),
                    options,
                )
                .await
                .unwrap();
                let mut out = Vec::new();
                while let Some(b) = s.next_batch().await.unwrap() {
                    out.push(b);
                }
                let threads = fault.take_get_threads();
                assert!(
                    threads.len() >= 5,
                    "one read per row group at least: {threads:?}"
                );
                assert!(
                    threads.iter().all(|t| t.as_deref() == Some("scan-test-io")),
                    "every data read runs on the I/O runtime: {threads:?}"
                );
                out
            })
        })
        .unwrap()
        .join()
        .unwrap();
    assert_eq!(rows(&batches), 5000);
}

/// Nulls at every level of deep nesting (null vs empty lists, null structs
/// inside lists, null elements, a struct in a struct, a map of lists) come
/// back as written. The interop fixtures only nest one level.
#[test]
fn deep_nested_nulls_round_trip() {
    use arrow_array::builder::{Int32Builder, ListBuilder, MapBuilder, StructBuilder};
    let env = Env::new();
    let inner_list = || {
        Field::new(
            "b",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        )
    };
    let elem_fields =
        arrow_schema::Fields::from(vec![Field::new("a", DataType::Int32, true), inner_list()]);
    let sb = StructBuilder::new(
        elem_fields.clone(),
        vec![
            Box::new(Int32Builder::new()),
            Box::new(ListBuilder::new(Int32Builder::new())),
        ],
    );
    let mut lst = ListBuilder::new(sb).with_field(Arc::new(Field::new(
        "item",
        DataType::Struct(elem_fields),
        true,
    )));
    // row 0: null list; row 1: empty; row 2: [null struct, {a:null,b:null}, {a:1,b:[]}, {a:2,b:[null,3]}]
    lst.append(false);
    lst.append(true);
    for (kind, a) in [(0, 0), (1, 0), (2, 1), (3, 2)] {
        let s = lst.values();
        match kind {
            0 => {
                s.field_builder::<Int32Builder>(0).unwrap().append_null();
                s.field_builder::<ListBuilder<Int32Builder>>(1)
                    .unwrap()
                    .append(false);
                s.append(false);
            }
            1 => {
                s.field_builder::<Int32Builder>(0).unwrap().append_null();
                s.field_builder::<ListBuilder<Int32Builder>>(1)
                    .unwrap()
                    .append(false);
                s.append(true);
            }
            2 => {
                s.field_builder::<Int32Builder>(0).unwrap().append_value(a);
                s.field_builder::<ListBuilder<Int32Builder>>(1)
                    .unwrap()
                    .append(true);
                s.append(true);
            }
            _ => {
                s.field_builder::<Int32Builder>(0).unwrap().append_value(a);
                let l = s.field_builder::<ListBuilder<Int32Builder>>(1).unwrap();
                l.values().append_null();
                l.values().append_value(3);
                l.append(true);
                s.append(true);
            }
        }
    }
    lst.append(true);
    lst.append(true);
    let lst = lst.finish();
    // struct<s: struct<i: int32>>: rows: outer null; outer ok inner null; inner ok leaf null; leaf 5.
    let leaf = arrow_schema::Fields::from(vec![Field::new("i", DataType::Int32, true)]);
    let mid = Arc::new(arrow_array::StructArray::new(
        leaf.clone(),
        vec![Arc::new(Int32Array::from(vec![None, None, None, Some(5)])) as ArrayRef],
        Some(vec![false, false, true, true].into()),
    ));
    let outer_f = arrow_schema::Fields::from(vec![Field::new("s", DataType::Struct(leaf), true)]);
    let outer = arrow_array::StructArray::new(
        outer_f,
        vec![mid as ArrayRef],
        Some(vec![false, true, true, true].into()),
    );
    let mut m = MapBuilder::new(
        None,
        Int32Builder::new(),
        ListBuilder::new(Int32Builder::new()),
    );
    m.append(false).unwrap();
    m.append(true).unwrap();
    m.keys().append_value(1);
    m.values().append(false);
    m.keys().append_value(2);
    m.values().values().append_null();
    m.values().values().append_value(9);
    m.values().append(true);
    m.append(true).unwrap();
    m.append(true).unwrap();
    let m = m.finish();
    // Rows: lst has 4 rows (null, empty, 4 elems, empty-after) ; pad others to 4.
    assert_eq!((lst.len(), outer.len(), m.len()), (4, 4, 4));
    let schema = Arc::new(Schema::new(vec![
        Field::new("l", lst.data_type().clone(), true),
        Field::new("o", outer.data_type().clone(), true),
        Field::new("m", m.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(lst), Arc::new(outer), Arc::new(m)],
    )
    .unwrap();
    let file = env.put(
        "deep.parquet",
        write(std::slice::from_ref(&batch), WriterProperties::default()),
    );
    let o = env.options(&[]);
    let meta = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), &o),
        )
        .unwrap();
    let out = env
        .scan(&file, &ScanRequest::all(meta.schema()), &o)
        .unwrap();
    let out = arrow_select::concat::concat_batches(&out[0].schema(), &out).unwrap();
    for c in 0..3 {
        assert_eq!(
            out.column(c).to_data(),
            batch.column(c).to_data(),
            "column {c}"
        );
    }
    // The same rows in row groups of one row keep their nulls too.
    let split = env.put(
        "deep_split.parquet",
        write(
            std::slice::from_ref(&batch),
            WriterProperties::builder()
                .set_max_row_group_row_count(Some(1))
                .build(),
        ),
    );
    let out = env
        .scan(&split, &ScanRequest::all(meta.schema()), &o)
        .unwrap();
    let out = arrow_select::concat::concat_batches(&out[0].schema(), &out).unwrap();
    for c in 0..3 {
        assert_eq!(
            out.column(c).to_data(),
            batch.column(c).to_data(),
            "split column {c}"
        );
    }
    // A row selection inside one page of nested columns (the skipped rows
    // hold the null list, the empty list and a list of structs) keeps
    // the rows that follow aligned.
    for (name, f) in [("one", &file), ("split", &split)] {
        for ranges in [&[(2, 4)][..], &[(0, 1), (3, 4)], &[(1, 3)]] {
            let mut req = ScanRequest::all(meta.schema());
            req.selection = Some(RowSelection::from_ranges(ranges.iter().map(|&(a, b)| a..b)));
            let out = env.scan(f, &req, &o).unwrap();
            let out = arrow_select::concat::concat_batches(&out[0].schema(), &out).unwrap();
            let idx =
                arrow_array::UInt64Array::from_iter_values(ranges.iter().flat_map(|&(a, b)| a..b));
            for c in 0..3 {
                let want = arrow_select::take::take(batch.column(c), &idx, None).unwrap();
                assert_eq!(
                    out.column(c).to_data(),
                    want.to_data(),
                    "{name} {ranges:?} column {c}"
                );
            }
        }
    }
}

/// Predicates over a path into a nullable struct (null struct rows, null
/// leaves, several row groups, row filter on and off) never drop a row the
/// predicate keeps, and `IS NULL` keeps rows whose struct is null.
#[test]
fn nested_path_predicates_keep_every_matching_row() {
    use crate::request::RowFilterPolicy;
    let env = Env::new();
    let n: i32 = 3000;
    let leaf = arrow_schema::Fields::from(vec![Field::new("x", DataType::Int32, true)]);
    // x is null every 7th row; the whole struct is null every 11th row.
    let x = Int32Array::from_iter((0..n).map(|i| (i % 7 != 0).then_some(i)));
    let st = arrow_array::StructArray::new(
        leaf.clone(),
        vec![Arc::new(x) as ArrayRef],
        Some((0..n).map(|i| i % 11 != 0).collect()),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("s", DataType::Struct(leaf), true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from_iter_values(0..n)), Arc::new(st)],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(500))
        .build();
    let file = env.put("np.parquet", write(&[batch], props));
    let o = env.options(&[]);
    let file_schema = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), &o),
        )
        .unwrap()
        .schema();
    let path = || ColumnPath::nested(["s", "x"]);
    let is_null = |i: i32| i % 11 == 0 || i % 7 == 0;
    type Keeps = Box<dyn Fn(i32) -> bool>;
    let cases: Vec<(&str, PushdownExpr, Keeps)> = vec![
        ("is null", PushdownExpr::IsNull(path()), Box::new(is_null)),
        (
            "is not null",
            PushdownExpr::IsNotNull(path()),
            Box::new(move |i| !is_null(i)),
        ),
        (
            "> 2500",
            PushdownExpr::cmp(path(), CmpOp::Gt, ScalarValue::Int32(2500)),
            Box::new(move |i| !is_null(i) && i > 2500),
        ),
        (
            "= 1234",
            PushdownExpr::cmp(path(), CmpOp::Eq, ScalarValue::Int32(1234)),
            Box::new(move |i| !is_null(i) && i == 1234),
        ),
    ];
    for policy in [
        RowFilterPolicy::Off,
        RowFilterPolicy::On,
        RowFilterPolicy::Auto,
    ] {
        for (name, pred, keeps) in &cases {
            let mut req = ScanRequest::new(
                Arc::clone(&file_schema),
                vec![ColumnPath::new("id"), ColumnPath::new("s")],
            );
            req.predicate = Some(pred.clone());
            req.row_filter = policy;
            let out = env.scan(&file, &req, &o).unwrap();
            let b = arrow_select::concat::concat_batches(&out[0].schema(), &out).unwrap();
            let ids = b.column(0).as_primitive::<Int32Type>();
            let got: std::collections::BTreeSet<i32> = ids.values().iter().copied().collect();
            let want: Vec<i32> = (0..n).filter(|i| keeps(*i)).collect();
            assert!(
                want.iter().all(|i| got.contains(i)),
                "{policy:?} {name}: lost {:?}",
                want.iter()
                    .filter(|i| !got.contains(i))
                    .take(5)
                    .collect::<Vec<_>>()
            );
            // Rows kept above the predicate are consistent with the data.
            let s = b.column(1).as_struct();
            for r in 0..b.num_rows() {
                let i = ids.value(r);
                assert_eq!(s.is_null(r), i % 11 == 0, "{policy:?} {name} id {i}");
            }
        }
    }
}
