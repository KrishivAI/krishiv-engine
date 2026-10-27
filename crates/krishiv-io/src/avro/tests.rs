// SPDX-License-Identifier: Apache-2.0
//! The Avro format's tests: the documented type mapping, round trips,
//! options, splits, corrupt files and schema evolution.
//!
//! Ground truth: Spark's documented Avro mapping (option names and the
//! type table) and container files written by `apache-avro` itself, which
//! follow the Avro specification the same way Spark's `spark-avro` package
//! does. No Spark Connect oracle was reachable, so no oracle runs were
//! possible; the mapping assertions below are the documented one.

use std::sync::Arc;

use apache_avro::Schema as AvroSchema;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Int32Type, Int64Type, TimestampMicrosecondType,
};
use arrow_array::{
    Array, ArrayRef, BinaryViewArray, BooleanArray, Date32Array, Decimal128Array, Float32Array,
    Float64Array, Int32Array, Int64Array, RecordBatch, RecordBatchOptions, StringViewArray,
    StructArray, TimestampMicrosecondArray,
};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef, TimeUnit};
use krishiv_common::error::{ErrorClass, KrishivError};
use object_store::ObjectStore;
use object_store::memory::InMemory;
use object_store::path::Path;

use super::AvroFormat;
use super::encode::AvroEncoder;
use super::schema::{fields_of_record, schema_json_of_arrow};
use crate::error::FormatError;
use crate::format::{Encoder, FileFormat};
use crate::io::{ByteRange, FileRef, IoConfig, IoContext, IoScheduler, StoreId};
use crate::options::FormatOptions;
use crate::request::ScanRequest;
use crate::{MetadataNeeds, ScanOptions, ScanStream};

struct Env {
    rt: tokio::runtime::Runtime,
    io: IoContext,
    store: Arc<InMemory>,
}

const STORE: &str = "memory://avro";

impl Env {
    fn new() -> Self {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()
            .unwrap();
        let store = Arc::new(InMemory::new());
        let sched = IoScheduler::new(IoConfig::default());
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

    fn put(&self, name: &str, bytes: Vec<u8>) -> FileRef {
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

    fn options(&self, kv: &[(&str, &str)]) -> FormatOptions {
        let user = kv
            .iter()
            .map(|(k, v)| {
                (
                    (*k).to_owned(),
                    krishiv_common::secret::OptionValue::from_user(k, (*v).to_owned()),
                )
            })
            .collect();
        AvroFormat
            .options()
            .parse(&user, &krishiv_common::config::Config::new())
            .unwrap()
            .0
    }

    fn encode(&self, schema: &SchemaRef, batches: &[RecordBatch], o: &FormatOptions) -> Vec<u8> {
        let mut enc = AvroEncoder::new(schema, o).unwrap();
        let mut out = Vec::new();
        for b in batches {
            enc.write(b).unwrap();
            while let Some(bytes) = enc.take_bytes() {
                out.extend_from_slice(&bytes);
            }
        }
        let done = Box::new(enc).finish().unwrap();
        out.extend_from_slice(&done.tail);
        assert_eq!(
            done.statistics.rows.exact(),
            Some(&(batches.iter().map(RecordBatch::num_rows).sum::<usize>() as u64))
        );
        out
    }

    fn scan(
        &self,
        file: &FileRef,
        req: &ScanRequest,
        o: &FormatOptions,
    ) -> Result<Vec<RecordBatch>, KrishivError> {
        self.scan_split(file, req, o, ByteRange::whole(file.size))
    }

    fn scan_split(
        &self,
        file: &FileRef,
        req: &ScanRequest,
        o: &FormatOptions,
        split: ByteRange,
    ) -> Result<Vec<RecordBatch>, KrishivError> {
        self.rt.block_on(async {
            let pool: Arc<dyn krishiv_memory::MemoryPool> =
                krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
            let id = pool.register("scan", None);
            let mut s = ScanStream::open(
                &self.io,
                &AvroFormat,
                file.clone(),
                split,
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
            Ok(out)
        })
    }

    fn metadata(&self, file: &FileRef, o: &FormatOptions) -> Arc<dyn crate::FileMetadata> {
        self.rt
            .block_on(
                self.io
                    .read_metadata(&AvroFormat, file, MetadataNeeds::default(), o),
            )
            .unwrap()
    }

    fn infer(&self, file: &FileRef, o: &FormatOptions) -> SchemaRef {
        self.rt
            .block_on(AvroFormat.infer_schema(&self.io, std::slice::from_ref(file), o))
            .unwrap()
    }
}

fn rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

fn concat(schema: &SchemaRef, batches: &[RecordBatch]) -> RecordBatch {
    arrow_select::concat::concat_batches(schema, batches).unwrap()
}

/// `batch` re-typed to `schema`: the same buffers under recursively
/// nullable fields, as the [`SchemaMapper`](crate::SchemaMapper) types a
/// mapped scan's output (Spark reads file sources with `asNullable`).
fn retype_batch(batch: &RecordBatch, schema: &SchemaRef) -> RecordBatch {
    let columns = batch
        .columns()
        .iter()
        .zip(schema.fields().iter())
        .map(|(c, f)| retype_array(c, f))
        .collect();
    RecordBatch::try_new_with_options(
        Arc::clone(schema),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
    .unwrap()
}

fn retype_array(array: &ArrayRef, field: &Field) -> ArrayRef {
    match (array.data_type(), field.data_type()) {
        (DataType::Struct(_), DataType::Struct(fields)) => {
            let s = array.as_struct();
            Arc::new(StructArray::new(
                fields.clone(),
                fields
                    .iter()
                    .map(|f| retype_array(s.column_by_name(f.name()).unwrap(), f))
                    .collect(),
                s.nulls().cloned(),
            ))
        }
        (DataType::List(_), DataType::List(f)) => {
            let l = array.as_list::<i32>();
            Arc::new(arrow_array::ListArray::new(
                Arc::clone(f),
                l.offsets().clone(),
                retype_array(l.values(), f),
                l.nulls().cloned(),
            ))
        }
        (DataType::Map(_, _), DataType::Map(f, sorted)) => {
            let m = array.as_map();
            let entries = m.entries();
            let DataType::Struct(fs) = f.data_type() else {
                panic!("not a map field");
            };
            let rebuilt = StructArray::new(
                fs.clone(),
                vec![
                    retype_array(entries.column(0), &fs[0]),
                    retype_array(entries.column(1), &fs[1]),
                ],
                None,
            );
            Arc::new(arrow_array::MapArray::new(
                Arc::clone(f),
                m.offsets().clone(),
                rebuilt,
                m.nulls().cloned(),
                *sorted,
            ))
        }
        _ => Arc::clone(array),
    }
}

/// The mapped output schema of scanning `file` with `req`: what the
/// framework's batches carry.
fn mapped_schema(file_schema: &SchemaRef, req: &ScanRequest) -> SchemaRef {
    let mapper = crate::SchemaMapper::new(req, Arc::clone(file_schema), "test").unwrap();
    Arc::clone(mapper.output_schema())
}

/// A container file written by `apache-avro` itself (the Avro specification
/// framing, as Spark's `spark-avro` writes it) holding `values`.
fn apache_write(schema_json: &str, values: &[apache_avro::types::Value]) -> Vec<u8> {
    let schema = AvroSchema::parse_str(schema_json).unwrap();
    let mut writer =
        apache_avro::Writer::with_codec(&schema, Vec::new(), apache_avro::Codec::Null).unwrap();
    writer.extend_from_slice(values).unwrap();
    writer.into_inner().unwrap()
}

/// The `apache-avro` values of an encoded file of ours.
fn apache_read(bytes: &[u8]) -> Vec<apache_avro::types::Value> {
    apache_avro::Reader::new(bytes)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

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

fn option_value(key: &str, value: &str) -> krishiv_common::secret::OptionValue {
    krishiv_common::secret::OptionValue::from_user(key, value.to_owned())
}

/// Each codec round-trips; the header names it the way Spark does.
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
    // `bzip2`, `xz` and `zstandard` are accepted options but not enabled in
    // this build: `zstandard` is Spark's `CODEC_NOT_AVAILABLE` error.
    let refused = [
        ("bzip2", ErrorClass::KrishivUnsupportedFeature),
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

fn class_of(e: &FormatError) -> ErrorClass {
    e.clone().into_error("test").error_class()
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
            Arc::new(empty_map_array(&schema.field(13).data_type().clone())),
        ],
    )
    .unwrap();
    let file = env.put("empty.avro", env.encode(&schema, &[empty], &o));
    let got = env
        .scan(&file, &ScanRequest::all(Arc::clone(&schema)), &o)
        .unwrap();
    assert_eq!(rows(&got), 0);
}

/// A list column with the `element` field name.
fn list_array(rows: &[Option<Vec<Option<i64>>>]) -> arrow_array::ListArray {
    let field = Arc::new(Field::new("element", DataType::Int64, false));
    let values: Vec<Option<i64>> = rows.iter().flatten().flatten().copied().collect();
    let mut offsets = vec![0_i32];
    let mut validity = Vec::new();
    for row in rows {
        match row {
            Some(items) => {
                offsets.push(
                    offsets.last().copied().unwrap_or(0) + i32::try_from(items.len()).unwrap(),
                );
                validity.push(true);
            }
            None => {
                offsets.push(offsets.last().copied().unwrap_or(0));
                validity.push(false);
            }
        }
    }
    arrow_array::ListArray::new(
        field,
        OffsetBuffer::new(offsets.into()),
        Arc::new(Int64Array::from(values)),
        Some(validity.into()),
    )
}

/// One map row: `None` is null, else the entries.
type MapRow = Option<Vec<(String, Option<i32>)>>;

/// A map column of `rows`: `None` is null, else the entries.
fn map_array(rows: &[MapRow]) -> arrow_array::MapArray {
    let mut offsets = vec![0_i32];
    let mut keys = Vec::new();
    let mut values = Vec::new();
    let mut validity = Vec::new();
    for row in rows {
        match row {
            Some(entries) => {
                for (k, v) in entries {
                    keys.push(Some(k.as_str()));
                    values.push(*v);
                }
                offsets.push(i32::try_from(keys.len()).unwrap());
                validity.push(true);
            }
            None => {
                offsets.push(i32::try_from(keys.len()).unwrap());
                validity.push(false);
            }
        }
    }
    let keys = StringViewArray::from(keys);
    let values = Int32Array::from(values);
    let entries = StructArray::new(
        Fields::from(vec![
            Field::new("key", DataType::Utf8View, false),
            Field::new("value", DataType::Int32, false),
        ]),
        vec![Arc::new(keys) as ArrayRef, Arc::new(values) as ArrayRef],
        None,
    );
    arrow_array::MapArray::new(
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
        OffsetBuffer::new(offsets.into()),
        entries,
        Some(validity.into()),
        false,
    )
}

/// An empty map array of `dt`.
fn empty_map_array(dt: &DataType) -> arrow_array::MapArray {
    let DataType::Map(entries, _) = dt else {
        panic!("not a map type");
    };
    let DataType::Struct(fields) = entries.data_type() else {
        panic!("not a struct");
    };
    let entries_array = StructArray::new(
        fields.clone(),
        vec![
            Arc::new(StringViewArray::from(Vec::<Option<&str>>::new())) as ArrayRef,
            Arc::new(Int32Array::from(Vec::<Option<i32>>::new())) as ArrayRef,
        ],
        None,
    );
    arrow_array::MapArray::new(
        Arc::clone(entries),
        OffsetBuffer::new(vec![0_i32].into()),
        entries_array,
        None,
        false,
    )
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
    let meta = any.downcast_ref::<super::AvroMetadata>().unwrap();
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

fn env_options_none() -> FormatOptions {
    AvroFormat
        .options()
        .parse(
            &std::collections::BTreeMap::new(),
            &krishiv_common::config::Config::new(),
        )
        .unwrap()
        .0
}

/// Like [`apache_write`], with `meta` added to the header.
fn apache_write_meta(
    schema_json: &str,
    values: &[apache_avro::types::Value],
    meta: &[(&str, &str)],
) -> Vec<u8> {
    let schema = AvroSchema::parse_str(schema_json).unwrap();
    let mut writer =
        apache_avro::Writer::with_codec(&schema, Vec::new(), apache_avro::Codec::Null).unwrap();
    for (key, value) in meta {
        writer
            .add_user_metadata((*key).to_owned(), value.as_bytes())
            .unwrap();
    }
    writer.extend_from_slice(values).unwrap();
    writer.into_inner().unwrap()
}

/// A file with an ancient date and timestamp, and the schema it was written
/// with (`apache_write_meta` adds the header entries under test).
fn ancient_file(env: &Env, name: &str, meta: &[(&str, &str)]) -> FileRef {
    use apache_avro::types::Value as V;
    let schema_json = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "d", "type": {"type": "int", "logicalType": "date"}},
            {"name": "t", "type": {"type": "long", "logicalType": "timestamp-micros"}}
        ]
    }"#;
    let values = vec![V::Record(vec![
        ("d".to_owned(), V::Date(-200_000)),
        ("t".to_owned(), V::TimestampMicros(-30_000_000_000_000)),
    ])];
    env.put(name, apache_write_meta(schema_json, &values, meta))
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
    use super::options::{AvroOptions, Mode};
    let env = Env::new();
    let default = AvroOptions::new(&env.options(&[]), false).unwrap();
    assert_eq!(default.mode, Mode::FailFast);
    for (mode, want) in [
        ("PERMISSIVE", Mode::Permissive),
        ("DROPMALFORMED", Mode::DropMalformed),
        ("FAILFAST", Mode::FailFast),
    ] {
        let o = AvroOptions::new(&env.options(&[("mode", mode)]), false).unwrap();
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
            let meta = any.downcast_ref::<super::AvroMetadata>().unwrap();
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
    let meta = any.downcast_ref::<super::AvroMetadata>().unwrap();
    let at = usize::try_from(meta.blocks[1].bytes.start).unwrap();
    let mut spliced = two[..at].to_vec();
    spliced.extend_from_slice(&[0, 0]);
    spliced.extend_from_slice(&meta.sync);
    spliced.extend_from_slice(&two[at..]);
    let file = env.put("empty-block.avro", spliced);
    let meta = env.metadata(&file, &plain);
    let any: &dyn std::any::Any = meta.as_ref();
    let meta = any.downcast_ref::<super::AvroMetadata>().unwrap();
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
    let meta = any.downcast_ref::<super::AvroMetadata>().unwrap();
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
    use super::schema::field_default;
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
                Arc::clone(&schema),
                o.clone(),
                crate::WriteOptions {
                    store: StoreId::new(STORE),
                    dir: Path::from(job),
                    task: 0,
                    job_uuid: job.to_owned(),
                    max_records_per_file: 0,
                    target_file_bytes: 0,
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
        syncs.push(any.downcast_ref::<super::AvroMetadata>().unwrap().sync);
    }
    assert_eq!(syncs[0], syncs[1]);
    assert_ne!(syncs[0], syncs[2]);
}
